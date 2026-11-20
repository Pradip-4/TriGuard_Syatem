"""
Face authentication module.

This version follows an attendance-style workflow:
- detect faces in each captured frame with MTCNN,
- extract ArcFace embeddings with DeepFace,
- match each embedding directly against all enrolled student embeddings (Top 5 K-NN),
- log recognized IDs per processed frame and vote across processed frames.

It still plugs into the shared TriGuard capture/orchestration flow.
"""

import logging
import math
import os
import pickle
import importlib
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.base_module import BaseModule, ModuleResult

logger = logging.getLogger(__name__)


def preload_face_runtime() -> None:
    """
    Dynamically patches the TensorFlow/Keras environment to mock missing layers 
    and bridge legacy properties (like KerasHistory.layer) required by DeepFace.
    """
    # 1. Patch missing LocallyConnected2D layer in Keras layers namespace
    try:
        layers_mod = importlib.import_module("tensorflow.keras.layers")
        if not hasattr(layers_mod, "LocallyConnected2D"):
            base_layer = getattr(layers_mod, "Layer")

            class LocallyConnected2D(base_layer):
                def __init__(self, *args, **kwargs):
                    super().__init__(**kwargs)

                def call(self, inputs):
                    return inputs

            setattr(layers_mod, "LocallyConnected2D", LocallyConnected2D)

        input_layer_cls = getattr(layers_mod, "InputLayer", None)
        if input_layer_cls is not None and not hasattr(input_layer_cls, "is_placeholder"):
            setattr(input_layer_cls, "is_placeholder", property(lambda self: False))
    except Exception as e:
        logger.warning(f"Could not apply structural Keras layers patch: {e}")

    # 2. Comprehensive deep-patch for KerasHistory / Node across Keras 3 path structures
    candidate_paths = ["keras.src.ops.node", "keras.ops.node", "keras.src.backend.tensorflow.layer"]
    for path in candidate_paths:
        try:
            mod = importlib.import_module(path)
            
            keras_history = getattr(mod, "KerasHistory", None)
            if keras_history is not None and not hasattr(keras_history, "layer"):
                setattr(keras_history, "layer", property(lambda self: getattr(self, "operation", None)))
                
            node_cls = getattr(mod, "Node", None)
            if node_cls is not None:
                if not hasattr(node_cls, "keras_inputs"):
                    setattr(node_cls, "keras_inputs", property(lambda self: getattr(self, "input_tensors", None)))
                if not hasattr(node_cls, "layer"):
                    setattr(node_cls, "layer", property(lambda self: getattr(self, "operation", None)))
        except Exception:
            pass

    # 3. Dynamic runtime sweeping fallback to catch any hidden or aliased copies of KerasHistory
    for mod_name, module in list(sys.modules.items()):
        if module and ("keras" in mod_name or "tensorflow" in mod_name):
            try:
                kh = getattr(module, "KerasHistory", None)
                if kh is not None and not hasattr(kh, "layer"):
                    try:
                        setattr(kh, "layer", property(lambda self: getattr(self, "operation", None)))
                    except Exception:
                        pass
                nc = getattr(module, "Node", None)
                if nc is not None:
                    if not hasattr(nc, "keras_inputs"):
                        try: setattr(nc, "keras_inputs", property(lambda self: getattr(self, "input_tensors", None)))
                        except Exception: pass
                    if not hasattr(nc, "layer"):
                        try: setattr(nc, "layer", property(lambda self: getattr(self, "operation", None)))
                        except Exception: pass
            except Exception:
                pass

    # Warm up dependencies into active memory
    try:
        importlib.import_module("mtcnn")
        importlib.import_module("deepface")
    except Exception:
        pass


class FaceModule(BaseModule):
    def __init__(self, module_config: Dict[str, Any]):
        super().__init__(module_config)

        self.embeddings_path: str = module_config["embeddings_path"]
        self.distance_threshold: float = module_config.get("distance_threshold", 4.1)
        self.detection_threshold: float = module_config.get("detection_threshold", 0.92)
        self.max_faces_allowed: int = module_config.get("max_faces_allowed", 1)
        self.model_name: str = module_config.get("model_name", "ArcFace")
        self.frame_skip: int = max(1, int(module_config.get("frame_skip", 15)))
        self.top_k_votes: int = module_config.get("top_k_votes", 5)

        self._flat_db: List[Tuple[str, np.ndarray]] = []
        self._detector = None

        self._load_resources()

    def _load_resources(self) -> None:
        preload_face_runtime()

        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(
                f"Face embeddings not found at '{self.embeddings_path}'. "
                "Run your enrollment script to generate the .pkl file."
            )

        with open(self.embeddings_path, "rb") as f:
            raw_db = pickle.load(f)

        self._flat_db = []
        identities_count = 0
        
        # Flatten the database into a list of (student_id, embedding) tuples
        for student_id, stored_data in raw_db.items():
            identities_count += 1
            if isinstance(stored_data, list):
                for emb in stored_data:
                    self._flat_db.append((student_id, np.asarray(emb, dtype=np.float32)))
            else:
                self._flat_db.append((student_id, np.asarray(stored_data, dtype=np.float32)))

        logger.info(
            f"[Face] Loaded {len(self._flat_db)} individual embeddings across "
            f"{identities_count} identities from '{self.embeddings_path}'."
        )

        try:
            from mtcnn import MTCNN
            self._detector = MTCNN()
            logger.info("[Face] MTCNN detector initialized.")
        except ImportError as e:
            raise RuntimeError(
                "Face module could not initialize MTCNN/TensorFlow."
            ) from e

        # Pre-build the face recognition model (ArcFace) so the first live
        # session does not pay the one-time model construction + weight load
        # (~10-15 s on CPU). DeepFace caches it process-wide, so this cost is
        # paid once at startup, exactly like the voice and lip module warm-ups.
        try:
            from deepface import DeepFace

            DeepFace.build_model(self.model_name)
            logger.info(f"[Face] {self.model_name} model pre-built (cached for sessions).")
        except Exception as warm_err:
            logger.warning(f"[Face] Model warm-up failed, will build lazily: {warm_err}")

    def run(
        self,
        frames: Optional[List[np.ndarray]] = None,
        face_abort: bool = False,
        **kwargs,
    ) -> ModuleResult:
        status_callback = kwargs.get("module_status_callback")

        if face_abort:
            self.emit_status(status_callback, "aborted", "Capture aborted due to multiple faces.")
            logger.warning("[Face] Hard abort received - multiple faces detected during capture.")
            return self.build_result(
                passed=False,
                score=0.0,
                identity="unknown",
                details={"reason": "multi_face_abort_from_capture_session"},
            )

        if not frames:
            self.emit_status(status_callback, "error", "No frames provided to face module.")
            logger.error("[Face] No frames provided.")
            return self.build_result(passed=False, score=0.0, error="No frames provided.")

        self.emit_status(
            status_callback,
            "running",
            "Face analysis started.",
            frames_total=len(frames),
            frame_skip=self.frame_skip,
        )
        logger.info(f"[Face] Processing {len(frames)} tracking frame(s).")

        try:
            return self._process_frames(frames, status_callback)
        except Exception as e:
            self.emit_status(status_callback, "error", f"Face analysis failed: {e}")
            logger.exception(f"[Face] Unexpected error: {e}")
            return self.build_result(passed=False, score=0.0, error=str(e))

    def _process_frames(self, frames: List[np.ndarray], status_callback=None) -> ModuleResult:
        preload_face_runtime()
        from deepface import DeepFace

        recognized_events: List[Dict[str, Any]] = []
        skipped_no_face = 0
        skipped_multi_face = 0
        detection_confidences: List[float] = []
        processed_frame_count = 0

        for idx, frame in enumerate(frames):
            if idx % self.frame_skip != 0:
                continue

            processed_frame_count += 1
            # OpenCV capture frames are BGR. MTCNN and DeepFace are given RGB,
            # matching the standalone face-testing workflow.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self._detector.detect_faces(rgb_frame)

            if len(faces) == 0:
                skipped_no_face += 1
                logger.debug(f"[Face] Frame {idx}: no face detected.")
                continue

            if len(faces) > self.max_faces_allowed:
                skipped_multi_face += 1
                logger.warning(
                    f"[Face] Frame {idx}: {len(faces)} faces detected - "
                    f"frame skipped (max allowed: {self.max_faces_allowed})."
                )
                continue

            for face_data in faces:
                conf = float(face_data.get("confidence", 0.0))

                # A confident face detection is required before attempting an
                # ArcFace embedding, exactly as in testing/face_testing.py.
                if conf < self.detection_threshold:
                    logger.debug(
                        "[Face] Frame %s: detection confidence %.3f below threshold %.3f.",
                        idx,
                        conf,
                        self.detection_threshold,
                    )
                    continue

                detection_confidences.append(conf)

                crop = self._extract_crop(rgb_frame, face_data.get("box"))
                if crop is None:
                    continue

                try:
                    embedding = DeepFace.represent(
                        crop,
                        model_name=self.model_name,
                        enforce_detection=False,
                    )[0]["embedding"]
                except Exception as e:
                    logger.warning(f"[Face] Frame {idx}: embedding failed - {e}.")
                    continue

                recognized_id, min_dist = self._match_embedding(
                    np.asarray(embedding, dtype=np.float32)
                )

                if recognized_id:
                    pass_reason = "distance_within_limit"
                    recognized_events.append(
                        {
                            "StudentID": recognized_id,
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "FrameIndex": idx,
                            "Distance": round(min_dist, 4),
                            "DetectionConfidence": round(conf, 4),
                            "Score": round(
                                self._dist_to_score(min_dist),
                                4,
                            ),
                            "PassReason": pass_reason,
                        }
                    )
                    self.emit_status(
                        status_callback,
                        "running",
                        f"Face frame {idx} matched '{recognized_id}' ({pass_reason}).",
                        frame_index=idx,
                        recognized_id=recognized_id,
                        distance=min_dist,
                        confidence=conf,
                        reason=pass_reason,
                    )
                else:
                    logger.info(
                        f"[Face] Frame {idx}: no valid match "
                        f"(best='{recognized_id}', dist={min_dist:.4f}, conf={conf:.3f})."
                    )

        if not recognized_events:
            reason = "all_frames_multi_face" if skipped_multi_face > 0 else "no_students_recognized"
            self.emit_status(
                status_callback,
                "finished",
                "Face analysis completed with no recognized identity.",
                frames_processed=processed_frame_count,
            )
            return self.build_result(
                passed=False,
                score=0.0,
                details={
                    "reason": reason,
                    "frames_total": len(frames),
                    "frames_processed": processed_frame_count,
                    "skipped_no_face": skipped_no_face,
                    "skipped_multi_face": skipped_multi_face,
                },
            )

        counts = Counter(event["StudentID"] for event in recognized_events)
        dists_by_id: Dict[str, List[float]] = defaultdict(list)
        confs_by_id: Dict[str, List[float]] = defaultdict(list)
        
        for event in recognized_events:
            dists_by_id[event["StudentID"]].append(float(event["Distance"]))
            confs_by_id[event["StudentID"]].append(float(event["DetectionConfidence"]))

        best_id = max(
            counts,
            key=lambda student_id: (counts[student_id], -float(np.mean(dists_by_id[student_id]))),
        )
        best_dist = float(np.mean(dists_by_id[best_id]))
        best_conf = float(np.mean(confs_by_id[best_id]))
        score = self._dist_to_score(best_dist)
        passed = best_dist <= self.distance_threshold

        avg_conf = float(np.mean(detection_confidences)) if detection_confidences else 0.0

        self.emit_status(
            status_callback,
            "finished",
            f"Face analysis complete. Final identity: {best_id}.",
            passed=passed,
            identity=best_id,
            score=score,
            votes=dict(counts),
        )

        return self.build_result(
            passed=passed,
            score=score,
            identity=best_id if passed else "unknown",
            details={
                "matched_id": best_id,
                "distance": round(best_dist, 4),
                "confidence": round(best_conf, 4),
                "threshold": self.distance_threshold,
                "detection_threshold": self.detection_threshold,
                "frames_total": len(frames),
                "frames_processed": processed_frame_count,
                "recognized_events": len(recognized_events),
                "recognized_students": dict(counts),
                "skipped_no_face": skipped_no_face,
                "skipped_multi_face": skipped_multi_face,
                "avg_detection_confidence": round(avg_conf, 4),
                "frame_skip": self.frame_skip,
            },
        )

    def _extract_crop(self, frame: np.ndarray, box: Optional[List[int]]) -> Optional[np.ndarray]:
        if not box or len(box) != 4:
            return None

        x, y, w, h = box
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(1, int(w))
        h = max(1, int(h))

        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        crop = frame[y:y2, x:x2]
        if crop.size == 0:
            return None
        return crop

    def _match_embedding(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        if not self._flat_db:
            return None, float("inf")

        distances = []
        for student_id, stored_emb in self._flat_db:
            dist = float(np.linalg.norm(embedding - stored_emb))
            distances.append((dist, student_id))

        distances.sort(key=lambda x: x[0])
        top_k_matches = distances[:self.top_k_votes]
        top_k_ids = [match[1] for match in top_k_matches]
        votes = Counter(top_k_ids)
        
        best_id = votes.most_common(1)[0][0]
        best_dist = min([match[0] for match in top_k_matches if match[1] == best_id])

        # Use the ArcFace L2 distance as the identity decision. MTCNN's
        # confidence only establishes that a face was detected, not who it is.
        if best_dist > self.distance_threshold:
            return None, best_dist

        return best_id, best_dist

    def _dist_to_score(self, distance: float) -> float:
        k = 10.0 / max(self.distance_threshold, 1e-6)
        return round(1.0 / (1.0 + math.exp(k * (distance - self.distance_threshold))), 6)
