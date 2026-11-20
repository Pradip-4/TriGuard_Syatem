import cv2
import logging
import math
import os
import pickle
import importlib
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# ---------------------------------------------------------
# DUMMY CORE CLASSES (To make the script standalone)
# ---------------------------------------------------------
class ModuleResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class BaseModule:
    def __init__(self, module_config: Dict[str, Any]):
        self.config = module_config
        
    def emit_status(self, *args, **kwargs):
        pass
        
    def build_result(self, **kwargs):
        return ModuleResult(**kwargs)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------
# FACE MODULE CONFIGURATION
# ---------------------------------------------------------
def preload_face_runtime() -> None:
    """
    Dynamically patches the TensorFlow/Keras environment to mock missing layers 
    and bridge legacy properties (like KerasHistory.layer) required by DeepFace.
    """
    try:
        layers_mod = importlib.import_module("tensorflow.keras.layers")
        if not hasattr(layers_mod, "LocallyConnected2D"):
            base_layer = getattr(layers_mod, "Layer")

            class LocallyConnected2D(base_layer):
                def __init__(self, *args, **kwargs): super().__init__(**kwargs)
                def call(self, inputs): return inputs

            setattr(layers_mod, "LocallyConnected2D", LocallyConnected2D)

        input_layer_cls = getattr(layers_mod, "InputLayer", None)
        if input_layer_cls is not None and not hasattr(input_layer_cls, "is_placeholder"):
            setattr(input_layer_cls, "is_placeholder", property(lambda self: False))
    except Exception as e:
        logger.warning(f"Could not apply structural Keras layers patch: {e}")

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

    for mod_name, module in list(sys.modules.items()):
        if module and ("keras" in mod_name or "tensorflow" in mod_name):
            try:
                kh = getattr(module, "KerasHistory", None)
                if kh is not None and not hasattr(kh, "layer"):
                    try: setattr(kh, "layer", property(lambda self: getattr(self, "operation", None)))
                    except Exception: pass
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

    try:
        importlib.import_module("mtcnn")
        importlib.import_module("deepface")
    except Exception:
        pass


class FaceModule(BaseModule):
    def __init__(self, module_config: Dict[str, Any]):
        super().__init__(module_config)
        self.embeddings_path: str = module_config["embeddings_path"]
        
        # LOWER values mean MORE STRICT face matching. For ArcFace L2 distance:
        # - 4.0 to 4.2 is ideal (very strict, blocks unknowns cleanly)
        # - 4.5+ starts accepting lookalikes or closer unknown faces
        self.distance_threshold: float = module_config.get("distance_threshold", 4.1)
        self.detection_threshold: float = module_config.get("detection_threshold", 0.90)
        self.model_name: str = module_config.get("model_name", "ArcFace")
        self.top_k_votes: int = module_config.get("top_k_votes", 5)

        self._flat_db: List[Tuple[str, np.ndarray]] = []
        self._detector = None
        self._load_resources()

    def _load_resources(self) -> None:
        preload_face_runtime()
        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(f"Database not found at '{self.embeddings_path}'.")

        with open(self.embeddings_path, "rb") as f:
            raw_db = pickle.load(f)

        self._flat_db = []
        identities_count = 0

        for student_id, stored_data in raw_db.items():
            identities_count += 1
            if isinstance(stored_data, list):
                for emb in stored_data:
                    self._flat_db.append((student_id, np.asarray(emb, dtype=np.float32)))
            else:
                self._flat_db.append((student_id, np.asarray(stored_data, dtype=np.float32)))

        logger.info(
            f"[Face] Loaded {len(self._flat_db)} raw embeddings across "
            f"{identities_count} distinct identities."
        )

        try:
            from mtcnn import MTCNN
            self._detector = MTCNN()
            logger.info("[Face] MTCNN detector initialized.")
        except ImportError as e:
            raise RuntimeError("Failed to load MTCNN.") from e

    def _extract_crop(self, frame: np.ndarray, box: Optional[List[int]]) -> Optional[np.ndarray]:
        if not box or len(box) != 4: return None
        x, y, w, h = [max(0, int(v)) for v in box]
        w, h = max(1, w), max(1, h)
        
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        crop = frame[y:y2, x:x2]
        return crop if crop.size > 0 else None

    def _match_embedding(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Calculates distances against every individual embedding array 
        and validates if the absolute closest match meets the strict threshold ceiling.
        """
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

        # CRITICAL FIX: If the closest variant is past our distance barrier, strip the identity
        if best_dist > self.distance_threshold:
            return None, best_dist

        return best_id, best_dist

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    config = {
        "embeddings_path": "embeddings.pkl", 
        "distance_threshold": 4.1,         # Strict threshold setting for ArcFace L2 Vector metrics
        "detection_threshold": 0.92,        # Early bounding gate check for MTCNN tracking quality
        "model_name": "ArcFace",
        "top_k_votes": 5
    }

    try:
        face_module = FaceModule(config)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    preload_face_runtime()
    from deepface import DeepFace
    
    cap = cv2.VideoCapture(0)
    logger.info("Starting live camera feed. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_module._detector.detect_faces(rgb_frame)

        for face_data in faces:
            box = face_data.get("box")
            conf = face_data.get("confidence", 0.0)
            if not box or len(box) != 4: continue

            # Gatekeeper Check: Drop the detection array if MTCNN is guessing or tracking noise
            if conf < config["detection_threshold"]:
                continue

            x, y, w, h = box
            crop = face_module._extract_crop(rgb_frame, box)
            identity_label = "Unknown"
            color = (0, 0, 255) # Default Red for strangers/unregistered faces

            if crop is not None:
                try:
                    embedding_data = DeepFace.represent(
                        crop, model_name=face_module.model_name, enforce_detection=False
                    )
                    embedding = embedding_data[0]["embedding"]
                    
                    best_id, min_dist = face_module._match_embedding(np.asarray(embedding, dtype=np.float32))
                    
                    # LOGIC FIX: The "or conf >= override" logic is eliminated. 
                    # Identification relies purely on authentic vector similarity boundaries now.
                    if best_id:
                        identity_label = f"{best_id} (Dist: {min_dist:.1f})"
                        color = (0, 255, 0) # Green for authorized matching status
                    else:
                        identity_label = f"Unknown (Dist: {min_dist:.1f})"

                except Exception:
                    pass

            # Draw Visual Bounding HUD elements
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(identity_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
            cv2.putText(frame, identity_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Live Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()