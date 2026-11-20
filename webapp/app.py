"""
webapp/app.py — TriGuard web console (local, single-user).

Runs the real TriGuard pipeline (CaptureSession + AuthOrchestrator) behind a
clean browser UI. Modules are preloaded once at startup so every auth session
after that reuses the loaded models.

Endpoints
---------
GET  /                   → the single-page UI
GET  /api/preview.mjpg   → MJPEG live preview (idle + during capture)
POST /api/start          → start an authentication session (background thread)
GET  /api/events         → Server-Sent Events stream (stages, countdown,
                           module status, result with timings)
GET  /api/status         → JSON snapshot of current state

Run from the project root:   python webapp/app.py   (then open http://127.0.0.1:5000)
"""

import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime

# Windows consoles often default to cp1252; the TriGuard capture pipeline prints
# box-drawing / Unicode glyphs (●, ✓, frames). Reconfigure stdout/stderr to UTF-8
# with replacement so a non-ASCII glyph can never crash a running session.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # system_main/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from flask import Flask, Response, jsonify, request  # noqa: E402

from core.capture_session import CaptureSession  # noqa: E402
from core.config_loader import load_config  # noqa: E402
from core.logger import setup_logger  # noqa: E402
from core.orchestrator import AuthOrchestrator  # noqa: E402
from main import build_modules  # noqa: E402

logger = logging.getLogger("webapp")
setup_logger(log_dir="logs/", log_level="INFO")

app = Flask(__name__, static_folder="static", static_url_path="/static")
PORT = int(os.environ.get("TRIGUARD_PORT", "5000"))
CONFIG_PATH = os.environ.get("TRIGUARD_CONFIG", "config.json")
CAMERA_INDEX = int(os.environ.get("TRIGUARD_CAMERA", "-1"))  # -1 → use config


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "loading"            # loading | ready | running
        self.config = None
        self.modules = None
        self.result = None
        self.timings = {}
        self.session_history = []          # last N session results (persistent)
        self.queue = queue.Queue(maxsize=1000)
        self.history = []                  # last events, replayed on connect
        self.prewarm_done = threading.Event()

    def emit(self, event: dict) -> None:
        event = dict(event)
        msg = json.dumps(event, ensure_ascii=False, default=str)
        self.history.append(event)
        if len(self.history) > 400:
            self.history = self.history[-400:]
        try:
            self.queue.put_nowait(msg)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(msg)
            except queue.Empty:
                pass


state = State()


# ---------------------------------------------------------------------------
# Preview feed (single camera, shared between idle preview and capture)
# ---------------------------------------------------------------------------
class PreviewFeed:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None
        self._snap_jpeg = None              # last frame of the most recent capture
        self._placeholder = self._make_placeholder()

    @staticmethod
    def _make_placeholder():
        img = np.full((360, 640, 3), (30, 46, 52), dtype=np.uint8)
        cv2.putText(img, "NO CAMERA SIGNAL", (150, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (216, 236, 233), 2, cv2.LINE_AA)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def push(self, bgr):
        try:
            jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        except Exception:
            return
        with self._lock:
            self._jpeg = jpg
            self._snap_jpeg = jpg           # keep the most recent frame (capture)

    def latest(self):
        with self._lock:
            return self._jpeg if self._jpeg is not None else self._placeholder

    def snapshot(self):
        with self._lock:
            return self._snap_jpeg if self._snap_jpeg is not None else self._placeholder


feed = PreviewFeed()


class IdlePreviewThread(threading.Thread):
    """Streams the camera into `feed` while the system is idle."""

    def __init__(self, camera_index: int):
        super().__init__(daemon=True, name="idle-preview")
        self.camera_index = camera_index
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.warning("Idle preview: camera %d unavailable.", self.camera_index)
            return
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                feed.push(frame)
                time.sleep(0.033)
        finally:
            cap.release()


preview_thread = None


def start_idle_preview():
    global preview_thread
    if preview_thread is not None and preview_thread.is_alive():
        return
    idx = state.config["system"]["camera_index"] if state.config else 0
    preview_thread = IdlePreviewThread(idx)
    preview_thread.start()


def stop_idle_preview(wait=True):
    global preview_thread
    if preview_thread is not None:
        preview_thread.stop()
        if wait:
            preview_thread.join(timeout=3)
    preview_thread = None


# ---------------------------------------------------------------------------
# Model preload (runs once at startup, off the request thread)
# ---------------------------------------------------------------------------
def preload_models():
    t0 = time.perf_counter()
    try:
        state.emit({"type": "log", "level": "info",
                    "message": "Loading configuration..."})
        config = load_config(CONFIG_PATH)
        state.config = config
        setup_logger(log_dir=config["system"].get("log_dir", "logs/"),
                     log_level=config["system"].get("log_level", "INFO"))
        state.emit({"type": "log", "level": "info",
                    "message": "Loading face / voice / lip models (one-time, ~1 min)..."})
        modules = build_modules(config, exit_on_error=False)
        state.modules = modules
        state.timings["preload"] = round(time.perf_counter() - t0, 2)
        for name in modules:
            state.emit({"type": "log", "level": "info",
                        "message": f"{name.title()} module ready."})
        state.emit({"type": "log", "level": "info",
                    "message": f"Models loaded in {state.timings['preload']}s."})
        with state.lock:
            state.status = "ready"
        start_idle_preview()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Preload failed")
        state.emit({"type": "log", "level": "error",
                    "message": f"Preload failed: {exc}"})
        with state.lock:
            state.status = "error"
    finally:
        state.emit({"type": "status", "status": state.status,
                    "preload": state.timings.get("preload")})
        state.prewarm_done.set()


# ---------------------------------------------------------------------------
# Authentication session (runs the real TriGuard pipeline)
# ---------------------------------------------------------------------------
def run_auth_session():
    t_start = time.perf_counter()
    timings = {"total_s": None, "capture_s": None, "countdown_s": None,
               "recording_s": None, "modules_s": {}}
    stage_stamp = {}
    module_stamp = {}

    def stamp(tag):
        stage_stamp[tag] = time.perf_counter()

    def on_capture(stage, message, details=None):
        stamp(stage)
        if "countdown_started" in stage_stamp and "recording_started" in stage_stamp:
            timings["countdown_s"] = round(stage_stamp["recording_started"]
                                           - stage_stamp["countdown_started"], 2)
        if "recording_started" in stage_stamp and "recording_finished" in stage_stamp:
            timings["recording_s"] = round(stage_stamp["recording_finished"]
                                           - stage_stamp["recording_started"], 2)
        state.emit({"type": "capture", "stage": stage, "message": message,
                    "details": details or {}})

    def on_preview(bgr):
        feed.push(bgr)

    def on_module(name, mod_stage, message, details=None):
        stamp(f"module:{name}:{mod_stage}")
        if mod_stage in ("starting", "finished"):
            module_stamp.setdefault(name, {})[mod_stage] = time.perf_counter()
        if mod_stage == "finished":
            if module_stamp.get(name, {}).get("starting"):
                timings["modules_s"][name] = round(
                    module_stamp[name]["finished"] - module_stamp[name]["starting"], 2)
        state.emit({"type": "module", "module": name, "stage": mod_stage,
                    "message": message, "details": details or {}})

    def finish(result):
        """Record + publish a session result (every exit path must use this)."""
        result = dict(result)
        result["timings"] = result.get("timings") or timings
        result["at"] = datetime.now().strftime("%H:%M:%S")
        state.result = result
        state.timings["last_session"] = result["timings"]
        state.session_history.append(result)
        if len(state.session_history) > 6:
            state.session_history = state.session_history[-6:]
        state.emit({"type": "result", "result": result})

    cfg, modules = state.config, state.modules
    session = None
    capture = None
    try:
        session = CaptureSession(
            system_cfg=cfg["system"],
            lip_cfg=cfg["modules"]["lip"],
            face_cfg=cfg["modules"]["face"],
            status_callback=on_capture,
            preview_callback=on_preview,
        )
        stamp("capture_start")
        capture = session.run()
        timings["capture_s"] = round(time.perf_counter() - t_start, 2)

        if capture.cancelled or capture.error:
            finish({
                "authenticated": False, "identity": "unknown", "final_score": 0.0,
                "summary": capture.error or "Capture cancelled by user.",
                "cancelled": True})
            return

        if capture.face_abort:
            finish({
                "authenticated": False, "identity": "unknown", "final_score": 0.0,
                "summary": "Authentication aborted: more than one face detected."})
            return

        inputs = {
            "frames": capture.frames,
            "face_abort": capture.face_abort,
            "audio_path": capture.audio_path,
            "video_path": capture.video_path,
            "chosen_sentence": capture.chosen_sentence,
        }
        state.emit({"type": "log", "level": "info",
                    "message": "Capture complete — running face, voice and lip in parallel."})
        orchestrator = AuthOrchestrator(cfg, modules)
        result = orchestrator.authenticate(inputs, status_callback=on_module)
        timings["analysis_s"] = round(time.perf_counter()
                                      - (stage_stamp.get("capture_complete") or t_start), 2)
        timings["total_s"] = round(time.perf_counter() - t_start, 2)

        # Wall-clock estimate: modules run in parallel, so analysis ≈ max(module times).
        if timings["modules_s"]:
            timings["analysis_parallel_s"] = round(max(timings["modules_s"].values()), 2)
            timings["analysis_sum_s"] = round(sum(timings["modules_s"].values()), 2)

        result["timings"] = timings
        finish(result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Auth session failed")
        finish({
            "authenticated": False, "identity": "unknown", "final_score": 0.0,
            "summary": f"Error: {exc}"})
    finally:
        if session and capture:
            session.cleanup(capture)
        with state.lock:
            state.status = "ready"
        state.emit({"type": "status", "status": "ready"})
        time.sleep(0.6)          # let the OS release the camera
        start_idle_preview()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": state.status,
        "preload": state.timings.get("preload"),
        "last_result": state.result,
        "history": state.session_history,
    })


@app.route("/api/snapshot.jpg")
def api_snapshot():
    """JPEG of the most recent capture frame (shown next to the result)."""
    return Response(feed.snapshot(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/api/preview.mjpg")
def preview_mjpg():
    def generate():
        last = None
        while True:
            jpg = feed.latest()
            if jpg is not last:
                last = jpg
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                       + jpg + b"\r\n")
            else:
                time.sleep(0.05)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


@app.route("/api/events")
def sse_events():
    def generate():
        # replay history so a late/refreshed page catches up
        for ev in state.history:
            yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        yield f"data: {json.dumps({'type': 'history_end'})}\n\n"
        q = state.queue
        while True:
            try:
                msg = q.get(timeout=20)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/start", methods=["POST"])
def api_start():
    with state.lock:
        if state.status == "running":
            return jsonify({"ok": False, "error": "A session is already running."})
        if state.status != "ready":
            return jsonify({"ok": False, "error": f"System not ready ({state.status})."})
        state.status = "running"
        state.result = None
    stop_idle_preview()
    state.emit({"type": "status", "status": "running"})
    state.emit({"type": "log", "level": "info", "message": "Authentication started."})
    threading.Thread(target=run_auth_session, daemon=True, name="auth-session").start()
    return jsonify({"ok": True})


@app.route("/api/keepalive", methods=["POST"])
def api_keepalive():
    # No-op: used by the page to check the server is alive.
    return jsonify({"ok": True})


def main():
    host = os.environ.get("TRIGUARD_HOST", "127.0.0.1")
    print("\n  TriGuard web console")
    print(f"  >> http://{host}:{PORT}\n")
    state.emit({"type": "status", "status": "loading",
                "message": "Loading models (one-time)…"})
    threading.Thread(target=preload_models, daemon=True, name="preload").start()
    try:
        from waitress import serve  # optional: robust local server if installed
    except ImportError:
        serve = None
    if serve is not None:
        serve(app, host=host, port=PORT, threads=8)
    else:
        app.run(host=host, port=PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()