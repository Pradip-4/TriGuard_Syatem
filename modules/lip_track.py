"""
modules/lip_track.py  —  DEPRECATED DUPLICATE.

This folder previously contained an older copy of lip_track.py that injected
`sys.modules['tensorflow'] = None`, which broke DeepFace (used by the Face
module) at runtime. The canonical implementation lives at the project root
(./lip_track.py) and is what modules/lip_module.py imports.

This shim simply re-exports the canonical implementation so that any stale
import of `modules.lip_track` behaves identically without code drift.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lip_track import (  # noqa: E402,F401
    LocalInferencePipeline,
    overlay_captions,
)