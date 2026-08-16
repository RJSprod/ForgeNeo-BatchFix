"""Forge Neo — Krea 2 Reference Batch Fix (extension entry point).

Loaded by Forge's script loader at startup. It only wires things up:

  * puts the extension root on ``sys.path`` so ``krea2_batchfix`` is importable;
  * registers the extension's Settings section;
  * installs the idempotent ``process_images`` wrapper (fail-closed).

Deliberately does *not* define a ``Script`` subclass: an AlwaysVisible script
would take up slots in ``p.script_args`` and shift every other script's argument
window for no benefit, since the fix needs no per-run UI.
"""

from __future__ import annotations

import os
import sys

_EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, _EXTENSION_ROOT)

from krea2_batchfix import patch, settings  # noqa: E402

settings.apply_on_ui_settings()
settings.sync_debug_level()

patch.install()
patch.install_late_hooks()
