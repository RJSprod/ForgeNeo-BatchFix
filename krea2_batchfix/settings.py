"""Extension settings, registered under their own Settings section.

The extension deliberately adds no UI components to the Img2Img page: doing so
would insert script arguments into ``p.script_args`` and shift every other
script's argument window. Everything is configured from Settings instead.
"""

from __future__ import annotations

from . import VERSION
from .logs import logger, set_debug

SECTION = ("krea2_refbatchfix", "Krea 2 Reference Batch Fix")

ENABLED = "krea2_refbatchfix_enabled"
FORCE_RECOND = "krea2_refbatchfix_force_recond"
INCLUDE_INPAINT = "krea2_refbatchfix_include_inpaint"
ADD_INFOTEXT = "krea2_refbatchfix_infotext"
DEBUG = "krea2_refbatchfix_debug"

DEFAULTS = {
    ENABLED: True,
    FORCE_RECOND: True,
    INCLUDE_INPAINT: False,
    ADD_INFOTEXT: True,
    DEBUG: False,
}


def get(key: str):
    """Read an option, falling back to the built-in default."""
    try:
        from modules.shared import opts

        return getattr(opts, key, DEFAULTS[key])
    except Exception:
        return DEFAULTS[key]


def register() -> None:
    """Add the extension's options. Safe to call more than once."""
    try:
        from modules import shared
    except Exception:
        return

    option = shared.OptionInfo

    shared.opts.add_option(
        ENABLED,
        option(DEFAULTS[ENABLED], f"Enable the Krea 2 reference batch fix (v{VERSION})", section=SECTION).info("fan Batch Count / Batch Size out into isolated single-output Krea jobs so every output keeps its primary Img2Img reference"),
    )
    shared.opts.add_option(
        FORCE_RECOND,
        option(DEFAULTS[FORCE_RECOND], "Re-encode conditioning for every isolated output", section=SECTION).info("<b>recommended:</b> Krea injects the reference latent while the prompt is encoded, so a cond-cache hit would reuse the previous output's reference"),
    )
    shared.opts.add_option(
        INCLUDE_INPAINT,
        option(DEFAULTS[INCLUDE_INPAINT], "Also isolate Inpaint / Sketch requests", section=SECTION).info("<b>untested:</b> V1 only claims support for plain Img2Img reference mode"),
    )
    shared.opts.add_option(
        ADD_INFOTEXT,
        option(DEFAULTS[ADD_INFOTEXT], "Add the 'Krea Ref Batch Fix' field to infotext", section=SECTION).info("one diagnostic field; all other metadata is unchanged"),
    )
    shared.opts.add_option(
        DEBUG,
        option(DEFAULTS[DEBUG], "Log one line per isolated output", section=SECTION).info("source index, logical output index, resolved seed and a short hash of the primary input image"),
    )


def sync_debug_level() -> None:
    set_debug(bool(get(DEBUG)))


def apply_on_ui_settings() -> None:
    """Register the options through Forge's settings callback."""
    try:
        from modules import script_callbacks
    except Exception:
        logger.warning("modules.script_callbacks unavailable; settings were not registered")
        return

    script_callbacks.on_ui_settings(register)
