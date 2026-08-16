"""
Forge Neo — Krea 2 Reference Batch Fix.

Makes every logical output of a native Krea 2 Img2Img / reference-image request
receive its own primary reference, for any combination of Batch Count and
Batch Size, without modifying any Forge Neo core file.

The package is deliberately split so that Forge-version-specific knowledge is
confined to `compat.py`, `detect.py` and `patch.py`; the generation logic in
`fanout.py`, `clone.py`, `aggregate.py` and `seeds.py` is independent of it.
"""

from __future__ import annotations

VERSION = "1.0"

INFOTEXT_FIELD = "Krea Ref Batch Fix"
"""Single diagnostic infotext key added to isolated outputs."""

UNIT_SENTINEL = "_krea2_reference_batchfix_unit"
"""Attribute set on a processing object that is already an isolated unit job."""

__all__ = ["VERSION", "INFOTEXT_FIELD", "UNIT_SENTINEL"]
