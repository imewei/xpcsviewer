"""The one wording and the one test for a file that runs on the placeholder Q map.

A file whose Q map is missing or unreadable gets a substituted calibration
(``QMap.is_default``). Every export that carries Q values of such a file, and the
status bar, use these two names so the warning cannot drift between writers.
"""

from __future__ import annotations

#: Written (as a comment line) into every export that carries Q values of a file
#: whose Q map was substituted by the placeholder calibration.
PLACEHOLDER_QMAP_WARNING = (
    "WARNING: placeholder Q map, Q values are NOT this file's calibration"
)


def has_placeholder_qmap(xf: object) -> bool:
    """True when ``xf`` runs on the substituted default Q map (``QMap.is_default``).

    ``is True``: a stand-in object whose ``qmap`` answers any attribute is no file.
    """
    return getattr(getattr(xf, "qmap", None), "is_default", False) is True
