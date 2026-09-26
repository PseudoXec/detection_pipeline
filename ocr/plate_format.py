"""
plate_format.py
----------------
Validates a cleaned OCR read against the plate formats the Philippine LTO
(Land Transportation Office) actually issues, so a read that comes out in
the wrong shape - digits and letters swapped/reversed, wrong grouping, a
stray leftover character - is caught and reported as unreadable instead
of being handed to the dashboard as a real plate number.

Formats covered (current + still-common older series - sources: LTO's
2014 "New Plate Number Series" standardization and its predecessor
format; see Wikipedia's "Vehicle registration plates of the Philippines"):

    ABC1234   3 letters + 4 digits   current private/PUV four-wheel format (2014-)
    ABC123    3 letters + 3 digits   pre-2014 four-wheel format, still common on
                                      older still-registered vehicles
    ABC12     3 letters + 2 digits   older "optional"/short-number plates (OMVSP)
    123ABC    3 digits + 3 letters   current motorcycle/tricycle format - digits
                                      come FIRST. This is a real, valid format,
                                      not a reversed four-wheel plate - don't
                                      reject it as "backwards".
    AB1234    2 letters + 4 digits   pre-2014 motorcycle format
    AB12345   2 letters + 5 digits   2014-era motorcycle format

Deliberately NOT covered: government (red), diplomatic (blue), and other
special-issue plates - their formats vary enough (see LTO Memorandum
Circulars on Optional/Special plates) that a strict regex here would
reject a lot of legitimate reads. This module is a sanity filter for
ordinary private/PUV/motorcycle plates, not an exhaustive or legal
validator; check with the LTO for the authoritative current spec.
"""

import re
from typing import Optional, Tuple

# (format name, compiled pattern) - matched in order, first hit wins.
_PATTERNS: Tuple[Tuple[str, "re.Pattern"], ...] = (
    ("car-current", re.compile(r"^[A-Z]{3}[0-9]{4}$")),
    ("car-pre2014", re.compile(r"^[A-Z]{3}[0-9]{3}$")),
    ("car-optional-short", re.compile(r"^[A-Z]{3}[0-9]{2}$")),
    ("motorcycle-current", re.compile(r"^[0-9]{3}[A-Z]{3}$")),
    ("motorcycle-pre2014", re.compile(r"^[A-Z]{2}[0-9]{4}$")),
    ("motorcycle-2014", re.compile(r"^[A-Z]{2}[0-9]{5}$")),
)


def classify_ph_plate(text: str) -> Optional[str]:
    """Returns the name of the matching format (see _PATTERNS), or None if
    `text` doesn't match any recognized Philippine plate format - including
    the right characters in the wrong/reversed order."""
    if not text:
        return None
    for name, pattern in _PATTERNS:
        if pattern.match(text):
            return name
    return None


def is_valid_ph_plate(text: str) -> bool:
    """True if `text` (already uppercased/cleaned) matches one of the
    formats above."""
    return classify_ph_plate(text) is not None
