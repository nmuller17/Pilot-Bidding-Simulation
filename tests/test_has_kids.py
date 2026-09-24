"""
Regression test for Pilot.has_kids: "Married, no kids" must not count as a
family pilot (it used to, because the substring 'kids' matched).

Run directly (no pytest needed):
    python tests/test_has_kids.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import FAMILY_OPTIONS
from models import Pilot

EXPECTED = {
    "Single":           False,
    "Married, no kids": False,
    "Married, 1 child": True,
    "Married, 2+ kids": True,
    "Single parent":    True,
}


def _pilot(family_status: str) -> Pilot:
    return Pilot(id=1, name="Test", age=40, family_status=family_status,
                 seniority=1, home_base="BOS", qualified_types=["B737"],
                 min_rest=10, base_pay=100.0)


def test_has_kids_covers_every_family_option():
    assert set(FAMILY_OPTIONS) == set(EXPECTED)
    for status, want in EXPECTED.items():
        assert _pilot(status).has_kids is want, status


if __name__ == "__main__":
    test_has_kids_covers_every_family_option()
    print("ok")
