"""Door/window/other classification of image segments.

Import policy, copied from EmissivityCalculation's hard-won one: the consensus
stage runs under the rosbags venv (no torch), so it imports openings.table and
openings.zone_prior as plain modules by path instead of through the package --
both of those are stdlib-only on purpose. Keep them that way.
"""

from .table import OpeningTable, OpeningRecord, OTHER_CLASS
from .zone_prior import OPENING_ZONE_CANDIDATES, restrict_opening_ranking
from . import geometry

__all__ = [
    "OpeningTable",
    "OpeningRecord",
    "OTHER_CLASS",
    "OPENING_ZONE_CANDIDATES",
    "restrict_opening_ranking",
    "geometry",
]
