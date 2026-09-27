"""
strategies
----------
The ranking methods, all implementing `harness.RankingMethod`.

The three pre-existing methods are adapters: they call the same prompt
builders, parsers and fitting code the CLI always used, so moving them behind
the common interface does not change what they do.
`tests/test_harness_identity.py` verifies that against the pre-existing code
paths directly.
"""

from strategies.pairwise import PairwiseBT
from strategies.rank_all import RankAll
from strategies.scoring import IndependentScoring

__all__ = ["RankAll", "IndependentScoring", "PairwiseBT"]
