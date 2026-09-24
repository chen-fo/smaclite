"""Optional reward interface for the two-sided environment.

The environment does not assume either cooperation or zero-sum competition.
Reward functions run outside the physics and receive detached transition data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Tuple

import numpy as np

TEAMS = ("team_0", "team_1")


@dataclass(frozen=True)
class RewardTransition:
    """Global features before/after a step; never passed to agent policies.

    Arrays are detached, read-only copies in the layout documented by
    ``env.get_env_info()['state_layout']``. Action tuples use stable team-local IDs.
    """

    state_before: np.ndarray
    state_after: np.ndarray
    actions: Mapping[str, Tuple[int, ...]]
    step_count: int
    winner: Optional[str]
    terminated: bool
    truncated: bool
    termination_reason: Optional[str]


class RewardFunction(Protocol):
    """Return one finite scalar per team, with no relation required between them."""

    def __call__(self, transition: RewardTransition) -> Mapping[str, float]: ...


def zero_rewards(transition: RewardTransition) -> Mapping[str, float]:
    """Placeholder until the algorithm project supplies its reward definition."""
    return {team: 0.0 for team in TEAMS}
