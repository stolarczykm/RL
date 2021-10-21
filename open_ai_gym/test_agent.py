import numpy as np
import pytest

from gym.spaces import Box, Discrete
from .agents import SarsaAgent


@pytest.fixture
def agent() -> SarsaAgent:
    return SarsaAgent(
        n_bins=2,
        action_space=Discrete(2),
        observation_space=Box(np.array([0,0]), np.array([1, 1])),
    )


@pytest.mark.parametrize(
    "state, expected",
    [
        ([0.2, 0.2], 0),
        ([0.2, 0.7], 1),
        ([0.7, 0.1], 2),
        ([1.0, 1.0], 3),
    ],
)
def test_get_active_features(agent: SarsaAgent, state, expected):
    result = agent._get_active_feature(np.array(state))
    assert result == expected
    assert isinstance(result, int)


