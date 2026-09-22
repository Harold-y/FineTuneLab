import pytest

from finetunelab.config import RewardSpec
from finetunelab.rewards.builtin import (
    build_reward_functions,
    exact_match_reward,
    numeric_reward,
)


def test_exact_match_reward() -> None:
    assert exact_match_reward(["4", " 5 "], answer=["4", "5"]) == [1.0, 1.0]


def test_numeric_reward_extracts_last_number() -> None:
    assert numeric_reward(["Reasoning... answer 4"], answer=["4"]) == [1.0]


def test_regex_reward_requires_pattern() -> None:
    with pytest.raises(Exception, match="pattern"):
        build_reward_functions([RewardSpec(name="regex")])


def test_weighted_reward() -> None:
    reward = build_reward_functions([RewardSpec(name="exact_match", weight=0.25)])[0]
    assert reward(["yes"], answer=["yes"]) == [0.25]
