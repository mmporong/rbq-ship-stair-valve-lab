import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "rbq_reward_residual", Path(__file__).parents[1] / "train" / "rbq_reward_residual.py"
)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_policy_gates_are_bounded_and_observation_dependent():
    parameters = np.array([-1.0, 1.0, -2.0])
    quiet = M.gates(parameters, 0.0)
    disturbed = M.gates(parameters, 0.09)
    assert 0 < quiet[0] < disturbed[0] < 1
    assert 0 < quiet[1] == disturbed[1] < 0.25


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("fixed_contact", (1.0, 0.0)),
        ("teacher_open_loop", (0.0, 0.0)),
        ("learned_no_ppo", None),
        ("learned_contact_off", None),
    ],
)
def test_ablation_gates(variant, expected):
    actual = M.variant_gates(variant, np.array([0.3, 0.2, -1.0]), 0.01)
    if expected is not None:
        assert actual == expected
    elif variant == "learned_no_ppo":
        assert actual[0] > 0 and actual[1] == 0
    else:
        assert actual[0] == 0 and actual[1] > 0


def test_candidate_score_is_mean_rollout_return():
    assert M.candidate_score([{"reward": 1.0}, {"reward": 5.0}]) == 3.0


def test_constants_freeze_bounded_cpu_study():
    assert M.CEM_GENERATIONS == 3
    assert M.CEM_POPULATION == 8
    assert M.CEM_ELITE == 3
    assert M.TRAIN_HORIZON_S == 20.0
    assert M.FULL_HORIZON_S == 127.75
    assert set(M.VARIANTS) == {
        "learned",
        "initial",
        "fixed_contact",
        "teacher_open_loop",
        "learned_no_ppo",
        "learned_contact_off",
    }
