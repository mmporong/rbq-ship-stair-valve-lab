import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "rbq_initial_no_ppo", Path(__file__).parents[1] / "tools" / "rbq_initial_no_ppo.py"
)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_initial_no_ppo_job_uses_frozen_initial_theta_and_existing_ablation():
    candidate_id, parameters, variant = M.initial_no_ppo_job()
    assert candidate_id == "initial_no_ppo"
    assert variant == "learned_no_ppo"
    assert np.array_equal(parameters, [-2.0, 0.0, -3.0])
    parameters[0] = 99
    assert M.INITIAL_PARAMETERS[0] == -2.0
