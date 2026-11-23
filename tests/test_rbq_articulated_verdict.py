import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))

from rbq_articulated_verdict import _validate_inputs, planned_stance  # noqa: E402


def test_validate_inputs_requires_uninterrupted_post_step_clock():
    model = SimpleNamespace(nq=3, nv=2, nu=1, opt=SimpleNamespace(timestep=0.001))
    state = {
        "time_s": np.array([0.001, 0.002]),
        "qpos": np.zeros((2, 3)),
        "qvel": np.zeros((2, 2)),
        "ctrl": np.zeros((2, 1)),
        "requested_robot_nm": np.zeros((2, 18)),
        "requested_finger_force_n": np.zeros(2),
    }
    reference = {
        "time_s": state["time_s"].copy(),
        "stage": np.array(["prewalk", "prewalk"]),
        "qpos": state["qpos"].copy(),
        "stance": np.ones((2, 4), dtype=bool),
    }
    assert _validate_inputs(model, state, reference) == 2
    state["time_s"][1] = 0.003
    with pytest.raises(ValueError, match="uninterrupted"):
        _validate_inputs(model, state, reference)


def test_planned_stance_prefers_frozen_per_step_flags(tmp_path):
    stance = np.array([[True, False, True, False], [True, True, True, True]])
    reference = {
        "time_s": np.array([0.001, 0.002]),
        "stage": np.array(["prewalk", "position"]),
        "qpos": np.zeros((2, 7)),
        "stance": stance,
    }
    missing_walk = tmp_path / "not-read.npz"
    np.testing.assert_array_equal(planned_stance(reference, missing_walk, 2), stance)
