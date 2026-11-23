import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.rbq_stair_interventions import blend_position_targets, interpolate_rows


def test_interpolate_rows_clamps_and_interpolates():
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([[0.0, 2.0], [2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_array_equal(interpolate_rows(times, values, -1.0), values[0])
    np.testing.assert_allclose(interpolate_rows(times, values, 1.5), [4.0, 6.0])
    np.testing.assert_array_equal(interpolate_rows(times, values, 3.0), values[-1])


def test_interpolate_rows_rejects_duplicate_times():
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_rows(np.array([0.0, 0.0]), np.zeros((2, 1)), 0.0)


def test_position_target_blend_preserves_actor_endpoints():
    actor = np.array([1.0, 2.0])
    teacher = np.array([3.0, 6.0])
    np.testing.assert_array_equal(blend_position_targets(actor, teacher, 0.0), actor)
    np.testing.assert_allclose(blend_position_targets(actor, teacher, 0.25), [1.5, 3.0])
    np.testing.assert_array_equal(blend_position_targets(actor, teacher, 1.0), teacher)
