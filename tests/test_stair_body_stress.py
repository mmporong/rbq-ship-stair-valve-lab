import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))

from stair_body_stress import (  # noqa: E402
    CHASSIS_DIAGNOSTIC_XY,
    HIP_INDICES,
    JOINT_FAMILIES,
    MPA_SCALE,
    RAINBOW,
    SECTION_FAMILIES,
    _section_properties,
    build_artifacts,
    chassis_field_diagnostics,
    chassis_point_stress,
    chassis_point_weights,
    colour_chassis_pixels,
    link_stress_series,
    paint_fixed_legend,
    rainbow_rgb,
)


def test_rainbow_uses_fixed_blue_to_red_scale():
    values = rainbow_rgb(np.linspace(0, MPA_SCALE, len(RAINBOW)))
    np.testing.assert_allclose(values, RAINBOW / 255)
    np.testing.assert_allclose(rainbow_rgb(-1), RAINBOW[0] / 255)
    np.testing.assert_allclose(rainbow_rgb(MPA_SCALE * 2), RAINBOW[-1] / 255)


def test_six_axis_loads_produce_per_link_equivalent_stress():
    samples = 2
    forces = np.zeros((samples, 18, 3))
    moments = np.zeros((samples, 18, 3))
    bending = np.zeros((samples, 18))
    forces[0, 0, 0] = 1000
    moments[1, 0, 2] = 10
    bending[1, 0] = 6
    result = link_stress_series(
        {
            "time_s": np.arange(samples, dtype=float),
            "joint_force_world_n": forces,
            "joint_moment_world_nm": moments,
            "joint_off_axis_bending_nm": bending,
        }
    )
    area, modulus, median_area = _section_properties(SECTION_FAMILIES[JOINT_FAMILIES[0]])
    expected_axial = 1000 / area / 1e6
    expected_bending = 6 / modulus / 1e6
    expected_torsion = 8 / (2 * median_area * SECTION_FAMILIES["hip"]["wall_m"]) / 1e6
    assert result["equivalent_mpa"].shape == (samples, 18)
    assert result["equivalent_mpa"][0, 0] == pytest.approx(expected_axial)
    assert result["equivalent_mpa"][1, 0] == pytest.approx(
        np.sqrt(expected_bending**2 + 3 * expected_torsion**2)
    )
    np.testing.assert_allclose(
        result["chassis_zone_mpa"], result["equivalent_mpa"][:, HIP_INDICES]
    )


def test_artifacts_state_assumptions_hash_and_peak(tmp_path):
    mission = tmp_path / "mission.npz"
    output = tmp_path / "link-stress.json"
    forces = np.zeros((2, 18, 3))
    moments = np.zeros((2, 18, 3))
    bending = np.zeros((2, 18))
    forces[1, 4, 0] = 900
    np.savez_compressed(
        mission,
        time_s=np.array([0.0, 0.5]),
        stage=np.array(["walk", "hold"]),
        joint_force_world_n=forces,
        joint_moment_world_nm=moments,
        joint_off_axis_bending_nm=bending,
        qpos=np.tile(np.array([0, 0, 0, 1, 0, 0, 0]), (2, 1)),
        rb1_mount_wrench_world=np.zeros((2, 6)),
    )
    summary = build_artifacts(mission, output, output.with_suffix(".csv"))
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert summary == saved
    assert saved["mission_sha256"]
    assert saved["scope"].startswith("conditional thin-wall")
    assert saved["peak"]["time_s"] == 0.5
    assert len(output.with_suffix(".csv").read_text(encoding="utf-8").splitlines()) == 47


def test_chassis_field_interpolates_at_body_local_points():
    np.testing.assert_allclose(chassis_point_weights(CHASSIS_DIAGNOSTIC_XY).sum(axis=1), 1)
    field = chassis_point_stress(
        CHASSIS_DIAGNOSTIC_XY, np.array([2.0, 8.0, 14.0, 20.0, 70.0])
    )
    assert np.ptp(field) > 5
    assert not np.allclose(field[:32], field[0])
    diagnostics = chassis_field_diagnostics(
        np.array([0.0, 1.0]),
        np.array([[2.0, 8.0, 14.0, 20.0, 70.0], [9.0, 9.0, 9.0, 9.0, 9.0]]),
    )
    assert diagnostics["maximum_spatial_range_time_s"] == 0.0
    assert diagnostics["maximum_spatial_range_mpa"] > 5
    assert diagnostics["mean_field_temporal_range_mpa"] > 0


def test_actual_geom_mask_pixels_are_reprojected_and_coloured():
    class Camera:
        pos = np.array([0.0, 0.0, -1.0])
        forward = np.array([0.0, 0.0, 1.0])
        up = np.array([0.0, 1.0, 0.0])
        frustum_top = 1.0
        frustum_near = 1.0

    class Body:
        xpos = np.zeros(3)
        xmat = np.eye(3).ravel()

    frame = np.full((6, 8, 3), 120, dtype=np.uint8)
    segmentation = np.full((6, 8, 2), -1, dtype=np.int32)
    segmentation[1:5, 1:7] = (14, 5)
    before = frame.copy()
    metrics = colour_chassis_pixels(
        frame,
        segmentation,
        np.ones((6, 8)),
        Camera(),
        Body(),
        (14, 15),
        np.array([2.0, 8.0, 14.0, 20.0, 70.0]),
    )
    np.testing.assert_array_equal(frame[0], before[0])
    assert metrics["mesh_pixel_count"] == 24
    assert metrics["unique_20mpa_colour_bands"] >= 2
    assert np.unique(frame[1:5, 1:7].reshape(-1, 3), axis=0).shape[0] > 2


def test_legend_paints_fixed_rainbow_bar():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    paint_fixed_legend(frame)
    np.testing.assert_allclose(frame[698, 933], RAINBOW[0], atol=4)
    np.testing.assert_allclose(frame[698, 1248], RAINBOW[-1], atol=7)
    np.testing.assert_allclose(frame[668, 933], RAINBOW[0], atol=4)
    np.testing.assert_allclose(frame[668, 1248], RAINBOW[-1], atol=7)
