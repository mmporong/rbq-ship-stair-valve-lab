import json

import numpy as np
import pytest

from go2_wall.loads import bracket_stress


def load_tool(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    import stair_stress_heatmap

    return stair_stress_heatmap


def mission_samples():
    return {
        "time_s": np.array([0.0, 0.5, 1.0]),
        "stage": np.array(["approach", "operate", "release"]),
        "qpos": np.tile(np.r_[np.zeros(3), 1.0, 0.0, 0.0, 0.0], (3, 1)),
        "rb1_mount_wrench_world": np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            ]
        ),
    }


def test_rainbow_scale_has_requested_endpoints_and_intermediates(monkeypatch):
    tool = load_tool(monkeypatch)

    assert tool.rainbow_colour(0, 100) == "#2646b7"
    assert tool.rainbow_colour(25, 100) == "#00acc1"
    assert tool.rainbow_colour(50, 100) == "#26b14c"
    assert tool.rainbow_colour(75, 100) == "#ffd600"
    assert tool.rainbow_colour(100, 100) == "#d52d26"
    assert tool.rainbow_colour(200, 100) == "#d52d26"


def test_stress_series_uses_recorded_wrench_and_base_quaternion(monkeypatch):
    tool = load_tool(monkeypatch)

    series = tool.stress_series(mission_samples())

    assert series["von_mises_mpa"] == pytest.approx([0.0, 3.125, 1.5625])
    assert series["bending_vertical_mpa"] == pytest.approx(series["von_mises_mpa"])
    assert np.allclose(series["bending_lateral_mpa"], 0)
    assert np.allclose(series["torsion_mpa"], 0)


def test_vector_stress_matches_existing_scalar_reference(monkeypatch):
    tool = load_tool(monkeypatch)
    mission = mission_samples()
    root = np.sqrt(0.5)
    mission["qpos"][1, 3:7] = [root, 0.0, 0.0, root]
    mission["rb1_mount_wrench_world"][1] = [3.0, -7.0, 11.0, 2.0, -5.0, 13.0]

    series = tool.stress_series(mission)

    rotation = tool._base_rotations(mission["qpos"])[1]
    reference = bracket_stress(
        -mission["rb1_mount_wrench_world"][1, :3],
        -mission["rb1_mount_wrench_world"][1, 3:],
        rotation,
        0.15,
    )
    assert series["von_mises_mpa"][1] == pytest.approx(reference["von_mises_mpa"])
    assert series["moment_bolt_line_nm"][1] == pytest.approx(reference["moment_bolt_line_nm"])


def test_stress_report_records_fixed_scale_peak_and_exclusions(tmp_path, monkeypatch):
    tool = load_tool(monkeypatch)
    mission = tmp_path / "mission.npz"
    np.savez_compressed(mission, **mission_samples())
    page = tmp_path / "stress.html"
    summary_path = tmp_path / "stress.json"
    csv_path = tmp_path / "stress.csv"

    summary = tool.build_report(mission, page, summary_path, csv_path)

    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved == summary
    assert saved["peak"]["time_s"] == 0.5
    assert saved["peak"]["von_mises_mpa"] == pytest.approx(3.125)
    assert [row["peak_mpa"] for row in saved["stage_peaks"]] == pytest.approx([0, 3.125, 1.5625])
    assert saved["scales_mpa"] == {
        "fixed_min": 0.0,
        "fixed_max": 276.0,
        "detail_min": 0.0,
        "detail_max": 5.0,
    }
    assert "not RBQ component specifications" in saved["provenance_warning"]
    assert [row["thickness_mm"] for row in saved["thickness_sensitivity"]] == [4.0, 6.0, 8.0]
    assert [row["peak_mpa"] for row in saved["thickness_sensitivity"]] == pytest.approx(
        [7.03125, 3.125, 1.7578125]
    )
    assert saved["calculation_check"]["expected_vertical_bending_mpa"] == 3.125
    assert "arm-link stress" in saved["excluded"]
    html = page.read_text(encoding="utf-8")
    for colour in ("#2646b7", "#00acc1", "#26b14c", "#ffd600", "#d52d26"):
        assert colour in html
    assert "항복강도 고정 범례" in html
    assert "관측 구간 확대 범례" in html
    assert "3.125 MPa" in html
    card = tool.peak_card_html(saved)
    field = tool.bending_section(saved)
    assert field.max() == pytest.approx(3.125)
    assert field[(len(field) - 1) // 2] == pytest.approx(np.zeros(field.shape[1]))
    assert np.allclose(field[0], field[-1])
    assert tool.section_svg(saved) in card
    assert "항복강도 고정 범례" in card
    raw_scene = tmp_path / "raw-scene.png"
    tool._write_rgb_png(raw_scene, np.zeros((3, 5, 3), dtype=np.uint8))
    assert raw_scene.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    pose_card = tool.pose_card_html(saved, raw_scene)
    assert "색 구체는 모델의 RB1_Base 원점" in pose_card
    assert "로봇과 계단은 원래 색" in pose_card
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 4


def test_stress_series_rejects_missing_or_malformed_load_channels(monkeypatch):
    tool = load_tool(monkeypatch)
    mission = mission_samples()
    mission.pop("rb1_mount_wrench_world")
    with pytest.raises(ValueError, match="missing load channels"):
        tool.stress_series(mission)
    mission = mission_samples()
    mission["rb1_mount_wrench_world"] = np.zeros((2, 6))
    with pytest.raises(ValueError, match="shapes are inconsistent"):
        tool.stress_series(mission)


def test_each_thickness_uses_its_own_peak_sample(tmp_path, monkeypatch):
    tool = load_tool(monkeypatch)
    samples = mission_samples()
    samples["rb1_mount_wrench_world"][2] = [0, 0, 0, 0, 0, 19.2]
    mission = tmp_path / "mission.npz"
    np.savez_compressed(mission, **samples)
    report = tool.build_report(
        mission, tmp_path / "a.html", tmp_path / "a.json", tmp_path / "a.csv"
    )
    rows = report["thickness_sensitivity"]
    assert [row["time_s"] for row in rows] == [0.5, 0.5, 1.0]
    assert rows[2]["peak_mpa"] == pytest.approx(2.25)
