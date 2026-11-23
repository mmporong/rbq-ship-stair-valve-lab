import numpy as np
import pytest


def test_corridor_requires_the_prescribed_tip_force_at_interpolated_times(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from check_stair_corridor import check_tip_forces

    forces = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -4.0]])
    left, right = np.array([0, 0, 1]), np.array([1, 1, 1])
    weight = np.array([0.0, 0.25, 0.0])
    prescribed = np.array([[0.0, 0.0, 0.0], [0.25, 0.5, -1.0], [1.0, 2.0, -4.0]])
    check_tip_forces(prescribed, forces, left, right, weight)

    no_load = np.zeros_like(forces)
    site_position_in_force_field = np.zeros_like(prescribed)
    site_position_in_force_field[1] = [0.63, 0.11, 1.36]
    with pytest.raises(ValueError, match="prescribed tip force"):
        check_tip_forces(site_position_in_force_field, no_load, left, right, weight)
    with pytest.raises(ValueError, match="prescribed tip force"):
        check_tip_forces(prescribed[:, :1], forces, left, right, weight)


def test_scheduled_landings_use_forward_plan_and_verify_archive(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from stair_mission_reference import (
        CORRIDOR_PROFILE,
        describe_reference,
        scheduled_landings,
        verified_landings,
    )
    from stair_mission_route import route

    _, _, _, _, ends, anchor = route(50.0, 0.19)
    nominal = np.arange(round(ends[-1][0] / 0.02) + 1) * 0.02
    work = ends[anchor][0]
    clock = nominal + np.maximum(nominal - work, 0)
    cache = dict(
        slope_deg=50.0,
        rise_m=0.19,
        footprint_profile=CORRIDOR_PROFILE,
        reference_time=nominal,
        time=clock,
    )
    path = tmp_path / "walking.npz"
    np.savez(path, **cache)
    info = describe_reference(path, CORRIDOR_PROFILE)
    mapped = verified_landings(info, path, ends, anchor, 50.0, 0.19)
    assert len(mapped) == len(ends)
    assert mapped[anchor][1] is ends[anchor][1]
    assert mapped[-1][0] == pytest.approx(2 * ends[-1][0] - work)
    # During upper-platform rotation the target centre moves laterally by one common translation.
    platform = next(
        i for i, (_, stance, _) in enumerate(ends) if min(p[0][0] for p in stance.values()) > 1.65
    )
    for leg in mapped[platform][1]:
        delta = mapped[platform][1][leg][0] - ends[platform][1][leg][0]
        np.testing.assert_allclose(delta, [0.0, 0.072, 0.0], atol=1e-12)
    np.savez(path, **{**cache, "time": nominal})
    with pytest.raises(ValueError, match="fingerprint"):
        verified_landings(info, path, ends, anchor, 50.0, 0.19)
    with pytest.raises(ValueError, match="before the fixed"):
        scheduled_landings({**cache, "time": nominal * 2}, ends, anchor, 50.0, 0.19)
    assert scheduled_landings({"time": nominal}, ends, anchor, 50.0, 0.19)[0] is ends


def test_refined_clock_preserves_every_original_knot_and_landing(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from stair_mission_reference import CORRIDOR_PROFILE, scheduled_landings
    from stair_mission_route import route

    _, _, _, _, ends, anchor = route(50.0, 0.19)
    grid = np.arange(round(ends[-1][0] / 0.02) + 1) * 0.02
    row = np.searchsorted(grid, ends[anchor + 1][0]) + 3
    phase = np.insert(grid, row, grid[row] - 0.01)
    cache = {"footprint_profile": CORRIDOR_PROFILE, "reference_time": phase, "time": phase}
    mapped, _ = scheduled_landings(cache, ends, anchor, 50.0, 0.19)
    np.testing.assert_allclose([m[0] for m in mapped], [e[0] for e in ends], atol=1e-8, rtol=0)
    missing = np.delete(phase, row + 1)
    with pytest.raises(ValueError, match="original route knot"):
        scheduled_landings(
            {**cache, "reference_time": missing, "time": missing}, ends, anchor, 50.0, 0.19
        )


def test_corridor_schedule_rejects_a_nonzero_assumed_tip_force(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from stair_mission_reference import CORRIDOR_PROFILE, scheduled_landings
    from stair_mission_route import route

    _, _, _, _, ends, anchor = route(50.0, 0.19)
    clock = np.arange(round(ends[-1][0] / 0.02) + 1) * 0.02
    cache = {
        "footprint_profile": CORRIDOR_PROFILE,
        "reference_time": clock,
        "time": clock,
        "tip": np.zeros((len(clock), 3)),
    }
    scheduled_landings(cache, ends, anchor, 50.0, 0.19)
    cache["tip"][4, 1] = 0.1
    with pytest.raises(ValueError, match="prescribed tip force"):
        scheduled_landings(cache, ends, anchor, 50.0, 0.19)


def test_replant_profile_adds_one_physical_landing_and_shifts_the_route(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from stair_mission_control_profile import REPLANT_SECONDS, replant_landings
    from stair_mission_route import route

    _, _, _, _, ends, anchor = route(50.0, 0.19)
    updated = replant_landings(ends)
    assert len(updated) == len(ends) + 1
    assert updated[16][0] == pytest.approx(81.6)
    assert updated[16][2] == "RL"
    assert updated[anchor + 1][0] == pytest.approx(ends[anchor][0] + REPLANT_SECONDS)
    assert updated[-1][0] == pytest.approx(ends[-1][0] + REPLANT_SECONDS)
