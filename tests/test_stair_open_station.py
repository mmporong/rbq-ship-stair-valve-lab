import numpy as np
import pytest

from go2_wall import stair_valve
from go2_wall.pose_search import set_base


@pytest.fixture(scope="module")
def scenes():
    base = [0.5, 0.03, 1.06, np.deg2rad(35)]
    mount = [0.90, 0, 1.58]
    return base, {
        mode: stair_valve.build_grasp_scene(50, 0.19, mount, start, sense, 0.12, 10)
        for mode, start, sense in (("opening", 180, 1), ("closing", 270, -1))
    }


def terminal_samples(base, model, data):
    model.opt.timestep = 0.005
    n = 1001  # Initial closed sample followed by a five-second stationary endpoint.
    set_base(model, data, base[3], np.array(base[:3]))
    q = np.tile(data.qpos, (n, 1))
    angle = np.full(n, 90.0)
    angle[0] = 0
    q[:, model.joint(stair_valve.VALVE_HINGE).qposadr[0]] = np.deg2rad(angle)
    stage = np.full(n, "station")
    stage[0] = "prewalk"
    trace = dict(
        stage=stage,
        gripper_open=np.ones(n, dtype=bool),
        time_s=(np.arange(n) + 1) * model.opt.timestep,
        valve_angle_deg=angle,
        pad_surface_gap_m=np.full((n, 2), 0.01),
        pad_normal_force_n=np.zeros((n, 2)),
        handle_nonpad_force_n=np.zeros(n),
    )
    loads = np.zeros((n, 4, 12))
    loads[:, np.arange(4), [5, 5, 3, 3]] = 20
    return trace, q, np.zeros((n, model.nv)), loads


@pytest.mark.parametrize("mode", ["opening", "closing"])
def test_station_checks_world_geometry_even_with_same_hinge_series(monkeypatch, scenes, mode):
    monkeypatch.syspath_prepend("tools")
    from stair_open_station import evaluate_station

    base, models = scenes
    model, data = models[mode]
    result = evaluate_station(
        model, {"start_deg": 270, "sense": -1}, base, *terminal_samples(base, model, data), 18, 18
    )
    assert result["verified"] == (mode == "opening")
    if mode == "closing":
        assert result["measurements"]["open_finish_error_deg"] == pytest.approx(90)


@pytest.mark.parametrize("fault", ["truncated_dwell", "unsupported_foot", "moving", "recontact"])
def test_station_requires_complete_loaded_quiet_endpoint(monkeypatch, scenes, fault):
    monkeypatch.syspath_prepend("tools")
    from stair_open_station import evaluate_station

    base, models = scenes
    model, data = models["opening"]
    trace, poses, velocity, loads = terminal_samples(base, model, data)
    if fault == "truncated_dwell":
        trace["stage"][-1] = "release"
    elif fault == "unsupported_foot":
        loads[-1, 2, 3] = 0.0
    elif fault == "moving":
        velocity[-1, 0] = 0.1
    else:
        trace["pad_normal_force_n"][-1, 1] = 5.0
    result = evaluate_station(
        model, {"start_deg": 270, "sense": -1}, base, trace, poses, velocity, loads, 18, 18
    )
    assert not result["verified"]
