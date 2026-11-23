import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools/triage_rbq_agent.py"
SPEC = importlib.util.spec_from_file_location("triage_rbq_agent", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def trial(speed, forward, treads, *, on_top=False):
    events = {"ascend_s": 0.0}
    if on_top:
        events["on_top_s"] = 10.0
    return {
        "speed_mps": speed,
        "maximum_forward_x_m": forward,
        "ascent_treads": treads,
        "events": events,
        "passed_round_trip": False,
        "termination": "illegal_contact",
        "first_illegal_contacts": [{"force_n": 12.0}],
    }


def result(checkpoint, trials):
    return {"checkpoint": checkpoint, "checkpoint_sha256": checkpoint + "-sha", "trials": trials}


def test_route_score_counts_platform_reach_and_progress():
    score = MODULE.route_score(result("a", [trial(0.3, -0.1, [1, 2], on_top=True)]))
    assert score["work_platform_reached"] == 1
    assert score["maximum_supported_tread_by_trial"] == [2]
    assert score["maximum_forward_progress_m_by_trial"] == [1.0]


def test_paired_effect_keeps_speed_pairing():
    before = result("a", [trial(0.3, -0.2, [1])])
    after = result("b", [trial(0.3, -0.1, [1, 2], on_top=True)])
    effect = MODULE.paired_effect(before, after)
    assert effect == [
        {
            "speed_mps": 0.3,
            "forward_progress_delta_m": 0.1,
            "supported_tread_delta": 1,
            "work_platform_reached_delta": 1,
        }
    ]


def test_paired_effect_rejects_different_speed_grids():
    before = result("a", [trial(0.3, -0.2, [1])])
    after = result("b", [trial(0.4, -0.1, [1])])
    try:
        MODULE.paired_effect(before, after)
    except ValueError as error:
        assert "same speeds" in str(error)
    else:
        raise AssertionError("different grids must fail")


def test_same_metrics_tolerates_platform_float_roundoff():
    assert MODULE.same_metrics({"x": [0.1 + 0.2]}, {"x": [0.3]})
    assert not MODULE.same_metrics({"x": [0.31]}, {"x": [0.3]})
