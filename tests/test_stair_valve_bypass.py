import numpy as np
import pytest


def test_adjusted_feet_preserves_landings_and_enforces_tread_width(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from stair_valve_bypass import adjusted_feet

    feet = np.array(
        [[0.7, 0.19, 0.98], [0.7, -0.19, 0.98], [0.38, 0.19, 0.60], [0.38, -0.19, 0.60]]
    )
    moved = adjusted_feet(feet, 0.04, -0.03)
    assert [moved[leg][0] for leg in moved] == pytest.approx(feet[:, 0])
    assert [moved[leg][2] for leg in moved] == pytest.approx(feet[:, 2])
    assert moved["FL"][1] == pytest.approx(0.20)
    assert moved["FR"][1] == pytest.approx(-0.12)
    with pytest.raises(ValueError, match="610 mm"):
        adjusted_feet(feet, 0.2, 0.0)


def test_support_margin_sign(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from stair_valve_bypass import support_margin_m

    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    assert support_margin_m(square, np.array([0.0, 0.0])) == pytest.approx(1.0)
    assert support_margin_m(square, np.array([2.0, 0.0])) == pytest.approx(-1.0)


def test_ship_treads_keep_upward_floor_contact_normal(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from stair_valve_bypass import horizontal_stance

    feet = {
        leg: np.array([index, 0.0, index * 0.19])
        for index, leg in enumerate(("FL", "FR", "RL", "RR"))
    }
    stance = horizontal_stance(feet, np.array([True, True, False, True]))
    assert set(stance) == {"FL", "FR", "RR"}
    assert {surface for _, surface in stance.values()} == {"floor"}
    assert stance["RR"][0][2] == pytest.approx(0.57)
