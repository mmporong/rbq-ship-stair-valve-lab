import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from go2_wall import grasp_motion, stair_valve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from stair_valve_swing_order import (
    OPEN_APERTURE_TOLERANCE_M,
    finger_trace,
    reorder_stances,
    set_open_hand,
    support_signature,
)


def stance(rl_x=0.54, rr_x=0.54):
    return {
        "FL": (np.array([0.86, 0.19, 1.17]), "floor"),
        "FR": (np.array([0.86, -0.19, 1.17]), "floor"),
        "RL": (np.array([rl_x, 0.19, 0.79]), "floor"),
        "RR": (np.array([rr_x, -0.19, 0.79]), "floor"),
    }


def test_reorder_moves_rr_then_rl_without_changing_targets():
    before = stance()
    rl_after = stance(rl_x=0.70)
    rr_after = stance(rl_x=0.70, rr_x=0.70)

    rr_first, final = reorder_stances(before, rl_after, rr_after)

    assert np.array_equal(rr_first["RL"][0], before["RL"][0])
    assert np.array_equal(rr_first["RR"][0], rr_after["RR"][0])
    assert np.array_equal(final["RL"][0], rl_after["RL"][0])
    assert np.array_equal(final["RR"][0], rr_after["RR"][0])
    assert support_signature(final) == support_signature(rr_after)


def test_support_signature_tracks_tread_multiset_not_in_tread_xy_offsets():
    original = stance(rl_x=0.70, rr_x=0.70)
    adjusted = {
        leg: (point + np.array([0.02, -0.03, 0.0]), surface)
        for leg, (point, surface) in original.items()
    }

    assert support_signature(adjusted) == support_signature(original)
    assert support_signature(original) == Counter({("floor", 6): 2, ("floor", 4): 2})


def test_open_hand_pose_sets_both_slides_and_aperture():
    model, _ = stair_valve.build_grasp_scene(50.0, 0.19, [0.9, 0.0, 1.58], 270.0, -1, 0.12, 10.0)
    qpos = model.qpos0.copy()
    set_open_hand(model, qpos)
    trace = finger_trace(model, np.tile(qpos, (3, 1)))

    np.testing.assert_allclose(trace["upper_m"], grasp_motion.OPEN_FINGER_M)
    np.testing.assert_allclose(trace["lower_m"], grasp_motion.OPEN_FINGER_M)
    np.testing.assert_allclose(trace["aperture_m"], 2 * grasp_motion.OPEN_FINGER_M)


def test_public_track_and_dynamic_raw_audit_actual_open_aperture():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "evidence/stair-mission-20261121/bypass/swing-order"
    with np.load(evidence / "static-result.track.npz", allow_pickle=False) as track:
        model, _ = stair_valve.build_grasp_scene(
            50.0, 0.19, [0.9, 0.0, 1.58], 270.0, -1, 0.12, 10.0
        )
        planned = finger_trace(model, track["qpos"])
        np.testing.assert_allclose(planned["upper_m"], grasp_motion.OPEN_FINGER_M)
        np.testing.assert_allclose(planned["lower_m"], grasp_motion.OPEN_FINGER_M)

    with np.load(evidence / "dynamic-result.raw.npz", allow_pickle=False) as raw:
        actual = finger_trace(model, raw["qpos"])
        np.testing.assert_allclose(actual["aperture_m"], raw["extra"][:, 10])
        assert np.isclose(actual["upper_m"][0], grasp_motion.OPEN_FINGER_M, atol=1e-6)
        assert np.isclose(actual["lower_m"][0], grasp_motion.OPEN_FINGER_M, atol=1e-6)
        assert actual["aperture_m"].min() >= (
            2 * grasp_motion.OPEN_FINGER_M - OPEN_APERTURE_TOLERANCE_M
        )
        np.testing.assert_allclose(np.diff(raw["physics_time_s"]), model.opt.timestep)

    report = json.loads((evidence / "dynamic-result.json").read_text(encoding="utf-8"))
    assert report["physics"]["actuation_period_s"] == 0.005
    assert report["finger_drive"]["watch_aperture_matches_qpos"]
    assert report["finger_drive"]["audit"]["qpos_write_count_after_start"] == 0
    assert report["finger_drive"]["audit"]["qvel_write_count_after_start"] == 0
    assert all(not Path(key).is_absolute() for key in report["hashes_sha256"])
