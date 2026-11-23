import doctest
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))

import link_surface_stress  # noqa: E402
from link_surface_stress import (  # noqa: E402
    PROFILE_POINTS,
    LinkState,
    LinkTopology,
    end_mismatch,
    link_profile,
    surface_stress,
)
from stair_body_stress import JOINT_FAMILIES, SECTION_FAMILIES, _section_properties  # noqa: E402

from go2_wall.stair_loads import JOINT_NAMES  # noqa: E402

LINKS = len(JOINT_NAMES)


def _topology() -> LinkTopology:
    sections = []
    for family in JOINT_FAMILIES:
        spec = SECTION_FAMILIES[family]
        area, modulus, median = _section_properties(spec)
        sections.append((area, modulus, median, spec["wall_m"]))
    empty = ((),) * LINKS
    return LinkTopology(
        tuple(range(LINKS)),
        tuple(range(LINKS)),
        (("tip", -1),) * LINKS,
        empty,
        (),
        (),
        tuple(sections),
    )


def _cantilever(tip_force: np.ndarray, length: float = 0.3) -> LinkState:
    """Massless links along -z with a free tip load: M_p = (c - p) x F, M_c = 0."""
    p = np.zeros((LINKS, 3))
    c = np.tile([0.0, 0.0, -length], (LINKS, 1))
    force = np.tile(tip_force, (LINKS, 1))
    moment_p = np.cross(c - p, force)
    return LinkState(p, c, force, moment_p, force.copy(), np.zeros((LINKS, 3)))


def test_tip_load_moment_peaks_at_the_proximal_end_and_vanishes_at_the_tip():
    topology = _topology()
    state = _cantilever(np.array([200.0, 0.0, 0.0]))
    calf = JOINT_NAMES.index("RL_calf_joint")
    profile = link_profile(state, topology, calf)
    assert np.argmax(profile) == 0
    area, modulus, _, _ = topology.sections[calf]
    shear_only = np.sqrt(3) * 2 * 200.0 / area / 1e6
    assert profile[-1] == pytest.approx(shear_only)
    bending_root = 200.0 * 0.3 / modulus / 1e6
    assert profile[0] == pytest.approx(np.sqrt(bending_root**2 + 3 * (2 * 200.0 / area / 1e6) ** 2))
    assert np.all(np.diff(profile) <= 1e-9)
    assert end_mismatch(state, calf) == pytest.approx(0.0, abs=1e-9)
    assert len(profile) == len(PROFILE_POINTS)


def test_bending_sides_are_equal_and_the_neutral_axis_is_lower():
    topology = _topology()
    state = _cantilever(np.array([200.0, 0.0, 0.0]))
    calf = JOINT_NAMES.index("RL_calf_joint")
    points = np.array(
        [
            [0.02, 0.0, -0.05],  # tension/compression face in the bending plane
            [-0.02, 0.0, -0.05],  # opposite face
            [0.0, 0.02, -0.05],  # on the neutral axis side
            [0.02, 0.0, -0.25],  # same face nearer the tip
        ]
    )
    stress = surface_stress(state, topology, calf, points)
    assert stress[0] == pytest.approx(stress[1])
    assert stress[2] < 0.2 * stress[0]
    assert stress[3] < stress[0]


def test_reversing_the_load_keeps_the_field():
    topology = _topology()
    calf = JOINT_NAMES.index("FL_calf_joint")
    points = np.array([[0.02, 0.01, -0.1], [-0.01, 0.02, -0.2]])
    forward = surface_stress(_cantilever(np.array([150.0, 40.0, -300.0])), topology, calf, points)
    backward = surface_stress(_cantilever(-np.array([150.0, 40.0, -300.0])), topology, calf, points)
    np.testing.assert_allclose(forward, backward)
    assert np.all(forward > 0)


def test_trunk_on_four_hips_matches_the_simply_supported_beam():
    """Four hips at +-0.3 m carry the trunk's own 164.4 N, spread over 0.8 m.

    Front half at the centre cut: hips +82.2 N at 0.3 m, own weight -82.2 N at 0.2 m,
    so |M| = 82.2 * (0.3 - 0.2) = 8.22 N m and the shear is zero there.
    """
    from link_surface_stress import TrunkState, trunk_profile

    area, modulus, median = _section_properties(
        {"width_m": 0.2, "height_m": 0.148, "wall_m": 0.003}
    )
    hips = np.array([[0.3, 0.1, 0.0], [0.3, -0.1, 0.0], [-0.3, 0.1, 0.0], [-0.3, -0.1, 0.0]])
    points = np.vstack([hips, [[0.238, 0.0, 0.06]]])
    forces = np.zeros((5, 3))
    forces[:4, 2] = 164.4 / 4
    trunk = TrunkState(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        0.4,
        points,
        forces,
        np.zeros((5, 3)),
        -forces.sum(axis=0),
        -np.cross(points, forces).sum(axis=0),
        (area, modulus, median, 0.003),
    )
    profile, s = trunk_profile(trunk, samples=81)
    centre = int(np.argmin(abs(s)))
    assert profile[centre] == pytest.approx(8.22 / modulus / 1e6, rel=1e-6)
    assert profile[0] == pytest.approx(0.0, abs=1e-9)
    assert profile[-1] == pytest.approx(0.0, abs=1e-9)


def test_labels_doctests():
    assert doctest.testmod(link_surface_stress).failed == 0


def test_studio_dressing_keeps_the_physical_model():
    sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
    from render_stair_mission import build_model

    folder = Path(__file__).parents[1] / "evidence/stair-mission-20261121/stance-feedback/full"
    scene_path = folder / "scene.json"
    if not scene_path.exists():
        pytest.skip("pinned stair scene not present")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    plain, _ = build_model(scene)
    dressed, _ = build_model(scene, studio=True)
    for field in ("nq", "nv", "nu", "nbody", "ngeom", "njnt", "neq"):
        assert getattr(plain, field) == getattr(dressed, field)
    np.testing.assert_array_equal(plain.geom_size, dressed.geom_size)
    np.testing.assert_array_equal(plain.body_mass, dressed.body_mass)
    np.testing.assert_array_equal(plain.geom_friction, dressed.geom_friction)
    assert dressed.geom_matid[dressed.geom("floor").id] >= 0
    assert dressed.ntex == plain.ntex + 2
