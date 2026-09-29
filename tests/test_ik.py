import numpy as np

from so101.ik import fk, grasp_pose, ik, load_chain, tilt_deg, top_down

CHAIN = load_chain()


def _random_q(rng, n):
    lo, hi = CHAIN.limits[:, 0], CHAIN.limits[:, 1]
    return rng.uniform(lo, hi, size=(n, 5))


def test_fk_zero_pose_matches_viam():
    # Arm extended forward at q=0; value cross-checked in the Phase 0 design review.
    p = fk(CHAIN, np.zeros(5))[:3, 3]
    np.testing.assert_allclose(p, [0.393, 0.0, 0.228], atol=0.002)


def test_ik_round_trip_full_pose():
    rng = np.random.default_rng(0)
    ok_count = 0
    for q in _random_q(rng, 100):
        T = fk(CHAIN, q)
        q_sol, ok = ik(CHAIN, T, np.zeros(5))
        T_sol = fk(CHAIN, q_sol)
        if ok:
            assert np.linalg.norm(T_sol[:3, 3] - T[:3, 3]) < 1e-3
        ok_count += ok
    assert ok_count >= 98, ok_count


def test_ik_position_only():
    # Random in-limit q includes rare folded poses near the limits; allow one miss.
    rng = np.random.default_rng(1)
    ok_count = sum(ik(CHAIN, fk(CHAIN, q), np.zeros(5), rot_weight=0)[1] for q in _random_q(rng, 100))
    assert ok_count >= 99, ok_count


def test_top_down_grasp_reachable_at_nominal_spawn():
    T = grasp_pose(np.array([0.20, 0.0, 0.015]), yaw=0.3)
    q, ok = ik(CHAIN, T, np.zeros(5))
    assert ok
    assert tilt_deg(fk(CHAIN, q)) < 1.0


def test_grasp_pose_offsets_along_tool_x_only():
    p = np.array([0.20, 0.05, 0.015])
    d = grasp_pose(p, 0.7)[:3, 3] - top_down(p, 0.7)[:3, 3]
    assert abs(np.linalg.norm(d) - 0.0139) < 1e-9 and abs(d[2]) < 1e-9
