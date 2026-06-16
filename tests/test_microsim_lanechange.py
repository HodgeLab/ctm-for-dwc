"""Tests for the per-subject N-lane lane-change decision (V2-V4 rework).

Newbolt eq 9-12 generalized to N lanes, evaluated per subject. Hand calcs use
length=5, s_o=25. Covers the eq-7 back-to-back backward gap (V2) and
lowest-feasible-index selection (V3).
"""

from __future__ import annotations

import numpy as np

from transportation_models.utils.microsim.lanechange import decide_lane_change

_LEN = 5.0
_SO = 25.0


def _rng():
    return np.random.default_rng(0)


def _decide(sub_pos, sub_lane, other_pos, other_lane, *, n_lanes=2, p=1.0):
    return decide_lane_change(
        sub_pos, sub_lane,
        np.asarray(other_pos, dtype=float), np.asarray(other_lane, dtype=int),
        length=_LEN, s_o=_SO, n_lanes=n_lanes, lane_change_prob=p, rng=_rng(),
    )


def test_blocked_changes_to_empty_adjacent_lane():
    # Subject at 100 blocked by leader at 120 (gap 15 <= 25); lane 1 empty.
    assert _decide(100.0, 0, [120.0], [0]) == 1


def test_unblocked_vehicle_stays():
    # Leader far ahead (gap 95 > 25) -> not blocked -> no change.
    assert _decide(100.0, 0, [200.0], [0]) == 0


def test_blocked_but_target_unsafe_stays():
    # Target lane 1 has a vehicle at 110 -> sF = 110-100-5 = 5 <= 25 -> unsafe.
    assert _decide(100.0, 0, [120.0, 110.0], [0, 1]) == 0


def test_prob_zero_never_changes():
    assert _decide(100.0, 0, [120.0], [0], p=0.0) == 0


def test_edge_lane_changes_inward_only():
    assert _decide(100.0, 0, [120.0], [0], n_lanes=2) == 1  # 0 -> 1
    assert _decide(100.0, 1, [120.0], [1], n_lanes=2) == 0  # top lane -> 0


def test_lowest_feasible_index_selected():
    # Subject in lane 1 blocked; lanes 0 and 2 both safe (empty) -> pick 0.
    assert _decide(100.0, 1, [120.0], [1], n_lanes=3) == 0


def test_eq7_backward_gap_is_back_to_back():
    # Subject 100 blocked; target lane 1 follower at 74 -> back-to-back
    # sB = 100-74 = 26 > 25 (feasible under eq 7). A bumper-gap convention
    # would give 26-5 = 21 <= 25 and forbid the change; eq 7 allows it.
    assert _decide(100.0, 0, [120.0, 74.0], [0, 1]) == 1


def test_eq7_backward_gap_blocks_when_follower_too_close():
    # Follower at 80 -> sB = 20 <= 25 -> unsafe -> stays.
    assert _decide(100.0, 0, [120.0, 80.0], [0, 1]) == 0


def test_reproducible_under_seed():
    args = (100.0, 1, np.array([118.0]), np.array([1]))
    kw = dict(length=_LEN, s_o=_SO, n_lanes=3, lane_change_prob=0.5)
    a = decide_lane_change(*args, rng=np.random.default_rng(7), **kw)
    b = decide_lane_change(*args, rng=np.random.default_rng(7), **kw)
    assert a == b


def test_output_in_range_and_adjacent():
    out = _decide(100.0, 2, [120.0], [2], n_lanes=4)
    assert 0 <= out < 4
    assert abs(out - 2) <= 1
