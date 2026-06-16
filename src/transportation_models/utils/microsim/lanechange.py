"""N-lane lane-change decision for one vehicle (Newbolt 2026 eq 6-14, Alg 3).

Generalizes Newbolt's two-lane incentive + random lane change to ``N`` lanes,
evaluated **per subject** so the main loop can apply it sequentially (Newbolt
Alg 1's per-vehicle ``for`` loop). A vehicle changes lanes only when:

* **alpha** (incentive, eq 9): blocked in its current lane — the forward gap
  to its leader is ``<= s_o``;
* **beta** (feasibility, eq 10): an adjacent lane is safe — forward gap
  (eq 6, bumper) and backward gap (eq 7, back-to-back) both exceed ``s_o``;
* **P** (eq 11): a weighted random draw (``lane_change_prob``) fires.

Selection among feasible adjacent lanes is by **lowest lane index** (a
deterministic first pass). Per spec decision 7 (DWPT in all lanes) the EV
charging-lane-seeking term (eq 13) is dropped: all vehicles use this rule. A
vehicle moves at most one lane per step.
"""

from __future__ import annotations

import numpy as np


def decide_lane_change(
    subject_pos: float,
    subject_lane: int,
    other_pos: np.ndarray,
    other_lane: np.ndarray,
    *,
    length: float,
    s_o: float,
    n_lanes: int,
    lane_change_prob: float,
    rng: np.random.Generator,
) -> int:
    """Return the subject's lane after a possible one-step change.

    Parameters
    ----------
    subject_pos : float
        Subject longitudinal position, meters.
    subject_lane : int
        Subject current lane (0-based).
    other_pos, other_lane : ndarray
        Positions (m) and lanes of all *other* active vehicles.
    length, s_o : float
        Vehicle length and desired safe spacing, meters.
    n_lanes : int
        Number of lanes.
    lane_change_prob : float
        Weighted probability a blocked, feasible change actually happens.
    rng : numpy.random.Generator
        RNG for the weighted draw.
    """
    # alpha: blocked in current lane (forward gap to own-lane leader <= s_o).
    same = other_pos[other_lane == subject_lane]
    ahead = same[same > subject_pos]
    own_gap = (ahead.min() - subject_pos - length) if ahead.size else np.inf
    if own_gap > s_o:
        return subject_lane  # not blocked -> no incentive

    # First feasible adjacent lane, lowest index first.
    target_found = -1
    for target in (subject_lane - 1, subject_lane + 1):
        if target < 0 or target >= n_lanes:
            continue
        tgt = other_pos[other_lane == target]
        leaders = tgt[tgt > subject_pos]
        # eq 6: forward bumper gap (leader back - subject front).
        sF = (leaders.min() - subject_pos - length) if leaders.size else np.inf
        followers = tgt[tgt <= subject_pos]
        # eq 7: backward back-to-back distance (lengths cancel).
        sB = (subject_pos - followers.max()) if followers.size else np.inf
        if sF > s_o and sB > s_o:
            target_found = target
            break

    if target_found >= 0 and rng.random() < lane_change_prob:
        return target_found
    return subject_lane
