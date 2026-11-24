import math

import pytest

from airpoint.lab.stats import aggregate, block_summary, sequence_stats


def trial(src, dst, start, click, t0, t1, hit=True, path=None):
    return {"src": list(src), "dst": list(dst), "start": list(start), "click": list(click),
            "t0": t0, "t1": t1, "hit": hit, "path": path or []}


def hand_sequence():
    """Two targets 100 px apart on the x axis, four selections back and forth, 0.8 s each.
    Along the task axis the clicks land +2, +1, -3, -4 px from the target centre."""
    a, b = (0, 0), (100, 0)
    return {"D": 100, "W": 20, "n": 4, "trials": [
        trial(a, b, (0, 0), (102, 3), 0, 800),
        trial(b, a, (102, 3), (-1, 0), 800, 1600),
        trial(a, b, (-1, 0), (97, -2), 1600, 2400, hit=False),
        trial(b, a, (97, -2), (4, 1), 2400, 3200),
    ]}


def test_sequence_statistics_match_the_hand_calculation():
    s = sequence_stats(hand_sequence())
    sd = math.sqrt(26 / 3)                       # dx = 2, 1, -3, -4: mean -1, sum of squares 26
    assert s["sdx"] == pytest.approx(sd)
    assert s["we"] == pytest.approx(4.133 * sd)
    assert s["ae"] == pytest.approx((102 + 103 + 98 + 93) / 4)
    assert s["ide"] == pytest.approx(math.log2(99.0 / (4.133 * sd) + 1))
    assert s["mt"] == pytest.approx(0.8)
    assert s["tp"] == pytest.approx(s["ide"] / 0.8)
    assert (s["n"], s["errors"], s["flat"]) == (4, 1, False)


def test_a_sequence_with_no_spread_falls_back_to_the_nominal_width_and_says_so():
    a, b = (0, 0), (100, 0)
    seq = {"D": 100, "W": 20, "n": 2, "trials": [trial(a, b, a, b, 0, 500), trial(b, a, b, a, 500, 1000)]}
    s = sequence_stats(seq)
    assert s["flat"] and s["we"] == 20
    assert s["ide"] == pytest.approx(math.log2(100 / 20 + 1))


def test_path_measures_see_a_wandering_pointer():
    a, b = (0, 0), (100, 0)
    straight = [[100, 25, 0], [200, 50, 0], [300, 75, 0]]
    wobbly = [[100, 30, 40], [200, 95, 0], [300, 130, 0], [400, 95, 0], [500, 130, 0]]   # leaves and re-enters twice
    seq = {"D": 100, "W": 20, "n": 2, "trials": [trial(a, b, a, (100, 0), 0, 400, path=straight),
                                                   trial(a, b, a, (100, 0), 400, 1000, path=wobbly)]}
    s = sequence_stats(seq)
    assert s["trials"][0]["path_ratio"] == pytest.approx(1.0)
    assert s["trials"][0]["reentries"] == 0
    assert s["trials"][1]["path_ratio"] > 1.5
    assert s["trials"][1]["reentries"] == 2


def block(participant, condition, scale=1.0):
    seq = hand_sequence()
    for tr in seq["trials"]:                       # slower or faster participant, same clicks
        tr["t0"], tr["t1"] = tr["t0"] * scale, tr["t1"] * scale
    return {"kind": "test", "participant": participant, "condition": condition, "sequences": [seq, hand_sequence()]}


def test_block_summary_averages_its_sequences_and_counts_errors():
    b = block("P1", "mouse", scale=2.0)
    s = block_summary(b)
    fast, slow = sequence_stats(hand_sequence())["tp"], sequence_stats(b["sequences"][0])["tp"]
    assert s["tp"] == pytest.approx((fast + slow) / 2)
    assert s["trials"] == 8 and s["error_rate"] == pytest.approx(2 / 8)
    assert s["mt"] == pytest.approx((1.6 * 4 + 0.8 * 4) / 8)


def test_tutorial_blocks_are_summed_not_scored():
    s = block_summary({"kind": "tutorial", "steps": [{"name": "move", "seconds": 4.0}, {"name": "click", "seconds": 6.5}]})
    assert s == {"kind": "tutorial", "seconds": 10.5, "steps": 2}


def test_aggregate_is_per_condition_with_each_participant_counted_once():
    blocks = [block("P1", "mouse"), block("P1", "mouse", scale=2.0), block("P2", "mouse"),
              block("P1", "airpoint sign-click"), {"kind": "tutorial", "steps": []}]
    rows = {r["condition"]: r for r in aggregate(blocks)}
    assert set(rows) == {"mouse", "airpoint sign-click"}
    m = rows["mouse"]
    p1 = (block_summary(blocks[0])["tp"] + block_summary(blocks[1])["tp"]) / 2
    p2 = block_summary(blocks[2])["tp"]
    assert (m["participants"], m["blocks"], m["trials"]) == (2, 3, 24)
    assert m["tp"] == pytest.approx((p1 + p2) / 2)
    assert m["tp_sd"] == pytest.approx(abs(p1 - p2) / math.sqrt(2))
    assert rows["airpoint sign-click"]["tp_sd"] is None           # one participant: no spread to report
