"""Hand-checked cases for advisory.metrics."""

import numpy as np
import pandas as pd
import pytest

from advisory.metrics import (
    bootstrap_task_success_ci,
    extra_cost_per_extra_correct,
    fixed_broken,
    summarize,
    task_success,
    verifier_metrics,
)


def _run(**overrides) -> dict:
    row = {
        "task_id": "t1",
        "condition": "dedicated",
        "run_number": 1,
        "initial_correct": False,
        "final_correct": False,
        "verdict": "accept",
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "latency_s": 0.0,
    }
    row.update(overrides)
    return row


def _bootstrap_interval(task_means: list[float], seed: int) -> tuple[float, float]:
    """Resample task means the way the metric is specified.

    ``numpy.random.default_rng(seed)`` draws 1,000 samples of the same length
    as ``task_means``, with replacement. Each sample's statistic is the mean
    of the drawn task means. The interval is NumPy's 2.5 and 97.5 percentiles.
    """
    means = np.asarray(task_means, dtype=float)
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(means), size=(1000, len(means)))
    stats = np.mean(means[drawn], axis=1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def test_verifier_that_accepts_everything():
    # Four runs, condition "weakened" (not a hard-coded name).
    # initial_correct: False, False, True, True.
    # Every verdict is "accept", so flagged is False on every run.
    #
    # actually_wrong = not initial_correct → True, True, False, False
    # confusion_matrix(..., labels=[False, True]) is [[TN, FP], [FN, TP]]:
    #   TN = 2  initially correct, not flagged
    #   FP = 0  initially correct, flagged
    #   FN = 2  initially wrong, not flagged
    #   TP = 0  initially wrong, flagged
    #
    # catch rate (recall) = TP / (TP + FN) = 0 / 2 = 0
    # false-alarm rate = FP / (FP + TN) = 0 / 2 = 0
    # precision = TP / (TP + FP) = 0 / 0 → None
    # abstention rate = 0 / 4 = 0
    # TP + FP = 0, so the MCC denominator is 0 and MCC is None.
    verdicts = ["accept", "accept", "accept", "accept"]
    initial = [False, False, True, True]
    metrics = verifier_metrics(verdicts, initial)

    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 2
    assert metrics["tp"] == 0
    assert metrics["catch_rate"] == 0
    assert metrics["false_alarm_rate"] == 0
    assert metrics["precision"] is None
    assert metrics["abstention_rate"] == 0
    assert metrics["mcc"] is None

    rows = [
        _run(condition="weakened", task_id="t1", run_number=1, initial_correct=False, verdict="accept"),
        _run(condition="weakened", task_id="t1", run_number=2, initial_correct=False, verdict="accept"),
        _run(condition="weakened", task_id="t2", run_number=1, initial_correct=True, verdict="accept"),
        _run(condition="weakened", task_id="t2", run_number=2, initial_correct=True, verdict="accept"),
    ]
    summary = summarize(rows)
    assert summary.loc["weakened", "mcc"] is None
    assert summary.loc["weakened", "catch_rate"] == 0


def test_perfect_verifier():
    # initial_correct False, False, True, True.
    # Verdicts reject, reject, accept, accept.
    # actually_wrong and flagged are both True, True, False, False.
    #
    #   TP = 2  wrong and flagged
    #   FN = 0  wrong and not flagged
    #   FP = 0  correct and flagged
    #   TN = 2  correct and not flagged
    #
    # catch rate = 2 / (2 + 0) = 1
    # false-alarm rate = 0 / (0 + 2) = 0
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    #     = (2*2 - 0) / sqrt(2*2*2*2) = 4/4 = 1
    verdicts = ["reject", "reject", "accept", "accept"]
    initial = [False, False, True, True]
    metrics = verifier_metrics(verdicts, initial)

    assert metrics["tp"] == 2
    assert metrics["fn"] == 0
    assert metrics["fp"] == 0
    assert metrics["tn"] == 2
    assert metrics["catch_rate"] == 1
    assert metrics["false_alarm_rate"] == 0
    assert metrics["mcc"] == 1

    rows = [
        _run(condition="cross_family", run_number=index + 1, initial_correct=correct, verdict=verdict)
        for index, (correct, verdict) in enumerate(
            [(False, "reject"), (False, "reject"), (True, "accept"), (True, "accept")]
        )
    ]
    summary = summarize(pd.DataFrame(rows))
    assert summary.loc["cross_family", "mcc"] == 1
    assert summary.loc["cross_family", "false_alarm_rate"] == 0
    assert summary.loc["cross_family", "fp"] == 0


def test_verifier_that_abstains_on_everything():
    # initial_correct False, False, True, True, and every verdict is "abstain".
    # Abstain is flagged, so flagged is True on all four runs.
    #
    #   TP = 2  wrong and flagged
    #   FP = 2  correct and flagged
    #   FN = 0
    #   TN = 0
    #
    # catch rate = 2 / (2 + 0) = 1
    # false-alarm rate = 2 / (2 + 0) = 1
    # abstention rate = 4 / 4 = 1
    # precision = 2 / (2 + 2) = 0.5
    # TN + FN = 0, so the MCC denominator is 0 and MCC is None.
    verdicts = ["abstain", "abstain", "abstain", "abstain"]
    initial = [False, False, True, True]
    metrics = verifier_metrics(verdicts, initial)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 2
    assert metrics["fn"] == 0
    assert metrics["tn"] == 0
    assert metrics["catch_rate"] == 1
    assert metrics["false_alarm_rate"] == 1
    assert metrics["abstention_rate"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["mcc"] is None

    rows = [
        _run(
            condition="dedicated",
            task_id="t1" if index < 2 else "t2",
            run_number=index + 1,
            initial_correct=correct,
            verdict="abstain",
        )
        for index, correct in enumerate(initial)
    ]
    summary = summarize(rows)
    assert summary.loc["dedicated", "catch_rate"] == 1
    assert summary.loc["dedicated", "false_alarm_rate"] == 1
    assert summary.loc["dedicated", "abstention_rate"] == 1
    assert summary.loc["dedicated", "mcc"] is None


def test_broken_when_correct_answer_becomes_wrong():
    # One run of each kind:
    #   1. initial correct, final wrong  → broken
    #   2. initial correct, final correct → neither
    #   3. initial wrong, final correct → fixed
    #   4. initial wrong, final wrong → neither
    #
    # broken count = 1, fixed count = 1
    # initially correct = 2, initially wrong = 2
    # fix rate = 1 / 2 = 0.5
    # break rate = 1 / 2 = 0.5
    # net gain = 0.5 - 0.5 = 0
    initial = [True, True, False, False]
    final = [False, True, True, False]
    metrics = fixed_broken(initial, final)

    assert metrics["broken"] == 1
    assert metrics["fixed"] == 1
    assert metrics["break_rate"] == 0.5
    assert metrics["fix_rate"] == 0.5
    assert metrics["net_gain"] == 0

    rows = [
        _run(
            condition="self",
            task_id=f"t{index}",
            run_number=1,
            initial_correct=initial_ok,
            final_correct=final_ok,
            verdict="accept",
        )
        for index, (initial_ok, final_ok) in enumerate(zip(initial, final), start=1)
    ]
    assert summarize(rows).loc["self", "broken"] == 1


def test_extra_cost_against_none():
    # Compared on per-run means, not totals.
    # none: 2 runs, one correct. costs 3 and 3.
    #   mean cost = 3, success rate = 1/2
    # dedicated: 1 run, correct, cost 9.
    #   mean cost = 9, success rate = 1
    #   extra cost = (9 - 3) / (1 - 1/2) = 12
    #   The old total formula would see 1 correct run against none's 1 and
    #   return None. The rate gap is what makes this 12.
    # self: one correct and one wrong, costs 8 and 12. success rate = 1/2,
    #   not higher than none, so None.
    #   mean tokens_in = (10 + 30) / 2 = 20
    #   mean tokens_out = (1 + 3) / 2 = 2
    # peer: both wrong. success rate = 0, which is below none, so None.
    # none against itself has no success-rate gap, so None.
    # none has no verifier, so MCC and n_missing_verdict are None.
    assert extra_cost_per_extra_correct(9, 1.0, 3, 0.5) == 12
    # Equal run counts: per-run means agree with the old total formula.
    # dedicated: 2 correct runs, costs 5 and 7 → mean 6, success 1
    # none: mean 3, success 1/2 → (6 - 3) / (1 - 1/2) = 6
    assert extra_cost_per_extra_correct(6, 1.0, 3, 0.5) == 6
    assert extra_cost_per_extra_correct(10, 0.5, 3, 0.5) is None
    assert extra_cost_per_extra_correct(4.5, 0.0, 3, 0.5) is None

    rows = [
        _run(condition="none", task_id="n1", final_correct=True, initial_correct=True, verdict=None, cost_usd=3),
        _run(condition="none", task_id="n2", final_correct=False, initial_correct=True, verdict=None, cost_usd=3),
        _run(
            condition="dedicated",
            task_id="d1",
            final_correct=True,
            initial_correct=True,
            verdict="accept",
            cost_usd=9,
        ),
        _run(
            condition="self",
            task_id="s1",
            final_correct=True,
            verdict="accept",
            cost_usd=8,
            tokens_in=10,
            tokens_out=1,
        ),
        _run(
            condition="self",
            task_id="s2",
            final_correct=False,
            verdict="accept",
            cost_usd=12,
            tokens_in=30,
            tokens_out=3,
        ),
        _run(condition="peer", task_id="p1", final_correct=False, verdict="accept", cost_usd=4),
        _run(condition="peer", task_id="p2", final_correct=False, verdict="accept", cost_usd=5),
    ]
    summary = summarize(rows)

    assert summary.loc["dedicated", "extra_cost_per_extra_correct"] == 12
    assert summary.loc["dedicated", "total_cost_usd"] == 9
    assert summary.loc["dedicated", "mean_cost_usd"] == 9
    assert summary.loc["self", "extra_cost_per_extra_correct"] is None
    assert summary.loc["self", "mean_tokens_in"] == 20
    assert summary.loc["self", "mean_tokens_out"] == 2
    assert summary.loc["peer", "extra_cost_per_extra_correct"] is None
    assert summary.loc["none", "extra_cost_per_extra_correct"] is None
    assert summary.loc["none", "total_cost_usd"] == 6
    assert summary.loc["none", "mean_cost_usd"] == 3
    assert summary.loc["none", "mcc"] is None
    assert summary.loc["none", "n_missing_verdict"] is None
    assert set(summary.index) == {"none", "dedicated", "self", "peer"}


def test_bootstrap_same_seed():
    # One task, two runs, final_correct True then False.
    # The task mean is (1 + 0) / 2 = 0.5.
    # Resampling that single mean always yields 0.5, so both percentiles
    # are 0.5. The same seed therefore returns the same interval.
    # Run-level task success is also 1/2 = 0.5.
    # A bootstrap of the two runs themselves would not be a point interval.
    point_rows = [
        _run(task_id="only", run_number=1, final_correct=True),
        _run(task_id="only", run_number=2, final_correct=False),
    ]
    assert task_success([True, False]) == 0.5
    assert bootstrap_task_success_ci(["only", "only"], [True, False], seed=0) == (0.5, 0.5)
    assert bootstrap_task_success_ci(["only", "only"], [True, False], seed=0) == (
        bootstrap_task_success_ci(["only", "only"], [True, False], seed=1)
    )
    point = summarize(point_rows, seed=0)
    point_again = summarize(point_rows, seed=0)
    assert point.loc["dedicated", "task_success"] == 0.5
    assert point.loc["dedicated", "task_success_ci_low"] == 0.5
    assert point.loc["dedicated", "task_success_ci_high"] == 0.5
    assert point_again.loc["dedicated", "task_success_ci_low"] == point.loc["dedicated", "task_success_ci_low"]
    assert point_again.loc["dedicated", "task_success_ci_high"] == point.loc["dedicated", "task_success_ci_high"]

    # Task "b" is listed first (final_correct 1) and task "a" second (final_correct 0).
    # Task ids are sorted before resampling, so the means are [0, 1] for a then b,
    # not the first-seen order [1, 0]. seed 0 and seed 1 are each checked against
    # that sorted procedure, and two summarize calls with seed 0 match each other.
    seeded_rows = [
        _run(task_id="b", run_number=1, final_correct=True),
        _run(task_id="a", run_number=1, final_correct=False),
    ]
    expected_seed_0 = _bootstrap_interval([0.0, 1.0], seed=0)
    expected_seed_1 = _bootstrap_interval([0.0, 1.0], seed=1)
    first = summarize(seeded_rows, seed=0)
    second = summarize(seeded_rows, seed=0)
    other = summarize(seeded_rows, seed=1)
    assert (
        first.loc["dedicated", "task_success_ci_low"],
        first.loc["dedicated", "task_success_ci_high"],
    ) == expected_seed_0
    assert (
        second.loc["dedicated", "task_success_ci_low"],
        second.loc["dedicated", "task_success_ci_high"],
    ) == expected_seed_0
    assert (
        other.loc["dedicated", "task_success_ci_low"],
        other.loc["dedicated", "task_success_ci_high"],
    ) == expected_seed_1

    # Unequal repeats. Task a has two incorrect runs, task b has one correct run.
    # Task success is the share of runs: 1/3.
    # The mean of the task means would be (0 + 1) / 2 = 0.5, which is not
    # the point estimate. Sorted by task id, the resampled means are [0, 1].
    unequal = [
        _run(task_id="a", run_number=1, final_correct=False),
        _run(task_id="a", run_number=2, final_correct=False),
        _run(task_id="b", run_number=1, final_correct=True),
    ]
    unequal_summary = summarize(unequal, seed=0)
    assert unequal_summary.loc["dedicated", "task_success"] == 1 / 3
    assert (
        unequal_summary.loc["dedicated", "task_success_ci_low"],
        unequal_summary.loc["dedicated", "task_success_ci_high"],
    ) == _bootstrap_interval([0.0, 1.0], seed=0)


def test_missing_verdict_is_left_out_of_verifier_metrics():
    # Condition "peer", three runs. The third verdict is missing.
    # Scored runs only:
    #   initial wrong, verdict reject → flagged → TP
    #   initial correct, verdict accept → not flagged → TN
    # Dropped: initial wrong, verdict None. n_missing_verdict = 1.
    #
    # On the two scored runs, TP = 1, FN = 0, FP = 0, TN = 1.
    # catch rate = 1 / (1 + 0) = 1
    # false-alarm rate = 0 / (0 + 1) = 0
    # abstention rate = 0 / 2 = 0
    # MCC = (1*1 - 0*0) / sqrt(1*1*1*1) = 1
    # If the missing run were kept and treated as not flagged, it would be an
    # FN and the catch rate would be 1 / 2 instead of 1.
    verdicts = ["reject", "accept", None]
    initial = [False, True, False]
    metrics = verifier_metrics(verdicts, initial, condition="peer")

    assert metrics["n_missing_verdict"] == 1
    assert metrics["tp"] == 1
    assert metrics["fn"] == 0
    assert metrics["fp"] == 0
    assert metrics["tn"] == 1
    assert metrics["catch_rate"] == 1
    assert metrics["false_alarm_rate"] == 0
    assert metrics["abstention_rate"] == 0
    assert metrics["mcc"] == 1

    rows = [
        _run(condition="peer", task_id="t1", run_number=1, initial_correct=False, verdict="reject"),
        _run(condition="peer", task_id="t2", run_number=1, initial_correct=True, verdict="accept"),
        _run(condition="peer", task_id="t3", run_number=1, initial_correct=False, verdict=None),
    ]
    summary = summarize(rows)
    assert summary.loc["peer", "n_missing_verdict"] == 1
    assert summary.loc["peer", "catch_rate"] == 1
    assert summary.loc["peer", "tp"] == 1
    assert summary.loc["peer", "fn"] == 0


def test_bootstrap_shuffled_rows_same_interval():
    # Task a is one incorrect run, so its mean is 0.
    # Task b is one correct run, so its mean is 1.
    # Sorted by task id the means are [0, 1] either way the rows are ordered.
    # The same seed draws the same indexes into that sorted array.
    rows = [
        _run(task_id="b", run_number=1, final_correct=True),
        _run(task_id="a", run_number=1, final_correct=False),
    ]
    shuffled = list(reversed(rows))
    expected = _bootstrap_interval([0.0, 1.0], seed=0)
    original = summarize(rows, seed=0)
    reordered = summarize(shuffled, seed=0)
    original_interval = (
        original.loc["dedicated", "task_success_ci_low"],
        original.loc["dedicated", "task_success_ci_high"],
    )
    assert original_interval == (
        reordered.loc["dedicated", "task_success_ci_low"],
        reordered.loc["dedicated", "task_success_ci_high"],
    )
    assert original_interval == expected
    assert bootstrap_task_success_ci(["b", "a"], [True, False], seed=0) == bootstrap_task_success_ci(
        ["a", "b"], [False, True], seed=0
    )


def test_success_delta_and_net_gain():
    # 100 runs. 90 start correct, so 10 start wrong.
    # 5 of the wrong drafts become correct (fixed).
    # 9 of the correct drafts become wrong (broken).
    # The other 81 correct drafts stay correct, and the other 5 wrong drafts stay wrong.
    #
    # fix rate = 5 / 10 = 0.5
    # break rate = 9 / 90 = 0.1
    # net gain = 0.5 - 0.1 = 0.40
    #
    # success_delta = (fixed - broken) / n_runs = (5 - 9) / 100 = -0.04
    # initial success = 90/100 = 0.90
    # final success = (90 - 9 + 5) / 100 = 0.86
    # 0.86 - 0.90 = -0.04
    initial = [True] * 90 + [False] * 10
    final = [True] * 81 + [False] * 9 + [True] * 5 + [False] * 5
    metrics = fixed_broken(initial, final)

    assert metrics["fixed"] == 5
    assert metrics["broken"] == 9
    assert metrics["net_gain"] == 0.40
    assert metrics["success_delta"] == -0.04

    rows = [
        _run(
            condition="dedicated",
            task_id=f"t{index}",
            run_number=1,
            initial_correct=initial_ok,
            final_correct=final_ok,
            verdict="accept",
        )
        for index, (initial_ok, final_ok) in enumerate(zip(initial, final))
    ]
    summary = summarize(rows)
    assert summary.loc["dedicated", "net_gain"] == 0.40
    assert summary.loc["dedicated", "success_delta"] == -0.04
    assert summary.loc["dedicated", "task_success"] == 0.86


def test_mcc_is_zero_when_defined():
    # One run in each confusion-matrix cell, so every margin is positive.
    #   wrong, reject   → TP
    #   wrong, accept   → FN
    #   correct, reject → FP
    #   correct, accept → TN
    # TP = FP = FN = TN = 1
    # numerator = TP*TN - FP*FN = 1 - 1 = 0
    # denominator = sqrt((2)*(2)*(2)*(2)) = 4
    # MCC = 0 / 4 = 0, which is defined, unlike the all-accept and all-abstain cases.
    verdicts = ["reject", "accept", "reject", "accept"]
    initial = [False, False, True, True]
    metrics = verifier_metrics(verdicts, initial, condition="peer")

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["mcc"] == 0

    rows = [
        _run(condition="peer", task_id=f"t{index}", run_number=1, initial_correct=correct, verdict=verdict)
        for index, (correct, verdict) in enumerate(zip(initial, verdicts))
    ]
    assert summarize(rows).loc["peer", "mcc"] == 0


def test_duplicate_run_is_rejected():
    # The same task, condition, and run number twice is not two runs.
    duplicate = [
        _run(task_id="t1", condition="peer", run_number=1),
        _run(task_id="t1", condition="peer", run_number=1, final_correct=True),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        summarize(duplicate)

    # The same task and run number under two conditions is allowed.
    both = [
        _run(task_id="t1", condition="peer", run_number=1),
        _run(task_id="t1", condition="self", run_number=1),
    ]
    assert set(summarize(both).index) == {"peer", "self"}


def test_unknown_verdict_is_rejected():
    # "approve" is not accept, reject, abstain, or a missing verdict.
    with pytest.raises(ValueError, match="unknown verdict"):
        summarize([_run(verdict="approve")])

    # cost_usd is one of the required columns.
    incomplete = _run()
    del incomplete["cost_usd"]
    with pytest.raises(ValueError, match="missing columns: cost_usd"):
        summarize([incomplete])


def test_two_models_share_a_file_and_compare_extra_cost_within_model():
    # Same task, condition and run number under two models is allowed once
    # model is present. Extra cost uses none for that model only.
    #
    # model a: none mean cost 3, success 1/2; dedicated mean cost 9, success 1
    #   extra = (9 - 3) / (1 - 1/2) = 12
    # model b: none mean cost 1, success 0; dedicated mean cost 2, success 1
    #   extra = (2 - 1) / (1 - 0) = 1
    rows = [
        _run(
            condition="none",
            model="a",
            task_id="t1",
            run_number=1,
            initial_correct=True,
            final_correct=True,
            verdict=None,
            cost_usd=3,
        ),
        _run(
            condition="none",
            model="a",
            task_id="t2",
            run_number=1,
            initial_correct=True,
            final_correct=False,
            verdict=None,
            cost_usd=3,
        ),
        _run(
            condition="dedicated",
            model="a",
            task_id="t1",
            run_number=1,
            initial_correct=True,
            final_correct=True,
            verdict="accept",
            cost_usd=9,
        ),
        _run(
            condition="none",
            model="b",
            task_id="t1",
            run_number=1,
            initial_correct=False,
            final_correct=False,
            verdict=None,
            cost_usd=1,
        ),
        _run(
            condition="dedicated",
            model="b",
            task_id="t1",
            run_number=1,
            initial_correct=False,
            final_correct=True,
            verdict="reject",
            cost_usd=2,
        ),
    ]
    summary = summarize(rows)
    assert set(summary.index) == {("none", "a"), ("dedicated", "a"), ("none", "b"), ("dedicated", "b")}
    assert summary.loc[("dedicated", "a"), "extra_cost_per_extra_correct"] == 12
    assert summary.loc[("dedicated", "b"), "extra_cost_per_extra_correct"] == 1
    assert summary.loc[("none", "a"), "n_failed"] == 0

    # The same model twice with the same identity is still a duplicate.
    with pytest.raises(ValueError, match="duplicate"):
        summarize(
            [
                _run(condition="peer", model="a", task_id="t1", run_number=1),
                _run(condition="peer", model="a", task_id="t1", run_number=1, final_correct=True),
            ]
        )


def test_failed_run_is_left_out_of_rates():
    # Three dedicated runs. Two are scored: one wrong-and-flagged, one
    # correct-and-accepted. The third is status failed and must not change
    # catch rate or task success, only n_failed.
    #
    # Scored: TP = 1, TN = 1 → catch rate = 1, task success = 1/2.
    # If the failed run (final_correct True) were kept, task success would be 2/3.
    rows = [
        _run(
            condition="dedicated",
            task_id="t1",
            run_number=1,
            initial_correct=False,
            final_correct=False,
            verdict="reject",
            status="unchecked",
        ),
        _run(
            condition="dedicated",
            task_id="t2",
            run_number=1,
            initial_correct=True,
            final_correct=True,
            verdict="accept",
            status="accepted",
        ),
        _run(
            condition="dedicated",
            task_id="t3",
            run_number=1,
            initial_correct=True,
            final_correct=True,
            verdict=None,
            status="failed",
            cost_usd=99,
            tokens_in=1000,
        ),
    ]
    summary = summarize(rows)
    assert summary.loc["dedicated", "n_failed"] == 1
    assert summary.loc["dedicated", "task_success"] == 0.5
    assert summary.loc["dedicated", "catch_rate"] == 1
    assert summary.loc["dedicated", "tp"] == 1
    assert summary.loc["dedicated", "tn"] == 1
    # Failed cost counts in the total; averages use scored runs only.
    assert summary.loc["dedicated", "total_cost_usd"] == 99
    assert summary.loc["dedicated", "mean_cost_usd"] == 0
    assert summary.loc["dedicated", "mean_tokens_in"] == 0


def test_all_failed_group_returns_n_failed_and_none_rates():
    # none has one scored run. self is entirely failed: still a row, with
    # n_failed set and every rate / CI / cost average as None. total cost
    # still sums the failed runs.
    rows = [
        _run(
            condition="none",
            model="m",
            task_id="t1",
            run_number=1,
            initial_correct=True,
            final_correct=True,
            verdict=None,
            status="unchecked",
            cost_usd=1,
        ),
        _run(
            condition="self",
            model="m",
            task_id="t1",
            run_number=1,
            initial_correct=False,
            final_correct=False,
            verdict=None,
            status="failed",
            cost_usd=4,
        ),
        _run(
            condition="self",
            model="m",
            task_id="t2",
            run_number=1,
            initial_correct=False,
            final_correct=False,
            verdict=None,
            status="failed",
            cost_usd=6,
        ),
    ]
    summary = summarize(rows)
    assert ("self", "m") in summary.index
    self_row = summary.loc[("self", "m")]
    assert self_row["n_failed"] == 2
    assert self_row["task_success"] is None
    assert self_row["task_success_ci_low"] is None
    assert self_row["task_success_ci_high"] is None
    assert self_row["catch_rate"] is None
    assert self_row["fix_rate"] is None
    assert self_row["mean_cost_usd"] is None
    assert self_row["mean_tokens_in"] is None
    assert self_row["median_latency_s"] is None
    assert self_row["extra_cost_per_extra_correct"] is None
    assert self_row["total_cost_usd"] == 10
    assert summary.loc[("none", "m"), "task_success"] == 1.0


def test_failed_run_cost_counts_in_total_not_mean():
    # One scored run (cost 2) and one failed run (cost 8). Total is 10;
    # mean cost is the scored mean 2.
    rows = [
        _run(
            condition="peer",
            task_id="t1",
            run_number=1,
            initial_correct=True,
            final_correct=True,
            verdict="accept",
            status="accepted",
            cost_usd=2,
            tokens_in=10,
            latency_s=1.0,
        ),
        _run(
            condition="peer",
            task_id="t2",
            run_number=1,
            initial_correct=False,
            final_correct=False,
            verdict=None,
            status="failed",
            cost_usd=8,
            tokens_in=999,
            latency_s=99.0,
        ),
    ]
    summary = summarize(rows)
    assert summary.loc["peer", "n_failed"] == 1
    assert summary.loc["peer", "total_cost_usd"] == 10
    assert summary.loc["peer", "mean_cost_usd"] == 2
    assert summary.loc["peer", "mean_tokens_in"] == 10
    assert summary.loc["peer", "median_latency_s"] == 1.0


def test_unknown_status_is_rejected():
    # "Failed" is not one of unchecked / failed / accepted / revised /
    # human_review.
    with pytest.raises(ValueError, match="unknown status"):
        summarize([_run(status="Failed")])


def test_single_model_rows_without_model_column_are_unchanged():
    # No model column: one summary row per condition, indexed by condition
    # alone, and none is the shared baseline for every condition.
    rows = [
        _run(condition="none", task_id="n1", final_correct=True, initial_correct=True, verdict=None, cost_usd=3),
        _run(condition="none", task_id="n2", final_correct=False, initial_correct=True, verdict=None, cost_usd=3),
        _run(
            condition="dedicated",
            task_id="d1",
            final_correct=True,
            initial_correct=True,
            verdict="accept",
            cost_usd=9,
        ),
    ]
    summary = summarize(rows)
    assert list(summary.index) == ["none", "dedicated"]
    assert summary.index.names == ["condition"]
    assert summary.loc["dedicated", "extra_cost_per_extra_correct"] == 12
    assert summary.loc["none", "n_failed"] == 0
    assert summary.loc["dedicated", "n_failed"] == 0
