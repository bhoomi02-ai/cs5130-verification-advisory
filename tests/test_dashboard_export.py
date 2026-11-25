"""The web dashboard's data export (src/advisory/dashboard_export.py, dashboard/build_data.py).

Builds from a tiny fixture written into tmp_path: 2 tasks (S01 and its
variant T01-v001) x 2 runs, with none, self and dedicated finished and peer
half done. The fixture deliberately contains answer-key values and account
identifiers, so the tests can show that neither reaches the exported files.
"""

import importlib.util
import json
import re

import pytest

from advisory import dashboard_export as ex

TASKS = ["S01", "T01-v001"]
REPEATS = 2
ORG = "org_01abcdefghijklmnopqrstuvwx"
GROQ_KEY = "gsk_ABCDEFGHIJKLMNOPQRSTUV123456"
EMAIL = "someone.private@example.com"
HOME = "/Users/someone/cs5130/results"
ERROR_ID = "S01_policy_violation_sell_aapl"  # in data/evaluation/seeded_error_labels.csv
CONTROL_ID = "S01_none_reference"

_spec = importlib.util.spec_from_file_location("build_data", ex.ROOT / "dashboard" / "build_data.py")
build_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_data)


def _row(task, condition, run, initial, final, status, verdict=None, second=None, cost=0.02):
    return {
        "run_id": f"{condition}-m-{task}-r{run}",
        "draft_run_id": f"draft-m-{task}-r{run}",
        "task_id": task,
        "condition": condition,
        "run_number": run,
        "model": "gemini-3.6-flash",
        "temperature": 0.7,
        "top_p": 0.9,
        "thinking_level": None,
        "initial_correct": initial,
        "final_correct": final,
        "initial_format_failure": False,
        "final_format_failure": False,
        "initial_status_correct": True,
        "final_status_correct": True,
        "verdict": verdict,
        "second_verdict": second,
        "verifier_calls": 0 if condition == "none" else 1,
        "status": status,
        "error": None,
        "tokens_in": 100,
        "tokens_out": 50,
        "cost_usd": cost,
        "latency_s": 1.0,
        "n_calls": 1,
        "n_failed_calls": 0,
    }


def _fixture_rows():
    rows = []
    # Drafts: S01 r1 right, S01 r2 wrong, T01-v001 r1 right, T01-v001 r2 right.
    drafts = {("S01", 1): True, ("S01", 2): False, ("T01-v001", 1): True, ("T01-v001", 2): True}
    for (task, run), right in drafts.items():
        rows.append(_row(task, "none", run, right, right, "unchecked", cost=0.01))
        # self: misses the wrong draft, false alarm on T01-v001 r2 that sends it to human review (revision not accepted)
        if (task, run) == ("T01-v001", 2):
            rows.append(_row(task, "self", run, True, False, "human_review", "reject", "reject"))
        else:
            rows.append(_row(task, "self", run, right, right, "accepted", "accept"))
        # dedicated: catches and fixes the wrong draft; one unreadable verdict goes to human review
        if not right:
            rows.append(_row(task, "dedicated", run, False, True, "revised", "reject", "accept", cost=0.05))
        elif (task, run) == ("T01-v001", 1):
            rows.append(_row(task, "dedicated", run, True, True, "human_review", None, cost=0.05))
        else:
            rows.append(_row(task, "dedicated", run, True, True, "accepted", "accept", cost=0.05))
    # peer: only 2 of 4 runs saved, one of them failed
    rows.append(_row("S01", "peer", 1, True, True, "accepted", "accept"))
    failed = _row("S01", "peer", 2, False, False, "failed")
    failed["error"] = f"ProviderError: 429 rate limit for organization `{ORG}`; key {GROQ_KEY}; contact {EMAIL}"
    rows.append(failed)
    return rows


def _answer(row):
    return {
        "run_id": row["run_id"],
        "task_id": row["task_id"],
        "condition": row["condition"],
        "run_number": row["run_number"],
        "draft": '{"metrics": {"observations": [{"measure": "VALUE", "value": 145470.34}]}}',
        "revision": None,
        "final": '```json\n{"metrics": {"observations": [{"measure": "VALUE", "value": 145469.35}]}}\n```',
        "verdict": row["verdict"],
        "second_verdict": row["second_verdict"],
        "status": row["status"],
        "verifier_calls": row["verifier_calls"],
        "reviews": [{"stage": "first", "verdict": row["verdict"], "issues": [f"saved under {HOME}"], "reason": "r",
                     "calls": 1, "raw": "raw text", "messages": [{"role": "user", "content": "long prompt"}]}],
        "draft_check": {"correct": row["initial_correct"], "format_failure": False, "checks": [
            {"check_id": "c1", "check_type": "INDEPENDENT_RESULTS", "passed": row["initial_correct"],
             "observed": {"fund_b_final": 145470.34},
             "details": {"reason": "value_mismatch", "items": [
                 {"reason": "value_mismatch", "metric": "fund_b_final", "observed": 145470.34, "expected": 145469.345}]}},
        ]},
        "final_check": {"correct": row["final_correct"], "format_failure": False, "checks": []},
    }


def _seeded(answer_id, condition, seeded_error, verdict):
    return {
        "run_id": f"seeded-{condition}-gemini-3.6-flash-{answer_id}-r1",
        "answer_id": answer_id,
        "scenario_id": "S01",
        "error_type": "policy_violation" if seeded_error else "none",
        "seeded_error": seeded_error,
        "condition": condition,
        "run_number": 1,
        "verifier_model": "gemini-3.6-flash",
        "verdict": verdict,
        "status": "reviewed",
        "reviews": [{"stage": "first", "verdict": verdict,
                     "issues": ["AAPL is sold but the policy says AAPL may not be sold"] if verdict == "reject" else [],
                     "reason": "", "error": None}],
        "error": None,
        "cost_usd": 0.01,
    }


@pytest.fixture
def built(tmp_path):
    results = tmp_path / "results"
    (results / "answers").mkdir(parents=True)
    (results / "seeded").mkdir()
    rows = _fixture_rows()
    (results / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for row in rows:
        if row["status"] != "failed":
            (results / "answers" / f"{row['run_id']}.json").write_text(json.dumps(_answer(row)))

    log = []
    for i, row in enumerate(rows):  # first calls in reverse file order, so replay order differs from file order
        log.append({"timestamp": f"2026-11-20T10:{59 - i:02d}:00+00:00", "run_id": row["run_id"], "task_id": row["task_id"],
                    "condition": row["condition"], "agent": "verifier", "model": "gemini-3.6-flash",
                    "cost_usd": 0.001, "tokens_in": 1, "tokens_out": 1, "error": row["error"], "raw_output": "{}"})
    log.append({"timestamp": "2026-11-17T09:00:00+00:00", "run_id": "live-dedicated-wrong", "task_id": "S04",
                "condition": "dedicated-check", "agent": "verifier", "model": "gemini-3.6-flash", "cost_usd": 0.002,
                "raw_output": '{"verdict": "reject", "issues": ["Fund B is wrong"], "reason": "recomputed"}'})
    log.append({"timestamp": "2026-11-17T09:05:00+00:00", "run_id": "live-weakened-wrong", "task_id": "S04",
                "condition": "weakened-check", "agent": "verifier", "model": "gemini-3.6-flash", "cost_usd": 0.001,
                "raw_output": '{"verdict": "accept", "issues": [], "reason": "looks right"}'})
    (results / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))

    for condition in ("self", "dedicated"):  # complete seeded sets (2 answers)
        (results / "seeded" / f"{condition}-error.json").write_text(json.dumps(_seeded(ERROR_ID, condition, True, "reject")))
        (results / "seeded" / f"{condition}-control.json").write_text(json.dumps(_seeded(CONTROL_ID, condition, False, "accept")))
    (results / "seeded" / "peer-error.json").write_text(json.dumps(_seeded(ERROR_ID, "peer", True, "reject")))

    injected = tmp_path / "injected"
    injected.mkdir()
    for answer_id in (ERROR_ID, CONTROL_ID):
        (injected / f"{answer_id}.json").write_text("{}")
    tasks_file = tmp_path / "tasks.txt"
    tasks_file.write_text("S01\nT01-v001  # variant\n")

    out = tmp_path / "out"
    meta = ex.build(results, out, tasks_file, REPEATS, injected_dir=injected)
    load = lambda name: json.loads((out / name).read_text())  # noqa: E731
    return {"meta": meta, "out": out, "load": load, "rows": rows}


def _all_exported(out):
    return sorted(out.rglob("*.json"))


def _keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from _keys(v)


# --------------------------------------------------------------------------- honesty and safety


def test_no_exported_file_contains_answer_key_fields(built):
    files = _all_exported(built["out"])
    assert len(files) > 5
    for path in files:
        keys = set(_keys(json.loads(path.read_text())))
        assert not keys & ex.FORBIDDEN_KEYS, f"{path.name} has {keys & ex.FORBIDDEN_KEYS}"
    detail = built["load"]("runs/self-m-S01-r2.json")
    item = detail["draft_check"]["checks"][0]["details"]["items"][0]
    assert item == {"reason": "value_mismatch", "metric": "fund_b_final", "observed": 145470.34}  # observed + reason only


def test_no_exported_file_contains_account_identifiers(built):
    text = "\n".join(p.read_text() for p in _all_exported(built["out"]))
    for secret in (ORG, GROQ_KEY, EMAIL, "/Users/someone"):
        assert secret not in text
    assert not re.search(r"org_[0-9A-Za-z]{8,}(?<!REDACTED)", text.replace("org_REDACTED", ""))
    assert "org_REDACTED" in text and "email@REDACTED" in text  # the failed row's error was redacted, not dropped


def test_scrub_drops_forbidden_keys_at_any_depth_and_cleans_numbers():
    value = {"a": [{"expected": 1, "b": {"correct_value": 2, "wrong_value": 3, "keep": float("nan")}}], "c": f"x {ORG}"}
    assert ex.scrub(value) == {"a": [{"b": {"keep": None}}], "c": "x org_REDACTED"}


def test_unfinished_conditions_are_pending_with_no_numbers(built):
    summary = built["load"]("summary.json")
    peer = summary["conditions"]["peer"]
    assert peer == {"status": "pending", "done": 2, "total": 4, "failed": 1, "label": "pending (2/4)"}
    assert summary["conditions"]["cross_family"]["label"] == "pending (0/4)"
    assert {c["condition"] for c in summary["comparisons"]} == {"self", "dedicated"}
    heat = built["load"]("heatmap.json")
    assert heat["groups"][0]["cells"]["peer"] == {"status": "pending"}
    seeded = built["load"]("seeded.json")["per_condition"]
    assert seeded["peer"]["label"] == "pending (1/2)" and "catch_rate" not in seeded["peer"]
    assert built["meta"]["complete_conditions"] == ["none", "self", "dedicated"]


# --------------------------------------------------------------------------- numbers


def test_rates_carry_counts_and_human_review_both_ways(built):
    c = built["load"]("summary.json")["conditions"]
    assert c["none"]["task_success"]["right"] == 3 and c["none"]["task_success"]["n"] == 4
    ded = c["dedicated"]
    assert (ded["task_success"]["right"], ded["task_success"]["n"]) == (4, 4)
    assert ded["human_review"]["routes"] == {"abstain": 0, "unreadable": 1, "revision_not_accepted": 0}
    assert (ded["task_success_excl_human_review"]["right"], ded["task_success_excl_human_review"]["n"]) == (3, 3)
    assert ded["verifier"]["catch_rate"] == {"value": 1.0, "caught": 1, "n": 1}
    assert ded["outcomes"]["fixed"] == 1 and ded["outcomes"]["broken"] == 0
    assert ded["cost"]["includes_shared_draft"] is True
    slf = c["self"]
    assert slf["human_review"]["routes"]["revision_not_accepted"] == 1
    assert slf["outcomes"]["broken"] == 1
    assert slf["verifier"]["false_alarm_rate"]["flagged"] == 1 and slf["verifier"]["false_alarm_rate"]["n"] == 3


def test_headline_is_built_from_the_data(built):
    head = built["load"]("summary.json")["headline"]
    # With 2 tasks the interval reaches zero, so the sentence must not claim an improvement.
    assert head["text"] == ("With no check the advisor is right 75.0% of the time. The best setting, a dedicated verifier "
                            "with a calculator, reaches 100.0%, +25.0 points, but the interval includes no change (95% CI +0.0 to +50.0).")
    assert head["best"] == "dedicated" and head["data_set"] == "2 tasks × 2 runs"


def test_headline_does_not_claim_a_gain_when_the_interval_includes_zero():
    metrics = {"none": {"status": "complete", "task_success": {"value": 0.8}},
               "self": {"status": "complete", "task_success": {"value": 0.85}}}
    comps = [{"condition": "self", "delta": 0.05, "ci_low": -0.02, "ci_high": 0.12, "sign_test_p": 0.5}]
    text = ex.headline(metrics, comps, 32, 3)["text"]
    assert "raises" not in text and "the interval includes no change (95% CI -2.0 to +12.0)" in text
    comps[0].update(ci_low=0.01)
    assert "Having the advisor re-check its own answer raises that to 85.0%, +5.0 points" in ex.headline(metrics, comps, 32, 3)["text"]


def test_heatmap_groups_variants_under_their_base_scenario(built):
    groups = built["load"]("heatmap.json")["groups"]
    assert [g["scenario"] for g in groups] == ["S01"]
    g = groups[0]
    assert g["tasks"] == ["S01", "T01-v001"]
    assert g["cells"]["none"] == {"right": 3, "n": 4, "failed": 0}
    rows = {r["task_id"]: r["cells"] for r in g["rows"]}
    assert rows["S01"]["none"] == {"right": 1, "n": 2, "failed": 0}
    assert rows["S01"]["dedicated"] == {"right": 2, "n": 2, "failed": 0}
    assert ex.base_scenario("T10-v003") == "S10" and ex.base_scenario("S04") == "S04"


def test_spend_counts_each_logged_call_once(built):
    spend = built["load"]("spend.json")
    assert spend["calls"] == len(built["rows"]) + 2
    assert spend["total_usd"] == pytest.approx(0.001 * len(built["rows"]) + 0.003)
    assert spend["budget_usd"] == 20.0 and spend["failed_calls"] == 1


def test_runs_are_in_replay_order_with_outcomes(built):
    runs = built["load"]("runs.json")["runs"]
    stamps = [r["first_call"] for r in runs]
    assert stamps == sorted(stamps)
    assert runs[0]["run_id"] == built["rows"][-1]["run_id"]  # the last row was logged first
    by_id = {r["run_id"]: r for r in runs}
    assert by_id["dedicated-m-S01-r2"]["outcome"] == "fixed"
    assert by_id["self-m-T01-v001-r2"]["outcome"] == "broken"
    assert by_id["self-m-T01-v001-r2"]["human_review_route"] == "revision_not_accepted"
    assert by_id["peer-m-S01-r2"]["outcome"] == "failed"


def test_run_detail_keeps_answers_and_reviews_but_not_prompts(built):
    detail = built["load"]("runs/dedicated-m-S01-r2.json")
    assert detail["final"]["metrics"]["observations"][0]["value"] == 145469.35  # fenced JSON parsed
    review = detail["reviews"][0]
    assert "messages" not in review and "raw" not in review
    assert review["issues"] == ["saved under /~/cs5130/results"]


def test_seeded_export_keeps_decisions_not_labels(built):
    seeded = built["load"]("seeded.json")
    ded = seeded["per_condition"]["dedicated"]
    assert ded["catch_rate"] == {"value": 1.0, "caught": 1, "n": 1}
    assert ded["false_alarm_rate"]["n"] == 1
    review = next(r for r in seeded["reviews"] if r["condition"] == "dedicated" and r["seeded_error"])
    assert review["decision"] in {"right_reason", "consequence_only", "wrong_reason"}
    assert set(review) == {"answer_id", "scenario_id", "error_type", "seeded_error", "condition", "verdict",
                           "decision", "issues", "status"}
    assert seeded["wording"].startswith("Sanity checks, not results")


def test_s04_sanity_table(built):
    cases = {c["case"]: c for c in built["load"]("sanity_s04.json")["cases"]}
    assert cases["live-dedicated-wrong"]["verdict"] == "reject" and cases["live-dedicated-wrong"]["verdict_right"] is True
    assert cases["live-weakened-wrong"]["verdict"] == "accept" and cases["live-weakened-wrong"]["verdict_right"] is False


def test_build_replaces_old_output(built, tmp_path):
    stale = built["out"] / "runs" / "stale.json"
    stale.write_text("{}")
    rebuilt = ex.build(tmp_path / "results", built["out"], tmp_path / "tasks.txt", REPEATS, injected_dir=tmp_path / "injected")
    assert not stale.exists() and rebuilt["run_details"] == 13  # 14 runs, the failed one has no answer file


def test_cli_refuses_a_folder_without_results(tmp_path, capsys):
    assert build_data.main(["--results", str(tmp_path), "--out", str(tmp_path / "out")]) == 1
    assert "No results.jsonl" in capsys.readouterr().err


# --------------------------------------------------------------------------- pairs and seeded by type


def test_pairs_cover_every_ordered_pair_of_finished_settings_and_mirror(built):
    pairs = built["load"]("pairs.json")
    assert pairs["complete"] == ["none", "self", "dedicated"]
    keyed = {(p["a"], p["b"]): p for p in pairs["pairs"]}
    assert set(keyed) == {(a, b) for a in pairs["complete"] for b in pairs["complete"] if a != b}
    fwd, back = keyed[("none", "dedicated")], keyed[("dedicated", "none")]
    assert fwd["delta"] == pytest.approx(0.25) and back["delta"] == pytest.approx(-0.25)
    assert (back["ci_low"], back["ci_high"]) == (-fwd["ci_high"], -fwd["ci_low"])
    assert fwd["tasks_b_better"] == back["tasks_a_better"]
    # dedicated fixed S01 r2, which none left wrong; nothing the other way round
    assert [(x["task_id"], x["run_number"], x["b_outcome"]) for x in fwd["runs_only_b_right"]] == [("S01", 2, "fixed")]
    assert fwd["runs_only_a_right"] == []
    assert back["runs_only_a_right"][0]["a_run_id"] == "dedicated-m-S01-r2"
    # self broke T01-v001 r2 (none kept it right)
    assert [(x["task_id"], x["a_outcome"], x["b_outcome"]) for x in keyed[("none", "self")]["runs_only_a_right"]] == [
        ("T01-v001", "kept_correct", "broken")]
    assert not any(p["a"] == "peer" or p["b"] == "peer" for p in pairs["pairs"])  # peer is unfinished


def test_seeded_grouped_by_error_type_with_pending_settings(built):
    groups = {g["error_type"]: g for g in built["load"]("seeded.json")["by_error_type"]}
    assert set(groups) == {"policy_violation", "none"}
    errors = groups["policy_violation"]
    assert errors["answers"] == [ERROR_ID] and errors["seeded_error"] is True
    assert errors["per_condition"]["dedicated"]["rejected"] == 1
    assert errors["per_condition"]["dedicated"]["right_reason"] in (0, 1)
    assert errors["per_condition"]["peer"] == {"status": "pending"}  # 1 of 2 seeded answers reviewed
    controls = groups["none"]
    assert controls["seeded_error"] is False and controls["per_condition"]["self"]["right_reason"] is None


def test_comparisons_carry_holm_p_and_spend_uses_report_names(built):
    comps = built["load"]("summary.json")["comparisons"]
    # The family is every planned comparison with none in the data, finished or not (metrics.compare_conditions).
    family = len({r["condition"] for r in _fixture_rows()} - {"none"})
    assert all("sign_test_p_holm" in c and c["n_comparisons"] == family for c in comps)
    assert all(c["sign_test_p_holm"] >= c["sign_test_p"] for c in comps)
    # Holm: the smallest p is multiplied by the number of comparisons (capped at 1)
    smallest = min(comps, key=lambda c: c["sign_test_p"])
    assert smallest["sign_test_p_holm"] == pytest.approx(min(1.0, smallest["n_comparisons"] * smallest["sign_test_p"]))
    assert ex.spend_label("weakened") == "No calculator" and ex.spend_label("cross-check") == "pre-study checks"


def test_a_setting_with_every_run_saved_is_complete_even_with_failed_runs(built, tmp_path):
    rows = [json.loads(l) for l in (tmp_path / "results" / "results.jsonl").read_text().splitlines()]
    for task, run in (("T01-v001", 1), ("T01-v001", 2)):  # peer: add the 2 missing runs, one of them failed
        row = next(r for r in rows if r["condition"] == "self" and r["task_id"] == task and r["run_number"] == run)
        rows.append({**row, "run_id": f"peer-m-{task}-r{run}", "condition": "peer",
                     **({"status": "failed", "error": "ProviderError: 400 parsing failed"} if run == 2 else {})})
    (tmp_path / "results" / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = ex.build(tmp_path / "results", tmp_path / "out2", tmp_path / "tasks.txt", REPEATS, injected_dir=tmp_path / "injected")
    assert "peer" in meta["complete_conditions"]
    peer = json.loads((tmp_path / "out2" / "summary.json").read_text())["conditions"]["peer"]
    assert peer["status"] == "complete" and peer["n_failed"] == 2 and peer["n_runs"] == 2  # rates over the 2 runs that ran


def test_regraded_file_exports_the_original_grading_as_a_sensitivity_check(built, tmp_path):
    rows = [json.loads(l) for l in (tmp_path / "results" / "results.jsonl").read_text().splitlines()]
    for r in rows:  # what scripts/regrade_results.py writes: originals kept, one final grade changed
        if r["status"] == "failed":
            continue
        r["original_initial_correct"], r["original_final_correct"] = r["initial_correct"], r["final_correct"]
        if r["run_id"] == "self-m-T01-v001-r2":
            r["final_correct"] = True
    regraded = tmp_path / "results" / "results_regraded.jsonl"
    regraded.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out3"
    meta = ex.build(tmp_path / "results", out, tmp_path / "tasks.txt", REPEATS, injected_dir=tmp_path / "injected",
                    results_file=regraded)
    assert meta["grading"]["regraded"] and meta["scoring_mode"] == "corrected"
    assert [c["run_id"] for c in meta["grading"]["changed_runs"]] == ["self-m-T01-v001-r2"]
    assert "Self run 2 was graded wrong" in meta["scoring_notes"]["T01-v001"]
    summary = json.loads((out / "summary.json").read_text())
    # Self: S01 r2 accepted wrong, T01-v001 r2 broken -> 2 of 4 right; the corrected grade makes it 3
    assert summary["conditions"]["self"]["task_success"]["right"] == 3  # corrected
    assert summary["sensitivity"]["conditions"]["self"]["right"] == 2  # original grading
    detail = json.loads((out / "runs" / "self-m-T01-v001-r2.json").read_text())
    assert detail["graded"]["final_correct"] is True and detail["regraded_from"]["final_correct"] is False
    # the plain file has no originals: no sensitivity block, scoring stays "under review"
    assert built["load"]("summary.json")["sensitivity"] is None and built["meta"]["scoring_mode"] == "under_review"


def test_follow_up_is_absent_by_default_and_never_pooled(built, tmp_path):
    assert built["load"]("follow_up.json") == {"available": False}
    # A follow-up on new tasks: none gets 1 of 2 right, dedicated fixes the wrong draft.
    rows = [_row("T02-v001", "none", 1, True, True, "unchecked"), _row("T02-v001", "none", 2, False, False, "unchecked"),
            _row("T02-v001", "dedicated", 1, True, True, "accepted", "accept"),
            _row("T02-v001", "dedicated", 2, False, True, "revised", "reject", "accept")]
    follow = tmp_path / "follow.jsonl"
    follow.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out4"
    ex.build(tmp_path / "results", out, tmp_path / "tasks.txt", REPEATS, injected_dir=tmp_path / "injected",
             follow_up_file=follow)
    fu = json.loads((out / "follow_up.json").read_text())
    assert fu["available"] and fu["tasks"] == ["T02-v001"] and fu["wrong_draft_tasks"] == ["T02-v001"]
    assert fu["conditions"]["none"]["task_success"]["right"] == 1
    ded = fu["conditions"]["dedicated"]
    assert ded["task_success"]["right"] == 2 and ded["wrong_drafts"] == {"n": 1, "flagged": 1, "fixed": 1}
    assert ded["vs_none"]["delta"] == pytest.approx(0.5)
    # the main study's numbers are unchanged by the follow-up
    assert json.loads((out / "summary.json").read_text())["conditions"] == built["load"]("summary.json")["conditions"]
