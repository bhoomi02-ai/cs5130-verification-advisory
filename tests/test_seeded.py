"""Seeded-answer reviews: IDs, record fields, what the verifier sees, errors, reruns.

No network: llm.call_model replies from a script. Uses the real seeded files
in data/injected/ and the real scenarios.
"""

import importlib.util
import json
from datetime import datetime, timezone

import pytest

from advisory import llm, seeded
from advisory.conditions import VERIFIED, make_run_id
from advisory.drafts import Settings

SETTINGS = Settings("gemini-3.6-flash")
_spec = importlib.util.spec_from_file_location("run_seeded", seeded.ROOT / "scripts" / "run_seeded.py")
run_seeded = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_seeded)
ACCEPT = '{"verdict": "accept", "issues": [], "reason": "All checks pass."}'
REJECT = '{"verdict": "reject", "issues": ["AAPL is sold but may not be"], "reason": "Breaks a restriction."}'
CALC = '{"calculate": ["20000 / 100000"]}'
S01_SEEDED = [a for a in seeded.all_answer_ids() if a.startswith("S01_")]
CONTROL = "S01_none_reference"
ERROR = "S01_policy_violation_sell_aapl"


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(seeded, "SEEDED_RESULTS_DIR", tmp_path / "seeded")
    return tmp_path / "seeded"


@pytest.fixture
def script(monkeypatch):
    calls = []

    def load(*replies):
        queue = list(replies)

        def call_model(**kwargs):
            calls.append(kwargs)
            reply = queue.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(llm, "call_model", call_model)
        return calls

    return load


def test_run_ids_differ_per_answer_and_from_the_main_run():
    assert len(S01_SEEDED) == 4
    ids = {seeded.seeded_run_id("dedicated", "gemini-3.6-flash", a, 1) for a in S01_SEEDED}
    assert len(ids) == 4
    assert make_run_id("dedicated", "S01", 1, "gemini-3.6-flash") not in ids
    assert seeded.seeded_run_id("dedicated", "gemini-3.6-flash", ERROR, 1) == (
        "seeded-dedicated-gemini-3.6-flash-S01_policy_violation_sell_aapl-r1"
    )


def test_every_seeded_file_has_a_scenario_and_a_response():
    ids = seeded.all_answer_ids()
    assert len(ids) == 31
    for answer_id in ids:
        entry = seeded.load_seeded(answer_id)
        assert (seeded.SCENARIO_DIR / f"{entry['scenario_id']}.json").exists()
        json.loads(seeded.answer_text(entry))  # the verifier gets valid JSON text


def test_record_has_answer_id_and_the_same_review_fields_as_the_main_runs(script, out):
    calls = script(REJECT)
    rec = seeded.run_seeded_one("peer", ERROR, 1, SETTINGS)
    assert len(calls) == 1
    assert (rec["answer_id"], rec["scenario_id"], rec["seeded_error"], rec["error_type"]) == (
        ERROR, "S01", True, "policy_violation")
    assert (rec["verdict"], rec["status"], rec["verifier_calls"]) == ("reject", "reviewed", 1)
    [review] = rec["reviews"]
    assert review["stage"] == "first"
    assert {"verdict", "issues", "reason", "trailing", "calls", "error", "raw"} <= set(review)
    assert review["issues"] == ["AAPL is sold but may not be"]
    assert json.loads((out / f"{rec['run_id']}.json").read_text()) == rec


def test_control_is_marked_and_accept_is_kept(script, out):
    script(ACCEPT)
    rec = seeded.run_seeded_one("self", CONTROL, 1, SETTINGS)
    assert (rec["seeded_error"], rec["error_type"], rec["verdict"]) == (False, "none", "accept")


def test_dedicated_counts_calculator_rounds_and_keeps_them(script, out):
    calls = script(CALC, REJECT)
    rec = seeded.run_seeded_one("dedicated", ERROR, 1, SETTINGS)
    assert len(calls) == 2 and rec["verifier_calls"] == 2
    assert rec["reviews"][0]["calculations"][0]["expression"] == "20000 / 100000"


def test_logged_calls_use_the_answer_id_and_seeded_run_id(script, out):
    calls = script(ACCEPT)
    rec = seeded.run_seeded_one("weakened", ERROR, 2, SETTINGS)
    assert calls[0]["task_id"] == ERROR
    assert calls[0]["run_id"] == rec["run_id"] == seeded.seeded_run_id("weakened", "gemini-3.6-flash", ERROR, 2)
    assert calls[0]["agent"] == "verifier"


def test_cross_family_uses_its_own_verifier_model(script, out):
    calls = script(ACCEPT)
    rec = seeded.run_seeded_one("cross_family", ERROR, 1, SETTINGS)
    assert calls[0]["model"] == rec["verifier_model"] == "openai/gpt-oss-20b"
    assert rec["model"] == "gemini-3.6-flash"


@pytest.mark.parametrize("condition", VERIFIED)
def test_verifier_never_sees_the_seeded_metadata(script, out, condition):
    calls = script(ACCEPT)
    seeded.run_seeded_one(condition, ERROR, 1, SETTINGS)
    entry = seeded.load_seeded(ERROR)
    sent = json.dumps(calls[0]["messages"]).lower()
    for leak in (entry["description"], entry["error_type"], entry["expected_check"], ERROR, "seeded", "injected"):
        assert leak.lower() not in sent, leak


def test_api_error_is_saved_as_failed(script, out):
    script(llm.ProviderError(500, "server error"))
    rec = seeded.run_seeded_one("peer", ERROR, 1, SETTINGS)
    assert (rec["status"], rec["verdict"], rec["reviews"]) == ("failed", None, [])
    assert "server error" in rec["error"]
    assert (out / f"{rec['run_id']}.json").exists()


def test_daily_limit_raises_and_saves_nothing(script, out):
    script(llm.ProviderError(429, "per-day limit", daily_quota=True))
    with pytest.raises(llm.ProviderError):
        seeded.run_seeded_one("cross_family", ERROR, 1, SETTINGS)
    assert not out.exists() or not list(out.iterdir())


def test_usage_counts_only_this_attempt(out, tmp_path, monkeypatch):
    log = tmp_path / "runs.jsonl"
    run_id = seeded.seeded_run_id("peer", "gemini-3.6-flash", ERROR, 1)
    old = {"timestamp": "2026-01-01T00:00:00+00:00", "run_id": run_id, "tokens_in": 999, "tokens_out": 999, "cost_usd": 9.0}
    log.write_text(json.dumps(old) + "\n")

    def reply_and_log(**kwargs):
        rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "run_id": kwargs["run_id"],
               "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.01}
        with log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return ACCEPT

    monkeypatch.setattr(llm, "call_model", reply_and_log)
    rec = seeded.run_seeded_one("peer", ERROR, 1, SETTINGS, log_path=log)
    assert (rec["n_calls"], rec["tokens_in"], rec["cost_usd"]) == (1, 10, 0.01)


def test_none_condition_is_refused(out):
    with pytest.raises(ValueError, match="no verifier"):
        seeded.run_seeded_one("none", ERROR, 1, SETTINGS)


def test_script_skips_saved_reviews_and_retries_failed_ones(script, out, capsys):
    script(llm.ProviderError(500, "boom"), ACCEPT)
    args = ["--condition", "peer", "--model", "gemini-3.6-flash", "--answers", ERROR, CONTROL]
    assert run_seeded.main(args) == 0  # ERROR fails and is saved, CONTROL is accepted
    assert run_seeded.main(args) == 0  # both saved: nothing called
    out_text = capsys.readouterr().out
    assert out_text.count("skip") == 2

    script(REJECT)
    assert run_seeded.main(args + ["--retry-failed"]) == 0
    rec = seeded.saved_record(seeded.seeded_run_id("peer", "gemini-3.6-flash", ERROR, 1))
    assert (rec["status"], rec["verdict"]) == ("reviewed", "reject")


def test_script_refuses_records_made_with_other_settings(script, out, capsys):
    script(ACCEPT)
    assert run_seeded.main(["--condition", "peer", "--model", "gemini-3.6-flash", "--answers", CONTROL]) == 0
    code = run_seeded.main(["--condition", "peer", "--model", "gemini-3.6-flash", "--answers", CONTROL, "--temperature", "0.2"])
    assert code == 1
    assert "other settings" in capsys.readouterr().out


def test_dry_run_calls_nothing_saves_nothing_and_estimates_from_saved_reviews(script, out, capsys):
    calls = script()
    args = ["--condition", "peer", "--model", "gemini-3.6-flash", "--answers", ERROR, CONTROL, "--dry-run"]
    assert run_seeded.main(args) == 0
    assert calls == [] and not out.exists()
    text = capsys.readouterr().out
    assert text.count("plan") == 2 and "no saved peer reviews" in text

    script(ACCEPT)
    run_seeded.main(["--condition", "peer", "--model", "gemini-3.6-flash", "--answers", CONTROL])
    path = out / f"{seeded.seeded_run_id('peer', 'gemini-3.6-flash', CONTROL, 1)}.json"
    rec = json.loads(path.read_text())
    rec["cost_usd"] = 0.01
    path.write_text(json.dumps(rec))
    capsys.readouterr()
    calls = script()  # the fixture keeps one list per test: the review above is already in it
    assert run_seeded.main(args) == 0
    text = capsys.readouterr().out
    assert len(calls) == 1
    assert "skip" in text and text.count("plan") == 1
    assert "1 review(s) would run" in text and "about $0.01 at $0.0100 per review (mean of 1 saved)" in text


def test_dry_run_still_refuses_records_made_with_other_settings(script, out, capsys):
    script(ACCEPT)
    run_seeded.main(["--condition", "peer", "--model", "gemini-3.6-flash", "--answers", CONTROL])
    code = run_seeded.main(["--condition", "peer", "--model", "gemini-3.6-flash", "--answers", CONTROL,
                            "--temperature", "0.2", "--dry-run"])
    assert code == 1 and "other settings" in capsys.readouterr().out
