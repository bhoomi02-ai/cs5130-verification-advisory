"""Spending report from the run log."""

import json

import pytest

from advisory.spend import format_summary, load_log, spend_summary

RECORDS = [
    {"timestamp": "2026-11-16T10:00:00+00:00", "model": "gemini-3.6-flash", "condition": "pilot", "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.02},
    {"timestamp": "2026-11-17T10:00:00+00:00", "model": "gemini-3.6-flash", "condition": "draft", "tokens_in": 200, "tokens_out": 70, "cost_usd": 0.03},
    {"timestamp": "2026-11-17T11:00:00+00:00", "model": "openai/gpt-oss-20b", "condition": "cross", "tokens_in": 300, "tokens_out": 10, "cost_usd": 0.001},
    {"timestamp": "2026-11-17T12:00:00+00:00", "model": "gemini-3.6-flash", "condition": "draft", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "error": "ServerError: 503"},
]


def test_totals_and_groupings():
    s = spend_summary(RECORDS)
    assert s["calls"] == 4 and s["failed"] == 1 and s["cost_usd"] == pytest.approx(0.051)
    by_model = {r["name"]: r for r in s["by_model"]}
    assert by_model["gemini-3.6-flash"]["calls"] == 3 and by_model["gemini-3.6-flash"]["cost_usd"] == pytest.approx(0.05)
    assert s["by_model"][0]["name"] == "gemini-3.6-flash"  # most expensive first
    assert [r["name"] for r in s["by_day"]] == ["2026-11-16", "2026-11-17"]  # days in order
    assert {r["name"]: r["failed"] for r in s["by_condition"]}["draft"] == 1


def test_report_shows_spend_against_the_budget():
    text = format_summary(spend_summary(RECORDS), 20.0)
    assert text.startswith("Spent $0.05 of $20.00 (0%); $19.95 left")
    assert "4 calls logged, 1 failed" in text and "openai/gpt-oss-20b" in text


def test_missing_log_means_nothing_spent(tmp_path):
    assert load_log(tmp_path / "runs.jsonl") == []
    assert format_summary(spend_summary([]), 20.0).startswith("Spent $0.00 of $20.00")


def test_load_log_reads_jsonl(tmp_path):
    log = tmp_path / "runs.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in RECORDS) + "\n")
    assert len(load_log(log)) == 4
