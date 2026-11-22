"""Turn the saved study results into the small static files the web dashboard loads.

Read-only: it reads results/ (results.jsonl, runs.jsonl, answers/, seeded/)
and calls the same metric code as scripts/analyze.py; it never calls a model.
Run it with `python dashboard/build_data.py`; the files go to dashboard/data/
(gitignored on main).

Honesty rules built in here, so the page cannot get them wrong:
- A condition is "complete" when every study run is saved. Failed runs (an API
  error after retries) count as saved: like metrics.summarize, the rates leave
  them out and report how many failed. Metrics are computed for complete
  conditions only; the others are exported as {"status": "pending", "done": n,
  "total": m} with no rates.
- The seeded metrics follow the same rule per condition (31 answers each).
- Answer-key values never leave the scorer side: every key named in
  FORBIDDEN_KEYS (the checker's `expected`, the seeded labels'
  `correct_value` / `wrong_value`) is dropped from every exported object, and
  the seeded labels contribute only the right-reason decision.
- Account identifiers (API keys, Groq org IDs, e-mail addresses, home
  folder paths) are redacted from every exported string.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from advisory import right_reason
from advisory.metrics import compare_conditions, summarize
from advisory.pipeline import STUDY_TASKS_PATH, read_task_file
from advisory.report import load_results
from advisory.verdict import parse_verdict

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "dashboard" / "data"
INJECTED_DIR = ROOT / "data" / "injected"

SCHEMA_VERSION = 1
STUDY_REPEATS = 3
BUDGET_USD = 20.0
CONDITION_ORDER = ("none", "self", "peer", "dedicated", "weakened", "cross_family")
VERIFIED = CONDITION_ORDER[1:]
CONDITION_LABELS = {
    "none": "None · no check",
    "self": "Self · re-reads its answer",
    "peer": "Peer · a second advisor call",
    "dedicated": "Dedicated · verifier + calculator",
    "weakened": "No calculator · same verifier",
    "cross_family": "Cross-family · gpt-oss-20b verifier",
}
# Pre-study sanity checks of the verifiers on S04, logged in runs.jsonl.
SANITY_CONDITIONS = {"dedicated-check": "dedicated", "weakened-check": "weakened", "cross-check": "cross_family"}
# Tasks whose scoring the team is reviewing; shown as a note, never changes a grade.
SCORING_UNDER_REVIEW = {
    "T06-v003": (
        "Scoring under review: the checker also requires the unrounded payment, which the "
        "client did not ask for; two dedicated revisions report the requested values but are graded wrong."
    ),
}

FORBIDDEN_KEYS = frozenset({"expected", "correct_value", "wrong_value"})
_REDACTIONS = (
    (re.compile(r"org_[0-9A-Za-z]{8,}"), "org_REDACTED"),
    (re.compile(r"gsk_[0-9A-Za-z]{8,}"), "gsk_REDACTED"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "AIza_REDACTED"),
    (re.compile(r"sk-[0-9A-Za-z_\-]{16,}"), "sk-REDACTED"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "email@REDACTED"),
    (re.compile(r"/(?:Users|home)/[^/\s\"']+"), "/~"),
)


# --------------------------------------------------------------------------- safety


def redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def scrub(value: Any) -> Any:
    """Drop FORBIDDEN_KEYS at any depth, redact identifiers in strings, make JSON-safe."""
    if isinstance(value, dict):
        return {str(k): scrub(v) for k, v in value.items() if k not in FORBIDDEN_KEYS}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if hasattr(value, "item"):  # numpy scalars
        return scrub(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scrub(payload), indent=1, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- inputs


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def base_scenario(task_id: str) -> str:
    """S04 -> S04; T04-v003 -> S04 (variants T01..T10 belong to S01..S10)."""
    match = re.fullmatch(r"T(\d{2})-v\d{3}", task_id)
    return f"S{match.group(1)}" if match else task_id


def human_review_route(row: dict) -> str | None:
    """Why a run went to human review: abstain, unreadable verdict, or a revision not accepted."""
    if row.get("status") != "human_review":
        return None
    verdict = row.get("verdict")
    if verdict == "abstain":
        return "abstain"
    if verdict == "reject":
        return "revision_not_accepted"
    return "unreadable"


def _first_calls(log: list[dict]) -> dict[str, str]:
    first: dict[str, str] = {}
    for call in log:
        run_id, ts = call.get("run_id"), call.get("timestamp")
        if run_id and ts and (run_id not in first or ts < first[run_id]):
            first[run_id] = ts
    return first


def _stat(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _int(value) -> int | None:
    number = _stat(value)
    return None if number is None else int(round(number))


# --------------------------------------------------------------------------- progress


def progress(rows: list[dict], tasks: list[str], seeded: list[dict], n_answers: int, repeats: int) -> dict:
    task_set = set(tasks)
    study = [r for r in rows if r.get("task_id") in task_set and 1 <= int(r.get("run_number") or 0) <= repeats]
    out = {}
    for condition in CONDITION_ORDER:
        mine = [r for r in study if r.get("condition") == condition]
        keys = {(r["task_id"], r["run_number"]) for r in mine}
        failed = sum(1 for r in mine if r.get("status") == "failed")
        expected = len(tasks) * repeats
        entry = {
            "runs_done": len(keys),
            "runs_expected": expected,
            "runs_failed": failed,
            "complete": len(keys) >= expected,
        }
        if condition in VERIFIED:
            reviews = [s for s in seeded if s.get("condition") == condition]
            seeded_failed = sum(1 for s in reviews if s.get("status") == "failed")
            entry.update(
                seeded_done=len({s.get("answer_id") for s in reviews}),
                seeded_expected=n_answers,
                seeded_failed=seeded_failed,
                seeded_complete=len({s.get("answer_id") for s in reviews}) >= n_answers,
            )
        out[condition] = entry
    return out


def _pending(entry: dict, kind: str = "runs") -> dict:
    # "total", not "expected": scrub() drops every key named "expected" (the checker's answer-key field).
    return {"status": "pending", "done": entry[f"{kind}_done"], "total": entry[f"{kind}_expected"],
            "failed": entry[f"{kind}_failed"], "label": f"pending ({entry[f'{kind}_done']}/{entry[f'{kind}_expected']})"}


# --------------------------------------------------------------------------- metrics


def condition_metrics(rows: list[dict], complete: list[str], prog: dict, seed: int = 0) -> dict:
    """Per condition: every rate with its sample size and CI; pending conditions get no numbers."""
    out: dict[str, dict] = {}
    usable = [r for r in rows if r.get("condition") in complete]
    table = summarize(usable, seed=seed).reset_index().set_index("condition") if usable else pd.DataFrame()
    for condition in CONDITION_ORDER:
        if condition not in complete:
            out[condition] = _pending(prog[condition])
            continue
        mine = [r for r in rows if r.get("condition") == condition and r.get("status") != "failed"]
        m = table.loc[condition]
        n_runs = len(mine)
        n_tasks = len({r["task_id"] for r in mine})
        wrong_drafts = sum(1 for r in mine if not r.get("initial_correct"))
        right_drafts = n_runs - wrong_drafts
        routes = Counter(human_review_route(r) for r in mine if r.get("status") == "human_review")
        hr_runs = [r for r in mine if r.get("status") == "human_review"]
        entry = {
            "status": "complete",
            "label": CONDITION_LABELS[condition],
            "n_runs": n_runs,
            "n_tasks": n_tasks,
            "task_success": {
                "value": _stat(m["task_success"]),
                "ci_low": _stat(m["task_success_ci_low"]),
                "ci_high": _stat(m["task_success_ci_high"]),
                "right": sum(1 for r in mine if r.get("final_correct")),
                "n": n_runs,
            },
            "task_success_excl_human_review": {
                "value": _stat(m["task_success_excl_human_review"]),
                "ci_low": _stat(m["task_success_excl_human_review_ci_low"]),
                "ci_high": _stat(m["task_success_excl_human_review_ci_high"]),
                "right": sum(1 for r in mine if r.get("final_correct") and r.get("status") != "human_review"),
                "n": n_runs - len(hr_runs),
            },
            "first_draft_success": {
                "value": right_drafts / n_runs if n_runs else None,
                "right": right_drafts,
                "n": n_runs,
            },
            "human_review": {
                "n": len(hr_runs),
                "rate": _stat(m["human_review_rate"]),
                "final_right": sum(1 for r in hr_runs if r.get("final_correct")),
                "routes": {
                    "abstain": routes.get("abstain", 0),
                    "unreadable": routes.get("unreadable", 0),
                    "revision_not_accepted": routes.get("revision_not_accepted", 0),
                },
            },
            "outcomes": {k: _int(m[k]) for k in ("kept_correct", "fixed", "broken", "still_wrong")},
            "cost": {
                "mean_cost_usd": _stat(m["mean_cost_usd"]),
                "total_cost_usd": _stat(m["total_cost_usd"]),
                "mean_tokens_in": _stat(m["mean_tokens_in"]),
                "mean_tokens_out": _stat(m["mean_tokens_out"]),
                "median_latency_s": _stat(m["median_latency_s"]),
                "extra_cost_per_extra_correct": _stat(m["extra_cost_per_extra_correct"]),
                "includes_shared_draft": True,
            },
            "n_failed": _int(m["n_failed"]) or 0,
        }
        if condition in VERIFIED:
            entry["verifier"] = {
                "catch_rate": {"value": _stat(m["catch_rate"]), "caught": _int(m["tp"]), "n": wrong_drafts},
                "false_alarm_rate": {"value": _stat(m["false_alarm_rate"]), "flagged": _int(m["fp"]), "n": right_drafts},
                "mcc": _stat(m["mcc"]),
                "missing_verdicts": _int(m["n_missing_verdict"]) or 0,
                "wrong_draft_fates": {k: _int(m[k]) for k in ("missed", "caught_fixed", "caught_not_fixed", "caught_escalated")},
                "mean_verifier_calls": sum(r.get("verifier_calls") or 0 for r in mine) / n_runs if n_runs else None,
            }
        out[condition] = entry
    return out


def comparisons(rows: list[dict], complete: list[str], seed: int = 0) -> list[dict]:
    if "none" not in complete:
        return []
    usable = [r for r in rows if r.get("condition") in complete]
    frame = compare_conditions(usable, baseline="none", seed=seed)
    order = {c: i for i, c in enumerate(CONDITION_ORDER)}
    out = []
    for _, c in frame.iterrows():
        out.append({
            "condition": c["condition_b"],
            "n_tasks": _int(c["n_tasks"]),
            "n_pairs": _int(c["n_pairs"]),
            "success_none": _stat(c["success_a"]),
            "success": _stat(c["success_b"]),
            "delta": _stat(c["delta"]),
            "ci_low": _stat(c["delta_ci_low"]),
            "ci_high": _stat(c["delta_ci_high"]),
            "runs_only_condition_right": _int(c["b_only_correct"]),
            "runs_only_none_right": _int(c["a_only_correct"]),
            "tasks_better": _int(c["task_b_better"]),
            "tasks_worse": _int(c["task_a_better"]),
            "sign_test_p": _stat(c["sign_test_p"]),
            "note": None if pd.isna(c.get("note")) else c.get("note"),
        })
    out = sorted(out, key=lambda x: order.get(x["condition"], 99))
    # Holm adjustment over the comparisons against none that are shown (finished settings only).
    usable = sorted((c for c in out if c["sign_test_p"] is not None), key=lambda c: c["sign_test_p"])
    m, running = len(usable), 0.0
    for i, c in enumerate(usable):
        running = max(running, min(1.0, (m - i) * c["sign_test_p"]))
        c["sign_test_p_holm"] = running
        c["n_comparisons"] = m
    return out


def pairwise(rows: list[dict], complete: list[str], seed: int = 0, notes: dict | None = None) -> list[dict]:
    """Every ordered pair of finished settings: paired change, CI, sign test, and the runs where they disagree.

    Pairs are matched on (task, run number), i.e. the same shared first draft. Each unordered
    comparison is computed once and stored in both directions, so the page only looks it up.
    """
    notes = SCORING_UNDER_REVIEW if notes is None else notes
    usable = [r for r in rows if r.get("condition") in complete]
    if len(complete) < 2 or not usable:
        return []
    frame = compare_conditions(usable, all_pairs=True, seed=seed)
    by_key = {(r["condition"], r["task_id"], r["run_number"]): r for r in usable if r.get("status") != "failed"}
    out = []
    for _, c in frame.iterrows():
        a, b = c["condition_a"], c["condition_b"]
        only_a, only_b = [], []
        for (cond, task, run), ra in sorted(by_key.items()):
            if cond != a:
                continue
            rb = by_key.get((b, task, run))
            if rb is None or bool(ra.get("final_correct")) == bool(rb.get("final_correct")):
                continue
            item = {"task_id": task, "run_number": run, "draft_correct": bool(ra.get("initial_correct")),
                    "a_run_id": ra["run_id"], "b_run_id": rb["run_id"],
                    "a_outcome": outcome(ra), "b_outcome": outcome(rb),
                    "a_status": ra.get("status"), "b_status": rb.get("status"),
                    "scoring_note": notes.get(task)}
            (only_a if ra.get("final_correct") else only_b).append(item)
        base = {
            "n_tasks": _int(c["n_tasks"]), "n_pairs": _int(c["n_pairs"]),
            "sign_test_p": _stat(c["sign_test_p"]),
            "note": None if pd.isna(c.get("note")) else c.get("note"),
        }
        forward = {**base, "a": a, "b": b, "success_a": _stat(c["success_a"]), "success_b": _stat(c["success_b"]),
                   "delta": _stat(c["delta"]), "ci_low": _stat(c["delta_ci_low"]), "ci_high": _stat(c["delta_ci_high"]),
                   "tasks_b_better": _int(c["task_b_better"]), "tasks_a_better": _int(c["task_a_better"]),
                   "runs_only_a_right": only_a, "runs_only_b_right": only_b}
        neg = lambda v: None if v is None else -v  # noqa: E731
        backward = {**base, "a": b, "b": a, "success_a": forward["success_b"], "success_b": forward["success_a"],
                    "delta": neg(forward["delta"]), "ci_low": neg(forward["ci_high"]), "ci_high": neg(forward["ci_low"]),
                    "tasks_b_better": forward["tasks_a_better"], "tasks_a_better": forward["tasks_b_better"],
                    "runs_only_a_right": [_swap(x) for x in only_b], "runs_only_b_right": [_swap(x) for x in only_a]}
        out += [forward, backward]
    order = {c: i for i, c in enumerate(CONDITION_ORDER)}
    return sorted(out, key=lambda x: (order.get(x["a"], 99), order.get(x["b"], 99)))


def _swap(item: dict) -> dict:
    return {**item, "a_run_id": item["b_run_id"], "b_run_id": item["a_run_id"], "a_outcome": item["b_outcome"],
            "b_outcome": item["a_outcome"], "a_status": item["b_status"], "b_status": item["a_status"]}


# How each setting reads as the subject of the headline sentence.
HEADLINE_SUBJECT = {
    "self": "Having the advisor re-check its own answer",
    "peer": "A second advisor call checking the answer",
    "dedicated": "A dedicated verifier with a calculator",
    "weakened": "A verifier without a calculator",
    "cross_family": "A verifier from another model family",
}


def headline(metrics: dict, comps: list[dict], n_tasks: int, repeats: int) -> dict:
    """The one-sentence summary, built only from finished settings.

    It names the setting with the largest gain over no check. If that gain's 95% interval
    includes zero, it says so instead of claiming an improvement.
    """
    none = metrics.get("none", {})
    if none.get("status") != "complete":
        return {"text": "Pending: the no-check baseline has not finished.", "best": None}
    best = max(
        (c for c in comps if c["delta"] is not None),
        key=lambda c: (c["delta"], -(c["sign_test_p"] or 1)),
        default=None,
    )
    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731
    pts = lambda v: f"{v * 100:+.1f}"  # noqa: E731
    text = f"With no check the advisor is right {pct(none['task_success']['value'])} of the time."
    if best is None or best["delta"] <= 0:
        text += " No finished setting improves on that yet."
    else:
        value = metrics[best["condition"]]["task_success"]["value"]
        interval = f"95% CI {pts(best['ci_low'])} to {pts(best['ci_high'])}"
        subject = HEADLINE_SUBJECT.get(best["condition"], best["condition"])
        if best["ci_low"] > 0:
            text += f" {subject} raises that to {pct(value)}, {pts(best['delta'])} points ({interval})."
        else:
            text += (f" The best setting, {subject[0].lower() + subject[1:]}, reaches {pct(value)}, "
                     f"{pts(best['delta'])} points, but the interval includes no change ({interval}).")
    return {"text": text, "best": best["condition"] if best else None,
            "data_set": f"{n_tasks} tasks × {repeats} runs"}


# --------------------------------------------------------------------------- heatmap


def heatmap(rows: list[dict], tasks: list[str], complete: list[str], prog: dict, notes: dict | None = None) -> dict:
    """Right final answers out of saved runs, per task and condition, grouped under S01..S10."""
    notes = SCORING_UNDER_REVIEW if notes is None else notes
    cells: dict[tuple, dict] = defaultdict(lambda: {"right": 0, "n": 0, "failed": 0})
    for r in rows:
        if r.get("task_id") not in tasks:
            continue
        cell = cells[(r["task_id"], r["condition"])]
        if r.get("status") == "failed":
            cell["failed"] += 1
        else:
            cell["n"] += 1
            cell["right"] += 1 if r.get("final_correct") else 0

    def cell(task_ids: list[str], condition: str) -> dict:
        if condition not in complete:
            return {"status": "pending"}
        right = sum(cells[(t, condition)]["right"] for t in task_ids)
        n = sum(cells[(t, condition)]["n"] for t in task_ids)
        failed = sum(cells[(t, condition)]["failed"] for t in task_ids)
        return {"right": right, "n": n, "failed": failed}

    groups: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        groups[base_scenario(task)].append(task)
    out = []
    for scenario in sorted(groups):
        members = sorted(groups[scenario], key=lambda t: (t != scenario, t))
        out.append({
            "scenario": scenario,
            "tasks": members,
            "cells": {c: cell(members, c) for c in CONDITION_ORDER},
            "rows": [
                {"task_id": t, "note": notes.get(t), "cells": {c: cell([t], c) for c in CONDITION_ORDER}}
                for t in members
            ],
        })
    return {
        "repeats": STUDY_REPEATS,
        "conditions": list(CONDITION_ORDER),
        "pending": {c: _pending(prog[c]) for c in CONDITION_ORDER if c not in complete},
        "groups": out,
    }


# --------------------------------------------------------------------------- spend


SPEND_LABELS = {
    "none": "None", "self": "Self", "peer": "Peer", "dedicated": "Dedicated", "weakened": "No calculator",
    "cross_family": "Cross-family", "draft": "first drafts", "pilot": "pilot",
}


def spend_label(name: str) -> str:
    """The name the dashboard, report and slides use for a spend line (code IDs stay as they are)."""
    if name in SPEND_LABELS:
        return SPEND_LABELS[name]
    return "pre-study checks" if name.endswith("-check") else name


def spend(log: list[dict], budget: float) -> dict:
    by: dict[str, dict] = {"condition": defaultdict(float), "model": defaultdict(float), "day": defaultdict(float)}
    calls: dict[str, Counter] = {"condition": Counter(), "model": Counter(), "day": Counter()}
    for call in log:
        cost = call.get("cost_usd") or 0.0
        for kind, key in (("condition", call.get("condition") or "?"), ("model", call.get("model") or "?"),
                          ("day", (call.get("timestamp") or "?")[:10])):
            by[kind][key] += cost
            calls[kind][key] += 1
    total = sum(call.get("cost_usd") or 0.0 for call in log)

    def rows(kind: str, sort_by_name: bool = False) -> list[dict]:
        items = [{"name": k, "label": spend_label(k) if kind == "condition" else k, "cost_usd": v, "calls": calls[kind][k]}
                 for k, v in by[kind].items()]
        return sorted(items, key=(lambda x: x["name"]) if sort_by_name else (lambda x: -x["cost_usd"]))

    return {
        "source": "results/runs.jsonl (every model call, as scripts/spend.py)",
        "total_usd": total,
        "budget_usd": budget,
        "left_usd": budget - total,
        "calls": len(log),
        "failed_calls": sum(1 for call in log if call.get("error")),
        "by_condition": rows("condition"),
        "by_model": rows("model"),
        "by_day": rows("day", sort_by_name=True),
        "note": "Each run's own cost also includes its share of the shared first draft, so summing run costs "
                "across settings counts drafts more than once; this total counts every call once.",
    }


# --------------------------------------------------------------------------- seeded


def seeded(records: list[dict], prog: dict) -> dict:
    complete = [c for c in VERIFIED if prog[c].get("seeded_complete")]
    labels = right_reason.load_labels()
    metrics = right_reason.right_reason_metrics(records, labels) if records else pd.DataFrame()
    audit = right_reason.audit_table(records, labels) if records else pd.DataFrame()
    per_condition = {}
    for condition in VERIFIED:
        if condition not in complete:
            per_condition[condition] = _pending(prog[condition], "seeded")
            continue
        m = metrics.set_index("condition").loc[condition]
        mine = [r for r in records if r.get("condition") == condition]
        per_condition[condition] = {
            "status": "complete",
            "n_errors": _int(m["n_errors"]),
            "n_controls": _int(m["n_controls"]),
            "n_failed": _int(m["n_failed"]),
            "missing_verdicts": _int(m["n_missing_verdict"]),
            "catch_rate": {"value": _stat(m["seeded_catch_rate"]),
                           "caught": sum(1 for r in mine if r.get("seeded_error") and r.get("verdict") == "reject"),
                           "n": _int(m["n_errors"])},
            "false_alarm_rate": {"value": _stat(m["seeded_false_alarm_rate"]),
                                 "flagged": sum(1 for r in mine if not r.get("seeded_error") and r.get("verdict") == "reject"),
                                 "n": _int(m["n_controls"])},
            "right_reason": {k: _int(m[k]) for k in ("right_reason", "consequence_only", "wrong_reason", "abstained")},
            "right_reason_rate": _stat(m["right_reason_rate"]),
            "cost_usd": sum(r.get("cost_usd") or 0.0 for r in mine),
        }
    by_type: dict[str, dict] = {}
    for r in records:
        et = r.get("error_type") or "?"
        group = by_type.setdefault(et, {"error_type": et, "seeded_error": bool(r.get("seeded_error")), "answers": set(), "per_condition": {}})
        group["answers"].add(r.get("answer_id"))
    decisions = {}
    if not audit.empty:
        decisions = {(row["answer_id"], row["condition"]): row.get("decision") for row in audit.to_dict("records")}
    for et, group in by_type.items():
        for condition in VERIFIED:
            if condition not in complete:
                group["per_condition"][condition] = {"status": "pending"}
                continue
            mine = [r for r in records if r.get("condition") == condition and (r.get("error_type") or "?") == et
                    and r.get("status") != "failed"]
            rejected = sum(1 for r in mine if r.get("verdict") == "reject")
            right = sum(1 for r in mine if decisions.get((r.get("answer_id"), condition)) == "right_reason")
            group["per_condition"][condition] = {"n": len(mine), "rejected": rejected,
                                                 "right_reason": right if group["seeded_error"] else None}
        group["answers"] = sorted(a for a in group["answers"] if a)
        group["n_answers"] = len(group["answers"])
    reviews = []
    if not audit.empty:
        meta = {(r.get("answer_id"), r.get("condition")): r for r in records}
        for row in audit.to_dict("records"):
            rec = meta.get((row["answer_id"], row["condition"]), {})
            reviews.append({
                "answer_id": row["answer_id"],
                "scenario_id": rec.get("scenario_id"),
                "error_type": rec.get("error_type"),
                "seeded_error": rec.get("seeded_error"),
                "condition": row["condition"],
                "verdict": row.get("verdict"),
                "decision": row.get("decision"),  # right_reason / consequence_only / wrong_reason / abstained / not_flagged
                "issues": rec.get("reviews", [{}])[0].get("issues") if rec.get("reviews") else [],
                "status": rec.get("status"),
            })
    return {
        "n_answers": len({r.get("answer_id") for r in records}) if records else 0,
        "per_condition": per_condition,
        "reviews": reviews,
        "by_error_type": sorted(by_type.values(), key=lambda g: (not g["seeded_error"], g["error_type"])),
        "wording": "Sanity checks, not results: 17 seeded answers with one known error and 14 correct controls, "
                   "each reviewed once per verifier.",
    }


# --------------------------------------------------------------------------- S04 sanity table


def sanity_s04(log: list[dict]) -> dict:
    """The pre-study verifier checks on two real S04 answers (one wrong, one right)."""
    by_run: dict[str, list[dict]] = defaultdict(list)
    for call in log:
        if call.get("condition") in SANITY_CONDITIONS:
            by_run[call.get("run_id") or "?"].append(call)
    cases = []
    for run_id, calls in sorted(by_run.items()):
        calls.sort(key=lambda c: c.get("timestamp") or "")
        verdicts = []
        for c in calls:
            try:
                parsed = parse_verdict(c.get("raw_output") or "")
            except ValueError:
                continue
            verdicts.append({"verdict": parsed.verdict, "issues": list(parsed.issues or []), "reason": parsed.reason})
        answer = "wrong" if run_id.endswith("wrong") else "correct" if "correct" in run_id else None
        final = verdicts[-1] if verdicts else None
        cases.append({
            "case": run_id,
            "setting": SANITY_CONDITIONS[calls[0]["condition"]],
            "verifier_model": calls[0].get("model"),
            "used_in_study": calls[0].get("model") in ("gemini-3.6-flash", "openai/gpt-oss-20b"),
            "answer_was": answer,
            "calls": len(calls),
            "verdict": final["verdict"] if final else "unreadable",
            "issues": final["issues"] if final else [],
            "reason": final["reason"] if final else None,
            "verdict_right": None if final is None or answer is None
            else (final["verdict"] == "reject") == (answer == "wrong"),
            "date": (calls[0].get("timestamp") or "")[:10],
        })
    return {"task": "S04", "wording": "Sanity checks, not results: one known-wrong and one known-right S04 answer, "
                                      "reviewed before the main study.", "cases": cases}


# --------------------------------------------------------------------------- runs and details


def run_rows(rows: list[dict], log: list[dict], tasks: list[str], notes: dict | None = None) -> list[dict]:
    """One row per run for the table and the replay, ordered by the run's first logged call."""
    notes = SCORING_UNDER_REVIEW if notes is None else notes
    first = _first_calls(log)
    out = []
    for r in rows:
        if r.get("task_id") not in tasks:
            continue
        started = first.get(r["run_id"]) or first.get(r.get("draft_run_id") or "")
        out.append({
            "run_id": r["run_id"],
            "task_id": r["task_id"],
            "scenario": base_scenario(r["task_id"]),
            "condition": r["condition"],
            "run_number": r["run_number"],
            "first_call": started,
            "initial_correct": r.get("initial_correct"),
            "final_correct": r.get("final_correct"),
            "verdict": r.get("verdict"),
            "second_verdict": r.get("second_verdict"),
            "status": r.get("status"),
            "human_review_route": human_review_route(r),
            "outcome": outcome(r),
            "verifier_calls": r.get("verifier_calls"),
            "cost_usd": r.get("cost_usd"),
            "tokens_in": r.get("tokens_in"),
            "tokens_out": r.get("tokens_out"),
            "error": r.get("error"),
            "scoring_note": notes.get(r["task_id"]),
            "regraded_from": _regrade_change(r),
        })
    return sorted(out, key=lambda x: (x["first_call"] or "9999", x["run_id"]))


def outcome(r: dict) -> str:
    if r.get("status") == "failed":
        return "failed"
    before, after = bool(r.get("initial_correct")), bool(r.get("final_correct"))
    return {(True, True): "kept_correct", (False, True): "fixed", (True, False): "broken", (False, False): "still_wrong"}[
        (before, after)
    ]


def _parse_answer(text: str | None) -> Any:
    if not text:
        return None
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end])
    except ValueError:
        return text


def _checks(check: dict | None) -> dict | None:
    """The checker's result without answer-key values: pass/fail, observed values, reason codes."""
    if not check:
        return None
    return {
        "correct": check.get("correct"),
        "format_failure": check.get("format_failure"),
        "checks": [
            {
                "check_id": c.get("check_id"),
                "check_type": c.get("check_type"),
                "passed": c.get("passed"),
                "observed": c.get("observed"),
                "details": c.get("details"),  # FORBIDDEN_KEYS (e.g. `expected`) are dropped by scrub()
            }
            for c in check.get("checks", [])
        ],
    }


def _regrade_change(row: dict) -> dict | None:
    """The original grades of a run whose grade changed when the results were re-graded, else None."""
    if "original_final_correct" not in row:
        return None
    before = (row.get("original_initial_correct"), row.get("original_final_correct"))
    after = (row.get("initial_correct"), row.get("final_correct"))
    if before == after:
        return None
    return {"initial_correct": before[0], "final_correct": before[1]}


def run_detail(record: dict, row: dict | None = None, notes: dict | None = None) -> dict:
    return {
        "run_id": record.get("run_id"),
        "task_id": record.get("task_id"),
        "condition": record.get("condition"),
        "run_number": record.get("run_number"),
        "status": record.get("status"),
        "verdict": record.get("verdict"),
        "second_verdict": record.get("second_verdict"),
        "verifier_calls": record.get("verifier_calls"),
        "draft": _parse_answer(record.get("draft")),
        "revision": _parse_answer(record.get("revision")),
        "final": _parse_answer(record.get("final")),
        "reviews": [
            {
                "stage": rv.get("stage"),
                "verdict": rv.get("verdict"),
                "issues": rv.get("issues") or [],
                "reason": rv.get("reason"),
                "calls": rv.get("calls"),
                "calculations": rv.get("calculations") or [],
                "error": rv.get("error"),
            }
            for rv in record.get("reviews") or []
        ],
        "draft_check": _checks(record.get("draft_check")),
        "final_check": _checks(record.get("final_check")),
        "scoring_note": (SCORING_UNDER_REVIEW if notes is None else notes).get(record.get("task_id") or ""),
        # The grades in results (possibly re-graded); the stored checks above are from when the run was graded.
        "graded": None if row is None else {"initial_correct": row.get("initial_correct"), "final_correct": row.get("final_correct")},
        "regraded_from": None if row is None else _regrade_change(row),
    }


# --------------------------------------------------------------------------- build


def build(
    results_dir: Path = RESULTS_DIR,
    out_dir: Path = OUT_DIR,
    tasks_file: Path = STUDY_TASKS_PATH,
    repeats: int = STUDY_REPEATS,
    injected_dir: Path = INJECTED_DIR,
    budget: float = BUDGET_USD,
    seed: int = 0,
    results_file: Path | None = None,
) -> dict:
    """Write every dashboard file into out_dir (replacing it) and return the manifest.

    `results_file` defaults to results_dir/results.jsonl. A re-graded file (scripts/regrade_results.py)
    keeps each run's original grades; then the original grading is exported as a sensitivity check and
    the notes say which runs changed.
    """
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    results_file = Path(results_file) if results_file else results_dir / "results.jsonl"
    rows, rejected = load_results(results_file)
    log = _read_jsonl(results_dir / "runs.jsonl")
    seeded_dir = results_dir / "seeded"
    seeded_records = right_reason.load_seeded_records(seeded_dir) if seeded_dir.exists() else []
    tasks = read_task_file(tasks_file)
    n_answers = len(list(Path(injected_dir).glob("*.json")))

    prog = progress(rows, tasks, seeded_records, n_answers, repeats)
    complete = [c for c in CONDITION_ORDER if prog[c]["complete"]]
    task_set = set(tasks)
    study_rows = [r for r in rows if r.get("task_id") in task_set and 1 <= int(r.get("run_number") or 0) <= repeats]
    regraded = any("original_final_correct" in r for r in study_rows)
    changes = regrade_changes(study_rows) if regraded else []
    notes = scoring_notes(changes) if regraded else SCORING_UNDER_REVIEW
    metrics = condition_metrics(study_rows, complete, prog, seed=seed)
    comps = comparisons(study_rows, complete, seed=seed)
    first_settings = next((r for r in study_rows if r.get("condition") == "none"), study_rows[0] if study_rows else {})

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "runs").mkdir(parents=True)

    files = {
        "summary.json": {
            "headline": headline(metrics, comps, len(tasks), repeats),
            "conditions": metrics,
            "comparisons": comps,
            "interval_note": f"Intervals are 95% bootstrap intervals over the {len(tasks)} tasks; with {len(tasks)} tasks "
                             f"and {repeats} runs each they are wide, so small differences are not conclusive.",
            "sensitivity": sensitivity(original_grading(study_rows), complete, prog, seed=seed) if changes else None,
        },
        "heatmap.json": heatmap(study_rows, tasks, complete, prog, notes),
        "pairs.json": {"complete": complete, "pairs": pairwise(study_rows, complete, seed=seed, notes=notes),
                       "note": "Paired on the same task and run number, i.e. the same shared first draft."},
        "spend.json": spend(log, budget),
        "seeded.json": seeded(seeded_records, prog),
        "sanity_s04.json": sanity_s04(log),
        "runs.json": {"runs": run_rows(study_rows, log, tasks, notes)},
    }
    for name, payload in files.items():
        _write(out_dir / name, payload)

    details = 0
    by_id = {r["run_id"]: r for r in study_rows}
    for row in files["runs.json"]["runs"]:
        path = results_dir / "answers" / f"{row['run_id']}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            _write(out_dir / "runs" / f"{_safe_name(row['run_id'])}.json", run_detail(record, by_id.get(row["run_id"]), notes))
            details += 1

    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study": {
            "question": "Which verification strategy makes multi-agent financial advisory systems more reliable?",
            "advisor_model": first_settings.get("model"),
            "temperature": first_settings.get("temperature"),
            "top_p": first_settings.get("top_p"),
            "thinking_level": first_settings.get("thinking_level") or "default",
            "n_tasks": len(tasks),
            "repeats": repeats,
            "n_scenarios": len({base_scenario(t) for t in tasks}),
            "n_seeded_answers": n_answers,
            "budget_usd": budget,
        },
        "conditions": [{"id": c, "label": CONDITION_LABELS[c], **prog[c]} for c in CONDITION_ORDER],
        "complete_conditions": complete,
        "grading": {"file": results_file.name, "regraded": regraded, "changed_runs": changes},
        "scoring_mode": "corrected" if changes else "under_review",
        "scoring_notes": notes,
        "rejected_rows": len(rejected),
        "files": sorted(files) + ["runs/<run_id>.json"],
        "run_details": details,
    }
    _write(out_dir / "meta.json", meta)
    return meta


SCORING_CORRECTED = (
    "Scoring corrected: the S06-family tasks ask only for the payment rounded upward to cents, so the checker no "
    "longer also requires the unrounded payment (same rule for every setting). Under the original rule {changes} "
    "graded wrong; the original grading is reported as a sensitivity check."
)


def original_grading(rows: list[dict]) -> list[dict]:
    """The same rows with the grades they had before re-grading (rows without originals are unchanged)."""
    out = []
    for r in rows:
        if "original_final_correct" in r:
            r = {**r, "initial_correct": r.get("original_initial_correct"), "final_correct": r.get("original_final_correct")}
        out.append(r)
    return out


def regrade_changes(rows: list[dict]) -> list[dict]:
    return [{"run_id": r["run_id"], "task_id": r["task_id"], "condition": r["condition"], "run_number": r["run_number"],
             "before": change, "after": {"initial_correct": r.get("initial_correct"), "final_correct": r.get("final_correct")}}
            for r in rows if (change := _regrade_change(r))]


def scoring_notes(changes: list[dict]) -> dict[str, str]:
    """One note per task whose grades changed, naming the runs."""
    by_task: dict[str, list[str]] = defaultdict(list)
    for c in changes:
        by_task[c["task_id"]].append(f"{CONDITION_LABELS[c['condition']].split(' ·')[0]} run {c['run_number']}")
    notes = {}
    for task, runs in by_task.items():
        changes_text = (runs[0] + " was") if len(runs) == 1 else (", ".join(runs[:-1]) + " and " + runs[-1] + " were")
        notes[task] = SCORING_CORRECTED.format(changes=changes_text)
    return notes


def sensitivity(rows: list[dict], complete: list[str], prog: dict, seed: int = 0) -> dict:
    """Task success and comparisons with None under the original grading."""
    metrics = condition_metrics(rows, complete, prog, seed=seed)
    return {
        "label": "Original grading (before the S06 scoring fix)",
        "conditions": {c: (m["task_success"] if m.get("status") == "complete" else m) for c, m in metrics.items()},
        "comparisons": comparisons(rows, complete, seed=seed),
    }


def _safe_name(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", run_id)
