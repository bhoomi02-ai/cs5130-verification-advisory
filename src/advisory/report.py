"""Turn pipeline result rows into report tables and one chart.

Reads ``results/results.jsonl`` as written by ``pipeline.run_one``. Malformed
or unusable rows are kept with a reason; they are never dropped silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from advisory.metrics import REQUIRED_COLUMNS, summarize, validate_runs

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "results" / "results.jsonl"
REPORT_DIR = ROOT / "results" / "report"

# Columns written into summary.csv / summary.md.
SUMMARY_COLUMNS = (
    "condition",
    "model",
    "task_success",
    "task_success_ci_low",
    "task_success_ci_high",
    "catch_rate",
    "false_alarm_rate",
    "mcc",
    "fixed",
    "broken",
    "success_delta",
    "mean_cost_usd",
    "p95_latency_s",
    "extra_cost_per_extra_correct",
    "n_failed",
    "n_missing_verdict",
)

INFEASIBLE_TASKS = ("S09", "S10")


def load_results(path: Path = RESULTS_PATH) -> tuple[list[dict], list[dict]]:
    """Load a JSONL results file.

    Returns ``(accepted, rejected)``. Every rejected entry is
    ``{"line": n, "reason": ..., "raw": ...}`` (and ``row`` when JSON parsed).
    """
    accepted: list[dict] = []
    rejected: list[dict] = []
    if not path.exists():
        raise FileNotFoundError(f"results file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append({"line": line_number, "reason": f"invalid JSON: {exc.msg}", "raw": line.rstrip("\n")})
                continue
            if not isinstance(row, dict):
                rejected.append({"line": line_number, "reason": "row is not a JSON object", "raw": line.rstrip("\n"), "row": row})
                continue
            reason = _row_problem(row, accepted)
            if reason:
                rejected.append({"line": line_number, "reason": reason, "raw": line.rstrip("\n"), "row": row})
                continue
            accepted.append(row)
    return accepted, rejected


def build_report(runs: list[dict], seed: int = 0) -> dict:
    """Build the summary, by-scenario and infeasible tables from accepted runs."""
    if not runs:
        raise ValueError("no accepted runs to summarise")
    summary = summarize(runs, seed=seed).reset_index()
    for column in SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = None if column == "model" else 0
    summary = summary[list(SUMMARY_COLUMNS)]
    return {
        "summary": summary,
        "by_scenario": _by_scenario(runs),
        "infeasible": _infeasible_status(runs),
    }


def write_report(
    report: dict,
    rejected: list[dict],
    out_dir: Path = REPORT_DIR,
) -> dict[str, Path]:
    """Write CSV/Markdown tables, a rejected-rows file and the success chart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    summary = report["summary"]
    paths["summary_csv"] = out_dir / "summary.csv"
    summary.to_csv(paths["summary_csv"], index=False)
    paths["summary_md"] = out_dir / "summary.md"
    paths["summary_md"].write_text(_markdown_table(summary, "Summary by condition") + "\n", encoding="utf-8")

    by_scenario = report["by_scenario"]
    paths["by_scenario_csv"] = out_dir / "by_scenario.csv"
    by_scenario.to_csv(paths["by_scenario_csv"], index=False)

    infeasible = report["infeasible"]
    paths["infeasible_csv"] = out_dir / "infeasible_status.csv"
    infeasible.to_csv(paths["infeasible_csv"], index=False)

    paths["rejected_csv"] = out_dir / "rejected_rows.csv"
    pd.DataFrame(rejected if rejected else [], columns=["line", "reason", "raw"]).to_csv(
        paths["rejected_csv"], index=False
    )

    paths["chart"] = out_dir / "task_success.png"
    _write_success_chart(summary, paths["chart"])

    paths["rejected_md"] = out_dir / "rejected.md"
    paths["rejected_md"].write_text(_rejected_markdown(rejected), encoding="utf-8")
    return paths


def analyze(
    results_path: Path = RESULTS_PATH,
    out_dir: Path = REPORT_DIR,
    seed: int = 0,
) -> dict:
    """Load results, summarise accepted rows and write the report files."""
    accepted, rejected = load_results(results_path)
    report = build_report(accepted, seed=seed)
    paths = write_report(report, rejected, out_dir=out_dir)
    return {"accepted": accepted, "rejected": rejected, "report": report, "paths": paths}


def _row_problem(row: dict, already_accepted: list[dict]) -> str | None:
    missing = [column for column in REQUIRED_COLUMNS if column not in row]
    if missing:
        return f"missing columns: {', '.join(missing)}"
    try:
        validate_runs(already_accepted + [row])
    except (TypeError, ValueError) as exc:
        return str(exc)
    return None


def _scored(runs: list[dict]) -> list[dict]:
    return [row for row in runs if row.get("status") != "failed"]


def _by_scenario(runs: list[dict]) -> pd.DataFrame:
    rows = []
    scored = _scored(runs)
    has_model = any("model" in row for row in scored)
    for row in scored:
        key = {
            "task_id": row["task_id"],
            "condition": row["condition"],
        }
        if has_model:
            key["model"] = row.get("model")
        rows.append({**key, "final_correct": bool(row["final_correct"])})
    frame = pd.DataFrame(rows)
    if frame.empty:
        columns = ["task_id", "condition", "model", "n_runs", "success_rate"] if has_model else [
            "task_id",
            "condition",
            "n_runs",
            "success_rate",
        ]
        return pd.DataFrame(columns=columns)
    group_cols = ["task_id", "condition", "model"] if has_model else ["task_id", "condition"]
    grouped = (
        frame.groupby(group_cols, sort=True)
        .agg(n_runs=("final_correct", "size"), success_rate=("final_correct", "mean"))
        .reset_index()
    )
    return grouped


def _infeasible_status(runs: list[dict]) -> pd.DataFrame:
    """Strict success vs answer-status-only success for S09 and S10.

    Status-only uses optional ``answer_status`` on the result row (the final
    answer's FEASIBLE / INFEASIBLE / ...). Pipeline rows without that field
    get a null status-only rate.
    """
    scored = [row for row in _scored(runs) if row.get("task_id") in INFEASIBLE_TASKS]
    has_model = any("model" in row for row in scored)
    records = []
    for row in scored:
        record = {
            "task_id": row["task_id"],
            "condition": row["condition"],
            "final_correct": bool(row["final_correct"]),
            "status_only_correct": (
                None
                if "answer_status" not in row
                else row.get("answer_status") == "INFEASIBLE"
            ),
        }
        if has_model:
            record["model"] = row.get("model")
        records.append(record)
    if not records:
        columns = ["task_id", "condition", "model", "n_runs", "strict_success", "status_only_success"]
        if not has_model:
            columns = ["task_id", "condition", "n_runs", "strict_success", "status_only_success"]
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(records)
    group_cols = ["task_id", "condition", "model"] if has_model else ["task_id", "condition"]

    def _mean_or_none(series: pd.Series):
        values = [value for value in series.tolist() if value is not None]
        if not values:
            return None
        return sum(bool(value) for value in values) / len(values)

    rows = []
    for key, group in frame.groupby(group_cols, sort=True):
        if has_model:
            task_id, condition, model = key
        else:
            task_id, condition = key
            model = None
        item = {
            "task_id": task_id,
            "condition": condition,
            "n_runs": len(group),
            "strict_success": float(group["final_correct"].mean()),
            "status_only_success": _mean_or_none(group["status_only_correct"]),
        }
        if has_model:
            item["model"] = model
            rows.append(item)
        else:
            rows.append(item)
    columns = ["task_id", "condition", "model", "n_runs", "strict_success", "status_only_success"] if has_model else [
        "task_id",
        "condition",
        "n_runs",
        "strict_success",
        "status_only_success",
    ]
    return pd.DataFrame(rows)[columns]


def _markdown_table(frame: pd.DataFrame, title: str) -> str:
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(_format_cell)
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in display.to_numpy())
    return f"# {title}\n\n{header}\n{sep}\n{body}\n"


def _format_cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _rejected_markdown(rejected: list[dict]) -> str:
    lines = [f"# Rejected rows\n\n{len(rejected)} row(s) were not summarised.\n"]
    if not rejected:
        return "\n".join(lines) + "\n"
    lines.append("| line | reason |")
    lines.append("| --- | --- |")
    for item in rejected:
        reason = str(item["reason"]).replace("|", "\\|")
        lines.append(f"| {item['line']} | {reason} |")
    return "\n".join(lines) + "\n"


def _write_success_chart(summary: pd.DataFrame, path: Path) -> None:
    labels = []
    for _, row in summary.iterrows():
        if row.get("model") not in (None, "") and not (isinstance(row.get("model"), float) and pd.isna(row.get("model"))):
            labels.append(f"{row['condition']}\n{row['model']}")
        else:
            labels.append(str(row["condition"]))
    success = summary["task_success"].astype(float).to_numpy()
    low = summary["task_success_ci_low"].astype(float).to_numpy()
    high = summary["task_success_ci_high"].astype(float).to_numpy()
    # Error bars are distance from the point to each CI end.
    yerr = [success - low, high - success]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(range(len(labels)), success, yerr=yerr, capsize=4, color="#4C78A8")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Task success")
    axis.set_title("Task success by condition")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
