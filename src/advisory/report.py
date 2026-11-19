"""Turn pipeline result rows into report tables and charts.

Reads ``results/results.jsonl`` as written by ``pipeline.run_one``. Malformed
or unusable rows are kept with a reason; they are never dropped silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from advisory.metrics import REQUIRED_COLUMNS, compare_conditions, summarize, validate_runs

# Fixed display order and colour-blind-safe palette (Okabe–Ito) for condition marks.
CONDITION_ORDER = ("none", "self", "peer", "dedicated", "weakened", "cross_family")
CONDITION_COLORS = {
    "none": "#0072B2",
    "self": "#E69F00",
    "peer": "#009E73",
    "dedicated": "#CC79A7",
    "weakened": "#D55E00",
    "cross_family": "#56B4E9",
}
# Outcome / catch / right-reason stacks use a separate palette (not condition colours).
OUTCOME_STACK = (
    ("Kept correct", "#7F7F7F"),
    ("Fixed", "#009E73"),
    ("Broken", "#D55E00"),
    ("Still wrong", "#CCCCCC"),
)
CATCH_STACK = (
    ("Missed", "#CCCCCC"),
    ("Caught and fixed", "#009E73"),
    ("Caught but not fixed", "#E69F00"),
    ("Escalated", "#56B4E9"),
)
RIGHT_REASON_STACK = (
    ("Right reason", "#009E73"),
    ("Consequence only", "#E69F00"),
    ("Wrong reason", "#D55E00"),
    ("Abstained", "#56B4E9"),
)
_FIGURE_DPI = 150
_PNG_METADATA = {"Software": "advisory.report", "Creation Time": "1970-01-01T00:00:00"}

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

INFEASIBLE_TASKS = ("S09", "S10", "T09-v001", "T09-v002", "T09-v003", "T10-v001", "T10-v002", "T10-v003")

# Outcome columns live in a separate table so summary.csv stays narrow.
OUTCOME_COLUMNS = (
    "condition",
    "model",
    "task_success",
    "task_success_excl_human_review",
    "human_review_rate",
    "task_success_excl_human_review_ci_low",
    "task_success_excl_human_review_ci_high",
    "kept_correct",
    "fixed",
    "broken",
    "still_wrong",
    "missed",
    "caught_fixed",
    "caught_not_fixed",
    "caught_escalated",
    "catch_fix_rate",
)


def load_results(path: Path = RESULTS_PATH) -> tuple[list[dict], list[dict]]:
    """Load a JSONL results file.

    Returns ``(accepted, rejected)``. Every rejected entry is
    ``{"line": n, "reason": ..., "raw": ...}`` (and ``row`` when JSON parsed).
    Each row is validated on its own; duplicates are checked with a set of
    identity keys so loading stays linear in the number of lines.
    """
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[tuple] = set()
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
            reason = _row_problem(row)
            if reason:
                rejected.append({"line": line_number, "reason": reason, "raw": line.rstrip("\n"), "row": row})
                continue
            key = _identity_key(row)
            if key in seen:
                rejected.append(
                    {
                        "line": line_number,
                        "reason": _duplicate_message(row),
                        "raw": line.rstrip("\n"),
                        "row": row,
                    }
                )
                continue
            seen.add(key)
            accepted.append(row)
    return accepted, rejected


def build_report(runs: list[dict], seed: int = 0) -> dict:
    """Build the summary, outcomes, by-scenario and infeasible tables from accepted runs."""
    if not runs:
        raise ValueError("no accepted runs to summarise")
    metrics = summarize(runs, seed=seed).reset_index()
    summary = metrics.copy()
    for column in SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = None
    summary = summary[list(SUMMARY_COLUMNS)]
    outcomes = metrics.copy()
    for column in OUTCOME_COLUMNS:
        if column not in outcomes.columns:
            outcomes[column] = None
    outcomes = outcomes[list(OUTCOME_COLUMNS)]
    comparisons = compare_conditions(runs, baseline="none", seed=seed)
    return {
        "summary": summary,
        "outcomes": outcomes,
        "comparisons": comparisons,
        "by_scenario": _by_scenario(runs),
        "infeasible": _infeasible_status(runs),
    }


def write_report(
    report: dict,
    rejected: list[dict],
    out_dir: Path = REPORT_DIR,
    *,
    seeded_dir: Path | None = None,
    labels_path: Path | None = None,
) -> dict[str, Path]:
    """Write CSV/Markdown tables, a rejected-rows file and the success chart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    summary = report["summary"]
    paths["summary_csv"] = out_dir / "summary.csv"
    summary.to_csv(paths["summary_csv"], index=False)
    paths["summary_md"] = out_dir / "summary.md"
    paths["summary_md"].write_text(_markdown_table(summary, "Summary by condition") + "\n", encoding="utf-8")

    outcomes = report["outcomes"]
    paths["outcomes_csv"] = out_dir / "outcomes.csv"
    outcomes.to_csv(paths["outcomes_csv"], index=False)
    paths["outcomes_md"] = out_dir / "outcomes.md"
    paths["outcomes_md"].write_text(
        _markdown_table(outcomes, "Outcomes by condition")
        + "\n"
        + "task_success_excl_human_review excludes runs the verifier escalated; "
        "read it together with human_review_rate.\n",
        encoding="utf-8",
    )

    comparisons = report["comparisons"]
    paths["comparisons_csv"] = out_dir / "comparisons.csv"
    comparisons.to_csv(paths["comparisons_csv"], index=False)
    paths["comparisons_md"] = out_dir / "comparisons.md"
    paths["comparisons_md"].write_text(
        _markdown_table(comparisons, "Paired comparisons vs none") + "\n",
        encoding="utf-8",
    )
    if "note" in comparisons.columns:
        for _, row in comparisons.iterrows():
            note = row.get("note")
            if note is None or (isinstance(note, float) and pd.isna(note)) or note == "":
                continue
            label = f"{row['condition_a']} vs {row['condition_b']}"
            if row.get("model") not in (None, "") and not (
                isinstance(row.get("model"), float) and pd.isna(row.get("model"))
            ):
                label = f"{label} ({row['model']})"
            print(f"warning: comparison {label}: {note}", file=sys.stderr)

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

    seeded_paths = _write_seeded_outputs(
        out_dir,
        seeded_dir=seeded_dir,
        labels_path=labels_path,
    )
    paths.update(seeded_paths)
    paths.update(
        _write_result_figures(
            report,
            out_dir,
            seeded_csv=seeded_paths.get("seeded_csv"),
        )
    )
    return paths


def analyze(
    results_path: Path = RESULTS_PATH,
    out_dir: Path = REPORT_DIR,
    seed: int = 0,
    *,
    seeded_dir: Path | None = None,
    labels_path: Path | None = None,
) -> dict:
    """Load results, summarise accepted rows and write the report files."""
    accepted, rejected = load_results(results_path)
    report = build_report(accepted, seed=seed)
    paths = write_report(
        report,
        rejected,
        out_dir=out_dir,
        seeded_dir=seeded_dir,
        labels_path=labels_path,
    )
    return {"accepted": accepted, "rejected": rejected, "report": report, "paths": paths}


def _write_seeded_outputs(
    out_dir: Path,
    *,
    seeded_dir: Path | None = None,
    labels_path: Path | None = None,
) -> dict[str, Path]:
    """Write seeded right-reason tables when ``results/seeded`` has JSON files."""
    from advisory.right_reason import (
        LABELS_PATH,
        SEEDED_DIR,
        audit_table,
        load_labels,
        load_seeded_records,
        right_reason_metrics,
    )

    directory = SEEDED_DIR if seeded_dir is None else Path(seeded_dir)
    if not directory.exists() or not any(directory.glob("*.json")):
        return {}
    labels = load_labels(LABELS_PATH if labels_path is None else labels_path)
    records = load_seeded_records(directory)
    metrics = right_reason_metrics(records, labels)
    audit = audit_table(records, labels)
    paths: dict[str, Path] = {}
    paths["seeded_csv"] = out_dir / "seeded.csv"
    metrics.to_csv(paths["seeded_csv"], index=False)
    paths["seeded_md"] = out_dir / "seeded.md"
    paths["seeded_md"].write_text(
        _markdown_table(metrics, "Seeded right-reason metrics") + "\n",
        encoding="utf-8",
    )
    paths["right_reason_audit_csv"] = out_dir / "right_reason_audit.csv"
    audit.to_csv(paths["right_reason_audit_csv"], index=False)
    return paths


def _row_problem(row: dict) -> str | None:
    missing = [column for column in REQUIRED_COLUMNS if column not in row]
    if missing:
        return f"missing columns: {', '.join(missing)}"
    try:
        validate_runs([row])
    except (TypeError, ValueError) as exc:
        return str(exc)
    return None


def _identity_key(row: dict) -> tuple:
    if "model" in row:
        return (row["task_id"], row["condition"], row["run_number"], row["model"])
    return (row["task_id"], row["condition"], row["run_number"])


def _duplicate_message(row: dict) -> str:
    if "model" in row:
        return "duplicate rows for the same task_id, condition, run_number, and model"
    return "duplicate rows for the same task_id, condition, and run_number"


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
    """Strict success vs status-only success for S09 and S10.

    Status-only uses ``final_status_correct`` from the pipeline (the checker's
    STATUS verdict on the final answer). Rows without that field get a null
    status-only rate.
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
                if "final_status_correct" not in row
                else bool(row["final_status_correct"])
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
    models = []
    if "model" in summary.columns:
        for value in summary["model"].tolist():
            if value in (None, "") or (isinstance(value, float) and pd.isna(value)):
                continue
            models.append(str(value))
    several_models = len(set(models)) > 1
    labels = []
    for _, row in summary.iterrows():
        model = row.get("model")
        has_model = model not in (None, "") and not (
            isinstance(model, float) and pd.isna(model)
        )
        if several_models and has_model:
            labels.append(f"{row['condition']}\n{model}")
        else:
            labels.append(str(row["condition"]))
    # All-failed groups leave rates as None; coerce so the chart still writes.
    success = pd.to_numeric(summary["task_success"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(summary["task_success_ci_low"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(summary["task_success_ci_high"], errors="coerce").to_numpy(dtype=float)
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


def _write_result_figures(
    report: dict,
    out_dir: Path,
    *,
    seeded_csv: Path | None = None,
) -> dict[str, Path]:
    """Write the paired, outcome, verifier, cost and optional right-reason figures."""
    paths: dict[str, Path] = {}
    writers = (
        ("comparisons.png", write_comparisons_figure, report["comparisons"]),
        ("outcomes.png", write_outcomes_figure, report["outcomes"]),
        ("catch_outcomes.png", write_catch_outcomes_figure, report["outcomes"]),
        ("verifier_rates.png", write_verifier_rates_figure, report["summary"]),
        ("cost_vs_success.png", write_cost_vs_success_figure, report["summary"]),
    )
    for name, writer, frame in writers:
        path = out_dir / name
        if writer(frame, path):
            paths[name] = path
    if seeded_csv is not None and Path(seeded_csv).exists():
        seeded = pd.read_csv(seeded_csv)
        path = out_dir / "right_reason.png"
        if write_right_reason_figure(seeded, path):
            paths["right_reason.png"] = path
    return paths


def write_comparisons_figure(comparisons: pd.DataFrame, path: Path) -> bool:
    """Forest plot of paired deltas versus none.

    Parameters
    ----------
    comparisons : DataFrame
        Output of ``compare_conditions`` (one row per pair).
    path : Path
        Destination PNG path.

    Returns
    -------
    bool
        True when a figure was written; False when skipped.
    """
    if comparisons is None or comparisons.empty:
        print(f"note: skipping {path.name}: no comparison rows", file=sys.stderr)
        return False
    frame = comparisons.copy()
    if "condition_b" not in frame.columns:
        print(f"note: skipping {path.name}: missing condition_b", file=sys.stderr)
        return False
    # One display row per condition_b in fixed order (mean over models if many).
    rows = []
    for condition in CONDITION_ORDER:
        if condition == "none":
            continue
        subset = frame.loc[frame["condition_b"] == condition]
        if subset.empty:
            continue
        keep = []
        notes = []
        for _, item in subset.iterrows():
            note = ""
            if "note" in subset.columns:
                raw_note = item.get("note")
                if raw_note is not None and not (
                    isinstance(raw_note, float) and pd.isna(raw_note)
                ):
                    note = str(raw_note)
            notes.append(note)
            delta_value = pd.to_numeric(item.get("delta"), errors="coerce")
            keep.append(note == "" and pd.notna(delta_value))
        usable = subset.iloc[[index for index, flag in enumerate(keep) if flag]]
        if usable.empty:
            note = next((value for value in notes if value), "no usable pairs")
            rows.append(
                {
                    "condition": condition,
                    "delta": None,
                    "low": None,
                    "high": None,
                    "sign_test_p": None,
                    "note": note,
                }
            )
            continue
        delta = float(pd.to_numeric(usable["delta"], errors="coerce").mean())
        low = float(pd.to_numeric(usable["delta_ci_low"], errors="coerce").mean())
        high = float(pd.to_numeric(usable["delta_ci_high"], errors="coerce").mean())
        p_value = pd.to_numeric(usable["sign_test_p"], errors="coerce").mean()
        rows.append(
            {
                "condition": condition,
                "delta": delta,
                "low": low,
                "high": high,
                "sign_test_p": None if pd.isna(p_value) else float(p_value),
                "note": "",
            }
        )
    if not rows:
        print(f"note: skipping {path.name}: no conditions to plot", file=sys.stderr)
        return False

    fig, axis = plt.subplots(figsize=(8.5, 0.6 * len(rows) + 1.5))
    y_positions = list(range(len(rows)))
    plotted_highs = []
    for index, row in enumerate(rows):
        color = CONDITION_COLORS[row["condition"]]
        y = y_positions[index]
        if row["note"]:
            axis.text(
                0.0,
                y,
                f"{row['condition']}: {row['note']}",
                va="center",
                ha="left",
                color=color,
                fontsize=9,
            )
            continue
        axis.errorbar(
            row["delta"],
            y,
            xerr=[[row["delta"] - row["low"]], [row["high"] - row["delta"]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=7,
        )
        plotted_highs.append(row["high"])
        p_text = (
            "p = —"
            if row["sign_test_p"] is None
            else f"p = {row['sign_test_p']:.2f}"
        )
        axis.text(
            row["high"],
            y,
            f"  {p_text}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )
    axis.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    axis.set_yticks(y_positions)
    axis.set_yticklabels([row["condition"] for row in rows])
    axis.set_xlabel("Delta task success (condition − none)")
    axis.set_title("Paired comparisons versus none")
    axis.invert_yaxis()
    if plotted_highs:
        low_bound = min(
            (row["low"] for row in rows if row["low"] is not None),
            default=-0.05,
        )
        high_bound = max(plotted_highs)
        span = max(high_bound - low_bound, 0.1)
        axis.set_xlim(low_bound - 0.05 * span, high_bound + 0.28 * span)
    # bbox_inches="tight" handles margins; tight_layout warns when p= labels
    # need extra right space.
    _savefig(fig, path)
    return True


def write_outcomes_figure(outcomes: pd.DataFrame, path: Path) -> bool:
    """100% stacked bars of kept / fixed / broken / still-wrong shares.

    Parameters
    ----------
    outcomes : DataFrame
        Per-condition outcome counts from ``summarize``.
    path : Path
        Destination PNG path.

    Returns
    -------
    bool
        True when a figure was written; False when skipped.
    """
    frame = _condition_frame(outcomes)
    if frame is None:
        print(f"note: skipping {path.name}: no outcome rows", file=sys.stderr)
        return False
    columns = ("kept_correct", "fixed", "broken", "still_wrong")
    if any(column not in frame.columns for column in columns):
        print(f"note: skipping {path.name}: missing outcome columns", file=sys.stderr)
        return False
    conditions = [name for name in CONDITION_ORDER if name in set(frame["condition"])]
    if not conditions:
        print(f"note: skipping {path.name}: no known conditions", file=sys.stderr)
        return False

    shares = []
    for condition in conditions:
        row = frame.loc[frame["condition"] == condition].iloc[0]
        counts = [
            float(pd.to_numeric(row[column], errors="coerce") or 0.0)
            for column in columns
        ]
        total = sum(counts)
        if total <= 0:
            shares.append([0.0, 0.0, 0.0, 0.0])
        else:
            shares.append([value / total for value in counts])
    shares = np.asarray(shares, dtype=float)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    _draw_stacked_bars(axis, conditions, shares, OUTCOME_STACK)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Share of scored runs")
    axis.set_title("Outcome mix by condition")
    _legend_below(axis, OUTCOME_STACK, ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _savefig(fig, path)
    return True


def write_catch_outcomes_figure(outcomes: pd.DataFrame, path: Path) -> bool:
    """Stacked counts of fates for initially wrong drafts under verified conditions.

    Parameters
    ----------
    outcomes : DataFrame
        Per-condition catch-outcome counts from ``summarize``.
    path : Path
        Destination PNG path.

    Returns
    -------
    bool
        True when a figure was written; False when skipped.
    """
    frame = _condition_frame(outcomes)
    if frame is None:
        print(f"note: skipping {path.name}: no outcome rows", file=sys.stderr)
        return False
    columns = ("missed", "caught_fixed", "caught_not_fixed", "caught_escalated")
    if any(column not in frame.columns for column in columns):
        print(f"note: skipping {path.name}: missing catch-outcome columns", file=sys.stderr)
        return False
    conditions = [
        name
        for name in CONDITION_ORDER
        if name != "none" and name in set(frame["condition"])
    ]
    plotted = []
    matrix = []
    for condition in conditions:
        row = frame.loc[frame["condition"] == condition].iloc[0]
        counts = [
            pd.to_numeric(row[column], errors="coerce")
            for column in columns
        ]
        if all(pd.isna(value) for value in counts):
            continue
        plotted.append(condition)
        matrix.append([0.0 if pd.isna(value) else float(value) for value in counts])
    if not plotted:
        print(f"note: skipping {path.name}: no verifier catch outcomes", file=sys.stderr)
        return False
    matrix = np.asarray(matrix, dtype=float)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    _draw_stacked_bars(axis, plotted, matrix, CATCH_STACK)
    axis.set_ylabel("Count of initially wrong drafts")
    axis.set_title("Catch outcomes by verified condition")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    _legend_below(axis, CATCH_STACK, ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _savefig(fig, path)
    return True


def write_verifier_rates_figure(summary: pd.DataFrame, path: Path) -> bool:
    """Grouped bars of catch rate and false-alarm rate for verified conditions.

    Parameters
    ----------
    summary : DataFrame
        Per-condition summary metrics from ``summarize``.
    path : Path
        Destination PNG path.

    Returns
    -------
    bool
        True when a figure was written; False when skipped.
    """
    frame = _condition_frame(summary)
    if frame is None:
        print(f"note: skipping {path.name}: no summary rows", file=sys.stderr)
        return False
    for column in ("catch_rate", "false_alarm_rate"):
        if column not in frame.columns:
            print(f"note: skipping {path.name}: missing {column}", file=sys.stderr)
            return False
    conditions = []
    catch_rates: list[float | None] = []
    false_alarms: list[float | None] = []
    for condition in CONDITION_ORDER:
        if condition == "none" or condition not in set(frame["condition"]):
            continue
        row = frame.loc[frame["condition"] == condition].iloc[0]
        catch = pd.to_numeric(row["catch_rate"], errors="coerce")
        false_alarm = pd.to_numeric(row["false_alarm_rate"], errors="coerce")
        if pd.isna(catch) and pd.isna(false_alarm):
            continue
        conditions.append(condition)
        catch_rates.append(None if pd.isna(catch) else float(catch))
        false_alarms.append(None if pd.isna(false_alarm) else float(false_alarm))
    if not conditions:
        print(f"note: skipping {path.name}: no verifier rates", file=sys.stderr)
        return False

    x = np.arange(len(conditions))
    width = 0.35
    fig, axis = plt.subplots(figsize=(8, 4.8))
    bar_colors = [CONDITION_COLORS[name] for name in conditions]
    catch_heights = [0.0 if value is None else value for value in catch_rates]
    false_heights = [0.0 if value is None else value for value in false_alarms]
    axis.bar(
        x - width / 2,
        catch_heights,
        width,
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.6,
    )
    axis.bar(
        x + width / 2,
        false_heights,
        width,
        color=bar_colors,
        alpha=0.45,
        edgecolor="#333333",
        linewidth=0.6,
        hatch="//",
    )
    for index, value in enumerate(catch_rates):
        height = 0.0 if value is None else value
        text = "n/a" if value is None else f"{value:.2f}"
        axis.text(
            x[index] - width / 2,
            height + 0.02,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
    for index, value in enumerate(false_alarms):
        height = 0.0 if value is None else value
        text = "n/a" if value is None else f"{value:.2f}"
        axis.text(
            x[index] + width / 2,
            height + 0.02,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
    axis.set_xticks(x)
    axis.set_xticklabels(conditions)
    axis.set_ylim(0, 1.18)
    axis.set_ylabel("Rate")
    axis.set_title("Verifier catch and false-alarm rates")
    handles = [
        Patch(facecolor="#888888", edgecolor="#333333", label="Catch rate"),
        Patch(
            facecolor="#888888",
            edgecolor="#333333",
            hatch="//",
            label="False-alarm rate",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _savefig(fig, path)
    return True


def write_cost_vs_success_figure(summary: pd.DataFrame, path: Path) -> bool:
    """Scatter of mean cost per run versus task success with CI error bars.

    Parameters
    ----------
    summary : DataFrame
        Per-condition summary metrics from ``summarize``.
    path : Path
        Destination PNG path.

    Returns
    -------
    bool
        True when a figure was written; False when skipped.
    """
    frame = _condition_frame(summary)
    if frame is None:
        print(f"note: skipping {path.name}: no summary rows", file=sys.stderr)
        return False
    needed = ("mean_cost_usd", "task_success", "task_success_ci_low", "task_success_ci_high")
    if any(column not in frame.columns for column in needed):
        print(f"note: skipping {path.name}: missing cost or success columns", file=sys.stderr)
        return False
    points = []
    for condition in CONDITION_ORDER:
        if condition not in set(frame["condition"]):
            continue
        row = frame.loc[frame["condition"] == condition].iloc[0]
        cost = pd.to_numeric(row["mean_cost_usd"], errors="coerce")
        success = pd.to_numeric(row["task_success"], errors="coerce")
        low = pd.to_numeric(row["task_success_ci_low"], errors="coerce")
        high = pd.to_numeric(row["task_success_ci_high"], errors="coerce")
        if pd.isna(cost) or pd.isna(success):
            continue
        points.append(
            {
                "condition": condition,
                "cost": float(cost),
                "success": float(success),
                "low": float(success if pd.isna(low) else low),
                "high": float(success if pd.isna(high) else high),
            }
        )
    if not points:
        print(f"note: skipping {path.name}: no cost/success points", file=sys.stderr)
        return False

    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    handles = []
    for point in points:
        color = CONDITION_COLORS[point["condition"]]
        axis.errorbar(
            point["cost"],
            point["success"],
            yerr=[
                [point["success"] - point["low"]],
                [point["high"] - point["success"]],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=8,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                linestyle="None",
                markersize=8,
                label=point["condition"],
            )
        )
    axis.set_xlabel("Mean cost per run (USD)")
    axis.set_ylabel("Task success")
    axis.set_ylim(0, 1.05)
    axis.set_title("Cost versus task success")
    axis.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    _savefig(fig, path)
    return True


def write_right_reason_figure(seeded: pd.DataFrame, path: Path) -> bool:
    """Stacked counts of right-reason decisions among flagged seeded errors.

    Parameters
    ----------
    seeded : DataFrame
        Output of ``right_reason_metrics`` (one row per condition).
    path : Path
        Destination PNG path.

    Returns
    -------
    bool
        True when a figure was written; False when skipped.
    """
    if seeded is None or seeded.empty:
        print(f"note: skipping {path.name}: no seeded metrics", file=sys.stderr)
        return False
    columns = ("right_reason", "consequence_only", "wrong_reason", "abstained")
    if any(column not in seeded.columns for column in columns):
        print(f"note: skipping {path.name}: missing right-reason columns", file=sys.stderr)
        return False
    frame = seeded.copy()
    if "condition" not in frame.columns:
        print(f"note: skipping {path.name}: missing condition column", file=sys.stderr)
        return False
    # Aggregate over verifier models if more than one row per condition.
    numeric = frame.groupby("condition", sort=False)[list(columns)].sum(min_count=1)
    conditions = [name for name in CONDITION_ORDER if name in numeric.index and name != "none"]
    if not conditions:
        print(f"note: skipping {path.name}: no verified seeded conditions", file=sys.stderr)
        return False
    matrix = []
    plotted = []
    for condition in conditions:
        counts = [
            float(numeric.loc[condition, column])
            if pd.notna(numeric.loc[condition, column])
            else 0.0
            for column in columns
        ]
        if sum(counts) <= 0:
            continue
        plotted.append(condition)
        matrix.append(counts)
    if not plotted:
        print(f"note: skipping {path.name}: no flagged seeded errors", file=sys.stderr)
        return False
    matrix = np.asarray(matrix, dtype=float)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    _draw_stacked_bars(axis, plotted, matrix, RIGHT_REASON_STACK)
    axis.set_ylabel("Count of flagged seeded errors")
    axis.set_title("Right-reason decisions on seeded errors")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    _legend_below(axis, RIGHT_REASON_STACK, ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _savefig(fig, path)
    return True


def _condition_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Collapse optional model rows to one row per condition (means of numerics)."""
    if frame is None or frame.empty or "condition" not in frame.columns:
        return None
    working = frame.copy()
    if "model" in working.columns and working["model"].notna().any():
        value_cols = [column for column in working.columns if column not in ("condition", "model")]
        for column in value_cols:
            if not pd.api.types.is_numeric_dtype(working[column]):
                working[column] = pd.to_numeric(working[column], errors="coerce")
        if not value_cols:
            return working.drop_duplicates(subset=["condition"])
        return (
            working.groupby("condition", sort=False)[value_cols]
            .mean()
            .reset_index()
        )
    return working


def _draw_stacked_bars(axis, conditions, matrix, stack) -> None:
    """Draw stacked bars with an explicit outcome palette; black x tick labels."""
    bottom = np.zeros(len(conditions))
    x = np.arange(len(conditions))
    for index, (_label, color) in enumerate(stack):
        heights = matrix[:, index]
        axis.bar(x, heights, bottom=bottom, color=color, width=0.7, linewidth=0)
        bottom = bottom + heights
    axis.set_xticks(x)
    axis.set_xticklabels(conditions)
    for tick in axis.get_xticklabels():
        tick.set_color("#000000")


def _legend_below(axis, stack, *, ncol: int) -> None:
    """Place Patch legend handles below the axes so every category has a swatch."""
    handles = [Patch(facecolor=color, edgecolor=color, label=label) for label, color in stack]
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
        frameon=False,
        fontsize=8,
    )


def _savefig(fig, path: Path) -> None:
    """Save a figure at 150 dpi with fixed PNG metadata, then close it."""
    fig.savefig(path, dpi=_FIGURE_DPI, metadata=_PNG_METADATA, bbox_inches="tight")
    plt.close(fig)
