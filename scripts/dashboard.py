"""Streamlit dashboard of the study results. Reads saved files only; no model calls.

    pip install -e ".[dashboard]"
    streamlit run scripts/dashboard.py
    streamlit run scripts/dashboard.py -- --results-dir path/to/results   # e.g. a study-results checkout
    (or set ADVISORY_RESULTS_DIR)

Tabs: overview (task success, paired gain vs none, cost), tasks (32 tasks x
conditions, with each run's draft, verdicts, verifier issues and failed
checks), verifiers (catch and false-alarm rates, outcome mix, seeded
right-reason results), and progress and spend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from advisory import dashboard_data as data
from advisory.llm import budget_usd

COLORS = {  # same palette as the report figures
    "none": "#0072B2",
    "self": "#E69F00",
    "peer": "#009E73",
    "dedicated": "#CC79A7",
    "weakened": "#D55E00",
    "cross_family": "#56B4E9",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default = Path(os.environ.get("ADVISORY_RESULTS_DIR") or data.RESULTS_DIR)
    parser.add_argument("--results-dir", type=Path, default=default)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def _mtime(path: Path) -> float:
    """Changes when a file is added or rewritten, so cached data refreshes after new runs."""
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_mtime
    return max([p.stat().st_mtime for p in path.iterdir()] + [path.stat().st_mtime])


@st.cache_data(show_spinner="Loading results…")
def load(results_dir: str, _stamp: tuple) -> dict:
    root = Path(results_dir)
    runs, rejected = data.load_runs(root)
    seeded = data.seeded_records(root)
    tasks = data.study_tasks()
    progress = data.progress(runs, tasks, seeded_records=seeded, n_answers=data.n_seeded_answers())
    seeded_metrics, seeded_audit = data.seeded_tables(seeded)
    return {
        "runs": runs,
        "rejected": rejected,
        "progress": progress,
        "seeded_metrics": seeded_metrics,
        "seeded_audit": seeded_audit,
        "spend": data.spend(root),
    }


@st.cache_data(show_spinner="Computing metrics…")
def headline(runs: pd.DataFrame, conditions: tuple[str, ...]) -> dict:
    return data.headline(runs, list(conditions))


def _color_scale(conditions: list[str]) -> alt.Scale:
    return alt.Scale(domain=conditions, range=[COLORS.get(c, "#888888") for c in conditions])


def _pct(value) -> str:
    return "" if value is None or pd.isna(value) else f"{value:.1%}"


def overview(summary: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    conditions = list(summary["condition"])
    base = summary.loc[summary["condition"] == "none", "task_success"]
    cols = st.columns(len(conditions))
    for col, (_, row) in zip(cols, summary.iterrows()):
        delta, color = None, "normal"
        if len(base) and row["condition"] != "none":
            points = (row["task_success"] - base.iloc[0]) * 100
            delta = f"{points:+.1f} pts vs none"
            color = "off" if abs(points) < 0.05 else "normal"
        col.metric(row["condition"], _pct(row["task_success"]), delta, delta_color=color)

    left, right = st.columns(2)
    with left:
        st.subheader("Task success (95% CI over tasks)")
        frame = summary[["condition", "task_success", "task_success_ci_low", "task_success_ci_high"]].copy()
        scale = _color_scale(conditions)
        x = alt.X("condition:N", sort=conditions, title=None, axis=alt.Axis(labelAngle=0))
        bars = alt.Chart(frame).mark_bar().encode(
            x=x, y=alt.Y("task_success:Q", scale=alt.Scale(domain=[0, 1]), title="Task success"),
            color=alt.Color("condition:N", scale=scale, legend=None),
            tooltip=["condition", alt.Tooltip("task_success:Q", format=".1%")],
        )
        errors = alt.Chart(frame).mark_errorbar().encode(
            x=x, y=alt.Y("task_success_ci_low:Q", title="Task success"), y2="task_success_ci_high:Q"
        )
        st.altair_chart(bars + errors, width="stretch")
    with right:
        st.subheader("Paired gain vs none")
        if comparisons is None or comparisons.empty:
            st.info("No paired comparisons yet (needs none and another condition).")
        else:
            comp = comparisons[comparisons["condition_a"] == "none"].copy()
            comp = comp[pd.to_numeric(comp["delta"], errors="coerce").notna()]
            order = [c for c in conditions if c in set(comp["condition_b"])]
            scale = _color_scale(order)
            y = alt.Y("condition_b:N", sort=order, title=None)
            points = alt.Chart(comp).mark_point(filled=True, size=90).encode(
                x=alt.X("delta:Q", title="Change in task success (condition − none)"),
                y=y, color=alt.Color("condition_b:N", scale=scale, legend=None),
                tooltip=[
                    alt.Tooltip("condition_b:N", title="condition"),
                    alt.Tooltip("delta:Q", format="+.3f"),
                    alt.Tooltip("delta_ci_low:Q", format="+.3f"),
                    alt.Tooltip("delta_ci_high:Q", format="+.3f"),
                    alt.Tooltip("sign_test_p:Q", format=".3f", title="sign test p"),
                    "task_b_better", "task_a_better",
                ],
            )
            whiskers = alt.Chart(comp).mark_rule().encode(
                x="delta_ci_low:Q", x2="delta_ci_high:Q", y=y,
                color=alt.Color("condition_b:N", scale=scale, legend=None),
            )
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
            st.altair_chart((whiskers + points + zero).properties(height=300), width="stretch")
            st.caption("Bars are 95% bootstrap CIs over tasks; hover for the task-level sign test.")

    st.subheader("Summary table")
    shown = summary[[
        "condition", "task_success", "catch_rate", "false_alarm_rate", "mcc", "fixed", "broken",
        "mean_cost_usd", "extra_cost_per_extra_correct", "n_failed",
    ]]
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        column_config={
            "task_success": st.column_config.NumberColumn(format="percent"),
            "catch_rate": st.column_config.NumberColumn(format="percent"),
            "false_alarm_rate": st.column_config.NumberColumn(format="percent"),
            "mcc": st.column_config.NumberColumn(format="%.2f"),
            "mean_cost_usd": st.column_config.NumberColumn("mean cost / run", format="$%.4f"),
            "extra_cost_per_extra_correct": st.column_config.NumberColumn("$ per extra correct", format="$%.2f"),
        },
    )

    st.subheader("Cost vs task success")
    scale = _color_scale(conditions)
    scatter = alt.Chart(summary).mark_circle(size=140).encode(
        x=alt.X("mean_cost_usd:Q", title="Mean cost per run (USD)", scale=alt.Scale(zero=False, padding=20)),
        y=alt.Y("task_success:Q", scale=alt.Scale(domain=[0.6, 1]), title="Task success"),
        color=alt.Color("condition:N", scale=scale),
        tooltip=["condition", alt.Tooltip("mean_cost_usd:Q", format="$.4f"), alt.Tooltip("task_success:Q", format=".1%")],
    )
    st.altair_chart(scatter, width="stretch")


def _verdict_block(review: dict) -> None:
    st.markdown(f"**{review.get('stage', '')} review:** `{review.get('verdict') or 'unreadable'}` · {review.get('calls', '?')} call(s)")
    if review.get("reason"):
        st.write(review["reason"])
    for issue in review.get("issues") or []:
        st.markdown(f"- {issue}")
    if review.get("calculations"):
        with st.expander(f"calculator ({len(review['calculations'])} expression(s))"):
            st.dataframe(pd.DataFrame(review["calculations"]), hide_index=True, width="stretch")


def _answer_json(text: str | None) -> None:
    if not text:
        st.caption("—")
        return
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        st.json(json.loads(text[start:end]), expanded=False)
    except ValueError:
        st.code(text)


def tasks_tab(runs: pd.DataFrame, conditions: list[str], results_dir: Path) -> None:
    matrix = data.task_matrix(runs, conditions)
    only_changed = st.toggle("Only tasks where a condition differs from none", value=True)
    if only_changed:
        matrix = matrix[(matrix["best_gain"] != 0) | (matrix["worst_loss"] != 0) | (matrix["none_right"] < 3)]
    st.dataframe(
        matrix.drop(columns=["none_right"]),
        hide_index=True,
        width="stretch",
        column_config={
            "best_gain": st.column_config.NumberColumn("best gain vs none", help="most extra right finals any condition has on this task"),
            "worst_loss": st.column_config.NumberColumn("worst loss vs none"),
        },
    )
    st.caption("Cells are right final answers out of saved (non-failed) runs.")

    task_ids = list(matrix["task_id"]) or sorted(runs["task_id"].unique())
    task_id = st.selectbox("Look at one task", task_ids)
    if not task_id:
        return
    task_runs = data.runs_for_task(runs, task_id)
    task_runs = task_runs[task_runs["condition"].isin(conditions)]
    st.dataframe(task_runs, hide_index=True, width="stretch")

    labels = [f"{r.condition} · run {r.run_number}" for r in task_runs.itertuples()]
    if not labels:
        return
    choice = st.selectbox("Run", labels)
    run_id = task_runs.iloc[labels.index(choice)]["run_id"]
    record = data.answer_record(run_id, results_dir)
    if record is None:
        st.warning(f"No saved answer file for {run_id} (failed runs have none).")
        return
    first, final = record.get("draft_check") or {}, record.get("final_check") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("First draft", "right" if first.get("correct") else "wrong")
    c2.metric("Final answer", "right" if final.get("correct") else "wrong")
    c3.metric("Status", record.get("status", "?"))
    for review in record.get("reviews") or []:
        _verdict_block(review)
    left, right = st.columns(2)
    with left:
        st.markdown("**First draft**")
        _answer_json(record.get("draft"))
        bad = data.failed_checks(first)
        if bad:
            st.markdown("Failed checks (draft)")
            st.json(bad, expanded=False)
    with right:
        st.markdown("**Final answer**" + (" (revision)" if record.get("revision") else ""))
        _answer_json(record.get("final"))
        bad = data.failed_checks(final)
        if bad:
            st.markdown("Failed checks (final)")
            st.json(bad, expanded=False)


def verifiers_tab(outcomes: pd.DataFrame, summary: pd.DataFrame, seeded_metrics: pd.DataFrame, seeded_audit: pd.DataFrame) -> None:
    verified = [c for c in summary["condition"] if c != "none"]
    if not verified:
        st.info("No verified condition in the selection.")
        return
    st.subheader("On the advisor's real errors")
    rates = summary[summary["condition"].isin(verified)][["condition", "catch_rate", "false_alarm_rate"]]
    long = rates.melt("condition", var_name="rate", value_name="value")
    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X("condition:N", sort=verified, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset="rate:N",
        y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title="Rate"),
        color=alt.Color("rate:N", scale=alt.Scale(domain=["catch_rate", "false_alarm_rate"], range=["#4C78A8", "#E45756"])),
        tooltip=["condition", "rate", alt.Tooltip("value:Q", format=".1%")],
    )
    st.altair_chart(chart, width="stretch")

    fates = outcomes[outcomes["condition"].isin(verified)][
        ["condition", "missed", "caught_fixed", "caught_not_fixed", "caught_escalated"]
    ].melt("condition", var_name="what happened", value_name="drafts")
    fates["drafts"] = pd.to_numeric(fates["drafts"], errors="coerce").fillna(0)
    stack = alt.Chart(fates).mark_bar().encode(
        x=alt.X("condition:N", sort=verified, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("drafts:Q", title="Initially wrong drafts"),
        color=alt.Color(
            "what happened:N",
            scale=alt.Scale(
                domain=["missed", "caught_fixed", "caught_not_fixed", "caught_escalated"],
                range=["#BBBBBB", "#2CA02C", "#F2B701", "#9467BD"],
            ),
        ),
        tooltip=["condition", "what happened", "drafts"],
    )
    st.altair_chart(stack, width="stretch")
    st.caption("caught_escalated counts abstains; a reject whose revision is not accepted is caught_not_fixed.")

    st.subheader("On the seeded answers (known errors and correct controls)")
    if seeded_metrics.empty:
        st.info("No seeded reviews saved yet.")
        return
    st.dataframe(
        seeded_metrics[[
            "condition", "verifier_model", "n_errors", "n_controls", "n_failed", "seeded_catch_rate",
            "seeded_false_alarm_rate", "right_reason", "consequence_only", "wrong_reason", "abstained", "right_reason_rate",
        ]],
        hide_index=True,
        width="stretch",
        column_config={
            "seeded_catch_rate": st.column_config.NumberColumn("catch rate", format="percent"),
            "seeded_false_alarm_rate": st.column_config.NumberColumn("false alarms", format="percent"),
            "right_reason_rate": st.column_config.NumberColumn("right reason", format="percent"),
        },
    )
    with st.expander("Every seeded review (what the verifier said, and whether it named the error)"):
        pick = st.multiselect("Conditions", sorted(seeded_audit["condition"].unique()), default=None)
        audit = seeded_audit if not pick else seeded_audit[seeded_audit["condition"].isin(pick)]
        st.dataframe(audit.astype(str), hide_index=True, width="stretch")


def progress_tab(progress: pd.DataFrame, spend: dict, rejected: list[dict]) -> None:
    st.subheader("Runs saved")
    shown = progress.copy()
    shown["runs"] = shown["runs_done"].astype(str) + " / " + shown["runs_expected"].astype(str)
    if "seeded_done" in shown:
        shown["seeded"] = [
            "" if pd.isna(d) else f"{int(d)} / {int(e)}" for d, e in zip(shown["seeded_done"], shown["seeded_expected"])
        ]
    st.dataframe(
        shown[[c for c in ("condition", "runs", "runs_failed", "seeded", "seeded_failed", "complete") if c in shown]],
        hide_index=True,
        width="stretch",
    )
    for row in progress.itertuples():
        st.progress(min(row.runs_done / row.runs_expected, 1.0), text=f"{row.condition}: {row.runs_done}/{row.runs_expected}")
    if rejected:
        st.warning(f"{len(rejected)} row(s) in results.jsonl were not used (see scripts/analyze.py's rejected.md).")

    st.subheader("Spend")
    budget = budget_usd()
    c1, c2, c3 = st.columns(3)
    c1.metric("Spent", f"${spend['cost_usd']:.2f}")
    c2.metric("Cap", f"${budget:.2f}", f"${budget - spend['cost_usd']:.2f} left")
    c3.metric("Model calls", f"{spend['calls']:,}", f"{spend['failed']} failed", delta_color="off")
    st.progress(min(spend["cost_usd"] / budget, 1.0))
    left, right = st.columns(2)
    with left:
        by_condition = pd.DataFrame(spend["by_condition"])
        if not by_condition.empty:
            st.altair_chart(
                alt.Chart(by_condition).mark_bar().encode(
                    y=alt.Y("name:N", sort="-x", title=None), x=alt.X("cost_usd:Q", title="USD"),
                    tooltip=["name", alt.Tooltip("cost_usd:Q", format="$.2f"), "calls"],
                ).properties(title="By condition (draft = shared first drafts)"),
                width="stretch",
            )
    with right:
        by_day = pd.DataFrame(spend["by_day"])
        if not by_day.empty:
            st.altair_chart(
                alt.Chart(by_day).mark_bar().encode(
                    x=alt.X("name:N", title=None), y=alt.Y("cost_usd:Q", title="USD"),
                    tooltip=["name", alt.Tooltip("cost_usd:Q", format="$.2f"), "calls"],
                ).properties(title="By day (UTC)"),
                width="stretch",
            )


def main() -> None:
    st.set_page_config(page_title="Verification study", layout="wide")
    args = _args()
    results_dir = Path(args.results_dir)
    st.title("Verification strategies for an AI financial advisor")
    if not (results_dir / "results.jsonl").exists():
        st.error(f"No results.jsonl in {results_dir}. Run a condition first, or pass --results-dir.")
        return

    stamp = tuple(_mtime(results_dir / name) for name in ("results.jsonl", "runs.jsonl", "seeded"))
    loaded = load(str(results_dir), stamp)
    runs, progress = loaded["runs"], loaded["progress"]
    complete = data.complete_conditions(progress)
    present = data.ordered(runs["condition"].unique())

    with st.sidebar:
        try:
            shown_dir = results_dir.resolve().relative_to(data.ROOT)
        except ValueError:
            shown_dir = results_dir.name
        st.caption(f"Results: `{shown_dir}`")
        only_complete = st.checkbox("Only finished conditions", value=True, help="Leave out conditions still running, so partial results don't mix in")
        default = [c for c in present if c in complete] if only_complete else present
        chosen = st.multiselect("Conditions", present, default=default)
        if not only_complete:
            unfinished = [c for c in chosen if c not in complete]
            if unfinished:
                st.warning(f"Not finished: {', '.join(unfinished)}. Their numbers will change.")
        if st.button("Reload results"):
            load.clear()
            headline.clear()
            st.rerun()

    if not chosen:
        st.info("Choose at least one condition in the sidebar.")
        return
    report = headline(runs, tuple(chosen))

    tab_overview, tab_tasks, tab_verifiers, tab_progress = st.tabs(["Overview", "Tasks", "Verifiers", "Progress & spend"])
    with tab_overview:
        overview(report["summary"], report["comparisons"])
    with tab_tasks:
        tasks_tab(runs, chosen, results_dir)
    with tab_verifiers:
        seeded_metrics = loaded["seeded_metrics"]
        seeded_audit = loaded["seeded_audit"]
        if not seeded_metrics.empty:
            seeded_metrics = seeded_metrics[seeded_metrics["condition"].isin(chosen)]
            seeded_audit = seeded_audit[seeded_audit["condition"].isin(chosen)]
        verifiers_tab(report["outcomes"], report["summary"], seeded_metrics, seeded_audit)
    with tab_progress:
        progress_tab(progress, loaded["spend"], loaded["rejected"])


main()
