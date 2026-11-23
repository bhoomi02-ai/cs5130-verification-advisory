"""Read-only checks that main-study results and seeded reviews are complete.

Never writes under ``results/`` and never calls a model. Used by
``scripts/check_study_data.py``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from advisory.conditions import CONDITIONS, VERIFIED, VERIFIER_PROMPTS, verifier_settings
from advisory.drafts import Settings
from advisory.pipeline import RESULTS_PATH, STUDY_TASKS_PATH, read_task_file
from advisory.prompts import PROMPT_VERSION, REVISION_PROMPT_VERSION
from advisory.seeded import SEEDED_DIR, SEEDED_RESULTS_DIR, all_answer_ids

# Advisor model chosen by the pilot for the main study (see docs/pilot.md).
# Temperature, top_p and thinking_level come from Settings defaults; verifier
# model and prompt versions come from conditions.verifier_settings / VERIFIER_PROMPTS.
STUDY_ADVISOR_MODEL = "gemini-3.6-flash"
STUDY_REPEATS = 3

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SectionSummary:
    """Counts for one check section."""

    name: str
    expected: int | None = None
    found: int = 0
    missing: int = 0
    duplicates: int = 0
    failed: int = 0
    mismatches: int = 0
    warnings: int = 0
    skipped: bool = False
    note: str = ""


@dataclass
class StudyCheckReport:
    """Full findings from a study data check."""

    ok: bool
    summaries: list[SectionSummary] = field(default_factory=list)
    missing_runs: list[dict] = field(default_factory=list)
    duplicate_runs: list[dict] = field(default_factory=list)
    extra_rows: list[dict] = field(default_factory=list)
    malformed_study_rows: list[dict] = field(default_factory=list)
    failed_runs: list[dict] = field(default_factory=list)
    failed_by_condition: dict[str, int] = field(default_factory=dict)
    draft_mismatches: list[dict] = field(default_factory=list)
    settings_mismatches: list[dict] = field(default_factory=list)
    missing_seeded: list[dict] = field(default_factory=list)
    duplicate_seeded: list[dict] = field(default_factory=list)
    failed_seeded: list[dict] = field(default_factory=list)
    failed_seeded_by_condition: dict[str, int] = field(default_factory=dict)
    malformed_seeded: list[dict] = field(default_factory=list)
    seeded_skipped: bool = False
    seeded_skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summaries": [asdict(item) for item in self.summaries],
            "missing_runs": self.missing_runs,
            "duplicate_runs": self.duplicate_runs,
            "extra_rows": self.extra_rows,
            "malformed_study_rows": self.malformed_study_rows,
            "failed_runs": self.failed_runs,
            "failed_by_condition": self.failed_by_condition,
            "draft_mismatches": self.draft_mismatches,
            "settings_mismatches": self.settings_mismatches,
            "missing_seeded": self.missing_seeded,
            "duplicate_seeded": self.duplicate_seeded,
            "failed_seeded": self.failed_seeded,
            "failed_seeded_by_condition": self.failed_seeded_by_condition,
            "malformed_seeded": self.malformed_seeded,
            "seeded_skipped": self.seeded_skipped,
            "seeded_skip_reason": self.seeded_skip_reason,
        }


def study_settings(model: str = STUDY_ADVISOR_MODEL) -> Settings:
    """Advisor settings for the main study (Settings defaults for temp / top_p / thinking)."""
    return Settings(model)


def expected_row_settings(condition: str, advisor: Settings | None = None) -> dict[str, Any]:
    """Expected settings fields for a study row under ``condition``."""
    advisor = advisor or study_settings()
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    verifier = verifier_settings(condition, advisor)
    return {
        "model": advisor.model,
        "temperature": advisor.temperature,
        "top_p": advisor.top_p,
        "thinking_level": advisor.thinking_level,
        "prompt_version": PROMPT_VERSION,
        "revision_prompt_version": (
            REVISION_PROMPT_VERSION if condition in VERIFIED else None
        ),
        "verifier_model": verifier.model if verifier else None,
        "verifier_thinking_level": (
            verifier.thinking_level if verifier else None
        ),
        "verifier_prompt_version": VERIFIER_PROMPTS.get(condition),
    }


def load_jsonl(path: Path) -> list[dict]:
    """Load every non-empty JSONL line; skip blank lines."""
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def load_seeded_json(seeded_dir: Path) -> list[dict]:
    """Load every ``*.json`` record from a seeded results directory."""
    if not seeded_dir.exists():
        return []
    records: list[dict] = []
    for path in sorted(seeded_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"{path}: expected a JSON object")
        record = dict(record)
        record["_path"] = str(path)
        records.append(record)
    return records


def check_study(
    *,
    results_path: Path = RESULTS_PATH,
    seeded_dir: Path = SEEDED_RESULTS_DIR,
    tasks_file: Path = STUDY_TASKS_PATH,
    repeats: int = STUDY_REPEATS,
    injected_dir: Path = SEEDED_DIR,
    advisor: Settings | None = None,
) -> StudyCheckReport:
    """Run all main-study and seeded checks. Read-only."""
    advisor = advisor or study_settings()
    tasks = read_task_file(tasks_file)
    conditions = tuple(CONDITIONS)
    report = StudyCheckReport(ok=True)

    rows = load_jsonl(Path(results_path))
    study_rows, extra_rows, malformed_rows = _split_study_rows(rows, tasks, repeats)
    report.extra_rows = extra_rows
    report.malformed_study_rows = malformed_rows
    report.summaries.append(
        SectionSummary(
            name="malformed_study_rows",
            found=len(malformed_rows),
            mismatches=len(malformed_rows),
            note="non-integer run_number (error)",
        )
    )
    if malformed_rows:
        report.ok = False

    missing, duplicates, found = _coverage_runs(study_rows, tasks, conditions, repeats)
    report.missing_runs = missing
    report.duplicate_runs = duplicates
    report.summaries.append(
        SectionSummary(
            name="coverage",
            expected=len(tasks) * len(conditions) * repeats,
            found=found,
            missing=len(missing),
            duplicates=len(duplicates),
        )
    )
    if missing or duplicates:
        report.ok = False

    extra_summary = SectionSummary(
        name="extra_rows",
        expected=0,
        found=len(extra_rows),
        warnings=len(extra_rows),
        note="not in study task list or run_number > repeats (warning only)",
    )
    report.summaries.append(extra_summary)

    failed_runs, failed_by = _failed_runs(study_rows)
    report.failed_runs = failed_runs
    report.failed_by_condition = failed_by
    report.summaries.append(
        SectionSummary(
            name="failed_runs",
            found=len(study_rows),
            failed=len(failed_runs),
            warnings=len(failed_runs),
            note="status=failed among study rows (warning only)",
        )
    )

    draft_mismatches = _draft_mismatches(study_rows)
    report.draft_mismatches = draft_mismatches
    report.summaries.append(
        SectionSummary(
            name="shared_drafts",
            found=len({(r.get("task_id"), r.get("run_number"), r.get("model")) for r in study_rows}),
            mismatches=len(draft_mismatches),
        )
    )
    if draft_mismatches:
        report.ok = False

    settings_mismatches = _settings_mismatches(study_rows, advisor)
    report.settings_mismatches = settings_mismatches
    report.summaries.append(
        SectionSummary(
            name="settings",
            found=len(study_rows),
            mismatches=len(settings_mismatches),
        )
    )
    if settings_mismatches:
        report.ok = False

    seeded_path = Path(seeded_dir)
    if not seeded_path.exists() or not any(seeded_path.glob("*.json")):
        reason = (
            f"{seeded_path} is absent"
            if not seeded_path.exists()
            else f"{seeded_path} has no .json files"
        )
        report.seeded_skipped = True
        report.seeded_skip_reason = reason
        report.summaries.append(
            SectionSummary(
                name="seeded",
                skipped=True,
                note=reason,
            )
        )
        return report

    records = load_seeded_json(seeded_path)
    answers = all_answer_ids(Path(injected_dir))
    verified = tuple(VERIFIED)
    seeded_missing, seeded_dupes, seeded_found = _coverage_seeded(
        records, answers, verified, run_number=1
    )
    report.missing_seeded = seeded_missing
    report.duplicate_seeded = seeded_dupes
    report.summaries.append(
        SectionSummary(
            name="seeded_coverage",
            expected=len(answers) * len(verified) * 1,
            found=seeded_found,
            missing=len(seeded_missing),
            duplicates=len(seeded_dupes),
        )
    )
    if seeded_missing or seeded_dupes:
        report.ok = False

    failed_seeded, failed_seeded_by = _failed_seeded(records)
    report.failed_seeded = failed_seeded
    report.failed_seeded_by_condition = failed_seeded_by
    report.summaries.append(
        SectionSummary(
            name="seeded_failed",
            found=len(records),
            failed=len(failed_seeded),
            warnings=len(failed_seeded),
            note="status=failed among seeded records (warning only)",
        )
    )

    malformed = _malformed_seeded(records)
    report.malformed_seeded = malformed
    report.summaries.append(
        SectionSummary(
            name="seeded_right_reason_fields",
            found=len(records),
            mismatches=len(malformed),
        )
    )
    if malformed:
        report.ok = False

    return report


def format_report(report: StudyCheckReport) -> str:
    """Short readable report: summary tables, then problem details."""
    lines: list[str] = ["Study data check", "================"]
    lines.append("")
    lines.append(
        f"{'section':<28} {'expected':>8} {'found':>8} {'missing':>8} "
        f"{'dupes':>8} {'failed':>8} {'mismatch':>8} {'warn':>6}"
    )
    lines.append("-" * 100)
    for item in report.summaries:
        if item.skipped:
            lines.append(f"{item.name:<28} skipped: {item.note}")
            continue
        expected = "—" if item.expected is None else str(item.expected)
        lines.append(
            f"{item.name:<28} {expected:>8} {item.found:>8} {item.missing:>8} "
            f"{item.duplicates:>8} {item.failed:>8} {item.mismatches:>8} {item.warnings:>6}"
        )
        if item.note:
            lines.append(f"  note: {item.note}")
    lines.append("")
    lines.append(f"Overall: {'OK' if report.ok else 'FAILED'}")

    def _block(title: str, rows: list[dict], limit: int = 50) -> None:
        if not rows:
            return
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))
        for row in rows[:limit]:
            lines.append(f"  {_fmt(row)}")
        if len(rows) > limit:
            lines.append(f"  … {len(rows) - limit} more")

    _block("Malformed study rows", report.malformed_study_rows)
    _block("Missing study runs", report.missing_runs)
    _block("Duplicate study runs", report.duplicate_runs)
    _block("Extra rows (warning)", report.extra_rows)
    if report.failed_runs:
        lines.append("")
        lines.append("Failed study runs (warning)")
        lines.append("---------------------------")
        lines.append(f"  by condition: {dict(sorted(report.failed_by_condition.items()))}")
        for row in report.failed_runs[:50]:
            lines.append(f"  {_fmt(row)}")
        if len(report.failed_runs) > 50:
            lines.append(f"  … {len(report.failed_runs) - 50} more")
    _block("Shared-draft mismatches", report.draft_mismatches)
    _block("Settings mismatches", report.settings_mismatches)

    if report.seeded_skipped:
        lines.append("")
        lines.append(f"Seeded checks skipped: {report.seeded_skip_reason}")
    else:
        _block("Missing seeded records", report.missing_seeded)
        _block("Duplicate seeded records", report.duplicate_seeded)
        if report.failed_seeded:
            lines.append("")
            lines.append("Failed seeded records (warning)")
            lines.append("-------------------------------")
            lines.append(
                f"  by condition: {dict(sorted(report.failed_seeded_by_condition.items()))}"
            )
            for row in report.failed_seeded[:50]:
                lines.append(f"  {_fmt(row)}")
        _block("Malformed seeded records", report.malformed_seeded)

    lines.append("")
    return "\n".join(lines)


def assert_json_path_outside_results(json_path: Path, results_path: Path) -> None:
    """Refuse to write the JSON report under the results directory."""
    results_root = Path(results_path).resolve().parent
    target = Path(json_path).resolve()
    if target == results_root or results_root in target.parents:
        raise ValueError(
            f"--json path {json_path} resolves under results/ ({results_root}); "
            "write it somewhere else"
        )


def _fmt(row: dict) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in row.items() if key != "_path")


def _split_study_rows(
    rows: list[dict], tasks: list[str], repeats: int
) -> tuple[list[dict], list[dict], list[dict]]:
    task_set = set(tasks)
    study: list[dict] = []
    extra: list[dict] = []
    malformed: list[dict] = []
    for row in rows:
        task_id = row.get("task_id")
        run_number = row.get("run_number")
        # Strict int only: a string like "2" must not enter coverage/sort keys.
        if type(run_number) is not int:
            malformed.append(
                {
                    "run_id": row.get("run_id"),
                    "task_id": task_id,
                    "condition": row.get("condition"),
                    "run_number": run_number,
                    "problem": (
                        f"run_number is {type(run_number).__name__} "
                        f"{run_number!r}, not int"
                    ),
                }
            )
            continue
        reasons = []
        if task_id not in task_set:
            reasons.append("task_not_in_study")
        if run_number > repeats:
            reasons.append("run_number_above_repeats")
        elif run_number < 1:
            reasons.append("run_number_below_one")
        if reasons:
            extra.append(
                {
                    "task_id": task_id,
                    "condition": row.get("condition"),
                    "run_number": run_number,
                    "model": row.get("model"),
                    "run_id": row.get("run_id"),
                    "reasons": reasons,
                }
            )
        else:
            study.append(row)
    return study, extra, malformed


def _coverage_runs(
    rows: list[dict],
    tasks: list[str],
    conditions: tuple[str, ...],
    repeats: int,
) -> tuple[list[dict], list[dict], int]:
    counts: Counter[tuple] = Counter()
    for row in rows:
        key = (row.get("task_id"), row.get("condition"), row.get("run_number"), row.get("model"))
        counts[key] += 1

    expected_keys = [
        (task_id, condition, run_number)
        for task_id in tasks
        for condition in conditions
        for run_number in range(1, repeats + 1)
    ]
    # Found = study rows that match an expected (task, condition, run) slot.
    found = 0
    missing: list[dict] = []
    for task_id, condition, run_number in expected_keys:
        matching = [
            (key, n)
            for key, n in counts.items()
            if key[0] == task_id and key[1] == condition and key[2] == run_number
        ]
        total = sum(n for _, n in matching)
        if total == 0:
            missing.append(
                {
                    "task_id": task_id,
                    "condition": condition,
                    "run_number": run_number,
                }
            )
        else:
            found += total

    duplicates: list[dict] = []
    for key, n in sorted(counts.items(), key=lambda item: (str(item[0][0]), str(item[0][1]), item[0][2] or 0)):
        if n > 1:
            duplicates.append(
                {
                    "task_id": key[0],
                    "condition": key[1],
                    "run_number": key[2],
                    "model": key[3],
                    "count": n,
                }
            )
    return missing, duplicates, found


def _failed_runs(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    failed = [
        {
            "task_id": row.get("task_id"),
            "condition": row.get("condition"),
            "run_number": row.get("run_number"),
            "model": row.get("model"),
            "run_id": row.get("run_id"),
            "error": row.get("error"),
        }
        for row in rows
        if row.get("status") == "failed"
    ]
    by_condition: dict[str, int] = defaultdict(int)
    for row in failed:
        by_condition[str(row["condition"])] += 1
    return failed, dict(by_condition)


def _draft_mismatches(rows: list[dict]) -> list[dict]:
    # Failed rows store initial_correct=False even when the draft was correct;
    # they are already listed under failed_runs, so leave them out here.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "failed":
            continue
        key = (row.get("task_id"), row.get("run_number"), row.get("model"))
        groups[key].append(row)

    mismatches: list[dict] = []
    for key, group in sorted(groups.items(), key=lambda item: (str(item[0][0]), item[0][1] or 0)):
        if len(group) < 2:
            continue
        has_draft_id = any("draft_run_id" in row for row in group)
        # One signature per condition (first row if a condition appears more than once).
        by_condition: dict[str, tuple] = {}
        for row in group:
            condition = str(row.get("condition"))
            if condition in by_condition:
                continue
            draft_id = row.get("draft_run_id") if has_draft_id else None
            by_condition[condition] = (draft_id, row.get("initial_correct"))

        signatures = list(by_condition.values())
        draft_ids = {sig[0] for sig in signatures}
        initials = {sig[1] for sig in signatures}
        problems = []
        if has_draft_id and len(draft_ids) > 1:
            problems.append("draft_run_id")
        if len(initials) > 1:
            problems.append("initial_correct")
        if not problems:
            continue

        majority = Counter(signatures).most_common(1)[0][0]
        parts = []
        disagreeing = []
        for condition in sorted(by_condition):
            draft_id, initial = by_condition[condition]
            if has_draft_id:
                piece = f"{condition}: {draft_id} (initial_correct={initial})"
            else:
                piece = f"{condition}: (initial_correct={initial})"
            if (draft_id, initial) != majority:
                piece += " [differs]"
                disagreeing.append(condition)
            parts.append(piece)
        mismatches.append(
            {
                "task_id": key[0],
                "run_number": key[1],
                "model": key[2],
                "fields": problems,
                "by_condition": ", ".join(parts),
                "disagreeing": disagreeing,
            }
        )
    return mismatches


def _settings_mismatches(rows: list[dict], advisor: Settings) -> list[dict]:
    mismatches: list[dict] = []
    fields = (
        "model",
        "temperature",
        "top_p",
        "thinking_level",
        "prompt_version",
        "revision_prompt_version",
        "verifier_model",
        "verifier_thinking_level",
        "verifier_prompt_version",
    )
    for row in rows:
        condition = row.get("condition")
        if condition not in CONDITIONS:
            mismatches.append(
                {
                    "run_id": row.get("run_id"),
                    "task_id": row.get("task_id"),
                    "condition": condition,
                    "field": "condition",
                    "expected": sorted(CONDITIONS),
                    "actual": condition,
                }
            )
            continue
        expected = expected_row_settings(condition, advisor)
        for name in fields:
            actual = row.get(name)
            wanted = expected[name]
            if actual != wanted:
                mismatches.append(
                    {
                        "run_id": row.get("run_id"),
                        "task_id": row.get("task_id"),
                        "condition": condition,
                        "run_number": row.get("run_number"),
                        "field": name,
                        "expected": wanted,
                        "actual": actual,
                    }
                )
    return mismatches


def _coverage_seeded(
    records: list[dict],
    answers: list[str],
    conditions: tuple[str, ...],
    *,
    run_number: int = 1,
) -> tuple[list[dict], list[dict], int]:
    counts: Counter[tuple] = Counter()
    for record in records:
        key = (
            record.get("answer_id"),
            record.get("condition"),
            record.get("run_number"),
        )
        counts[key] += 1

    missing: list[dict] = []
    found = 0
    for answer_id in answers:
        for condition in conditions:
            key = (answer_id, condition, run_number)
            n = counts.get(key, 0)
            if n == 0:
                missing.append(
                    {
                        "answer_id": answer_id,
                        "condition": condition,
                        "run_number": run_number,
                    }
                )
            else:
                found += n

    duplicates: list[dict] = []
    for key, n in sorted(counts.items(), key=lambda item: (str(item[0][0]), str(item[0][1]), item[0][2] or 0)):
        if n > 1:
            duplicates.append(
                {
                    "answer_id": key[0],
                    "condition": key[1],
                    "run_number": key[2],
                    "count": n,
                }
            )
    return missing, duplicates, found


def _failed_seeded(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    failed = [
        {
            "answer_id": record.get("answer_id"),
            "condition": record.get("condition"),
            "run_number": record.get("run_number"),
            "run_id": record.get("run_id"),
            "error": record.get("error"),
        }
        for record in records
        if record.get("status") == "failed"
    ]
    by_condition: dict[str, int] = defaultdict(int)
    for row in failed:
        by_condition[str(row["condition"])] += 1
    return failed, dict(by_condition)


def _malformed_seeded(records: list[dict]) -> list[dict]:
    """Records that lack fields right_reason metrics need (skip status=failed)."""
    malformed: list[dict] = []
    for record in records:
        if record.get("status") == "failed":
            continue
        problems: list[str] = []
        for name in ("answer_id", "error_type", "condition", "verifier_model"):
            if record.get(name) in (None, ""):
                problems.append(f"missing_{name}")
        reviews = record.get("reviews")
        if not isinstance(reviews, list):
            problems.append("reviews_not_list")
        elif len(reviews) != 1:
            problems.append(f"reviews_count_{len(reviews)}")
        elif not isinstance(reviews[0], dict) or reviews[0].get("stage") != "first":
            problems.append("review_stage_not_first")
        if problems:
            malformed.append(
                {
                    "run_id": record.get("run_id"),
                    "answer_id": record.get("answer_id"),
                    "condition": record.get("condition"),
                    "path": record.get("_path"),
                    "problems": problems,
                }
            )
    return malformed
