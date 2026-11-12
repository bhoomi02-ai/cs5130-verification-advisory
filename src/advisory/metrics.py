"""Evaluation metrics computed from per-run scorer results.

The scorer decides whether an answer is correct. This module only aggregates
those decisions. Every distinct ``condition`` value gets a row; the condition
list is not hard-coded. When an optional ``model`` column is present, each
``(condition, model)`` pair gets its own row and extra cost compares with
``none`` for the same model. An optional ``status`` column may mark runs as
``failed``; those runs are left out of the rates and counted in ``n_failed``.
Condition ``none`` has no verifier, so its verifier metrics, including
``n_missing_verdict``, are ``None``. Every other condition is scored on the
runs that have a verdict, and ``n_missing_verdict`` counts the rest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef

REQUIRED_COLUMNS = (
    "task_id",
    "condition",
    "run_number",
    "initial_correct",
    "final_correct",
    "verdict",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_s",
)

# Abstain counts as a flag: the verifier did not accept the draft.
_FLAGGED_VERDICTS = ("reject", "abstain")
_KNOWN_VERDICTS = ("accept", "reject", "abstain")
_N_RESAMPLES = 1000

METRIC_COLUMNS = (
    "task_success",
    "task_success_ci_low",
    "task_success_ci_high",
    "tp",
    "fp",
    "fn",
    "tn",
    "catch_rate",
    "precision",
    "false_alarm_rate",
    "mcc",
    "abstention_rate",
    "n_missing_verdict",
    "fixed",
    "broken",
    "fix_rate",
    "break_rate",
    "net_gain",
    "success_delta",
    "total_cost_usd",
    "mean_cost_usd",
    "mean_tokens_in",
    "mean_tokens_out",
    "median_latency_s",
    "p95_latency_s",
    "extra_cost_per_extra_correct",
    "n_failed",
)

_NULLABLE_COLUMNS = (
    "tp",
    "fp",
    "fn",
    "tn",
    "catch_rate",
    "precision",
    "false_alarm_rate",
    "mcc",
    "abstention_rate",
    "n_missing_verdict",
    "fix_rate",
    "break_rate",
    "net_gain",
    "extra_cost_per_extra_correct",
)

__all__ = [
    "REQUIRED_COLUMNS",
    "METRIC_COLUMNS",
    "validate_runs",
    "task_success",
    "verifier_metrics",
    "fixed_broken",
    "cost_metrics",
    "extra_cost_per_extra_correct",
    "bootstrap_task_success_ci",
    "summarize",
]


def validate_runs(runs: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of ``runs``, or raise if the input is unusable.

    Missing required columns, unknown verdicts, and duplicate identity rows
    are errors. With a ``model`` column the identity is
    ``(task_id, condition, run_number, model)``; without it the identity is
    ``(task_id, condition, run_number)``. Missing verdicts are kept as
    ``None``.
    """
    if isinstance(runs, pd.DataFrame):
        frame = runs.copy()
    elif isinstance(runs, list):
        frame = pd.DataFrame(runs)
    else:
        raise TypeError("runs must be a list of dicts or a pandas DataFrame")
    if frame.empty:
        raise ValueError("runs must contain at least one row")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    frame = frame.reset_index(drop=True)
    if frame["task_id"].isna().any():
        raise ValueError("task_id must be present on every run")
    if frame["condition"].isna().any():
        raise ValueError("condition must be present on every run")
    if "model" in frame.columns:
        if frame["model"].isna().any():
            raise ValueError("model must be present on every run")
        identity = ["task_id", "condition", "run_number", "model"]
        duplicate_message = "duplicate rows for the same task_id, condition, run_number, and model"
    else:
        identity = ["task_id", "condition", "run_number"]
        duplicate_message = "duplicate rows for the same task_id, condition, and run_number"
    if frame.duplicated(subset=identity).any():
        raise ValueError(duplicate_message)

    frame["initial_correct"] = _as_bool(frame["initial_correct"], "initial_correct")
    frame["final_correct"] = _as_bool(frame["final_correct"], "final_correct")
    frame["verdict"] = _checked_verdicts(frame["verdict"])
    for column in ("tokens_in", "tokens_out", "cost_usd", "latency_s"):
        frame[column] = _as_float(frame[column], column)
    return frame


def task_success(final_correct) -> float:
    """Share of runs whose final answer is correct."""
    finals = _as_bool(_as_series(final_correct), "final_correct")
    if finals.empty:
        raise ValueError("final_correct is empty")
    return float(finals.mean())


def verifier_metrics(verdict, initial_correct, condition=None) -> dict[str, float | int | None]:
    """Confusion-matrix metrics for the runs of one condition.

    The positive class is an initial answer that is wrong. A run is flagged
    when its verdict is ``reject`` or ``abstain``. Catch rate is recall on
    that class. The false-alarm rate is the share of initially correct
    answers that were flagged. A rate whose denominator is zero is ``None``.
    Condition ``none`` is not a verifier, so its rates and
    ``n_missing_verdict`` stay ``None``. Any other condition is scored only
    on runs that have a verdict, and ``n_missing_verdict`` counts the runs
    left out. MCC is scikit-learn's coefficient when the denominator is
    positive, and ``None`` when that denominator is zero.
    """
    labels = _checked_verdicts(_as_series(verdict))
    initial = _as_bool(_as_series(initial_correct), "initial_correct")
    if len(labels) != len(initial):
        raise ValueError("verdict and initial_correct must have the same length")
    if labels.empty:
        raise ValueError("verifier metrics need at least one run")

    missing = np.array([value is None for value in labels.tolist()], dtype=bool)
    n_missing = int(missing.sum())
    scored = ~missing
    if _python_scalar(condition) == "none":
        return _blank_verifier(None)
    if not scored.any():
        return _blank_verifier(n_missing)

    label_np = labels.to_numpy()[scored]
    actually_wrong = ~initial.to_numpy(dtype=bool)[scored]
    flagged = np.isin(label_np, _FLAGGED_VERDICTS)
    # labels=[False, True] forces [[TN, FP], [FN, TP]] even if a cell is zero.
    matrix = confusion_matrix(actually_wrong, flagged, labels=[False, True])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    mcc = _mcc(tp, fp, fn, tn, actually_wrong, flagged)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "catch_rate": _rate(tp, tp + fn),
        "precision": _rate(tp, tp + fp),
        "false_alarm_rate": _rate(fp, fp + tn),
        "mcc": mcc,
        "abstention_rate": float(np.mean(label_np == "abstain")),
        "n_missing_verdict": n_missing,
    }


def fixed_broken(initial_correct, final_correct) -> dict[str, float | int | None]:
    """Counts and rates for drafts that verification fixes or breaks.

    Fixed means initially wrong and finally correct. Broken means initially
    correct and finally wrong. Fix rate divides by the number of initially
    wrong runs, and break rate by the number of initially correct runs.
    Either rate is ``None`` when its denominator is zero, and then net gain
    is ``None`` too. ``success_delta`` is ``(fixed - broken) / n_runs``,
    which is final success minus initial success.
    """
    initial = _as_bool(_as_series(initial_correct), "initial_correct")
    final = _as_bool(_as_series(final_correct), "final_correct")
    if len(initial) != len(final):
        raise ValueError("initial_correct and final_correct must have the same length")
    if initial.empty:
        raise ValueError("fixed vs broken needs at least one run")

    initial_np = initial.to_numpy(dtype=bool)
    final_np = final.to_numpy(dtype=bool)
    fixed = int((~initial_np & final_np).sum())
    broken = int((initial_np & ~final_np).sum())
    initially_wrong = int((~initial_np).sum())
    initially_correct = int(initial_np.sum())
    fix_rate = _rate(fixed, initially_wrong)
    break_rate = _rate(broken, initially_correct)
    net_gain = None if fix_rate is None or break_rate is None else fix_rate - break_rate
    n_runs = len(initial_np)
    return {
        "fixed": fixed,
        "broken": broken,
        "fix_rate": fix_rate,
        "break_rate": break_rate,
        "net_gain": net_gain,
        "success_delta": (fixed - broken) / n_runs,
    }


def cost_metrics(cost_usd, tokens_in, tokens_out, latency_s) -> dict[str, float]:
    """Total and mean cost, mean tokens, and median and 95th-percentile latency.

    The latency percentile uses NumPy's default linear interpolation.
    """
    cost, incoming, outgoing, latency = _aligned_floats(
        ("cost_usd", cost_usd),
        ("tokens_in", tokens_in),
        ("tokens_out", tokens_out),
        ("latency_s", latency_s),
    )
    return {
        "total_cost_usd": float(cost.sum()),
        "mean_cost_usd": float(cost.mean()),
        "mean_tokens_in": float(incoming.mean()),
        "mean_tokens_out": float(outgoing.mean()),
        "median_latency_s": float(np.median(latency)),
        "p95_latency_s": float(np.percentile(latency, 95)),
    }


def extra_cost_per_extra_correct(
    mean_cost: float,
    success_rate: float,
    none_mean_cost: float,
    none_success_rate: float,
) -> float | None:
    """Extra mean dollars per extra point of success rate, versus ``none``.

    Uses per-run means, so conditions with different numbers of runs compare
    on the same scale. Returns ``None`` when this success rate is not higher
    than ``none``.
    """
    extra_success = success_rate - none_success_rate
    if extra_success <= 0:
        return None
    return (mean_cost - none_mean_cost) / extra_success


def bootstrap_task_success_ci(task_id, final_correct, seed: int = 0) -> tuple[float, float]:
    """95% interval for task success, resampling tasks rather than runs.

    Average ``final_correct`` within each task, sort those means by task id,
    then draw them with replacement 1,000 times using
    ``numpy.random.default_rng(seed)``. The interval is the 2.5 and 97.5
    percentiles of the resampled means, so row order does not change it.
    """
    ids = _as_series(task_id).reset_index(drop=True)
    finals = _as_bool(_as_series(final_correct), "final_correct")
    if len(ids) != len(finals):
        raise ValueError("task_id and final_correct must have the same length")
    if ids.empty:
        raise ValueError("bootstrap needs at least one run")

    task_means = (
        pd.DataFrame({"task_id": ids, "final_correct": finals.to_numpy(dtype=float)})
        .groupby("task_id", sort=True)["final_correct"]
        .mean()
        .to_numpy(dtype=float)
    )
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(task_means), size=(_N_RESAMPLES, len(task_means)))
    stats = np.mean(task_means[drawn], axis=1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def summarize(runs: list[dict] | pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """One row per condition (or per condition and model) and one column per metric.

    ``seed`` fixes the task bootstrap for ``task_success_ci_low`` and
    ``task_success_ci_high``. Task success itself is the share of scored runs
    with ``final_correct``, which matches the mean of the task means when
    every task has the same number of scored runs. Runs with ``status``
    ``failed`` are left out of every rate and counted in ``n_failed``. Extra
    cost is ``None`` when the matching ``none`` baseline is absent or when a
    condition's success rate is not higher than that baseline. With a
    ``model`` column the baseline is ``none`` for the same model; without it
    the baseline is every ``none`` run.
    """
    frame = validate_runs(runs)
    has_model = "model" in frame.columns
    group_columns = ["condition", "model"] if has_model else "condition"
    none_baselines = _none_baselines(frame, has_model)

    rows: list[dict] = []
    for key, group in frame.groupby(group_columns, sort=False):
        condition_name, model_name, scored, n_failed = _group_parts(key, group, has_model)
        if scored.empty:
            raise ValueError(
                f"no scored runs for condition {condition_name!r}"
                + (f" and model {model_name!r}" if has_model else "")
            )
        low, high = bootstrap_task_success_ci(
            scored["task_id"], scored["final_correct"], seed=seed
        )
        success = task_success(scored["final_correct"])
        baseline = none_baselines.get(model_name) if has_model else none_baselines.get(None)
        if baseline is None:
            extra = None
        else:
            none_mean_cost, none_success = baseline
            extra = extra_cost_per_extra_correct(
                float(scored["cost_usd"].mean()),
                success,
                none_mean_cost,
                none_success,
            )
        row = {
            "condition": condition_name,
            "task_success": success,
            "task_success_ci_low": low,
            "task_success_ci_high": high,
            **verifier_metrics(
                scored["verdict"], scored["initial_correct"], condition_name
            ),
            **fixed_broken(scored["initial_correct"], scored["final_correct"]),
            **cost_metrics(
                scored["cost_usd"],
                scored["tokens_in"],
                scored["tokens_out"],
                scored["latency_s"],
            ),
            "extra_cost_per_extra_correct": extra,
            "n_failed": n_failed,
        }
        if has_model:
            row["model"] = model_name
        rows.append(row)
    return _metrics_frame(rows, has_model=has_model)


def _blank_verifier(n_missing: int | None) -> dict[str, float | int | None]:
    blank: dict[str, float | int | None] = {
        column: None
        for column in (
            "tp",
            "fp",
            "fn",
            "tn",
            "catch_rate",
            "precision",
            "false_alarm_rate",
            "mcc",
            "abstention_rate",
        )
    }
    blank["n_missing_verdict"] = n_missing
    return blank


def _none_baselines(frame: pd.DataFrame, has_model: bool) -> dict:
    """Mean cost and success for each ``none`` baseline used by extra cost."""
    baselines: dict = {}
    none = frame.loc[frame["condition"] == "none"]
    if none.empty:
        return baselines
    if has_model:
        for model, group in none.groupby("model", sort=False):
            scored = _scored_runs(group)
            if scored.empty:
                continue
            baselines[_python_scalar(model)] = (
                float(scored["cost_usd"].mean()),
                task_success(scored["final_correct"]),
            )
    else:
        scored = _scored_runs(none)
        if not scored.empty:
            baselines[None] = (
                float(scored["cost_usd"].mean()),
                task_success(scored["final_correct"]),
            )
    return baselines


def _group_parts(key, group: pd.DataFrame, has_model: bool):
    if has_model:
        condition_name = _python_scalar(key[0])
        model_name = _python_scalar(key[1])
    else:
        condition_name = _python_scalar(key)
        model_name = None
    scored = _scored_runs(group)
    n_failed = int(len(group) - len(scored))
    return condition_name, model_name, scored, n_failed


def _scored_runs(group: pd.DataFrame) -> pd.DataFrame:
    if "status" not in group.columns:
        return group
    return group.loc[group["status"] != "failed"]


def _metrics_frame(rows: list[dict], has_model: bool = False) -> pd.DataFrame:
    if has_model:
        frame = pd.DataFrame(
            {
                "condition": [row["condition"] for row in rows],
                "model": [row["model"] for row in rows],
            }
        )
    else:
        frame = pd.DataFrame({"condition": [row["condition"] for row in rows]})
    nullable = set(_NULLABLE_COLUMNS)
    for column in METRIC_COLUMNS:
        values = [row[column] for row in rows]
        if column in nullable:
            frame[column] = pd.Series(values, dtype=object)
        else:
            frame[column] = values
    if has_model:
        return frame.set_index(["condition", "model"])
    return frame.set_index("condition")


def _as_series(values) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(list(values))


def _as_bool(series: pd.Series, name: str) -> pd.Series:
    converted: list[bool] = []
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            converted.append(bool(value))
        elif isinstance(value, (int, np.integer)) and value in (0, 1):
            converted.append(bool(value))
        else:
            raise ValueError(f"{name} must be a boolean, got {value!r}")
    return pd.Series(converted, index=series.index, dtype=bool)


def _as_float(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{name} must be numeric on every run")
    return pd.Series(numeric.to_numpy(dtype=float), index=series.index)


def _checked_verdicts(series: pd.Series) -> pd.Series:
    normalized: list = []
    for value in _as_series(series).tolist():
        if _is_missing(value):
            normalized.append(None)
        else:
            normalized.append(value)
    unknown = sorted(
        {repr(value) for value in normalized if value is not None and value not in _KNOWN_VERDICTS}
    )
    if unknown:
        raise ValueError(f"unknown verdict value(s): {', '.join(unknown)}")
    return pd.Series(normalized, dtype=object)


def _aligned_floats(*columns: tuple[str, object]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    length: int | None = None
    for name, values in columns:
        numeric = pd.to_numeric(_as_series(values), errors="coerce")
        array = numeric.to_numpy(dtype=float)
        if length is None:
            length = int(array.size)
        elif array.size != length:
            raise ValueError(f"{name} must have one value per run")
        if length == 0 or np.isnan(array).any():
            raise ValueError(f"{name} must be numeric on every run")
        arrays.append(array)
    return arrays


def _mcc(tp: int, fp: int, fn: int, tn: int, actually_wrong, flagged) -> float | None:
    """Matthews correlation, or ``None`` when its denominator is zero."""
    if (tp + fp) == 0 or (tp + fn) == 0 or (tn + fp) == 0 or (tn + fn) == 0:
        return None
    return float(matthews_corrcoef(actually_wrong, flagged))


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    try:
        missing = pd.isna(value)
    except TypeError:
        return False
    return bool(missing) if np.ndim(missing) == 0 else False


def _python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value
