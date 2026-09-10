from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from repoexec_baseline.common import (
    DEFAULT_DATASET,
    DEFAULT_SUBSET,
    compute_dir_for_prediction,
    estimate_pass_at_k,
    iter_jsonl,
    load_repoexec_dataset,
    safe_mean,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RepoExec baseline metrics for all tasks and cross_context tasks.")
    parser.add_argument("--prediction-dir", required=True, help="Directory produced by repoexec_baseline.generate.")
    parser.add_argument("--execution-dir", default=None, help="Directory produced by repoexec_baseline.execute.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subset", default=DEFAULT_SUBSET, choices=["full_context", "medium_context", "small_context"])
    return parser.parse_args()


def build_group(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["task_id"])].append(row)
    return grouped


def first_prediction(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: int(row["prediction_id"]))[0]


def aggregate_pass_at_k(rows: list[dict[str, Any]], k: int) -> float | None:
    if not rows:
        return None

    estimates: list[float] = []
    for row in rows:
        estimate = estimate_pass_at_k(int(row["num_predictions"]), int(row["num_correct_predictions"]), k)
        if estimate is None:
            return None
        estimates.append(estimate)

    return safe_mean(estimates)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    estimated_pass_at_1 = aggregate_pass_at_k(rows, 1)
    summary = {
        "num_tasks": len(rows),
        "first_pass@1": safe_mean(1.0 if row["pass@1"] else 0.0 for row in rows),
        "DIR": safe_mean(row["DIR"] for row in rows),
        "mean_input_tokens": safe_mean(row["input_tokens"] for row in rows),
        "mean_output_tokens": safe_mean(row["output_tokens"] for row in rows),
        "mean_generation_seconds": safe_mean(row["generation_seconds"] for row in rows),
        "mean_peak_vram_mb": safe_mean(row["peak_vram_mb"] for row in rows),
        "missing_execution_results": sum(1 for row in rows if row["missing_execution"]),
        "mean_num_predictions": safe_mean(row["num_predictions"] for row in rows),
        "mean_num_correct_predictions": safe_mean(row["num_correct_predictions"] for row in rows),
    }
    if estimated_pass_at_1 is not None:
        summary["pass@1"] = estimated_pass_at_1
    for k in (5, 10):
        pass_at_k = aggregate_pass_at_k(rows, k)
        if pass_at_k is not None:
            summary[f"pass@{k}"] = pass_at_k
    return summary


def main() -> None:
    args = parse_args()

    prediction_dir = Path(args.prediction_dir).resolve()
    execution_dir = Path(args.execution_dir).resolve() if args.execution_dir else prediction_dir / "execution"
    workspace_root = prediction_dir.parent.parent if prediction_dir.parent.name == "runs" else Path.cwd().resolve()
    hf_home = Path(os.environ.get("HF_HOME", workspace_root / ".hf")).resolve()
    datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE", hf_home / "datasets")).resolve()
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    hf_home.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    dataset = load_repoexec_dataset(args.dataset, args.subset, hf_home)

    processed_entries = {int(row["task_id"]): row for row in iter_jsonl(prediction_dir / "processed_generations.jsonl")}
    metric_rows = list(iter_jsonl(prediction_dir / "task_metrics.jsonl"))
    task_index_rows = list(iter_jsonl(prediction_dir / "task_index.jsonl"))

    metrics_by_task = build_group(metric_rows)
    execution_rows: list[dict[str, Any]] = []
    for result_file in sorted(execution_dir.glob("results_*.jsonl")):
        execution_rows.extend(iter_jsonl(result_file))
    execution_by_task = build_group(execution_rows)

    per_task_rows: list[dict[str, Any]] = []
    for index_row in task_index_rows:
        task_id = int(index_row["task_id"])
        example = dataset[task_id]
        processed = processed_entries[task_id]
        metric = first_prediction(metrics_by_task.get(task_id, []))
        execution = first_prediction(execution_by_task.get(task_id, []))
        execution_rows_for_task = sorted(execution_by_task.get(task_id, []), key=lambda row: int(row["prediction_id"]))

        prediction = processed["predictions"][0] if processed["predictions"] else ""
        dir_value = compute_dir_for_prediction(example, prediction)
        per_task_rows.append(
            {
                "task_id": task_id,
                "dataset_id": index_row.get("dataset_id"),
                "entry_point": index_row["entry_point"],
                "project": index_row["project"],
                "module": index_row["module"],
                "cross_context": bool(index_row["cross_context"]),
                "pass@1": bool(execution["passed"]) if execution else False,
                "DIR": dir_value,
                "input_tokens": metric["input_tokens"] if metric else None,
                "output_tokens": metric["output_tokens"] if metric else None,
                "generation_seconds": metric["generation_seconds"] if metric else None,
                "peak_vram_mb": metric["peak_vram_mb"] if metric else None,
                "missing_execution": execution is None,
                "num_predictions": len(execution_rows_for_task),
                "num_correct_predictions": sum(1 for row in execution_rows_for_task if row.get("passed")),
            }
        )

    cross_context_rows = [row for row in per_task_rows if row["cross_context"]]
    summary = {
        "dataset": args.dataset,
        "subset": args.subset,
        "prediction_dir": str(prediction_dir),
        "execution_dir": str(execution_dir),
        "all_tasks": aggregate(per_task_rows),
        "cross_context_true": aggregate(cross_context_rows),
    }

    write_jsonl(prediction_dir / "per_task_metrics.jsonl", per_task_rows)
    write_json(prediction_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
