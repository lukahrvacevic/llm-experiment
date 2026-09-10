from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, median
from typing import Any

from repoexec_baseline.common import DEFAULT_DATASET, ensure_dir, load_repoexec_dataset, write_json, write_jsonl
from repoexec_baseline.llm_client import LLMClient
from repoexec_baseline.representations import build_prompt


REPRESENTATIONS = (
    ("raw_full_context", "full_context", "raw"),
    ("raw_medium_context", "medium_context", "raw"),
    ("raw_small_context", "small_context", "raw"),
    ("ast", "full_context", "ast"),
    ("reduced_ast", "full_context", "reduced_ast"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-task prompt token statistics for RepoExec prompt representations.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="runs/prompt-token-stats")
    parser.add_argument("--tokenizer-model", default="qwen2.5-coder:1.5b-base")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--ollama-keep-alive", default="30m")
    parser.add_argument("--ollama-num-ctx", type=int, default=None)
    return parser.parse_args()


def summarize(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
        }

    sorted_values = sorted(values)
    p95_index = min(len(sorted_values) - 1, max(0, int(round(0.95 * (len(sorted_values) - 1)))))
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": float(mean(sorted_values)),
        "median": float(median(sorted_values)),
        "p95": sorted_values[p95_index],
    }


def main() -> None:
    args = parse_args()

    output_dir = ensure_dir(Path(args.output_dir).resolve())
    workspace_root = output_dir.parent.parent if output_dir.parent.name == "runs" else Path.cwd().resolve()
    hf_home = Path(os.environ.get("HF_HOME", workspace_root / ".hf")).resolve()
    datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE", hf_home / "datasets")).resolve()
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    ensure_dir(hf_home)
    ensure_dir(datasets_cache)

    datasets_by_subset = {
        subset: load_repoexec_dataset(args.dataset, subset, hf_home)
        for subset in {"full_context", "medium_context", "small_context"}
    }

    client = LLMClient(
        model=args.tokenizer_model,
        base_url=args.ollama_base_url,
        timeout_seconds=args.ollama_timeout_seconds,
        keep_alive=args.ollama_keep_alive,
        num_ctx=args.ollama_num_ctx,
        raw=True,
    )

    full_dataset = datasets_by_subset["full_context"]
    num_tasks = len(full_dataset)
    per_task_rows: list[dict[str, Any]] = []

    for task_id in range(num_tasks):
        base_example = full_dataset[task_id]
        row: dict[str, Any] = {
            "task_id": task_id,
            "dataset_id": base_example.get("id"),
            "entry_point": base_example["entry_point"],
            "project": base_example["project"],
            "module": base_example["module"],
            "cross_context": bool(base_example.get("cross_context", False)),
        }

        for label, subset, representation in REPRESENTATIONS:
            example = datasets_by_subset[subset][task_id]
            prompt = build_prompt(example, representation)
            prompt_tokens = client.count_prompt_tokens(prompt)
            row[label] = prompt_tokens
            row[f"{label}_chars"] = len(prompt)

        per_task_rows.append(row)
        print(
            f"[{task_id + 1}/{num_tasks}] {base_example['entry_point']} | "
            f"full={row['raw_full_context']} medium={row['raw_medium_context']} "
            f"small={row['raw_small_context']} ast={row['ast']} reduced_ast={row['reduced_ast']}"
        )

    write_jsonl(output_dir / "per_task_token_stats.jsonl", per_task_rows)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "tokenizer_model": args.tokenizer_model,
        "num_tasks": num_tasks,
        "representations": {},
        "cross_context_true": {},
    }
    cross_rows = [row for row in per_task_rows if row["cross_context"]]
    for label, _, _ in REPRESENTATIONS:
        summary["representations"][label] = summarize([row[label] for row in per_task_rows if row[label] is not None])
        summary["cross_context_true"][label] = summarize([row[label] for row in cross_rows if row[label] is not None])

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
