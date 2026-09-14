from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator, TypeVar

from repoexec_baseline.common import (
    DEFAULT_DATASET,
    DEFAULT_SUBSET,
    ensure_dir,
    extract_solution_function,
    get_actual_solution,
    load_repoexec_dataset,
    safe_mean,
    write_json,
    write_jsonl,
)
from repoexec_baseline.llm_client import GenerationResult, LLMClient
from repoexec_baseline.representations import SUPPORTED_REPRESENTATIONS, build_prompt


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def ordered_parallel_map(
    function: Callable[[InputT], OutputT],
    values: Iterable[InputT],
    max_workers: int,
) -> Iterator[OutputT]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if max_workers == 1:
        yield from map(function, values)
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        yield from executor.map(function, values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a raw/full_context RepoExec baseline via Ollama.")
    parser.add_argument("--model", required=True, help="Ollama model name.")
    parser.add_argument("--output-dir", required=True, help="Directory for generations and metrics.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subset", default=DEFAULT_SUBSET, choices=["full_context", "medium_context", "small_context"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--ollama-keep-alive", default="30m")
    parser.add_argument("--ollama-num-ctx", type=int, default=None)
    parser.add_argument(
        "--ollama-parallel-requests",
        type=int,
        default=1,
        help="Concurrent candidate requests. Keep at 1 for serial generation.",
    )
    parser.add_argument(
        "--parallel-tasks",
        type=int,
        default=1,
        help="Tasks generated concurrently. Total possible requests are this value times --ollama-parallel-requests.",
    )
    parser.add_argument("--ollama-raw", action="store_true", default=True)
    parser.add_argument("--representation", default="raw", choices=SUPPORTED_REPRESENTATIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_return_sequences <= 0:
        raise ValueError("--num-return-sequences must be positive")
    if args.task_limit is not None and args.task_limit <= 0:
        raise ValueError("--task-limit must be positive")
    if args.ollama_parallel_requests <= 0:
        raise ValueError("--ollama-parallel-requests must be positive")
    if args.parallel_tasks <= 0:
        raise ValueError("--parallel-tasks must be positive")
    random.seed(args.seed)

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

    dataset = load_repoexec_dataset(args.dataset, args.subset, hf_home)
    task_count = len(dataset) if args.task_limit is None else min(args.task_limit, len(dataset))
    client = LLMClient(
        model=args.model,
        base_url=args.ollama_base_url,
        timeout_seconds=args.ollama_timeout_seconds,
        keep_alive=args.ollama_keep_alive,
        num_ctx=args.ollama_num_ctx,
        parallel_requests=args.ollama_parallel_requests,
        raw=args.ollama_raw,
    )

    generations: list[list[str]] = []
    task_metrics: list[dict[str, object]] = []
    processed_generations: list[dict[str, object]] = []
    task_index: list[dict[str, object]] = []

    task_inputs: list[tuple[int, dict[str, Any], str]] = []
    for task_id in range(task_count):
        example = dataset[task_id]
        task_inputs.append((task_id, example, build_prompt(example, args.representation)))

    def generate_task(
        task_input: tuple[int, dict[str, Any], str],
    ) -> tuple[int, dict[str, Any], list[GenerationResult], float]:
        task_id, example, prompt = task_input
        task_started = perf_counter()
        try:
            results = client.generate(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.num_return_sequences,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + (task_id * args.num_return_sequences),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Generation failed for task_id={task_id}, entry_point={example['entry_point']}"
            ) from exc
        task_wall_seconds = perf_counter() - task_started
        return task_id, example, results, task_wall_seconds

    generated_tasks = ordered_parallel_map(generate_task, task_inputs, args.parallel_tasks)
    for task_id, example, results, task_wall_seconds in generated_tasks:
        decoded = [result.text for result in results]
        generations.append(decoded)

        actual_solution = get_actual_solution(example)
        extracted_predictions: list[str] = []
        patched_tests: list[str] = []

        for prediction_id, result in enumerate(results):
            extracted_prediction = extract_solution_function(
                generation_text=result.text,
                target_function_prompt=example["target_function_prompt"],
                entry_point=example["entry_point"],
            )
            extracted_predictions.append(extracted_prediction)
            patched_tests.append(example["check"].replace(actual_solution, extracted_prediction))

            task_metrics.append(
                {
                    "task_id": task_id,
                    "prediction_id": prediction_id,
                    "dataset_id": example.get("id"),
                    "entry_point": example["entry_point"],
                    "project": example["project"],
                    "module": example["module"],
                    "cross_context": bool(example.get("cross_context", False)),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "generation_seconds": result.generation_seconds,
                    "task_wall_seconds": task_wall_seconds,
                    "peak_vram_mb": result.peak_vram_mb,
                }
            )

        processed_generations.append(
            {
                "task_id": task_id,
                "project": example["project"],
                "module": example["module"],
                "predictions": extracted_predictions,
                "test": patched_tests,
            }
        )
        task_index.append(
            {
                "task_id": task_id,
                "dataset_id": example.get("id"),
                "entry_point": example["entry_point"],
                "project": example["project"],
                "module": example["module"],
                "cross_context": bool(example.get("cross_context", False)),
            }
        )

        candidate_seconds = [result.generation_seconds for result in results]
        input_tokens = results[0].input_tokens
        output_tokens = [result.output_tokens for result in results if result.output_tokens is not None]
        peak_vram_values = [result.peak_vram_mb for result in results if result.peak_vram_mb is not None]
        mean_output_tokens = safe_mean(output_tokens)
        output_text = f"{mean_output_tokens:.1f}" if mean_output_tokens is not None else "n/a"
        print(
            f"[{task_id + 1}/{task_count}] {example['entry_point']} | "
            f"candidates={len(results)} task_parallel={args.parallel_tasks} "
            f"candidate_parallel={min(args.ollama_parallel_requests, len(results))} "
            f"in={input_tokens} out_mean={output_text} "
            f"task_time={task_wall_seconds:.2f}s request_mean={safe_mean(candidate_seconds):.2f}s "
            f"peak_vram={max(peak_vram_values) if peak_vram_values else None}"
        )

    write_json(output_dir / "generations.json", generations)
    write_jsonl(output_dir / "processed_generations.jsonl", processed_generations)
    write_jsonl(output_dir / "task_metrics.jsonl", task_metrics)
    write_jsonl(output_dir / "task_index.jsonl", task_index)

    cross_context_task_ids = {row["task_id"] for row in task_index if row["cross_context"]}
    first_predictions = [row for row in task_metrics if row["prediction_id"] == 0]

    write_json(
        output_dir / "run_config.json",
        {
            "model": args.model,
            "dataset": args.dataset,
            "subset": args.subset,
            "task_count": task_count,
            "max_new_tokens": args.max_new_tokens,
            "num_return_sequences": args.num_return_sequences,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "ollama_base_url": args.ollama_base_url,
            "ollama_timeout_seconds": args.ollama_timeout_seconds,
            "ollama_keep_alive": args.ollama_keep_alive,
            "ollama_num_ctx": args.ollama_num_ctx,
            "ollama_parallel_requests": args.ollama_parallel_requests,
            "parallel_tasks": args.parallel_tasks,
            "ollama_raw": args.ollama_raw,
            "representation": args.representation,
        },
    )
    write_json(
        output_dir / "pre_eval_summary.json",
        {
            "all_tasks": {
                "num_tasks": len(first_predictions),
                "mean_input_tokens": safe_mean(row["input_tokens"] for row in first_predictions),
                "mean_output_tokens": safe_mean(row["output_tokens"] for row in first_predictions),
                "mean_generation_seconds": safe_mean(row["generation_seconds"] for row in first_predictions),
                "mean_task_wall_seconds": safe_mean(row["task_wall_seconds"] for row in first_predictions),
                "mean_peak_vram_mb": safe_mean(row["peak_vram_mb"] for row in first_predictions),
            },
            "cross_context_true": {
                "num_tasks": sum(1 for row in first_predictions if row["task_id"] in cross_context_task_ids),
                "mean_input_tokens": safe_mean(
                    row["input_tokens"] for row in first_predictions if row["task_id"] in cross_context_task_ids
                ),
                "mean_output_tokens": safe_mean(
                    row["output_tokens"] for row in first_predictions if row["task_id"] in cross_context_task_ids
                ),
                "mean_generation_seconds": safe_mean(
                    row["generation_seconds"] for row in first_predictions if row["task_id"] in cross_context_task_ids
                ),
                "mean_task_wall_seconds": safe_mean(
                    row["task_wall_seconds"] for row in first_predictions if row["task_id"] in cross_context_task_ids
                ),
                "mean_peak_vram_mb": safe_mean(
                    row["peak_vram_mb"] for row in first_predictions if row["task_id"] in cross_context_task_ids
                ),
            },
        },
    )


if __name__ == "__main__":
    main()
