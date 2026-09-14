from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, time
from typing import Any

from repoexec_baseline.common import ensure_dir, iter_jsonl, write_json
from repoexec_baseline.representations import SUPPORTED_REPRESENTATIONS
from repoexec_baseline.run_matrix import DEFAULT_MODELS, slugify


DEFAULT_REPRESENTATIONS = ["raw", "ast", "reduced_ast"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a RepoExec model/representation matrix without executing generated code."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--representations",
        nargs="+",
        default=DEFAULT_REPRESENTATIONS,
        choices=SUPPORTED_REPRESENTATIONS,
    )
    parser.add_argument("--subset", default="full_context", choices=["full_context", "medium_context", "small_context"])
    parser.add_argument("--task-limit", type=int, default=30)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-prefix", default="fmle-generation30")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-return-sequences", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--ollama-keep-alive", default="30m")
    parser.add_argument("--ollama-num-ctx", type=int, default=None)
    parser.add_argument("--ollama-parallel-requests", type=int, default=1)
    parser.add_argument("--parallel-tasks", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument(
        "--session-start-epoch",
        type=float,
        default=None,
        help="Optional launcher start time, used to include environment setup and model pulls in total time.",
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument("--do-sample", dest="do_sample", action="store_true")
    sample_group.add_argument("--greedy", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=True)
    return parser.parse_args()


def utc_timestamp(epoch: float | None = None) -> str:
    value = time() if epoch is None else epoch
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in iter_jsonl(path))


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def is_complete_run(
    output_dir: Path,
    model: str,
    subset: str,
    representation: str,
    task_limit: int,
    num_return_sequences: int,
) -> bool:
    config = load_json(output_dir / "run_config.json")
    if config is None:
        return False

    expected = {
        "model": model,
        "subset": subset,
        "representation": representation,
        "task_count": task_limit,
        "num_return_sequences": num_return_sequences,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        return False

    return (
        count_rows(output_dir / "processed_generations.jsonl") == task_limit
        and count_rows(output_dir / "task_index.jsonl") == task_limit
        and count_rows(output_dir / "task_metrics.jsonl") == task_limit * num_return_sequences
    )


def build_generate_command(args: argparse.Namespace, model: str, representation: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "repoexec_baseline.generate",
        "--model",
        model,
        "--output-dir",
        str(output_dir),
        "--subset",
        args.subset,
        "--representation",
        representation,
        "--task-limit",
        str(args.task_limit),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--num-return-sequences",
        str(args.num_return_sequences),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
        "--ollama-base-url",
        args.ollama_base_url,
        "--ollama-timeout-seconds",
        str(args.ollama_timeout_seconds),
        "--ollama-keep-alive",
        args.ollama_keep_alive,
        "--ollama-parallel-requests",
        str(args.ollama_parallel_requests),
        "--parallel-tasks",
        str(args.parallel_tasks),
    ]
    if args.do_sample:
        command.append("--do-sample")
    if args.ollama_num_ctx is not None:
        command.extend(["--ollama-num-ctx", str(args.ollama_num_ctx)])
    return command


def write_matrix_summary(
    path: Path,
    args: argparse.Namespace,
    status: str,
    started_epoch: float,
    invocation_started_epoch: float,
    rows: list[dict[str, Any]],
    error: str | None = None,
) -> None:
    finished_epoch = time()
    recorded_generation_seconds = sum(
        float(row.get("generation_wall_seconds") or 0.0)
        for row in rows
        if row.get("status") in {"completed", "skipped_existing"}
    )
    current_invocation_generation_seconds = sum(
        float(row.get("generation_wall_seconds") or 0.0)
        for row in rows
        if row.get("status") == "completed"
    )
    write_json(
        path,
        {
            "schema_version": 1,
            "status": status,
            "error": error,
            "started_at_utc": utc_timestamp(started_epoch),
            "invocation_started_at_utc": utc_timestamp(invocation_started_epoch),
            "updated_at_utc": utc_timestamp(finished_epoch),
            "session_wall_seconds": finished_epoch - started_epoch,
            "invocation_wall_seconds": finished_epoch - invocation_started_epoch,
            "recorded_generation_wall_seconds": recorded_generation_seconds,
            "current_invocation_generation_wall_seconds": current_invocation_generation_seconds,
            "configuration": {
                "models": args.models,
                "representations": args.representations,
                "subset": args.subset,
                "task_limit": args.task_limit,
                "max_new_tokens": args.max_new_tokens,
                "num_return_sequences": args.num_return_sequences,
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "ollama_base_url": args.ollama_base_url,
                "ollama_num_ctx": args.ollama_num_ctx,
                "ollama_parallel_requests": args.ollama_parallel_requests,
                "parallel_tasks": args.parallel_tasks,
            },
            "runs": rows,
        },
    )


def create_bundle(bundle_path: Path, summary_path: Path, runs_root: Path, rows: list[dict[str, Any]]) -> None:
    generation_files = (
        "generations.json",
        "processed_generations.jsonl",
        "task_metrics.jsonl",
        "task_index.jsonl",
        "run_config.json",
        "pre_eval_summary.json",
        "generation_timing.json",
    )
    temporary_path = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    with tarfile.open(temporary_path, "w:gz") as archive:
        archive.add(summary_path, arcname=summary_path.name)
        for row in rows:
            run_dir = runs_root / str(row["run_dir"])
            if run_dir.exists() and row.get("status") in {"completed", "skipped_existing"}:
                for filename in generation_files:
                    source = run_dir / filename
                    if source.exists():
                        archive.add(source, arcname=f"{run_dir.name}/{filename}")
    temporary_path.replace(bundle_path)


def main() -> None:
    args = parse_args()
    if args.task_limit <= 0:
        raise ValueError("--task-limit must be positive")
    if args.num_return_sequences <= 0:
        raise ValueError("--num-return-sequences must be positive")
    if args.ollama_parallel_requests <= 0:
        raise ValueError("--ollama-parallel-requests must be positive")
    if args.parallel_tasks <= 0:
        raise ValueError("--parallel-tasks must be positive")

    workspace_root = Path.cwd().resolve()
    runs_root = ensure_dir(Path(args.runs_root).resolve())
    summary_path = runs_root / f"{args.run_prefix}-generation-summary.json"
    bundle_path = runs_root / f"{args.run_prefix}-generation-bundle.tar.gz"
    invocation_started_epoch = time()
    started_epoch = args.session_start_epoch or invocation_started_epoch
    rows: list[dict[str, Any]] = []

    child_env = os.environ.copy()
    child_env.setdefault("HF_DATASETS_OFFLINE", "0")
    child_env.setdefault("HF_HUB_OFFLINE", "0")

    try:
        for model in args.models:
            for representation in args.representations:
                run_name = (
                    f"{args.run_prefix}-{slugify(model)}-{args.subset}-{representation}-n{args.task_limit}"
                )
                output_dir = ensure_dir(runs_root / run_name)
                row: dict[str, Any] = {
                    "model": model,
                    "subset": args.subset,
                    "representation": representation,
                    "task_limit": args.task_limit,
                    "num_return_sequences": args.num_return_sequences,
                    "ollama_parallel_requests": args.ollama_parallel_requests,
                    "parallel_tasks": args.parallel_tasks,
                    "run_dir": run_name,
                    "status": "pending",
                    "generation_wall_seconds": None,
                }
                rows.append(row)

                complete = is_complete_run(
                    output_dir,
                    model,
                    args.subset,
                    representation,
                    args.task_limit,
                    args.num_return_sequences,
                )
                if complete and not args.overwrite:
                    previous_timing = load_json(output_dir / "generation_timing.json") or {}
                    previous_config = load_json(output_dir / "run_config.json") or {}
                    row["status"] = "skipped_existing"
                    row["generation_wall_seconds"] = previous_timing.get("generation_wall_seconds")
                    row["ollama_parallel_requests"] = previous_config.get(
                        "ollama_parallel_requests", row["ollama_parallel_requests"]
                    )
                    row["parallel_tasks"] = previous_config.get("parallel_tasks", row["parallel_tasks"])
                    print(f"Skipping complete run: {run_name}", flush=True)
                    write_matrix_summary(
                        summary_path, args, "running", started_epoch, invocation_started_epoch, rows
                    )
                    continue

                print(f"=== {model} | {representation} | first {args.task_limit} {args.subset} tasks ===", flush=True)
                row["status"] = "running"
                write_matrix_summary(summary_path, args, "running", started_epoch, invocation_started_epoch, rows)
                run_started_at = time()
                run_timer = perf_counter()
                process = subprocess.run(
                    build_generate_command(args, model, representation, output_dir),
                    cwd=str(workspace_root),
                    env=child_env,
                )
                elapsed = perf_counter() - run_timer
                row["generation_wall_seconds"] = elapsed
                row["started_at_utc"] = utc_timestamp(run_started_at)
                row["finished_at_utc"] = utc_timestamp()

                if process.returncode != 0:
                    row["status"] = "failed"
                    row["exit_code"] = process.returncode
                    raise RuntimeError(f"Generation failed with exit code {process.returncode}: {run_name}")

                if not is_complete_run(
                    output_dir,
                    model,
                    args.subset,
                    representation,
                    args.task_limit,
                    args.num_return_sequences,
                ):
                    row["status"] = "failed"
                    raise RuntimeError(f"Generation command finished but output is incomplete: {run_name}")

                row["status"] = "completed"
                write_json(
                    output_dir / "generation_timing.json",
                    {
                        "model": model,
                        "subset": args.subset,
                        "representation": representation,
                        "task_limit": args.task_limit,
                        "num_return_sequences": args.num_return_sequences,
                        "ollama_parallel_requests": args.ollama_parallel_requests,
                        "parallel_tasks": args.parallel_tasks,
                        "started_at_utc": row["started_at_utc"],
                        "finished_at_utc": row["finished_at_utc"],
                        "generation_wall_seconds": elapsed,
                    },
                )
                write_matrix_summary(summary_path, args, "running", started_epoch, invocation_started_epoch, rows)
                print(f"Completed {run_name} in {elapsed / 60:.2f} min", flush=True)

        write_matrix_summary(summary_path, args, "completed", started_epoch, invocation_started_epoch, rows)
        if not args.no_archive:
            create_bundle(bundle_path, summary_path, runs_root, rows)
            print(f"Transfer bundle: {bundle_path}", flush=True)
        print(f"Generation summary: {summary_path}", flush=True)
    except (KeyboardInterrupt, Exception) as exc:
        error = "Interrupted by user" if isinstance(exc, KeyboardInterrupt) else str(exc)
        write_matrix_summary(summary_path, args, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", started_epoch, invocation_started_epoch, rows, error)
        print(f"Partial summary: {summary_path}", file=sys.stderr, flush=True)
        if isinstance(exc, KeyboardInterrupt):
            raise SystemExit(130) from exc
        raise


if __name__ == "__main__":
    main()
