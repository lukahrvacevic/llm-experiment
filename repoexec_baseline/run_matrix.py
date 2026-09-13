from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from repoexec_baseline.common import ensure_dir
from repoexec_baseline.representations import SUPPORTED_REPRESENTATIONS


DEFAULT_MODELS = [
    "qwen2.5-coder:1.5b-base",
    "qwen2.5-coder:3b-base",
    "qwen2.5-coder:7b-base",
    "deepseek-coder:1.3b-base-q4_K_M",
    "deepseek-coder:6.7b-base-q4_K_M",
    "deepseek-coder-v2:16b-lite-base-q4_K_M",
    "codegemma:2b-code-q4_K_M",
    "codegemma:7b-code-q4_K_M",
]

DEFAULT_SUBSETS = ["full_context", "medium_context", "small_context"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a RepoExec model/subset matrix and collect timing + summary data.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS, choices=DEFAULT_SUBSETS)
    parser.add_argument("--representation", default="raw", choices=SUPPORTED_REPRESENTATIONS)
    parser.add_argument("--task-limit", type=int, default=30)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-prefix", default="matrix30")
    parser.add_argument("--repoexec-dir", default="RepoExec")
    parser.add_argument("--docker-image", default="codeeval-runner:latest")
    parser.add_argument(
        "--docker-bin",
        default=r"C:\Users\admin\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe",
    )
    parser.add_argument("--docker-config-dir", default=".docker-config")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--ollama-keep-alive", default="30m")
    parser.add_argument("--ollama-num-ctx", type=int, default=None)
    parser.add_argument("--ollama-parallel-requests", type=int, default=1)
    parser.add_argument("--parallel-tasks", type=int, default=1)
    parser.add_argument("--execution-timeout", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    return value.replace(":", "-").replace("/", "-")


def run_checked(command: list[str], cwd: Path, env: dict[str, str]) -> float:
    start = perf_counter()
    process = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = perf_counter() - start
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")
    return elapsed


def main() -> None:
    args = parse_args()

    workspace_root = Path.cwd().resolve()
    runs_root = ensure_dir((workspace_root / args.runs_root).resolve())
    repoexec_dir = (workspace_root / args.repoexec_dir).resolve()
    docker_config_dir = ensure_dir((workspace_root / args.docker_config_dir).resolve())
    env = os.environ.copy()
    env["DOCKER_CONFIG"] = str(docker_config_dir)

    matrix_rows: list[dict[str, object]] = []

    for model in args.models:
        for subset in args.subsets:
            run_name = f"{args.run_prefix}-{slugify(model)}-{subset}-{args.representation}-n{args.task_limit}"
            output_dir = ensure_dir(runs_root / run_name)

            generate_cmd = [
                sys.executable,
                "-m",
                "repoexec_baseline.generate",
                "--model",
                model,
                "--output-dir",
                str(output_dir),
                "--subset",
                subset,
                "--representation",
                args.representation,
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
                generate_cmd.append("--do-sample")
            if args.ollama_num_ctx is not None:
                generate_cmd.extend(["--ollama-num-ctx", str(args.ollama_num_ctx)])

            execute_cmd = [
                sys.executable,
                "-m",
                "repoexec_baseline.execute",
                "--repoexec-dir",
                str(repoexec_dir),
                "--prediction-dir",
                str(output_dir),
                "--docker-image",
                args.docker_image,
                "--docker-bin",
                args.docker_bin,
                "--docker-config-dir",
                str(docker_config_dir),
                "--timeout",
                str(args.execution_timeout),
            ]
            if args.overwrite:
                execute_cmd.append("--overwrite")

            summarize_cmd = [
                sys.executable,
                "-m",
                "repoexec_baseline.summarize",
                "--prediction-dir",
                str(output_dir),
                "--subset",
                subset,
            ]

            print(
                f"=== {model} | {subset} | representation={args.representation} | first {args.task_limit} tasks ===",
                flush=True,
            )
            total_start = perf_counter()
            generate_seconds = run_checked(generate_cmd, workspace_root, env)
            execute_seconds = run_checked(execute_cmd, workspace_root, env)
            summarize_seconds = run_checked(summarize_cmd, workspace_root, env)
            total_seconds = perf_counter() - total_start

            timing = {
                "model": model,
                "subset": subset,
                "representation": args.representation,
                "task_limit": args.task_limit,
                "ollama_parallel_requests": args.ollama_parallel_requests,
                "parallel_tasks": args.parallel_tasks,
                "ollama_num_ctx": args.ollama_num_ctx,
                "generate_seconds": generate_seconds,
                "execute_seconds": execute_seconds,
                "summarize_seconds": summarize_seconds,
                "total_seconds": total_seconds,
            }
            (output_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            row = {
                "model": model,
                "subset": subset,
                "representation": args.representation,
                "task_limit": args.task_limit,
                "output_dir": str(output_dir),
                "timing": timing,
                "all_tasks": summary["all_tasks"],
                "cross_context_true": summary["cross_context_true"],
            }
            matrix_rows.append(row)
            print(
                f"Completed {model} | {subset} | representation={args.representation} | pass@1={summary['all_tasks']['pass@1']} | "
                f"total={total_seconds:.2f}s",
                flush=True,
            )

    matrix_path = runs_root / f"{args.run_prefix}-summary.json"
    matrix_path.write_text(json.dumps(matrix_rows, indent=2), encoding="utf-8")
    print(f"Saved matrix summary to {matrix_path}", flush=True)


if __name__ == "__main__":
    main()
