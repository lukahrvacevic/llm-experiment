from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RepoExec matrix sequentially for AST and reduced AST.")
    parser.add_argument("--task-limit", type=int, default=100)
    parser.add_argument("--subsets", nargs="+", default=["full_context"])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--repoexec-dir", default="RepoExec")
    parser.add_argument("--docker-image", default="codeeval-runner:latest")
    parser.add_argument(
        "--docker-bin",
        default=r"C:\Users\admin\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe",
    )
    parser.add_argument("--docker-config-dir", default=".docker-config")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-return-sequences", type=int, default=5)
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--ollama-keep-alive", default="30m")
    parser.add_argument("--ollama-num-ctx", type=int, default=None)
    parser.add_argument("--execution-timeout", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-prefix-ast", default="matrix100-pass5-full-ast-q4")
    parser.add_argument("--run-prefix-reduced", default="matrix100-pass5-full-reduced-ast-q4")
    return parser.parse_args()


def build_command(args: argparse.Namespace, representation: str, run_prefix: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "repoexec_baseline.run_matrix",
        "--subsets",
        *args.subsets,
        "--representation",
        representation,
        "--task-limit",
        str(args.task_limit),
        "--runs-root",
        args.runs_root,
        "--run-prefix",
        run_prefix,
        "--repoexec-dir",
        args.repoexec_dir,
        "--docker-image",
        args.docker_image,
        "--docker-bin",
        args.docker_bin,
        "--docker-config-dir",
        args.docker_config_dir,
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
        "--execution-timeout",
        str(args.execution_timeout),
    ]
    if args.do_sample:
        command.append("--do-sample")
    if args.ollama_num_ctx is not None:
        command.extend(["--ollama-num-ctx", str(args.ollama_num_ctx)])
    if args.overwrite:
        command.append("--overwrite")
    return command


def run_checked(command: list[str], cwd: Path) -> None:
    process = subprocess.run(command, cwd=str(cwd))
    if process.returncode != 0:
        raise SystemExit(process.returncode)


def main() -> None:
    args = parse_args()
    workspace_root = Path.cwd().resolve()

    run_checked(build_command(args, "ast", args.run_prefix_ast), workspace_root)
    run_checked(build_command(args, "reduced_ast", args.run_prefix_reduced), workspace_root)


if __name__ == "__main__":
    main()
