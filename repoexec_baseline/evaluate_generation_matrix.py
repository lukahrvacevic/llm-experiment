from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from repoexec_baseline.common import ensure_dir, write_json


def default_docker_bin() -> str:
    windows_path = Path(r"C:\Users\admin\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe")
    if os.name == "nt" and windows_path.exists():
        return str(windows_path)
    return os.environ.get("DOCKER_BIN", "docker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a generation-only RepoExec matrix locally with Docker.")
    parser.add_argument("--matrix-summary", required=True, help="Transferred *-generation-summary.json file.")
    parser.add_argument("--repoexec-dir", default="RepoExec")
    parser.add_argument("--docker-image", default="codeeval-runner:latest")
    parser.add_argument("--docker-bin", default=default_docker_bin())
    parser.add_argument("--docker-config-dir", default=".docker-config")
    parser.add_argument("--execution-timeout", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_checked(command: list[str], cwd: Path, env: dict[str, str]) -> float:
    started = perf_counter()
    process = subprocess.run(command, cwd=str(cwd), env=env)
    elapsed = perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")
    return elapsed


def main() -> None:
    args = parse_args()
    workspace_root = Path.cwd().resolve()
    matrix_path = Path(args.matrix_summary).resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    runs_root = matrix_path.parent
    repoexec_dir = Path(args.repoexec_dir).resolve()
    docker_config_dir = ensure_dir(Path(args.docker_config_dir).resolve())
    env = os.environ.copy()
    env["DOCKER_CONFIG"] = str(docker_config_dir)

    evaluated_rows: list[dict[str, Any]] = []
    total_started = perf_counter()
    for source_row in matrix.get("runs", []):
        if source_row.get("status") not in {"completed", "skipped_existing"}:
            continue

        prediction_dir = runs_root / str(source_row["run_dir"])
        processed_path = prediction_dir / "processed_generations.jsonl"
        if not processed_path.exists():
            raise FileNotFoundError(f"Missing transferred generation output: {processed_path}")

        execute_command = [
            sys.executable,
            "-m",
            "repoexec_baseline.execute",
            "--repoexec-dir",
            str(repoexec_dir),
            "--prediction-dir",
            str(prediction_dir),
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
            execute_command.append("--overwrite")

        summarize_command = [
            sys.executable,
            "-m",
            "repoexec_baseline.summarize",
            "--prediction-dir",
            str(prediction_dir),
            "--subset",
            str(source_row["subset"]),
        ]

        print(f"=== Evaluating {source_row['model']} | {source_row['representation']} ===", flush=True)
        execute_seconds = run_checked(execute_command, workspace_root, env)
        summarize_seconds = run_checked(summarize_command, workspace_root, env)
        metrics = json.loads((prediction_dir / "summary.json").read_text(encoding="utf-8"))
        evaluated_rows.append(
            {
                **source_row,
                "prediction_dir": str(prediction_dir),
                "execute_seconds": execute_seconds,
                "summarize_seconds": summarize_seconds,
                "all_tasks": metrics["all_tasks"],
                "cross_context_true": metrics["cross_context_true"],
            }
        )

    evaluation_path = runs_root / matrix_path.name.replace(
        "-generation-summary.json", "-evaluation-summary.json"
    )
    write_json(
        evaluation_path,
        {
            "source_generation_summary": matrix_path.name,
            "total_evaluation_wall_seconds": perf_counter() - total_started,
            "runs": evaluated_rows,
        },
    )
    print(f"Evaluation summary: {evaluation_path}", flush=True)


if __name__ == "__main__":
    main()
