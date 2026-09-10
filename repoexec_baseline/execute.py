from __future__ import annotations

import argparse
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from repoexec_baseline.common import ensure_dir, iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RepoExec processed generations through the official Docker runner.")
    parser.add_argument("--repoexec-dir", required=True, help="Path to the cloned RepoExec repository.")
    parser.add_argument("--prediction-dir", required=True, help="Directory produced by repoexec_baseline.generate.")
    parser.add_argument("--execution-dir", default=None, help="Directory for results_<task>.jsonl files.")
    parser.add_argument("--docker-image", default="codeeval-runner:latest")
    parser.add_argument("--docker-bin", default=os.environ.get("DOCKER_BIN", "docker"))
    parser.add_argument("--docker-config-dir", default=os.environ.get("DOCKER_CONFIG"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_dir(data_root: Path, project: str) -> Path:
    direct = data_root / project
    if direct.exists():
        return direct

    plus_encoded = data_root / project.replace("/", "+")
    if plus_encoded.exists():
        return plus_encoded

    return direct


def resolve_package_dir(repoexec_root: Path, project: str) -> Path:
    direct = repoexec_root / project
    if direct.exists():
        return direct

    plus_encoded = repoexec_root / project.replace("/", "+")
    if plus_encoded.exists():
        return plus_encoded

    data_encoded = repoexec_root / "data_with_test_case" / project.replace("/", "+")
    if data_encoded.exists():
        return data_encoded

    return direct


def run_checked(command: list[str], env: dict[str, str] | None = None) -> None:
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )


def group_pending_tasks(
    tasks: list[dict[str, object]],
    repoexec_dir: Path,
    execution_dir: Path,
    overwrite: bool,
) -> list[tuple[str, Path, list[int]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    package_dirs: dict[str, Path] = {}

    for row in tasks:
        task_id = int(row["task_id"])
        project = str(row["project"])
        result_file = execution_dir / f"results_{task_id}.jsonl"
        if result_file.exists() and not overwrite:
            continue

        grouped[project].append(task_id)
        if project not in package_dirs:
            package_dirs[project] = resolve_package_dir(repoexec_dir, project)

    grouped_items: list[tuple[str, Path, list[int]]] = []
    for project, task_ids in grouped.items():
        grouped_items.append((project, package_dirs[project], sorted(task_ids)))

    return sorted(grouped_items, key=lambda item: (item[0], item[2][0]))


def main() -> None:
    args = parse_args()

    repoexec_dir = Path(args.repoexec_dir).resolve()
    prediction_dir = Path(args.prediction_dir).resolve()
    execution_dir = Path(args.execution_dir).resolve() if args.execution_dir else ensure_dir(prediction_dir / "execution")
    execution_dir = ensure_dir(execution_dir)
    pip_cache_dir = ensure_dir(execution_dir / ".pip_cache")
    docker_env = os.environ.copy()
    if args.docker_config_dir:
        docker_config_dir = ensure_dir(Path(args.docker_config_dir).resolve())
        docker_env["DOCKER_CONFIG"] = str(docker_config_dir)

    processed_path = prediction_dir / "processed_generations.jsonl"
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed generations file: {processed_path}")

    run_checked([args.docker_bin, "image", "inspect", args.docker_image], env=docker_env)

    tasks = list(iter_jsonl(processed_path))
    grouped_tasks = group_pending_tasks(tasks, repoexec_dir, execution_dir, args.overwrite)
    log_path = execution_dir / "docker_execution.log"

    with log_path.open("a", encoding="utf-8") as log_handle:
        for project, package_dir, task_ids in grouped_tasks:
            package_txt = package_dir / "package.txt"

            if not package_txt.exists():
                raise FileNotFoundError(f"Missing package.txt for project {project}: {package_txt}")

            command = [
                args.docker_bin,
                "run",
                "--rm",
                "-v",
                f"{prediction_dir}:/pred_dir:ro",
                "-v",
                f"{execution_dir}:/rs_dir",
                "-v",
                f"{repoexec_dir}:/input:ro",
                "-v",
                f"{(repoexec_dir / 'data_with_test_case')}:/output:ro",
                "-v",
                f"{package_dir}:/package:ro",
                "-v",
                f"{pip_cache_dir}:/tmp/pip_cache",
                "-e",
                "PIP_CACHE_DIR=/tmp/pip_cache",
                "--entrypoint",
                "/codegendata/run_project_batch.sh",
                args.docker_image,
                "/pred_dir/processed_generations.jsonl",
                "/rs_dir",
                str(args.timeout),
            ]
            command.extend(str(task_id) for task_id in task_ids)

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=docker_env,
            )
            log_handle.write(f"===== project {project} | tasks {task_ids[0]}-{task_ids[-1]} ({len(task_ids)}) =====\n")
            log_handle.write(process.stdout or "")
            log_handle.write(process.stderr or "")
            log_handle.flush()

            if process.returncode != 0:
                raise RuntimeError(
                    f"Docker execution failed for project {project}.\n"
                    f"Tasks: {task_ids}\n"
                    f"STDOUT:\n{process.stdout}\n"
                    f"STDERR:\n{process.stderr}"
                )

            for task_id in task_ids:
                result_file = execution_dir / f"results_{task_id}.jsonl"
                if not result_file.exists():
                    raise RuntimeError(f"Missing execution result after successful run: {result_file}")
                print(f"Executed task {task_id} -> {result_file}")


if __name__ == "__main__":
    main()
