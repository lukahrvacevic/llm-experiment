from __future__ import annotations

import argparse
import os
from pathlib import Path

from repoexec_baseline.common import DEFAULT_DATASET, ensure_dir, load_repoexec_dataset
from repoexec_baseline.representations import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview RepoExec prompt representations for selected tasks.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subset", default="full_context", choices=["full_context", "medium_context", "small_context"])
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 63, 166])
    parser.add_argument("--output-dir", default="runs/representation-previews")
    return parser.parse_args()


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

    dataset = load_repoexec_dataset(args.dataset, args.subset, hf_home)

    for task_id in args.task_ids:
        example = dataset[task_id]
        file_name = f"task_{task_id:03d}_{example['entry_point']}.md"
        output_path = output_dir / file_name

        raw_prompt = build_prompt(example, "raw")
        ast_prompt = build_prompt(example, "ast")
        reduced_ast_prompt = build_prompt(example, "reduced_ast")

        body = "\n".join(
            [
                f"# Task {task_id}: {example['entry_point']}",
                "",
                f"- Project: `{example['project']}`",
                f"- Module: `{example['module']}`",
                f"- Cross context: `{bool(example.get('cross_context', False))}`",
                "",
                "## Target Function Prompt",
                "```python",
                example["target_function_prompt"].rstrip(),
                "```",
                "",
                "## Raw Prompt",
                "```python",
                raw_prompt.rstrip(),
                "```",
                "",
                "## AST Prompt",
                "```text",
                ast_prompt.rstrip(),
                "```",
                "",
                "## Reduced AST Prompt",
                "```text",
                reduced_ast_prompt.rstrip(),
                "```",
                "",
            ]
        )
        output_path.write_text(body, encoding="utf-8")
        print(f"[ok] task={task_id} entry_point={example['entry_point']} -> {output_path}")


if __name__ == "__main__":
    main()
