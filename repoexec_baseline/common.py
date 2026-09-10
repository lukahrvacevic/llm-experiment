from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Iterator


DEFAULT_DATASET = "Fsoft-AIC/RepoExec"
DEFAULT_SUBSET = "full_context"


def load_repoexec_dataset(dataset_name: str, subset: str, hf_home: Path):
    local_cache_root = hf_home / "datasets" / "Fsoft-AIC___repo_exec" / "default" / "0.0.0"
    arrow_matches = sorted(local_cache_root.glob(f"*/repo_exec-{subset}.arrow"))
    if arrow_matches:
        from datasets import Dataset

        return Dataset.from_file(str(arrow_matches[-1]))

    from datasets import load_dataset

    return load_dataset(dataset_name, split=subset, cache_dir=str(hf_home))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_mean(values: Iterable[float]) -> float | None:
    materialized = [value for value in values if value is not None]
    if not materialized:
        return None
    return float(mean(materialized))


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float | None:
    if num_samples < k:
        return None
    if num_samples - num_correct < k:
        return 1.0

    estimate = 1.0
    for value in range(num_samples - num_correct + 1, num_samples + 1):
        estimate *= 1.0 - (k / value)
    return 1.0 - estimate


def get_source_segment(code: str, node: ast.AST) -> str | None:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return None

    lines = code.splitlines(keepends=True)
    start_line = node.lineno - 1
    end_line = node.end_lineno - 1
    start_col = getattr(node, "col_offset", 0)
    end_col = getattr(node, "end_col_offset", None)

    if start_line == end_line:
        return lines[start_line][start_col:end_col]

    first = lines[start_line][start_col:]
    middle = lines[start_line + 1:end_line]
    last = lines[end_line][:end_col]
    return "".join([first, *middle, last])


def get_function_source(code: str, entry_point: str) -> str | None:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            source = get_source_segment(code, node)
            if source:
                return source.strip()
    return None


def get_actual_solution(example: dict[str, Any]) -> str:
    solution = example["solution"]
    check = example["check"]
    if solution in check:
        return solution

    extracted = get_function_source(check, example["entry_point"])
    if extracted is None:
        raise ValueError(f"Cannot extract ground-truth function for task {example.get('id', 'unknown')}.")
    return extracted


def _slice_target_function_block(code: str, entry_point: str) -> str:
    lines = code.splitlines(keepends=True)
    header_re = re.compile(rf"^(async\s+def|def)\s+{re.escape(entry_point)}\s*\(")
    boundary_prefixes = ("def ", "async def ", "class ", "@", "```")

    start_line: int | None = None
    for index, line in enumerate(lines):
        if header_re.match(line):
            start_line = index
            break

    if start_line is None:
        return ""

    end_line = len(lines)
    for index in range(start_line + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped:
            continue

        is_top_level = lines[index] == lines[index].lstrip()
        if is_top_level and stripped.startswith(boundary_prefixes):
            end_line = index
            break

    return "".join(lines[start_line:end_line]).strip()


def extract_solution_function(
    generation_text: str,
    target_function_prompt: str,
    entry_point: str,
) -> str:
    extracted = get_function_source(generation_text, entry_point)
    if extracted:
        return extracted.strip()

    stripped_prompt = target_function_prompt.strip()
    if stripped_prompt and stripped_prompt in generation_text:
        candidate = generation_text[generation_text.index(stripped_prompt):]
        sliced = _slice_target_function_block(candidate, entry_point)
        if sliced:
            return sliced

    return _slice_target_function_block(generation_text, entry_point)


def get_function_body(function_text: str, target_function_prompt: str) -> str:
    stripped_prompt = target_function_prompt.strip()
    stripped_function = function_text.strip()
    if not stripped_function:
        return ""

    if stripped_prompt in stripped_function:
        return stripped_function.replace(stripped_prompt, "", 1).strip()

    function_lines = stripped_function.splitlines()
    prompt_lines = stripped_prompt.splitlines()
    if len(function_lines) > len(prompt_lines):
        return "\n".join(function_lines[len(prompt_lines):]).strip()
    return stripped_function


def _collect_target_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            names.update(_collect_target_names(element))
    return names


class IdentifierCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.identifiers: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.identifiers.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.identifiers.add(node.attr)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.identifiers.add(node.name)
        for arg in node.args.args + node.args.kwonlyargs:
            self.identifiers.add(arg.arg)
        if node.args.vararg:
            self.identifiers.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.identifiers.add(node.args.kwarg.arg)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.identifiers.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.identifiers.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.identifiers.add(alias.asname or alias.name)


def extract_identifiers_from_code(code: str) -> set[str]:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return set()

    collector = IdentifierCollector()
    collector.visit(module)
    return collector.identifiers


def extract_top_level_dependencies(prompt_code: str, entry_point: str, solution_body: str) -> set[str]:
    solution_identifiers = extract_identifiers_from_code(solution_body)
    if not solution_identifiers:
        return set()

    try:
        prompt_module = ast.parse(prompt_code)
    except SyntaxError:
        return set()

    dependencies: set[str] = set()

    for node in prompt_module.body:
        candidates: set[str] = set()
        if isinstance(node, ast.Import):
            for alias in node.names:
                candidates.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                candidates.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            candidates.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                candidates.update(_collect_target_names(target))
        elif isinstance(node, ast.AnnAssign):
            candidates.update(_collect_target_names(node.target))

        for candidate in candidates:
            if candidate != entry_point and candidate in solution_identifiers:
                dependencies.add(candidate)

    return dependencies


def compute_dir_for_prediction(example: dict[str, Any], prediction: str) -> float | None:
    solution_body = get_function_body(example["solution"], example["target_function_prompt"])
    prediction_body = get_function_body(prediction, example["target_function_prompt"])
    dependencies = extract_top_level_dependencies(example["prompt"], example["entry_point"], solution_body)
    if not dependencies:
        return None

    prediction_identifiers = extract_identifiers_from_code(prediction_body)
    return len(dependencies.intersection(prediction_identifiers)) / len(dependencies)
