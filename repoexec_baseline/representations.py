from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any


SUPPORTED_REPRESENTATIONS = ("raw", "ast", "reduced_ast")


@dataclass(frozen=True)
class PromptParts:
    context: str
    target: str


def split_prompt(example: dict[str, Any]) -> PromptParts:
    prompt = example["prompt"]
    target = example["target_function_prompt"]
    entry_point = example["entry_point"]
    header_re = re.compile(rf"(?m)^(?:async\s+def|def)\s+{re.escape(entry_point)}\s*\(")
    match = header_re.search(prompt)
    if match is not None:
        boundary = match.start()
        target = prompt[boundary:]
    elif target in prompt:
        boundary = prompt.index(target)
    else:
        raise ValueError(f"Target function prompt not found in prompt for task {example.get('id')}.")
    return PromptParts(context=prompt[:boundary], target=target)


def build_prompt(example: dict[str, Any], representation: str) -> str:
    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(f"Unsupported representation: {representation}")

    if representation == "raw":
        return example["prompt"]

    parts = split_prompt(example)
    context = parts.context.strip()
    if not context:
        return parts.target

    if representation == "ast":
        rendered_context = render_ast_context(context)
    else:
        rendered_context = render_reduced_ast_context(context)

    if not rendered_context:
        return parts.target

    return f"{rendered_context.rstrip()}\n\n{parts.target}"


def render_ast_context(context: str) -> str:
    blocks: list[str] = []
    for block in _split_top_level_blocks(context):
        try:
            module = ast.parse(block)
        except SyntaxError:
            rendered = _render_unparsed_block(block, reduced=False)
            if rendered:
                blocks.append(rendered)
            continue

        for node in module.body:
            rendered = _render_node(node, reduced=False)
            if rendered:
                blocks.append(rendered)
    return "\n\n".join(blocks)


def render_reduced_ast_context(context: str) -> str:
    blocks: list[str] = []
    for block in _split_top_level_blocks(context):
        try:
            module = ast.parse(block)
        except SyntaxError:
            rendered = _render_unparsed_block(block, reduced=True)
            if rendered:
                blocks.append(rendered)
            continue

        for node in module.body:
            rendered = _render_node(node, reduced=True)
            if rendered:
                blocks.append(rendered)
    return "\n".join(blocks)


def _split_top_level_blocks(context: str) -> list[str]:
    lines = context.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    paren_balance = 0

    for line in lines:
        stripped = line.strip()
        is_top_level = bool(stripped) and line == line.lstrip()
        starts_declaration = stripped.startswith(("def ", "async def ", "class "))
        previous_is_decorator = any(existing.strip() for existing in current) and next(
            existing.strip() for existing in reversed(current) if existing.strip()
        ).startswith("@")

        should_start_new_block = (
            current
            and is_top_level
            and paren_balance <= 0
            and not (previous_is_decorator and starts_declaration)
        )
        if should_start_new_block:
            blocks.append(current)
            current = []

        current.append(line)
        paren_balance += _paren_delta(line)

    if current:
        blocks.append(current)

    return ["\n".join(block).strip("\n") for block in blocks if any(line.strip() for line in block)]


def _paren_delta(line: str) -> int:
    return sum(line.count(ch) for ch in "([{") - sum(line.count(ch) for ch in ")]}")


def _render_unparsed_block(block: str, reduced: bool) -> str:
    headline = next((line.strip() for line in block.splitlines() if line.strip()), "")
    if not headline:
        return ""
    if len(headline) > 120:
        headline = headline[:117] + "..."
    if reduced:
        return f"UnparsedBlock[head={headline!r}]"
    return f"UnparsedBlock\n  Head: {headline!r}"


def _render_node(node: ast.AST, reduced: bool) -> str:
    if isinstance(node, ast.Import):
        names = ", ".join(_format_alias(alias) for alias in node.names)
        return f"Import[{names}]"

    if isinstance(node, ast.ImportFrom):
        module = "." * node.level + (node.module or "")
        names = ", ".join(_format_alias(alias) for alias in node.names)
        return f"ImportFrom[module={module or '<relative>'}; names=[{names}]]"

    if isinstance(node, ast.Assign):
        targets = ", ".join(ast.unparse(target) for target in node.targets)
        if reduced:
            return f"Assign[target={targets}]"
        return f"Assign[target={targets}; value={ast.unparse(node.value)}]"

    if isinstance(node, ast.AnnAssign):
        target = ast.unparse(node.target)
        annotation = ast.unparse(node.annotation)
        if reduced:
            return f"AnnAssign[target={target}; annotation={annotation}]"
        value = ast.unparse(node.value) if node.value is not None else "<none>"
        return f"AnnAssign[target={target}; annotation={annotation}; value={value}]"

    if isinstance(node, ast.ClassDef):
        return _render_class(node, reduced)

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _render_function(node, reduced)

    expr = ast.unparse(node).strip()
    node_type = type(node).__name__
    return f"{node_type}[{expr}]"


def _render_class(node: ast.ClassDef, reduced: bool) -> str:
    bases = [ast.unparse(base) for base in node.bases]
    decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
    methods = [child for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assignments = [child for child in node.body if isinstance(child, (ast.Assign, ast.AnnAssign))]

    if reduced:
        method_names = ", ".join(method.name for method in methods)
        assignment_targets = ", ".join(_collect_assignment_targets(assignments))
        details: list[str] = [f"name={node.name}"]
        if bases:
            details.append(f"bases=[{', '.join(bases)}]")
        if method_names:
            details.append(f"methods=[{method_names}]")
        if assignment_targets:
            details.append(f"attrs=[{assignment_targets}]")
        return f"ClassDef[{'; '.join(details)}]"

    lines = [f"ClassDef name={node.name}"]
    if bases:
        lines.append(f"  Bases: {', '.join(bases)}")
    if decorators:
        lines.append(f"  Decorators: {', '.join(decorators)}")
    docstring = ast.get_docstring(node)
    if docstring:
        lines.append(f"  Docstring: {docstring!r}")
    for assignment in assignments:
        lines.append(f"  {_render_node(assignment, reduced=False)}")
    for method in methods:
        method_lines = _render_function(method, reduced=False).splitlines()
        lines.extend(f"  {line}" for line in method_lines)
    return "\n".join(lines)


def _render_function(node: ast.FunctionDef | ast.AsyncFunctionDef, reduced: bool) -> str:
    args = _format_args(node)
    returns = ast.unparse(node.returns) if node.returns is not None else None
    decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
    docstring = ast.get_docstring(node)
    calls = sorted(_collect_calls(node))
    assigns = sorted(_collect_assigned_names(node))

    if reduced:
        details = [f"name={node.name}", f"args=[{', '.join(args)}]"]
        if returns:
            details.append(f"returns={returns}")
        if decorators:
            details.append(f"decorators=[{', '.join(decorators)}]")
        if calls:
            details.append(f"calls=[{', '.join(calls)}]")
        if assigns:
            details.append(f"assigns=[{', '.join(assigns)}]")
        if docstring:
            details.append(f"doc={docstring!r}")
        return f"{type(node).__name__}[{'; '.join(details)}]"

    lines = [f"{type(node).__name__} name={node.name}"]
    lines.append(f"  Args: {', '.join(args) if args else '<none>'}")
    if returns:
        lines.append(f"  Returns: {returns}")
    if decorators:
        lines.append(f"  Decorators: {', '.join(decorators)}")
    if docstring:
        lines.append(f"  Docstring: {docstring!r}")
    if calls:
        lines.append(f"  Calls: {', '.join(calls)}")
    if assigns:
        lines.append(f"  Assigns: {', '.join(assigns)}")
    body_nodes = [child for child in node.body if not _is_docstring_expr(child)]
    for child in body_nodes[:12]:
        statement = ast.unparse(child).strip().replace("\n", " ")
        lines.append(f"  Body: {statement}")
    if len(body_nodes) > 12:
        lines.append(f"  Body: ... ({len(body_nodes) - 12} more statements)")
    return "\n".join(lines)


def _format_alias(alias: ast.alias) -> str:
    return alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"


def _format_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args: list[str] = []
    positional = list(getattr(node.args, "posonlyargs", [])) + list(node.args.args)
    for arg in positional:
        args.append(_format_arg(arg))
    if node.args.vararg:
        args.append("*" + _format_arg(node.args.vararg))
    for arg in node.args.kwonlyargs:
        args.append(_format_arg(arg))
    if node.args.kwarg:
        args.append("**" + _format_arg(node.args.kwarg))
    return args


def _format_arg(arg: ast.arg) -> str:
    if arg.annotation is None:
        return arg.arg
    return f"{arg.arg}: {ast.unparse(arg.annotation)}"


def _collect_calls(node: ast.AST) -> set[str]:
    calls: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            try:
                calls.add(ast.unparse(child.func))
            except Exception:
                continue
    return calls


def _collect_assigned_names(node: ast.AST) -> set[str]:
    assigned: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                assigned.update(_collect_target_names(target))
        elif isinstance(child, ast.AnnAssign):
            assigned.update(_collect_target_names(child.target))
    return assigned


def _collect_assignment_targets(nodes: list[ast.Assign | ast.AnnAssign]) -> list[str]:
    output: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                output.extend(sorted(_collect_target_names(target)))
        else:
            output.extend(sorted(_collect_target_names(node.target)))
    return output


def _collect_target_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            names.update(_collect_target_names(element))
    return names


def _is_docstring_expr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(getattr(node, "value", None), ast.Constant)
        and isinstance(node.value.value, str)
    )
