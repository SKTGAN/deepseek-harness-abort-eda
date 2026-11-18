"""Read candidate replies and explain failures from the official validator."""

import ast

from local_bench.program_safety import normalized_program
from local_tasks.tp_gnn_local_task_definitions import (
    DECODER_ARGS,
    DECODER_HELPERS,
    DECODER_TASK,
    LEGACY_LOSS_ARGS,
    LOSS_HELPERS,
    validate_program,
)


def model_text(response):
    if not isinstance(response, dict):
        raise ValueError("The model returned a malformed response")
    status = response.get("status")
    if status != "completed":
        details = response.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        usage = response.get("usage")
        token_count = usage.get("output_tokens") if isinstance(usage, dict) else None
        message = "The model returned an incomplete response"
        if isinstance(status, str):
            message += f" (status: {status[:40]})"
        if isinstance(reason, str):
            message += f", reason: {reason[:80]}"
        if type(token_count) is int:
            message += f", output tokens: {token_count}"
        if reason == "max_output_tokens":
            message += ". Return a shorter complete answer"
        raise ValueError(message)

    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("The completed model response is missing a valid output list")
    parts = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text":
                value = part.get("text")
                if isinstance(value, str):
                    parts.append(value)

    answer = "\n".join(parts).strip()
    if not answer:
        raise ValueError("The completed model response contained no output_text")
    return answer


def _python_source(text):
    lines = text.splitlines()
    fences = [index for index, line in enumerate(lines) if line.strip().startswith("```")]
    if not fences:
        return text
    try:
        ast.parse(text)
    except SyntaxError:
        pass
    else:
        # A valid raw program can contain backticks inside a docstring.
        return text

    if len(fences) == 2:
        start, end = fences
        language = lines[start].strip()[3:].strip().casefold()
        if language in {"", "python", "py"} and lines[end].strip() == "```":
            return "\n".join(lines[start + 1:end])
    raise ValueError("Return raw Python or exactly one complete Python code block")


def _binding_detail(source):
    tree = ast.parse(normalized_program(source))
    contracts = ((LEGACY_LOSS_ARGS, LOSS_HELPERS), (DECODER_ARGS, DECODER_HELPERS))
    for function, (arguments, helpers) in zip(tree.body, contracts):
        reserved = set(arguments) | helpers
        offenders = (
            node for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            and (node.id in reserved or node.id.startswith("_"))
        )
        offender = min(offenders, key=lambda node: (node.lineno, node.col_offset), default=None)
        if offender is not None:
            name = repr(offender.id[:60])
            return (f"{function.name} line {offender.lineno} assigns to {name}. "
                    "Use a fresh local name with no leading underscore.")
    return None


def generated_program(response):
    source = _python_source(model_text(response))
    try:
        return validate_program(source, DECODER_TASK)
    except ValueError as error:
        if str(error) == "Candidate cannot rebind inputs or trusted helpers":
            detail = _binding_detail(source)
            if detail:
                raise ValueError(f"{error}. {detail}") from error
        raise
