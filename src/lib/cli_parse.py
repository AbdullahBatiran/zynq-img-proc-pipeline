"""Small CLI expression parser for GStreamer-like pipeline strings."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from .pipeline import ConnectionSpec, ElementSpec, PipelineSpec


@dataclass
class _Segment:
    element_id: str
    element_type: str | None
    params: dict[str, Any]
    input_port: str = "in"
    output_port: str = "out"
    creates_element: bool = False


def parse_pipeline_expression(expression: str) -> PipelineSpec:
    """Parse a compact CLI pipeline expression into a PipelineSpec.

    Supports simple linear chains and named graph references such as
    ``ra.out ! combine.left name=c`` and ``rb.out ! c.right``.
    """
    elements: list[ElementSpec] = []
    connections: list[ConnectionSpec] = []
    known_ids: set[str] = set()
    auto_counts: dict[str, int] = {}

    for statement in _statements(expression):
        segments = [_parse_segment(raw, known_ids, auto_counts) for raw in statement]
        for segment in segments:
            if segment.creates_element and segment.element_id not in known_ids:
                elements.append(
                    ElementSpec(
                        id=segment.element_id,
                        type=segment.element_type or segment.element_id,
                        params=segment.params,
                    )
                )
                known_ids.add(segment.element_id)
        for left, right in zip(segments, segments[1:]):
            connections.append(
                ConnectionSpec(
                    left.element_id,
                    left.output_port,
                    right.element_id,
                    right.input_port,
                )
            )

    return PipelineSpec(elements=elements, connections=connections)


def _statements(expression: str) -> list[list[str]]:
    statements: list[list[str]] = []
    pending = ""
    for raw_line in expression.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        is_continuation_line = line.startswith("!")
        if pending and (line.startswith("!") or _ends_with_unquoted(pending, "!")):
            pending = f"{pending} {line}"
        else:
            _append_statement(statements, pending)
            pending = line
        if not is_continuation_line and _is_complete_statement(pending):
            _append_statement(statements, pending)
            pending = ""
    _append_statement(statements, pending)

    if not statements:
        _append_statement(statements, expression.strip())
    return statements


def _append_statement(statements: list[list[str]], statement: str) -> None:
    if not statement:
        return
    pieces = [piece.strip() for piece in _split_unquoted(statement, "!")]
    if len(pieces) >= 2 and pieces[0] and pieces[-1]:
        statements.append(pieces)


def _is_complete_statement(statement: str) -> bool:
    pieces = [piece.strip() for piece in _split_unquoted(statement, "!")]
    return len(pieces) >= 2 and bool(pieces[0]) and bool(pieces[-1])


def _ends_with_unquoted(text: str, separator: str) -> bool:
    pieces = _split_unquoted(text, separator)
    return len(pieces) >= 2 and pieces[-1].strip() == ""


def _parse_segment(
    raw: str, known_ids: set[str], auto_counts: dict[str, int]
) -> _Segment:
    tokens = shlex.split(raw)
    if not tokens:
        raise ValueError("Empty pipeline segment")
    head = tokens[0]
    params = _parse_params(tokens[1:])
    explicit_name = params.pop("name", None)

    if "." in head:
        base, port = head.split(".", 1)
        if explicit_name is not None:
            return _Segment(
                element_id=str(explicit_name),
                element_type=base,
                params=params,
                input_port=port,
                output_port="out",
                creates_element=True,
            )
        if base in known_ids:
            return _Segment(
                element_id=base,
                element_type=None,
                params={},
                input_port=port,
                output_port=port,
                creates_element=False,
            )
        return _Segment(
            element_id=base,
            element_type=None,
            params={},
            input_port=port,
            output_port=port,
            creates_element=False,
        )

    element_type = head
    element_id = str(explicit_name) if explicit_name is not None else _auto_id(
        element_type, auto_counts
    )
    return _Segment(
        element_id=element_id,
        element_type=element_type,
        params=params,
        creates_element=True,
    )


def _auto_id(element_type: str, auto_counts: dict[str, int]) -> str:
    count = auto_counts.get(element_type, 0)
    auto_counts[element_type] = count + 1
    if count == 0:
        return element_type
    return f"{element_type}_{count}"


def _parse_params(tokens: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Expected key=value parameter, got {token!r}")
        key, value = token.split("=", 1)
        params[key] = _coerce_value(value)
    return params


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _split_unquoted(text: str, separator: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for char in text:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
            current.append(char)
            continue
        if char == separator and quote is None:
            pieces.append("".join(current))
            current = []
            continue
        current.append(char)
    pieces.append("".join(current))
    return pieces
