"""Command-line interface for running stream graph pipelines."""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

from src.lib.cli_parse import parse_pipeline_expression
from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.opencv_qt import configure_opencv_qt_environment
from src.lib.pipeline import Pipeline
from src.lib.registry import default_registry, register_builtin_elements


def main(argv: list[str] | None = None) -> int:
    configure_opencv_qt_environment()
    register_builtin_elements()
    parser = argparse.ArgumentParser(prog="zpipe")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a pipeline expression")
    run_parser.add_argument(
        "expression",
        nargs="?",
        help="Pipeline expression, or a script file path when it exists",
    )
    run_parser.add_argument(
        "script_args",
        nargs="*",
        help="Arguments used to replace $1, $2, ... in a script file",
    )
    run_parser.add_argument(
        "-f",
        "--file",
        dest="expression_file",
        default=None,
        help="Read the pipeline expression from a script file",
    )
    run_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames per source",
    )

    list_parser = subparsers.add_parser("list-elements", help="List registered elements")
    list_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show descriptions and parameter names for every element",
    )

    describe_parser = subparsers.add_parser(
        "describe", help="Show details for one element"
    )
    describe_parser.add_argument("element", help="Element name")

    args = parser.parse_args(argv)
    if args.command == "list-elements":
        for name, element_cls in default_registry.items():
            contract = element_cls.contract()
            if args.verbose:
                print(_format_verbose_element(name, contract))
            else:
                print(f"{name}: {contract.description}")
        return 0
    if args.command == "describe":
        try:
            element_cls = default_registry.get(args.element)
        except KeyError:
            print(f"Unknown element {args.element!r}", file=sys.stderr)
            return 1
        print(_format_element_description(args.element, element_cls.contract()))
        return 0
    if args.command == "run":
        try:
            expression = _resolve_run_expression(
                args.expression, args.expression_file, args.script_args
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        spec = parse_pipeline_expression(expression)
        Pipeline.from_spec(spec).run(max_frames=args.max_frames)
        return 0

    parser.print_help()
    return 2


def _resolve_run_expression(
    expression: str | None, expression_file: str | None, script_args: list[str]
) -> str:
    if expression_file is not None:
        placeholder_args = (
            [expression] if expression is not None else []
        ) + script_args
        return _load_pipeline_expression(Path(expression_file), placeholder_args)

    if expression is None:
        raise ValueError("run requires a pipeline expression or --file path")

    expression_path = Path(expression)
    if expression_path.is_file():
        return _load_pipeline_expression(expression_path, script_args)

    if script_args:
        raise ValueError("Extra run arguments are only supported with script files")
    return expression


def _load_pipeline_expression(path: Path, args: list[str]) -> str:
    try:
        expression = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read pipeline script {str(path)!r}: {exc}") from exc
    return _substitute_placeholders(expression, args)


def _substitute_placeholders(expression: str, args: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index <= 0:
            raise ValueError("Pipeline script placeholders start at $1")
        try:
            value = args[index - 1]
        except IndexError as exc:
            raise ValueError(
                f"Missing value for pipeline script placeholder ${index}"
            ) from exc
        return shlex.quote(value)

    return re.sub(r"\$(\d+)", replace, expression)


def _format_verbose_element(name: str, contract: ElementContract) -> str:
    parameters = ", ".join(contract.parameters) if contract.parameters else "none"
    return f"{name}: {contract.description} params=[{parameters}]"


def _format_element_description(name: str, contract: ElementContract) -> str:
    lines = [
        f"Element: {name}",
        f"Description: {contract.description or '(none)'}",
        "",
        "Parameters:",
    ]
    if contract.parameters:
        for parameter in contract.parameters.values():
            lines.append(f"  {_format_parameter(parameter)}")
    else:
        lines.append("  none")

    lines.extend(["", "Input ports:"])
    lines.extend(_format_ports(contract.input_ports))
    lines.extend(["", "Output ports:"])
    lines.extend(_format_ports(contract.output_ports))

    rules = _format_rules(contract)
    if rules:
        lines.extend(["", "Compatibility rules:"])
        lines.extend(f"  {rule}" for rule in rules)
    return "\n".join(lines)


def _format_parameter(parameter: ParameterContract) -> str:
    required = "required" if parameter.required else "optional"
    parts = [f"{parameter.name}: {parameter.type_name}", required]
    if not parameter.required:
        parts.append(f"default={parameter.default}")
    if parameter.choices:
        choices = ", ".join(str(choice) for choice in parameter.choices)
        parts.append(f"choices=[{choices}]")
    if parameter.description:
        parts.append(parameter.description)
    return " | ".join(parts)


def _format_ports(ports: dict[str, PortContract]) -> list[str]:
    if not ports:
        return ["  none"]
    lines: list[str] = []
    for port in ports.values():
        constraints: list[str] = []
        if port.formats is not None:
            constraints.append(f"formats=[{', '.join(sorted(port.formats))}]")
        if port.depths is not None:
            constraints.append(f"depths=[{', '.join(str(depth) for depth in sorted(port.depths))}]")
        suffix = f" | {'; '.join(constraints)}" if constraints else ""
        lines.append(f"  {port.name}: {port.packet_type.__name__}{suffix}")
    return lines


def _format_rules(contract: ElementContract) -> list[str]:
    rules: list[str] = []
    if contract.require_same_size:
        rules.append("requires matching input width and height")
    if contract.require_same_format:
        rules.append("requires matching input formats")
    if contract.require_same_depth:
        rules.append("requires matching input depths")
    if contract.require_same_index:
        rules.append("requires matching input frame indexes")
    if contract.require_same_pts:
        rules.append("requires matching input timestamps")
    if contract.synchronized_inputs:
        rules.append("synchronizes multiple input streams")
    return rules


if __name__ == "__main__":
    sys.exit(main())
