"""Command-line interface for running stream graph pipelines."""

from __future__ import annotations

import argparse
import sys

from src.lib.cli_parse import parse_pipeline_expression
from src.lib.pipeline import Pipeline
from src.lib.registry import default_registry, register_builtin_elements


def main(argv: list[str] | None = None) -> int:
    register_builtin_elements()
    parser = argparse.ArgumentParser(prog="zpipe")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a pipeline expression")
    run_parser.add_argument("expression", help="Pipeline expression")
    run_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames per source",
    )

    subparsers.add_parser("list-elements", help="List registered elements")

    args = parser.parse_args(argv)
    if args.command == "list-elements":
        for name, element_cls in default_registry.items():
            print(f"{name}: {element_cls.contract().description}")
        return 0
    if args.command == "run":
        spec = parse_pipeline_expression(args.expression)
        Pipeline.from_spec(spec).run(max_frames=args.max_frames)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
