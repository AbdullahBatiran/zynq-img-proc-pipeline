"""Element registry used by Python code and the CLI."""

from __future__ import annotations

from typing import Any

from .elements import Element


class ElementRegistry:
    def __init__(self) -> None:
        self._types: dict[str, type[Element]] = {}

    def register(self, element_cls: type[Element]) -> type[Element]:
        name = element_cls.type_name
        if name in self._types:
            raise ValueError(f"Element {name!r} is already registered")
        self._types[name] = element_cls
        return element_cls

    def create(
        self, type_name: str, instance_id: str, params: dict[str, Any] | None = None
    ) -> Element:
        try:
            element_cls = self._types[type_name]
        except KeyError as exc:
            raise KeyError(f"Unknown element type {type_name!r}") from exc
        return element_cls(instance_id=instance_id, params=params or {})

    def get(self, type_name: str) -> type[Element]:
        try:
            return self._types[type_name]
        except KeyError as exc:
            raise KeyError(f"Unknown element type {type_name!r}") from exc

    def names(self) -> list[str]:
        return sorted(self._types)

    def items(self) -> list[tuple[str, type[Element]]]:
        return sorted(self._types.items())


default_registry = ElementRegistry()
_builtins_registered = False


def register_builtin_elements(registry: ElementRegistry = default_registry) -> None:
    """Register built-in source, transformer, and sink elements once."""
    global _builtins_registered
    if registry is default_registry and _builtins_registered:
        return

    from src.sinks.displaysink import DisplaySink
    from src.sinks.filesink import FileSink
    from src.sources.filesrc import FileSource
    from src.transformers.bilateral import Bilateral
    from src.transformers.combine import Combine
    from src.transformers.debug import Debug
    from src.transformers.fan_out import FanOut
    from src.transformers.gaussian import Gaussian
    from src.transformers.hist_equalize import HistEqualize
    from src.transformers.laplacian_sharp import LaplacianSharp
    from src.transformers.linear_scale import LinearScale
    from src.transformers.median import Median
    from src.transformers.resize import Resize
    from src.transformers.text_overlay import TextOverlay
    from src.transformers.unsharp import Unsharp

    for element_cls in (
        FileSource,
        Resize,
        HistEqualize,
        LinearScale,
        Unsharp,
        Median,
        Gaussian,
        Bilateral,
        LaplacianSharp,
        Debug,
        FanOut,
        Combine,
        TextOverlay,
        FileSink,
        DisplaySink,
    ):
        if element_cls.type_name not in registry.names():
            registry.register(element_cls)

    if registry is default_registry:
        _builtins_registered = True
