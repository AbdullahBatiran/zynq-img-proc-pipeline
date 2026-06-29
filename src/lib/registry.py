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
    from src.transformers.bit_shift import BitShift
    from src.transformers.bilateral import Bilateral
    from src.transformers.clahe import Clahe
    from src.transformers.bypass import Bypass
    from src.transformers.combine import Combine
    from src.transformers.debug import Debug
    from src.transformers.dtype_convert import DtypeConvert
    from src.transformers.deflicker import Deflicker
    from src.transformers.dog import Dog
    from src.transformers.edge_enhance import EdgeEnhance
    from src.transformers.fan_out import FanOut
    from src.transformers.frame_diff_debug import FrameDiffDebug
    from src.transformers.gaussian import Gaussian
    from src.transformers.guided_filter import GuidedFilter
    from src.transformers.hist_equalize import HistEqualize
    from src.transformers.interlace_mimic_test import InterlaceMimicTest
    from src.transformers.laplacian_sharp import LaplacianSharp
    from src.transformers.linear_scale import LinearScale
    from src.transformers.local_contrast import LocalContrast
    from src.transformers.log_filter import LogFilter
    from src.transformers.meam import Meam
    from src.transformers.median import Median
    from src.transformers.morphology import Morphology
    from src.transformers.nl_means import NlMeans
    from src.transformers.progress import Progress
    from src.transformers.retinex import Retinex
    from src.transformers.mono_to_color import MonoToColor
    from src.transformers.resize import Resize
    from src.transformers.text_overlay import TextOverlay
    from src.transformers.rolling_background import RollingBackground
    from src.transformers.sharpen_kernel import SharpenKernel
    from src.transformers.temporal_denoise import TemporalDenoise
    from src.transformers.tone_curve import ToneCurve
    from src.transformers.unsharp import Unsharp
    from src.transformers.tv_denoise import TvDenoise
    from src.transformers.wavelet_denoise import WaveletDenoise

    for element_cls in (
        FileSource,
        BitShift,
        Resize,
        HistEqualize,
        LinearScale,
        Clahe,
        Meam,
        ToneCurve,
        Retinex,
        LocalContrast,
        RollingBackground,
        Unsharp,
        Median,
        Gaussian,
        Bilateral,
        LaplacianSharp,
        SharpenKernel,
        Morphology,
        Dog,
        LogFilter,
        EdgeEnhance,
        GuidedFilter,
        NlMeans,
        TvDenoise,
        WaveletDenoise,
        DtypeConvert,
        Bypass,
        Debug,
        Progress,
        FrameDiffDebug,
        InterlaceMimicTest,
        FanOut,
        TemporalDenoise,
        Deflicker,
        Combine,
        MonoToColor,
        TextOverlay,
        FileSink,
        DisplaySink,
    ):
        if element_cls.type_name not in registry.names():
            registry.register(element_cls)

    if registry is default_registry:
        _builtins_registered = True
