"""Core stream graph framework."""

from .packets import FrameMetadata, FramePacket
from .pipeline import ConnectionSpec, ElementSpec, Pipeline, PipelineSpec
from .registry import default_registry, register_builtin_elements

__all__ = [
    "ConnectionSpec",
    "ElementSpec",
    "FrameMetadata",
    "FramePacket",
    "Pipeline",
    "PipelineSpec",
    "default_registry",
    "register_builtin_elements",
]

