"""Qt environment defaults for OpenCV HighGUI."""

from __future__ import annotations

import os
from pathlib import Path


_FONT_DIR_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype"),
    Path("/usr/share/fonts/opentype"),
    Path("/usr/share/fonts"),
)


def configure_opencv_qt_environment() -> None:
    """Reduce noisy Qt warnings from OpenCV's bundled HighGUI backend."""
    if (
        os.environ.get("XDG_SESSION_TYPE") == "wayland"
        and "QT_QPA_PLATFORM" not in os.environ
    ):
        # The OpenCV wheel only bundles xcb in this environment. Setting xcb
        # explicitly avoids Qt warning that it ignored Gnome's Wayland session.
        os.environ["QT_QPA_PLATFORM"] = "xcb"

    font_dir = os.environ.get("QT_QPA_FONTDIR")
    if font_dir and Path(font_dir).is_dir():
        return

    for candidate in _FONT_DIR_CANDIDATES:
        if candidate.is_dir():
            os.environ["QT_QPA_FONTDIR"] = str(candidate)
            return
