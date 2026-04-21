from __future__ import annotations

from typing import Sequence


COLOR_POOL: list[str] = [
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "cyan",
    "pink",
    "black",
    "white",
]

COLOR_RGB: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 0.8, 0.0),      # Standard visible green
    "blue": (0.0, 0.0, 1.0),       # Pure blue
    "yellow": (1.0, 1.0, 0.0),     # Pure bright yellow
    "orange": (1.0, 0.5, 0.0),     # Distinct mid-orange
    "purple": (0.6, 0.0, 0.8),     # High-contrast purple
    "cyan": (0.0, 1.0, 1.0),       # Distinct from blue
    "pink": (1.0, 0.4, 0.7),       # Light warm pink
    "black": (0.15, 0.15, 0.15),   # Dark enough to be black, light enough to render safely
    "white": (0.95, 0.95, 0.95),   # Standard white
}


def color_names() -> Sequence[str]:
    return tuple(COLOR_POOL)


def rgba_for_color(name: str, *, alpha: float = 1.0) -> list[float]:
    rgb = COLOR_RGB.get(str(name).lower(), (0.6, 0.6, 0.6))
    return [float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha)]
