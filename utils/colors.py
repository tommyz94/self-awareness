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
    "magenta",
    "lime",
    "teal",
    "pink",
    "brown",
    "gray",
    "white",
    "navy",
]

COLOR_RGB: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 0.8, 0.0),
    "blue": (0.0, 0.3, 1.0),
    "yellow": (1.0, 0.85, 0.1),
    "orange": (1.0, 0.55, 0.0),
    "purple": (0.55, 0.25, 0.75),
    "cyan": (0.0, 0.85, 0.85),
    "magenta": (0.85, 0.1, 0.75),
    "lime": (0.65, 0.9, 0.1),
    "teal": (0.0, 0.55, 0.5),
    "pink": (0.95, 0.55, 0.75),
    "brown": (0.55, 0.35, 0.2),
    "gray": (0.55, 0.55, 0.55),
    "white": (0.95, 0.95, 0.95),
    "navy": (0.1, 0.2, 0.55),
}


def color_names() -> Sequence[str]:
    return tuple(COLOR_POOL)


def rgba_for_color(name: str, *, alpha: float = 1.0) -> list[float]:
    rgb = COLOR_RGB.get(str(name).lower(), (0.6, 0.6, 0.6))
    return [float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha)]
