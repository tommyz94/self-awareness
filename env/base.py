"""
Abstract base environment for the mirror self-recognition simulator.

The MuJoCo backend implements this interface.
The Controller uses only the methods defined here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from experiments.specs.types import RunSetup
    from env.observation import ObservationMetadata

# (x, z) cube slot positions — backend-agnostic
CUBE_SLOTS = [
    (-10.0, -4.0), (-8.0, -4.0), (-6.0, -4.0),
    (-10.0, -2.0), (-8.0, -2.0), (-6.0, -2.0),
]


class BaseEnv(ABC):

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self, run_setup: RunSetup) -> None:
        """Reset robot pose/color and scene objects for a new run."""
        ...

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def position(self) -> Tuple[float, float]:
        """Current (x, z) of the robot."""
        ...

    @property
    @abstractmethod
    def direction(self) -> str:
        """Cardinal direction the robot faces: 'N', 'E', 'S', 'W'."""
        ...

    @property
    @abstractmethod
    def object_positions_2d(self) -> dict:
        """{'red': (x,z), ...} for all named objects currently in the scene."""
        ...

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, action: str) -> Tuple[float, float, str, bool]:
        """
        Execute one action (w/a/s/d).
        Returns (x, z, direction, bumped_wall).
        """
        ...

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @abstractmethod
    def capture_image(self, save_path: str) -> str:
        """Render the current first- or third-person view to a JPEG. Returns actual path."""
        ...

    @abstractmethod
    def get_observation_metadata(self) -> "ObservationMetadata":
        """Return metadata describing the next frame that capture_image() will render."""
        ...

    # ------------------------------------------------------------------
    # Per-frame update (physics, mirror sync, camera)
    # ------------------------------------------------------------------

    def update_frame(self):
        """Called every iteration. Returns False to signal the loop should stop (e.g. window closed). Default: no-op."""

    # ------------------------------------------------------------------
    # Rate limiter
    # ------------------------------------------------------------------

    def tick(self, fps: int = 30) -> None:
        """Frame-rate limiter. Default: no-op (runs as fast as possible)."""

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def set_caption(self, text: str) -> None:
        """Overlay text. Default: no-op."""

    # ------------------------------------------------------------------
    # Human navigation
    # ------------------------------------------------------------------

    def bind_keys(self, callback) -> None:
        """Bind keyboard input. Default: no-op."""

    def jump(self) -> None:
        """Optional jump action. Default: no-op."""

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources. Default: no-op."""
