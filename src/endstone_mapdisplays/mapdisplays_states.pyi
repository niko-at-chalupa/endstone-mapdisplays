"""
Type stubs for mapdisplays_states — a PyO3-based Rust extension module.

Provides high-performance display state management using direct FFmpeg 
bindings (FFI) for in-memory frame decoding and scaling.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class PyDisplayState:
    """
    Base display state class.

    Holds a reference to an inner Rust ``DisplayState`` trait object.
    Subclassed by concrete states such as :class:`PyIdleState`.
    """

    def get_full_frame(self) -> Tuple[NDArray[np.uint8], int]:
        """
        Return the latest decoded frame along with its frame ID.

        Returns
        -------
        frame : numpy.ndarray, shape (height, width, 3), dtype uint8
            The current RGB pixel data stored in the shared buffer.
        frame_id : int
            A 16-bit monotonically increasing counter. Increments 
            whenever a new frame is rendered to the buffer.
        """
        ...

    def stop(self) -> None:
        """
        Signal the background decoding thread to shut down.
        """
        ...

class PyIdleState(PyDisplayState):
    """
    Idle display state — decodes and loops a video file in a background thread.

    Uses a time-aware loop to synchronize playback with a target FPS. 
    If the video's native FPS is higher than the target, frames are 
    dropped before the expensive scaling and memory-copy steps.

    Parameters
    ----------
    width : int
        Target frame width in pixels (uint16).
    height : int
        Target frame height in pixels (uint16).
    video_path : str
        Path to the video file.
    target_fps : float
        The desired playback rate. If higher than the video's native
        FPS, the video will play at its native speed. If lower, 
        the background thread will downsample the video.

    Raises
    ------
    RuntimeError
        If the FFmpeg libraries fail to initialize, the file cannot 
        be found, or the video stream is invalid.
    """

    def __new__(
        cls,
        width: int,
        height: int,
        video_path: str,
        target_fps: float,
    ) -> PyIdleState: ...

    @property
    def target_fps(self) -> float:
        """
        The immutable target frames per second set during initialization.
        """
        ...

    def get_full_frame(self) -> Tuple[NDArray[np.uint8], int]:
        """
        Return the latest decoded RGB frame and its current ID.
        """
        ...

    def stop(self) -> None:
        """
        Signal the background thread to exit and stop decoding.
        """
        ...