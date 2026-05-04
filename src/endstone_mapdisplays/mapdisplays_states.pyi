"""
Type stubs for mapdisplays_states — a PyO3-based Rust extension module.

Provides display state management with frame decoding via ffmpeg,
exposing RGB frames as NumPy arrays to Python.
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
            Packed RGB pixel data.
        frame_id : int
            A 16-bit monotonically increasing counter (0-65535). 
            Increments each time a new frame is decoded.
        """
        ...

    def stop(self) -> None:
        """
        Signal the background decoding thread to shut down and 
        release system resources (FFmpeg processes and temp files).
        """
        ...

class PyIdleState(PyDisplayState):
    """
    Idle display state — loops a video file via ffmpeg in a background thread.

    The video is decoded to 8-bit RGB, scaled to ``width × height``,
    and written into a shared frame buffer. When the video ends, it 
    restarts automatically until :meth:`stop` is called.

    Parameters
    ----------
    width : int
        Target frame width in pixels (uint16).
    height : int
        Target frame height in pixels (uint16).
    video_path : str
        Path to the video file or resource identifier for ffmpeg.

    Raises
    ------
    RuntimeError
        If ffmpeg fails to probe the file, fails to resize, 
        or cannot be spawned.
    """

    def __new__(
        cls,
        width: int,
        height: int,
        video_path: str,
    ) -> PyIdleState: ...

    def get_full_frame(self) -> Tuple[NDArray[np.uint8], int]:
        """
        Return the latest decoded RGB frame and its ID.
        
        Returns
        -------
        frame : numpy.ndarray, shape (height, width, 3), dtype uint8
        frame_id : int
        """
        ...

    def stop(self) -> None:
        """
        Signal the background ffmpeg decode loop to exit.
        """
        ...