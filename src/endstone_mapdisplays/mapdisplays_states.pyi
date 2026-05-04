"""
Type stubs for mapdisplays_states — a PyO3-based Rust extension module.

Provides display state management with frame decoding via ffmpeg,
exposing grayscale frames as NumPy arrays to Python.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

def add(left: int, right: int) -> int:
    """Add two unsigned 64-bit integers. (Test function.)"""
    ...

class PyDisplayState:
    """
    Base display state class.

    Holds a reference to an inner Rust ``DisplayState`` trait object.
    Subclassed by concrete states such as :class:`PyIdleState`.

    Not intended to be instantiated directly from Python.
    """

    def get_full_frame(self) -> tuple[NDArray[np.uint8], int]:
        """
        Return the latest decoded frame along with its frame ID.

        Returns
        -------
        frame : numpy.ndarray, shape (height, width), dtype uint8
            Grayscale pixel data.
        frame_id : int
            Monotonically increasing counter; incremented each time a new
            frame is written by the background thread.  Useful for detecting
            whether the frame has changed since the last call.
        """
        ...

    def stop(self) -> None:
        """
        Signal the background decoding thread to shut down.

        After calling this the state object should be considered unusable.
        Calling :meth:`get_full_frame` afterwards has undefined behaviour.
        """
        ...

class PyIdleState(PyDisplayState):
    """
    Idle display state — loops a video file via ffmpeg in a background thread.

    Inherits :class:`PyDisplayState` and exposes the same
    :meth:`get_full_frame` / :meth:`stop` interface.

    The video is decoded to 8-bit grayscale, scaled to ``width × height``,
    and written into a shared frame buffer at ~30 fps (33 ms per frame).
    When the video ends it restarts automatically until :meth:`stop` is called.

    Parameters
    ----------
    width : int
        Target frame width in pixels (must be > 0, fits in uint16).
    height : int
        Target frame height in pixels (must be > 0, fits in uint16).
    resource_path : str
        Path to the video file passed directly to ffmpeg's ``-i`` argument.
        Any container/codec that ffmpeg supports is accepted.

    Raises
    ------
    OSError
        If ffmpeg is not found on ``PATH`` or fails to spawn.

    Examples
    --------
    >>> state = PyIdleState(320, 240, "/path/to/idle.mp4")
    >>> frame, fid = state.get_full_frame()
    >>> frame.shape
    (240, 320)
    >>> frame.dtype
    dtype('uint8')
    >>> state.stop()
    """

    def __new__(
        cls,
        width: int,
        height: int,
        resource_path: str,
    ) -> PyIdleState: ...

    def get_full_frame(self) -> tuple[NDArray[np.uint8], int]:
        """
        Return the latest decoded frame along with its frame ID.

        Returns
        -------
        frame : numpy.ndarray, shape (height, width), dtype uint8
            Grayscale pixel data for the most recently decoded video frame.
        frame_id : int
            Frame counter; incremented once per decoded frame (~30 fps).
        """
        ...

    def stop(self) -> None:
        """
        Signal the background ffmpeg decode loop to exit.

        Equivalent to the base-class method; provided here for convenience
        so callers do not need to up-cast to :class:`PyDisplayState`.
        """
        ...
