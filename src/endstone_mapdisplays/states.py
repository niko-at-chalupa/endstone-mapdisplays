from abc import ABC, abstractmethod
import time
import av
from importlib.resources import files
import threading
import numpy as np
from endstone import Logger
from enum import Enum
import yt_dlp
import cv2


class DisplayState(ABC):
    @abstractmethod
    def get_full_frame(self) -> tuple[np.ndarray, int]: ...


class IdleState(DisplayState):
    def __init__(
        self,
        width: int,
        height: int,
        logger: Logger,
        resource_path: str = "resources/mapdisplays_idle.webm",
    ) -> None:
        self.width = width
        self.height = height
        self.logger = logger
        self._current_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._frame_id = 0
        self._running = True
        self._thread = threading.Thread(target=self._decode_loop, daemon=True)
        self.resource_path = resource_path
        self._thread.start()

    def _decode_loop(self):
        try:
            resource_path = files("endstone_mapdisplays").joinpath(self.resource_path)
            while self._running:
                with av.open(str(resource_path)) as container:
                    stream = container.streams.video[0]
                    fps = float(stream.average_rate)  # type:ignore
                    frame_time = 1.0 / fps if fps > 0 else 0.033

                    for frame in container.decode(video=0):
                        if not self._running:
                            break

                        start_time = time.perf_counter()
                        img = frame.to_ndarray(format="rgb24")
                        resized = cv2.resize(
                            img,
                            (self.width, self.height),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        self._current_frame = resized
                        self._frame_id += 1

                        elapsed = time.perf_counter() - start_time
                        sleep_time = max(0, frame_time - elapsed)
                        time.sleep(sleep_time)

        except Exception:
            self._current_frame[:, :, :] = 40

    def get_full_frame(self) -> tuple[np.ndarray, int]:
        return self._current_frame, self._frame_id

    def stop(self):
        self._running = False