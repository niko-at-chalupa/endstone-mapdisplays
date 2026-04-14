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
    def get_full_frame(self) -> tuple[np.ndarray, int]:
        ...

class IdleState(DisplayState):
    def __init__(self, width: int, height: int, logger: Logger, resource_path: str = "resources/mapdisplays_idle.webm") -> None:
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
                    fps = float(stream.average_rate) #type:ignore
                    frame_time = 1.0 / fps if fps > 0 else 0.033
                    
                    for frame in container.decode(video=0):
                        if not self._running:
                            break
                        
                        start_time = time.perf_counter()
                        img = frame.to_ndarray(format="rgb24")
                        resized = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
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

class YoutubeState(DisplayState):
    def __init__(self, width: int, height: int, logger: Logger, url: str = "https://youtu.be/NEUCpotovEc?si=VDWECOWTVxFOWkOK"):
        self.width = width
        self.height = height
        self.logger = logger
        self.url = url
        self._current_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._frame_id = 0
        self._running = True
        self._substate = self.Substate.IDLE
        self._stream_url = None
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._video_loop, daemon=True)
        self._thread.start()

        self._loader_thread = threading.Thread(target=self._load_url, daemon=True)
        self._loader_thread.start()

    class Substate(Enum):
        IDLE = 0
        LOADING = 1
        PLAYING = 2

    def _load_url(self):
        try:
            self._substate = self.Substate.LOADING
            ydl_opts = {
                "format": f"bestvideo[height<={self.height}][width<={self.width}][ext=mp4]/bestvideo[height<=144][ext=mp4]/best[height<=144]",
                "quiet": True,
                "no_warnings": True,
                "cookiesfrombrowser": ("firefox",),
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                self._stream_url = info.get("url")
        except Exception as e:
            self.logger.error(f"failed to get youtube stream: {e}")
            self._substate = self.Substate.IDLE

    def _video_loop(self):
        idle = IdleState(self.width, self.height, self.logger, "resources/mapdisplays_loading.webm")

        while self._running:
            if not self._stream_url:
                frame, fid = idle.get_full_frame()
                with self._lock:
                    self._current_frame = frame
                    self._frame_id = fid
                time.sleep(0.05)
                continue

            idle.stop()

            try:
                self._substate = self.Substate.PLAYING
                options = {"fflags": "nobuffer", "flags": "low_delay", "analyzeduration": "0", "probesize": "32768"}
                with av.open(self._stream_url, options=options) as container:
                    stream = container.streams.video[0]
                    stream.thread_type = "AUTO"
                    stream.codec_context.width = self.width
                    stream.codec_context.height = self.height

                    fps = float(stream.average_rate) if stream.average_rate else 20.0
                    source_frame_time = 1.0 / fps
                    deadline = time.perf_counter()

                    for packet in container.demux(stream):
                        if not self._running:
                            break
                        try:
                            frames = list(packet.decode())
                            if not frames:
                                continue

                            now = time.perf_counter()
                            if now > deadline:
                                dropped = len(frames) - 1
                                frames = [frames[-1]]
                                deadline += dropped * source_frame_time

                            for frame in frames:
                                if not self._running:
                                    break
                                img = frame.to_ndarray(format="rgb24", width=self.width, height=self.height)
                                with self._lock:
                                    self._current_frame = img
                                    self._frame_id += 1

                            deadline += 1.0 / 20
                            now = time.perf_counter()
                            gap = deadline - now
                            if gap > 0:
                                time.sleep(gap)
                            else:
                                deadline = now
                        except Exception as e:
                            if "Invalid data found when processing input" in str(e):
                                continue
                            self.logger.warning(f"bad packet skipped: {e}")
                            continue
            except Exception as e:
                self.logger.error(f"error during playback: {e}")
                self._stream_url = None
                self._substate = self.Substate.IDLE
                time.sleep(2)

    def get_full_frame(self) -> tuple[np.ndarray, int]:
        with self._lock:
            return self._current_frame, self._frame_id

    def stop(self):
        self._running = False
