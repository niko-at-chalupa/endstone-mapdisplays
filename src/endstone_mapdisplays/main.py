from numpy.ma import isin
from endstone import ColorFormat, Logger, Player, asyncio
import asyncio as aio
from endstone.event import event_handler, PlayerJoinEvent
from endstone.map import MapRenderer, MapCanvas, MapView
from endstone.plugin import Plugin
from endstone.inventory import ItemStack, MapMeta
import threading
import numpy as np
import time
import av
import cv2
from typing import cast, Any
from abc import ABC, abstractmethod
from importlib.resources import files
from enum import Enum
import tempfile
import subprocess
import yt_dlp
import os

class CabinetMapRenderer(MapRenderer):
    def __init__(self, logger: Logger, row: int, col: int) -> None:
        super().__init__(is_contextual=False)
        self.logger = logger
        self.row = row
        self.col = col
        self.lock = threading.Lock()
        self.buffer = np.zeros((128, 128, 4), dtype=np.uint8)
        self._has_frame = False
        self._last_frame_id = -1

    def update(self, array: np.ndarray, frame_id: int):
        with self.lock:
            if array.shape[2] == 3:
                rgba = np.empty((128, 128, 4), dtype=np.uint8)
                rgba[:, :, :3] = array
                rgba[:, :, 3] = 255
                np.copyto(self.buffer, rgba)
            else:
                np.copyto(self.buffer, array)
            self._has_frame = True
            self._last_frame_id = frame_id

    def render(self, view: MapView, canvas: MapCanvas, player: Player) -> None:
        with self.lock:
            if self._has_frame:
                canvas.draw_image(0, 0, cast(Any, self.buffer))

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
        self._tmp_path = None

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
                "format": f"bestvideo[height<={self.height}][ext=mp4]/bestvideo[height<=360][ext=mp4]/best",
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                stream_url = info.get("url")

            self.logger.info(f"Transcoding stream to {self.width}x{self.height}...")
            self._tmp_path = f"/tmp/mapdisplay_{self.width}x{self.height}.mp4"

            with av.open(stream_url) as inp:
                with av.open(self._tmp_path, "w", format="rawvideo") as out:
                    in_stream = inp.streams.video[0]
                    in_stream.thread_type = "AUTO"
                    out_stream = out.add_stream("rawvideo", rate=20)
                    out_stream.width = self.width
                    out_stream.height = self.height
                    out_stream.pix_fmt = "rgb24"

                    for packet in inp.demux(in_stream):
                        for frame in packet.decode():
                            frame = frame.reformat(width=self.width, height=self.height, format="rgb24")
                            enc = out_stream.encode(frame)
                            out.mux(enc)

            self.logger.info("Transcoding complete, starting playback.")
            self._stream_url = self._tmp_path

        except Exception as e:
            self.logger.error(f"failed to get youtube stream: {e}")
            self._substate = self.Substate.IDLE

    def _video_loop(self):
        max_fps = 20
        target_frame_time = 1.0 / max_fps

        idle = IdleState(self.width, self.height, self.logger, "resources/mapdisplays_loading.webm")

        while self._running:
            if not self._stream_url:
                self._current_frame = idle.get_full_frame()[0]
                self._frame_id = idle.get_full_frame()[1]
                time.sleep(target_frame_time)
                continue

            idle.stop()

            try:
                self._substate = self.Substate.PLAYING
                with av.open(self._stream_url, format="rawvideo", options={
                    "video_size": f"{self.width}x{self.height}",
                    "pixel_format": "rgb24",
                }) as container:
                    stream = container.streams.video[0]
                    stream.thread_type = "AUTO"

                    for packet in container.demux(stream):
                        if not self._running:
                            break

                        start_time = time.perf_counter()
                        for frame in packet.decode():
                            img = frame.to_ndarray(format="rgb24")
                            if img.shape[1] != self.width or img.shape[0] != self.height:
                                self.logger.warning("BAD VIDEO!!!")
                                img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
                            self._current_frame = img
                            self._frame_id += 1

                        elapsed = time.perf_counter() - start_time
                        time.sleep(max(0, target_frame_time - elapsed))
            except Exception as e:
                self.logger.error(f"error during playback: {e}")
                self._stream_url = None
                self._substate = self.Substate.IDLE
                time.sleep(2)

    def get_full_frame(self) -> tuple[np.ndarray, int]:
        return self._current_frame, self._frame_id

    def stop(self):
        self._running = False
        if self._tmp_path:
            try:
                os.remove(self._tmp_path)
            except Exception:
                pass

class MapDisplay:
    def __init__(self, plugin: Plugin, cols: int, rows: int) -> None:
        self.plugin = plugin
        self.cols = cols
        self.rows = rows
        self.width = cols * 128
        self.height = rows * 128
        self.logger = plugin.logger
        
        self.renderers: list[list[CabinetMapRenderer]] = []
        self.views: list[list[MapView]] = []
        
        for r in range(rows):
            row_renderers = []
            row_views = []
            for c in range(cols):
                renderer = CabinetMapRenderer(self.logger, r, c)
                view = plugin.server.create_map(plugin.server.level.get_dimension("Overworld"))
                for old_renderer in list(view.renderers):
                    view.remove_renderer(old_renderer)
                view.add_renderer(renderer)
                row_renderers.append(renderer)
                row_views.append(view)
            self.renderers.append(row_renderers)
            self.views.append(row_views)
            
        self.state = IdleState(self.width, self.height, self.logger)

    def update(self):
        full_frame, frame_id = self.state.get_full_frame()
        for r in range(self.rows):
            for c in range(self.cols):
                sub_frame = full_frame[r*128:(r+1)*128, c*128:(c+1)*128]
                self.renderers[r][c].update(sub_frame, frame_id)
                
                def task(view=self.views[r][c], row=r, col=c):
                    for player in self.plugin.server.online_players:
                        player.send_map(view)
                        time.sleep(0.0001) # without this, we'd send out way too much per frame and stuff like block breaking and such wouldn't get sent to the server.
                        # uncomment this and test it out, idk if it's just me
                    #self.logger.info(f"map {row*self.cols + col + 1} full cycle finished")

                self.plugin.server.scheduler.run_task(self.plugin, task)

class EntryForPlugin(Plugin):
    commands = {
        "getdisplay": {
            "description": "get maps for a tiled display",
            "usages": ["/getdisplay <width: int> <height: int>"],
            "permissions": ["mapdisplay.command.get"],
        }
    }

    def on_enable(self) -> None:
        self.displays: list[MapDisplay] = []
        self._running = True
        asyncio.submit(self._loop())

    def on_disable(self) -> None:
        self._running = False
        for d in self.displays:
            if hasattr(d.state, "stop"):
                d.state.stop()

    async def _loop(self):
        while self._running:
            for display in self.displays:
                display.update()
            await aio.sleep(0.05)

    def on_command(self, sender: Any, command: Any, args: list[str]) -> bool:
        if not isinstance(sender, Player) or command.name != "getdisplay":
            return False
            
        try:
            cols, rows = int(args[0]), int(args[1])
            display = MapDisplay(self, cols, rows)
            self.displays.append(display)
            
            for r in range(rows):
                for c in range(cols):
                    item = ItemStack("minecraft:filled_map")
                    meta = item.item_meta
                    if isinstance(meta, MapMeta):
                        meta.map_view = display.views[r][c]
                        item.set_item_meta(meta)
                    sender.inventory.add_item(item)
            
            sender.send_message(f"here are your {cols*rows} display maps.")

            def set_to_youtube(self, link: str, display):
                display.state = YoutubeState(display.width, display.height, self.logger,link)
            self.server.scheduler.run_task(plugin=self, task=lambda: set_to_youtube(self, link="https://youtu.be/pmoKnB3DALc?si=AlK_dpBMIsCzZiaB", display=display), delay=375)
            return True
        except Exception:
            return False