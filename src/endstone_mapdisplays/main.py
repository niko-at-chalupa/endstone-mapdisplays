from endstone import ColorFormat, Logger, Player, asyncio
import asyncio as aio
from endstone.event import event_handler, PlayerJoinEvent, PlayerChatEvent
from endstone.map import MapRenderer, MapCanvas, MapView
from endstone.plugin import Plugin
import threading
import numpy as np
from typing import cast, Any
from abc import ABC, abstractmethod
from importlib.resources import files

class CabinetMapRenderer:
    """
    Renderer for a single "cabinet" *(singular map within a bigger `MapDisplay`)*.
    """
    MAP_SIZE = 128

    def __init__(self, logger: Logger) -> None:
        self.logger = logger
        self.lock = threading.Lock()
        self.buffer = np.zeros((self.MAP_SIZE, self.MAP_SIZE, 4), dtype=np.uint8)

        self._has_frame = False

    def update(self, array: np.ndarray):
        with self.lock:
            np.copyto(self.buffer, array)
            self._has_frame = True

    def render(self, view: MapView, canvas: MapCanvas, player: Player) -> None:
        with self.lock:
            if self._has_frame:
                canvas.draw_image(0, 0, cast(Any, self.buffer))
            else:
                self.logger.warning("render() called with no frame available")

class DisplayState(ABC):
    @abstractmethod
    async def get_frame(self) -> np.ndarray:
        ...

class IdleState(DisplayState):
    async def get_frame(self) -> np.ndarray:
        # play idle animation
        ...

class VideoState(DisplayState):
    def __init__(self, source: ...) -> None:
        self.source = source

    async def get_frame(self) -> np.ndarray:
        # pull next video frame
        ...

class MapDisplay:
    views: tuple[tuple[MapView, ...], ...]
    plugin: Plugin

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self._dirty = aio.Event()
        self._running = False
        self._thread = threading.Thread(target=self._render_loop, daemon=True)

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._dirty.set()  # wake thread so it can exit
        self._thread.join()

    def update(self, frame: np.ndarray) -> None:
        self._dirty.set()  # signal thread a new frame is ready

    def _render_loop(self) -> None:
        while self._running:
            self._dirty.wait()
            self._dirty.clear()
            # flush all cabinets sequentially

class EntryForPlugin(Plugin):
    def on_enable(self) -> None:
        self.register_events(self)
    
    