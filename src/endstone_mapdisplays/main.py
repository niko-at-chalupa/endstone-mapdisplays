from endstone import ColorFormat, Logger, Player
from endstone.event import event_handler, PlayerJoinEvent, PlayerChatEvent
from endstone.map import MapRenderer, MapCanvas, MapView
from endstone.plugin import Plugin
import threading
import numpy as np
from typing import cast, Any

class CabinetMapRenderer:
    """
    Renderer for a single "cabinet" *(singular map within a bigger MapDisplay)*.
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

class EntryForPlugin(Plugin):
    def on_enable(self) -> None:
        self.register_events(self)