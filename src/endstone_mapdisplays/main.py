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
from .states import IdleState

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
                        #time.sleep(0.0001) # without this, we'd send out way too much per frame and stuff like block breaking and such wouldn't get sent to the server.
                        # comment this and test it out, idk if it's just me
                        # it was just that day, i'm so geeked
                    #self.logger.info(f"map {row*self.cols + col + 1} full cycle finished")

                self.plugin.server.scheduler.run_task(self.plugin, task)

class EntryForPlugin(Plugin):
    commands = {
        "get_display": {
            "description": "Get maps for a tiled display",
            "usages": ["/get_display <width: int> <height: int>"],
            "permissions": ["mapdisplays.command.modify"],
        },
        "remove_displays": {
            "description": "Remove ALL displays! This will make them freeze and stop working.",
            "usages": ["/remove_displays"],
            "permissions": ["mapdisplays.command.modify"],
        }
    }

    permissions = {
        "mapdisplays.command.modify": {
            "description": "Allow users to modify MapDisplays.",
            "default": "op", 
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
        if not isinstance(sender, Player):
            return False
            
        if command.name == "get_display":
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
                            meta.display_name = f"Map {c+1}x{r+1} of MapDisplay {self.displays.index(display)}"
                            item.set_item_meta(meta)
                        sender.inventory.add_item(item)
                
                sender.send_message(f"here are your {cols*rows} display maps.")

                return True
            except Exception:
                return False
        if command.name == "remove_displays":
            self.displays.clear()

        return False