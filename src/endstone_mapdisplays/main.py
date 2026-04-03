"""
endstone-mapdisplays  —  Server-local video & image display system for Endstone.

Folder layout (relative to plugins/mapdisplays/):
    videos/         ← drop MP4 / WEBM / MKV here
    images/         ← drop PNG / JPG here
    idle.webm       ← replaceable idle animation (auto-copied from package on first run)
    config.json     ← plugin configuration (world_folder, display_fps, etc.)
    displays.json   ← persisted display state
    resourcepack/   ← auto-generated Bedrock resource pack with extracted OGG audio
        manifest.json
        sounds.json
        sounds/mapdisplays/<stem>.ogg

Stream support:
    /setdisplay <id> stream <url>  — stream any YouTube/Twitch/HTTP URL via yt-dlp.
    Sound is intentionally disabled for streams (no stable sync is possible).

Auto resource pack registration:
    Set 'world_folder' in config.json to the path of your world folder.
    The plugin will automatically update world_resource_packs.json and bump the
    manifest version each time audio is added or removed.
"""

from __future__ import annotations

import asyncio as aio
import json
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
from PIL import Image

from endstone import Player
from endstone import asyncio as endstone_aio
from endstone.event import PlayerJoinEvent, event_handler
from endstone.inventory import ItemStack, MapMeta
from endstone.map import MapCanvas, MapRenderer, MapView
from endstone.plugin import Plugin

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_VALID_VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}
_VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

# UUID used for our resource pack in manifest.json and world_resource_packs.json
_RP_HEADER_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_RP_MODULE_UUID = "b2c3d4e5-f6a7-8901-bcde-f12345678901"

_DEFAULT_CONFIG: dict = {
    # Path to your Bedrock world folder.
    # Can be relative to the server root (e.g. "worlds/Bedrock level")
    # or an absolute path (e.g. "C:/servers/myserver/worlds/Bedrock level").
    # Leave empty ("") to disable auto resource pack registration.
    "world_folder": "worlds/Bedrock level",

    # Target frame rate for pushing map updates (frames per second).
    "display_fps": 20,
}


def _resize_rgb(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an HxWx3 uint8 RGB array using Pillow (no OpenCV needed)."""
    if img.shape[:2] == (height, width):
        return img
    pil = Image.fromarray(img, mode="RGB").resize((width, height), Image.BILINEAR)
    return np.asarray(pil)


# ──────────────────────────────────────────────────────────────────────────────
# Renderer — one 128×128 tile of a display
# ──────────────────────────────────────────────────────────────────────────────

class CabinetMapRenderer(MapRenderer):
    """Thread-safe renderer for a single 128×128 map tile."""

    def __init__(self) -> None:
        super().__init__(is_contextual=False)
        self._lock = threading.Lock()
        self._buffer = np.zeros((128, 128, 4), dtype=np.uint8)
        self._has_frame = False
        self._frame_id = -1

    def push(self, rgb_crop: np.ndarray, frame_id: int) -> bool:
        """
        Push a new 128×128 RGB or RGBA crop.
        Returns True if the frame was new (i.e. the display needs a send_map call).
        """
        if frame_id == self._frame_id:
            return False
        with self._lock:
            if rgb_crop.ndim == 3 and rgb_crop.shape[2] == 3:
                buf = np.empty((128, 128, 4), dtype=np.uint8)
                buf[:, :, :3] = rgb_crop
                buf[:, :, 3] = 255
                np.copyto(self._buffer, buf)
            else:
                np.copyto(self._buffer, rgb_crop[:128, :128])
            self._has_frame = True
            self._frame_id = frame_id
        return True

    def render(self, view: MapView, canvas: MapCanvas, player: Player) -> None:
        with self._lock:
            if self._has_frame:
                canvas.draw_image(0, 0, cast(Any, self._buffer))


# ──────────────────────────────────────────────────────────────────────────────
# Display States
# ──────────────────────────────────────────────────────────────────────────────

class DisplayState(ABC):
    """Abstract base for anything that can drive a map display."""

    @abstractmethod
    def get_full_frame(self) -> tuple[np.ndarray, int]:
        """Return (HxWx3 RGB frame, frame_id). frame_id increments on each new frame."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Signal the state to stop any background threads."""
        ...

    @property
    def duration(self) -> float | None:
        """Duration in seconds; None if infinite / not applicable."""
        return None

    @property
    def sound_name(self) -> str | None:
        """Bedrock sound event name to play alongside this state (e.g. 'mapdisplays.intro')."""
        return None


class IdleState(DisplayState):
    """
    Loops a webm/mp4 animation as an idle screen.

    Resolution order:
    1. data_folder/idle.webm     (user-replaceable)
    2. Package resource fallback (resources/idle.webm baked into wheel)
    """

    _FALLBACK_RESOURCE = "resources/idle.webm"

    def __init__(self, width: int, height: int, logger: Any, data_folder: Path) -> None:
        self._width = width
        self._height = height
        self._logger = logger
        self._lock = threading.Lock()
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._frame_id = 0
        self._running = True

        # Prefer the user-replaceable copy in the data folder
        data_copy = data_folder / "idle.webm"
        if data_copy.exists():
            self._path: str | None = str(data_copy)
        else:
            try:
                from importlib.resources import files as _res_files
                res = _res_files("endstone_mapdisplays").joinpath(self._FALLBACK_RESOURCE)
                self._path = str(res)
            except Exception:
                self._path = None

        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="mapdisplay-idle"
        )
        self._thread.start()

    def _loop(self) -> None:
        back_off = 1.0
        while self._running:
            if not self._path:
                time.sleep(1.0)
                continue
            try:
                with av.open(self._path) as container:
                    stream = container.streams.video[0]
                    fps = float(stream.average_rate) if stream.average_rate else 20.0
                    frame_time = 1.0 / fps
                    for frame in container.decode(video=0):
                        if not self._running:
                            return
                        img = frame.to_ndarray(format="rgb24")
                        img = _resize_rgb(img, self._width, self._height)
                        with self._lock:
                            self._frame = img
                            self._frame_id += 1
                        time.sleep(frame_time)
                back_off = 1.0  # clean loop — reset back-off
            except Exception as exc:
                self._logger.warning(f"[MapDisplays] IdleState decode error: {exc}")
                time.sleep(back_off)
                back_off = min(back_off * 2, 30.0)

    def get_full_frame(self) -> tuple[np.ndarray, int]:
        with self._lock:
            return self._frame, self._frame_id

    def stop(self) -> None:
        self._running = False


class ImageState(DisplayState):
    """Displays a static image (PNG, JPG, etc.) — no background thread needed."""

    def __init__(self, width: int, height: int, logger: Any, path: Path) -> None:
        self._logger = logger
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._frame_id = 0
        try:
            img = (
                Image.open(path)
                .convert("RGB")
                .resize((width, height), Image.LANCZOS)
            )
            self._frame = np.asarray(img)
            self._frame_id = 1
        except Exception as exc:
            logger.error(f"[MapDisplays] ImageState failed to load '{path}': {exc}")

    def get_full_frame(self) -> tuple[np.ndarray, int]:
        return self._frame, self._frame_id

    def stop(self) -> None:
        pass  # stateless — nothing to clean up


class VideoFileState(DisplayState):
    """
    Streams a local video file, looping indefinitely.

    Tracks video duration for sound loop synchronisation.
    Calls on_loop() each time the video restarts; the plugin uses this to
    restart the Bedrock sound event so audio stays in sync.
    """

    def __init__(
        self,
        width: int,
        height: int,
        logger: Any,
        path: Path,
        sound: str | None = None,
    ) -> None:
        self._width = width
        self._height = height
        self._logger = logger
        self._path = path
        self._sound = sound
        self._lock = threading.Lock()
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._frame_id = 0
        self._running = True
        self._duration: float | None = None

        # Callback invoked each time the video loops (set by plugin for sound sync)
        self.on_loop: Any = None

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"mapdisplay-video-{path.stem}",
        )
        self._thread.start()

    @property
    def duration(self) -> float | None:
        return self._duration

    @property
    def sound_name(self) -> str | None:
        return self._sound

    def _loop(self) -> None:
        back_off = 1.0
        while self._running:
            try:
                with av.open(str(self._path)) as container:
                    v_stream = container.streams.video[0]
                    fps = float(v_stream.average_rate) if v_stream.average_rate else 20.0
                    frame_time = 1.0 / fps

                    # Capture duration once
                    if self._duration is None and container.duration:
                        self._duration = float(container.duration) / 1_000_000.0

                    deadline = time.perf_counter()
                    for frame in container.decode(video=0):
                        if not self._running:
                            return
                        img = frame.to_ndarray(format="rgb24")
                        img = _resize_rgb(img, self._width, self._height)

                        # Frame-rate limiter — don't burn CPU
                        now = time.perf_counter()
                        wait = deadline - now
                        if wait > 0:
                            time.sleep(wait)
                        deadline = time.perf_counter() + frame_time

                        with self._lock:
                            self._frame = img
                            self._frame_id += 1

                # Video ended naturally — fire loop callback, then continue loop
                if self._running and self.on_loop is not None:
                    try:
                        self.on_loop()
                    except Exception:
                        pass
                back_off = 1.0

            except Exception as exc:
                self._logger.warning(
                    f"[MapDisplays] VideoFileState '{self._path.name}' error: {exc}"
                )
                time.sleep(back_off)
                back_off = min(back_off * 2, 30.0)

    def get_full_frame(self) -> tuple[np.ndarray, int]:
        with self._lock:
            return self._frame, self._frame_id

    def stop(self) -> None:
        self._running = False


class StreamState(DisplayState):
    """
    Streams video from any URL supported by yt-dlp (YouTube, Twitch, direct HLS,
    plain HTTP video streams, etc.).

    Resolution order:
    1. yt-dlp extracts the best direct stream URL (preferred — handles all platforms)
    2. Falls back to av.open(url) directly (works for raw HTTP/HLS/RTSP streams)

    Sound is intentionally NOT supported — audio sync across a network stream is
    not reliably achievable with the Bedrock sound event system.
    """

    # Prefer low-resolution streams to reduce server CPU load
    _YDL_FORMAT = (
        "bestvideo[height<=144][ext=mp4]/"
        "bestvideo[height<=240][ext=mp4]/"
        "bestvideo[height<=144]/"
        "bestvideo[height<=360]/"
        "best[height<=144]/best"
    )

    def __init__(
        self,
        width: int,
        height: int,
        logger: Any,
        url: str,
        data_folder: Path,
    ) -> None:
        self._width = width
        self._height = height
        self._logger = logger
        self._url = url
        self._data_folder = data_folder

        self._lock = threading.Lock()
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._frame_id = 0
        self._running = True

        # Show idle animation while the URL is being resolved
        self._idle = IdleState(width, height, logger, data_folder)
        self._stream_url: str | None = None
        self._resolving = True

        self._resolve_thread = threading.Thread(
            target=self._resolve_url, daemon=True, name="mapdisplay-stream-resolve"
        )
        self._resolve_thread.start()

        self._decode_thread = threading.Thread(
            target=self._decode_loop, daemon=True, name="mapdisplay-stream-decode"
        )
        self._decode_thread.start()

    # sound_name intentionally returns None — no audio for streams

    def _resolve_url(self) -> None:
        """Try yt-dlp first; fall back to using the raw URL directly with av."""
        try:
            import yt_dlp  # optional dependency

            ydl_opts = {
                "format": self._YDL_FORMAT,
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                # No cookiesfrombrowser — fails on headless servers.
                # Public videos work without it; age-restricted content will fail gracefully.
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self._url, download=False)
                if info:
                    # Prefer direct URL; for formats list, pick the first entry
                    url = info.get("url")
                    if not url and info.get("formats"):
                        url = info["formats"][0].get("url")
                    if url:
                        self._stream_url = url
                        self._logger.info(f"[MapDisplays] Stream URL resolved via yt-dlp.")
                        self._resolving = False
                        return

        except ImportError:
            self._logger.warning(
                "[MapDisplays] yt-dlp not installed — attempting direct stream open."
            )
        except Exception as exc:
            self._logger.warning(
                f"[MapDisplays] yt-dlp resolution failed: {exc} — trying direct open."
            )

        # Fallback: let av try to open the URL directly (works for plain HLS/HTTP)
        self._stream_url = self._url
        self._logger.info("[MapDisplays] Using URL directly with av (no yt-dlp resolution).")
        self._resolving = False

    def _decode_loop(self) -> None:
        back_off = 2.0
        while self._running:
            # While resolving, serve idle frames
            if self._resolving or not self._stream_url:
                frame, fid = self._idle.get_full_frame()
                with self._lock:
                    self._frame = frame
                    self._frame_id = fid
                time.sleep(0.05)
                continue

            try:
                self._logger.info("[MapDisplays] StreamState: opening stream…")
                # Low-latency flags for HTTP/HLS streams
                options = {
                    "fflags": "nobuffer",
                    "flags": "low_delay",
                    "analyzeduration": "1000000",
                    "probesize": "65536",
                }
                with av.open(self._stream_url, options=options) as container:
                    v_stream = container.streams.video[0]
                    v_stream.thread_type = "AUTO"
                    fps = float(v_stream.average_rate) if v_stream.average_rate else 20.0
                    frame_time = 1.0 / fps
                    deadline = time.perf_counter()

                    for packet in container.demux(v_stream):
                        if not self._running:
                            return
                        try:
                            frames = list(packet.decode())
                            if not frames:
                                continue

                            # Drop stale frames if we're falling behind
                            now = time.perf_counter()
                            if now > deadline and len(frames) > 1:
                                frames = [frames[-1]]

                            for frame in frames:
                                if not self._running:
                                    return
                                img = frame.to_ndarray(format="rgb24")
                                img = _resize_rgb(img, self._width, self._height)
                                with self._lock:
                                    self._frame = img
                                    self._frame_id += 1

                            deadline += frame_time
                            gap = deadline - time.perf_counter()
                            if gap > 0:
                                time.sleep(gap)
                            else:
                                deadline = time.perf_counter()

                        except Exception:
                            continue  # skip bad packets

                # Stream ended — re-resolve (live streams reconnect; VODs restart)
                self._logger.info("[MapDisplays] Stream ended — reconnecting…")
                self._resolving = True
                self._stream_url = None
                self._resolve_thread = threading.Thread(
                    target=self._resolve_url, daemon=True, name="mapdisplay-stream-resolve"
                )
                self._resolve_thread.start()
                back_off = 2.0

            except Exception as exc:
                self._logger.warning(f"[MapDisplays] StreamState error: {exc}")
                time.sleep(back_off)
                back_off = min(back_off * 2, 60.0)

    def get_full_frame(self) -> tuple[np.ndarray, int]:
        with self._lock:
            return self._frame, self._frame_id

    def stop(self) -> None:
        self._running = False
        self._idle.stop()


# ──────────────────────────────────────────────────────────────────────────────
# MapDisplay — a grid of tiles driven by a DisplayState
# ──────────────────────────────────────────────────────────────────────────────

class MapDisplay:
    """Manages a rows×cols grid of CabinetMapRenderers backed by a single DisplayState."""

    def __init__(
        self,
        plugin: "EntryForPlugin",
        display_id: int,
        cols: int,
        rows: int,
    ) -> None:
        self.plugin = plugin
        self.display_id = display_id
        self.cols = cols
        self.rows = rows
        self.width = cols * 128
        self.height = rows * 128

        self._state_lock = threading.Lock()
        self._state: DisplayState = IdleState(
            self.width, self.height, plugin.logger, Path(plugin.data_folder)
        )
        self._state_name = "idle"
        self._state_arg: str | None = None

        # Pre-build a flat (renderer, view) list — avoids re-allocation every frame
        self._grid: list[list[tuple[CabinetMapRenderer, MapView]]] = []
        self._tiles: list[tuple[CabinetMapRenderer, MapView]] = []

        for r in range(rows):
            row: list[tuple[CabinetMapRenderer, MapView]] = []
            for c in range(cols):
                renderer = CabinetMapRenderer()
                view = plugin.server.create_map(
                    plugin.server.level.get_dimension("Overworld")
                )
                # Remove default renderer so only our renderer runs
                for old in list(view.renderers):
                    view.remove_renderer(old)
                view.add_renderer(renderer)
                row.append((renderer, view))
                self._tiles.append((renderer, view))
            self._grid.append(row)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def state(self) -> DisplayState:
        with self._state_lock:
            return self._state

    @property
    def map_ids(self) -> list[list[int]]:
        return [
            [self._grid[r][c][1].id for c in range(self.cols)]
            for r in range(self.rows)
        ]

    # ── State management ────────────────────────────────────────────────────

    def set_state(
        self,
        new_state: DisplayState,
        name: str,
        arg: str | None = None,
    ) -> None:
        """Thread-safe state swap. Stops the old state after releasing the lock."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            self._state_name = name
            self._state_arg = arg
        # Stop old state outside the lock so it doesn't deadlock with its own lock
        try:
            old_state.stop()
        except Exception:
            pass

    # ── Frame push ──────────────────────────────────────────────────────────

    def update(self) -> None:
        """
        Pull the current frame from the active state and push it to every tile.
        Only schedules a send_map task if the frame actually changed.
        Called from the main async loop at ~20fps.
        """
        with self._state_lock:
            state = self._state

        full_frame, frame_id = state.get_full_frame()

        for r in range(self.rows):
            for c in range(self.cols):
                renderer, view = self._grid[r][c]
                crop = full_frame[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128]
                if renderer.push(crop, frame_id):
                    # Default-parameter capture freezes view/plugin at definition time
                    def _send(v=view, plg=self.plugin):
                        for player in plg.server.online_players:
                            try:
                                player.send_map(v)
                            except Exception:
                                pass
                    self.plugin.server.scheduler.run_task(self.plugin, _send)

    # ── Inventory helpers ───────────────────────────────────────────────────

    def give_maps_to(self, player: Player) -> None:
        """Give all map items (one per tile) to a player."""
        for r in range(self.rows):
            for c in range(self.cols):
                _, view = self._grid[r][c]
                item = ItemStack("minecraft:filled_map")
                meta = item.item_meta
                if isinstance(meta, MapMeta):
                    meta.map_view = view
                    item.set_item_meta(meta)
                player.inventory.add_item(item)

    def send_all_maps_to(self, player: Player) -> None:
        """Force-send every current frame to a specific player (e.g. on join)."""
        for _, view in self._tiles:
            try:
                player.send_map(view)
            except Exception:
                pass

    # ── Persistence helper ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.display_id,
            "cols": self.cols,
            "rows": self.rows,
            "state_name": self._state_name,
            "state_arg": self._state_arg,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Plugin Entry Point
# ──────────────────────────────────────────────────────────────────────────────

class EntryForPlugin(Plugin):
    # Required class-level metadata
    name = "mapdisplays"
    api_version = "0.5"

    commands = {
        "getdisplay": {
            "description": "Create a tiled map display and receive the map items",
            "usages": ["/getdisplay <cols: int> <rows: int>"],
            "permissions": ["mapdisplays.command.get"],
        },
        "setdisplay": {
            "description": "Change what a display shows",
            "usages": [
                "/setdisplay <id: int> video <file: str>",
                "/setdisplay <id: int> image <file: str>",
                "/setdisplay <id: int> stream <url: str>",
                "/setdisplay <id: int> idle",
            ],
            "permissions": ["mapdisplays.command.set"],
        },
        "listvideos": {
            "description": "List video files available in the videos folder",
            "usages": ["/listvideos"],
            "permissions": ["mapdisplays.command.get"],
        },
        "stopdisplay": {
            "description": "Stop a display and reset it to idle",
            "usages": ["/stopdisplay <id: int>"],
            "permissions": ["mapdisplays.command.set"],
        },
        "getmaps": {
            "description": "Re-receive map items for all active displays (use after server restart)",
            "usages": ["/getmaps"],
            "permissions": ["mapdisplays.command.get"],
        },
        "listdisplays": {
            "description": "List all active displays and their current state",
            "usages": ["/listdisplays"],
            "permissions": ["mapdisplays.command.get"],
        },
        "removevideo": {
            "description": "Remove a video and its extracted audio from the server",
            "usages": ["/removevideo <filename: str>"],
            "permissions": ["mapdisplays.command.admin"],
        },
        "reloadconfig": {
            "description": "Reload config.json without restarting the plugin",
            "usages": ["/reloadconfig"],
            "permissions": ["mapdisplays.command.admin"],
        },
    }

    permissions = {
        "mapdisplays.command.get": {
            "description": "Receive or list map displays",
            "default": "op",
        },
        "mapdisplays.command.set": {
            "description": "Control what displays show",
            "default": "op",
        },
        "mapdisplays.command.admin": {
            "description": "Administrative commands (reload config, remove video)",
            "default": "op",
        },
    }

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_load(self) -> None:
        self.displays: dict[int, MapDisplay] = {}
        self._next_id: int = 0
        self._running: bool = False
        self._config: dict = dict(_DEFAULT_CONFIG)

    def on_enable(self) -> None:
        self._running = True
        self._setup_data_folder()
        self._load_config()
        self._load_persistence()
        self.register_events(self)
        endstone_aio.submit(self._update_loop())
        self.logger.info(
            f"[MapDisplays] Enabled — {len(self.displays)} display(s) restored."
        )
        # Register resource pack with world on startup if configured
        if self._get_world_folder():
            self._update_world_resource_packs()

    def on_disable(self) -> None:
        self._running = False
        self._save_persistence()
        for d in self.displays.values():
            try:
                d.state.stop()
            except Exception:
                pass
        self.logger.info("[MapDisplays] Disabled.")

    # ── Data folder setup ────────────────────────────────────────────────────

    def _setup_data_folder(self) -> None:
        df = Path(self.data_folder)
        (df / "videos").mkdir(parents=True, exist_ok=True)
        (df / "images").mkdir(parents=True, exist_ok=True)
        (df / "resourcepack" / "sounds" / "mapdisplays").mkdir(parents=True, exist_ok=True)

        # Copy idle.webm from wheel resources to data_folder so it's user-replaceable
        idle_dest = df / "idle.webm"
        if not idle_dest.exists():
            try:
                from importlib.resources import files as _res_files
                src = _res_files("endstone_mapdisplays").joinpath("resources/idle.webm")
                idle_dest.write_bytes(src.read_bytes())
                self.logger.info("[MapDisplays] Copied default idle.webm to data folder.")
            except Exception as exc:
                self.logger.warning(
                    f"[MapDisplays] Could not copy default idle.webm: {exc}"
                )

        self._ensure_resourcepack_skeleton(df)

    # ── Config ───────────────────────────────────────────────────────────────

    def _load_config(self) -> None:
        """Load config.json, writing defaults if it doesn't exist."""
        path = Path(self.data_folder) / "config.json"
        if not path.exists():
            self._save_config()
            self.logger.info(
                "[MapDisplays] Created default config.json. "
                "Set 'world_folder' to enable auto resource pack registration."
            )
            return
        try:
            loaded = json.loads(path.read_text())
            # Merge loaded values over defaults so new keys are always present
            self._config = {**_DEFAULT_CONFIG, **loaded}
        except Exception as exc:
            self.logger.error(f"[MapDisplays] Failed to read config.json: {exc} — using defaults.")
            self._config = dict(_DEFAULT_CONFIG)

    def _save_config(self) -> None:
        path = Path(self.data_folder) / "config.json"
        try:
            path.write_text(json.dumps(self._config, indent=2))
        except Exception as exc:
            self.logger.error(f"[MapDisplays] Failed to write config.json: {exc}")

    def _get_world_folder(self) -> Path | None:
        """Return the resolved world folder Path, or None if not configured / not found."""
        raw = str(self._config.get("world_folder", "")).strip()
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            # Relative to server root: data_folder is plugins/mapdisplays,
            # so walk up two levels to reach the server working directory.
            server_root = Path(self.data_folder).parent.parent
            p = server_root / p
        if p.is_dir():
            return p
        self.logger.warning(
            f"[MapDisplays] world_folder '{p}' does not exist or is not a directory. "
            f"Check config.json."
        )
        return None

    def _ensure_resourcepack_skeleton(self, df: Path) -> None:
        """Write manifest.json and sounds.json if they don't already exist."""
        manifest_path = df / "resourcepack" / "manifest.json"
        sounds_path = df / "resourcepack" / "sounds.json"

        if not manifest_path.exists():
            manifest = {
                "format_version": 2,
                "header": {
                    "description": "MapDisplays Plugin Audio Pack",
                    "name": "MapDisplays Audio",
                    "uuid": _RP_HEADER_UUID,
                    "version": [1, 0, 0],
                    "min_engine_version": [1, 20, 0],
                },
                "modules": [
                    {
                        "description": "MapDisplays custom sounds",
                        "type": "resources",
                        "uuid": _RP_MODULE_UUID,
                        "version": [1, 0, 0],
                    }
                ],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))

        if not sounds_path.exists():
            sounds_path.write_text(json.dumps({}, indent=2))

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_persistence(self) -> None:
        try:
            data = {
                "next_id": self._next_id,
                "displays": [d.to_dict() for d in self.displays.values()],
            }
            path = Path(self.data_folder) / "displays.json"
            path.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            self.logger.error(f"[MapDisplays] Failed to save displays.json: {exc}")

    def _load_persistence(self) -> None:
        path = Path(self.data_folder) / "displays.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self._next_id = data.get("next_id", 0)
            for entry in data.get("displays", []):
                display = self._restore_display(entry)
                if display is not None:
                    self.displays[display.display_id] = display
        except Exception as exc:
            self.logger.error(f"[MapDisplays] Failed to load displays.json: {exc}")

    def _restore_display(self, entry: dict) -> MapDisplay | None:
        """Re-create a MapDisplay from a saved entry (maps get new IDs — players use /getmaps)."""
        try:
            display = MapDisplay(
                self,
                entry["id"],
                entry["cols"],
                entry["rows"],
            )

            state_name = entry.get("state_name", "idle")
            state_arg = entry.get("state_arg")

            if state_name == "video" and state_arg:
                video_path = Path(self.data_folder) / "videos" / state_arg
                if video_path.exists():
                    sound = self._get_existing_sound(state_arg)
                    vs = VideoFileState(
                        display.width, display.height, self.logger, video_path, sound
                    )
                    self._bind_sound_loop(display, vs)
                    display.set_state(vs, "video", state_arg)
                else:
                    self.logger.warning(
                        f"[MapDisplays] Video '{state_arg}' not found — display #{entry['id']} set to idle."
                    )
            elif state_name == "image" and state_arg:
                image_path = Path(self.data_folder) / "images" / state_arg
                if image_path.exists():
                    display.set_state(
                        ImageState(display.width, display.height, self.logger, image_path),
                        "image",
                        state_arg,
                    )
            elif state_name == "stream" and state_arg:
                ss = StreamState(
                    display.width, display.height, self.logger,
                    state_arg, Path(self.data_folder)
                )
                display.set_state(ss, "stream", state_arg)
                self.logger.info(
                    f"[MapDisplays] Display #{entry['id']} restoring stream: {state_arg}"
                )

            return display
        except Exception as exc:
            self.logger.error(
                f"[MapDisplays] Failed to restore display #{entry.get('id', '?')}: {exc}"
            )
            return None

    # ── Main update loop ─────────────────────────────────────────────────────

    async def _update_loop(self) -> None:
        """~20fps frame push loop. Runs on the Endstone async executor."""
        while self._running:
            for display in list(self.displays.values()):
                try:
                    display.update()
                except Exception as exc:
                    self.logger.warning(
                        f"[MapDisplays] Display #{display.display_id} update error: {exc}"
                    )
            await aio.sleep(0.05)

    # ── Events ───────────────────────────────────────────────────────────────

    @event_handler
    def on_player_join(self, event: PlayerJoinEvent) -> None:
        """Send current map frames and restart any playing sound for the joining player."""
        player = event.player

        def _welcome():
            try:
                for display in self.displays.values():
                    display.send_all_maps_to(player)
                    sound = display.state.sound_name
                    if sound:
                        try:
                            player.play_sound(player.location, sound, 1.0, 1.0)
                        except Exception:
                            pass
            except Exception as exc:
                self.logger.warning(f"[MapDisplays] Join handler error for {player.name}: {exc}")

        # Small delay so the player is fully loaded before we flood them with packets
        self.server.scheduler.run_task(self, _welcome, delay=40)

    # ── Commands ─────────────────────────────────────────────────────────────

    def on_command(self, sender: Any, command: Any, args: list[str]) -> bool:
        if not isinstance(sender, Player):
            sender.send_message("§c[MapDisplays] This command is player-only.")
            return True
        try:
            n = command.name
            if n == "getdisplay":
                return self._cmd_getdisplay(sender, args)
            elif n == "setdisplay":
                return self._cmd_setdisplay(sender, args)
            elif n == "listvideos":
                return self._cmd_listvideos(sender)
            elif n == "stopdisplay":
                return self._cmd_stopdisplay(sender, args)
            elif n == "getmaps":
                return self._cmd_getmaps(sender)
            elif n == "listdisplays":
                return self._cmd_listdisplays(sender)
            elif n == "removevideo":
                return self._cmd_removevideo(sender, args)
            elif n == "reloadconfig":
                return self._cmd_reloadconfig(sender)
        except Exception as exc:
            self.logger.error(f"[MapDisplays] Command '{command.name}' error: {exc}")
            sender.send_message(f"§c[MapDisplays] Error: {exc}")
        return True

    def _cmd_getdisplay(self, player: Player, args: list[str]) -> bool:
        if len(args) < 2:
            player.send_message("§cUsage: /getdisplay <cols> <rows>")
            return True
        try:
            cols, rows = int(args[0]), int(args[1])
        except ValueError:
            player.send_message("§cCols and rows must be integers.")
            return True
        if not (1 <= cols <= 8 and 1 <= rows <= 8):
            player.send_message("§cCols and rows must each be between 1 and 8.")
            return True

        display_id = self._next_id
        self._next_id += 1
        display = MapDisplay(self, display_id, cols, rows)
        self.displays[display_id] = display
        display.give_maps_to(player)
        self._save_persistence()

        player.send_message(
            f"§aDisplay §f#{display_id} §acreated (§f{cols}§a×§f{rows}§a). "
            f"§7You received §f{cols * rows} §7map item(s). "
            f"Place them on item frames in a §f{cols} wide §7× §f{rows} tall §7grid."
        )
        return True

    def _cmd_setdisplay(self, player: Player, args: list[str]) -> bool:
        if len(args) < 2:
            player.send_message("§cUsage: /setdisplay <id> video <file> | image <file> | idle")
            return True
        try:
            display_id = int(args[0])
        except ValueError:
            player.send_message("§cDisplay ID must be an integer.")
            return True

        display = self.displays.get(display_id)
        if display is None:
            player.send_message(f"§cNo display with ID {display_id}. Use /listdisplays.")
            return True

        mode = args[1].lower()

        if mode == "idle":
            self._stop_sound_for(display)
            display.set_state(
                IdleState(display.width, display.height, self.logger, Path(self.data_folder)),
                "idle",
            )
            self._save_persistence()
            player.send_message(f"§aDisplay §f#{display_id} §areset to idle.")

        elif mode == "video":
            if len(args) < 3:
                player.send_message("§cUsage: /setdisplay <id> video <filename>")
                return True
            filename = args[2]
            video_path = Path(self.data_folder) / "videos" / filename
            if not video_path.exists():
                player.send_message(
                    f"§cFile not found: §fvideos/{filename}\n"
                    f"§7Upload the file to §fplugins/mapdisplays/videos/ §7then retry."
                )
                return True

            player.send_message(f"§7Loading '§f{filename}§7'… (audio extraction may take a moment)")

            def _load_video():
                sound = self._prepare_sound(video_path)
                vs = VideoFileState(
                    display.width, display.height, self.logger, video_path, sound
                )
                self._bind_sound_loop(display, vs)
                self._stop_sound_for(display)
                display.set_state(vs, "video", filename)
                self._save_persistence()

                def _notify():
                    player.send_message(
                        f"§aDisplay §f#{display_id} §anow playing: §f{filename}"
                    )
                    if sound:
                        self._play_sound_all(sound)
                    else:
                        player.send_message(
                            "§7§o(No audio — FFmpeg not found or extraction failed. "
                            "Install FFmpeg to enable sound.)"
                        )
                self.server.scheduler.run_task(self, _notify)

            threading.Thread(
                target=_load_video,
                daemon=True,
                name=f"mapdisplay-load-{filename}",
            ).start()

        elif mode == "image":
            if len(args) < 3:
                player.send_message("§cUsage: /setdisplay <id> image <filename>")
                return True
            filename = args[2]
            image_path = Path(self.data_folder) / "images" / filename
            if not image_path.exists():
                player.send_message(
                    f"§cFile not found: §fimages/{filename}\n"
                    f"§7Upload the file to §fplugins/mapdisplays/images/ §7then retry."
                )
                return True
            self._stop_sound_for(display)
            display.set_state(
                ImageState(display.width, display.height, self.logger, image_path),
                "image",
                filename,
            )
            self._save_persistence()
            player.send_message(
                f"§aDisplay §f#{display_id} §anow showing image: §f{filename}"
            )
        elif mode == "stream":
            if len(args) < 3:
                player.send_message("§cUsage: /setdisplay <id> stream <url>")
                return True
            url = args[2]
            # Basic URL sanity check
            if not (url.startswith("http://") or url.startswith("https://") or url.startswith("rtmp://")):
                player.send_message(
                    "§cURL must start with http://, https://, or rtmp://"
                )
                return True
            self._stop_sound_for(display)
            ss = StreamState(
                display.width, display.height, self.logger,
                url, Path(self.data_folder)
            )
            display.set_state(ss, "stream", url)
            self._save_persistence()
            player.send_message(
                f"§aDisplay §f#{display_id} §anow streaming: §f{url}\n"
                f"§7§o(Resolving stream URL via yt-dlp — idle animation shown until ready. "
                f"No sound for streams.)"
            )
        else:
            player.send_message(
                "§cUnknown mode. Valid options: §fvideo§c, §fimage§c, §fstream§c, §fidle§c."
            )

        return True

    def _cmd_listvideos(self, player: Player) -> bool:
        video_dir = Path(self.data_folder) / "videos"
        files = sorted(video_dir.iterdir()) if video_dir.exists() else []
        valid = [f for f in files if f.suffix.lower() in _VALID_VIDEO_EXTS]
        images = sorted((Path(self.data_folder) / "images").iterdir()) if (Path(self.data_folder) / "images").exists() else []
        valid_img = [f for f in images if f.suffix.lower() in _VALID_IMAGE_EXTS]

        if not valid and not valid_img:
            player.send_message(
                "§7No media files found. Upload to:\n"
                "  §fplugins/mapdisplays/videos/\n"
                "  §fplugins/mapdisplays/images/"
            )
        else:
            if valid:
                player.send_message(f"§a§l{len(valid)}§r §avideos:")
                for f in valid:
                    has_audio = (
                        Path(self.data_folder) / "resourcepack" / "sounds" / "mapdisplays" / f"{f.stem}.ogg"
                    ).exists()
                    audio_tag = " §a[audio ready]" if has_audio else " §7[no audio yet]"
                    player.send_message(f"  §f{f.name}{audio_tag}")
            if valid_img:
                player.send_message(f"§a§l{len(valid_img)}§r §aimages:")
                for f in valid_img:
                    player.send_message(f"  §f{f.name}")
        return True

    def _cmd_stopdisplay(self, player: Player, args: list[str]) -> bool:
        if not args:
            player.send_message("§cUsage: /stopdisplay <id>")
            return True
        try:
            display_id = int(args[0])
        except ValueError:
            player.send_message("§cDisplay ID must be an integer.")
            return True

        display = self.displays.get(display_id)
        if display is None:
            player.send_message(f"§cNo display with ID {display_id}.")
            return True

        self._stop_sound_for(display)
        display.set_state(
            IdleState(display.width, display.height, self.logger, Path(self.data_folder)),
            "idle",
        )
        self._save_persistence()
        player.send_message(f"§aDisplay §f#{display_id} §astopped.")
        return True

    def _cmd_getmaps(self, player: Player) -> bool:
        if not self.displays:
            player.send_message("§7No active displays. Use /getdisplay to create one.")
            return True
        for display in self.displays.values():
            display.give_maps_to(player)
        player.send_message(
            f"§aGiven map items for §f{len(self.displays)} §adisplay(s). "
            f"§7Place them in item frames to restore your boards."
        )
        return True

    def _cmd_listdisplays(self, player: Player) -> bool:
        if not self.displays:
            player.send_message("§7No active displays.")
            return True
        player.send_message(f"§a§l{len(self.displays)}§r §aactive display(s):")
        for d in self.displays.values():
            state_desc = d._state_name
            if d._state_arg:
                state_desc += f": §f{d._state_arg}"
            player.send_message(
                f"  §f#{d.display_id} §7({d.cols}×{d.rows}) — {state_desc}"
            )
        return True

    def _cmd_removevideo(self, player: Player, args: list[str]) -> bool:
        if not args:
            player.send_message("§cUsage: /removevideo <filename>")
            return True
        filename = args[0]
        video_path = Path(self.data_folder) / "videos" / filename
        stem = Path(filename).stem

        if not video_path.exists():
            player.send_message(f"§cVideo not found: §f{filename}")
            return True

        # Stop any displays currently showing this video first
        for display in self.displays.values():
            if display._state_name == "video" and display._state_arg == filename:
                self._stop_sound_for(display)
                display.set_state(
                    IdleState(display.width, display.height, self.logger, Path(self.data_folder)),
                    "idle",
                )
                player.send_message(
                    f"§7Display §f#{display.display_id} §7reset to idle (was showing this video)."
                )

        # Delete the video file
        try:
            video_path.unlink()
            player.send_message(f"§7Deleted video: §f{filename}")
        except Exception as exc:
            player.send_message(f"§cFailed to delete video file: {exc}")
            return True

        # Remove OGG + sounds.json entry + bump version
        has_audio = (
            Path(self.data_folder) / "resourcepack" / "sounds" / "mapdisplays" / f"{stem}.ogg"
        ).exists()

        if has_audio:
            player.send_message(f"§7Removing audio and bumping resource pack version…")
            self._unregister_sound_event(stem)  # also calls _bump_resourcepack_version
            player.send_message(
                f"§aVideo §f{filename} §aremoved. Resource pack version bumped and "
                f"world_resource_packs.json updated. "
                f"§7Players need to rejoin to receive the updated pack."
            )
        else:
            player.send_message(f"§aVideo §f{filename} §aremoved §7(no audio was extracted).")

        self._save_persistence()
        return True

    def _cmd_reloadconfig(self, player: Player) -> bool:
        self._load_config()
        world = self._get_world_folder()
        if world:
            self._update_world_resource_packs()
            player.send_message(
                f"§aConfig reloaded. World folder: §f{world}\n"
                f"§7world_resource_packs.json synced."
            )
        else:
            player.send_message(
                "§aConfig reloaded. §7world_folder is not set or not found — "
                "auto pack registration disabled."
            )
        return True

    # ── Sound helpers ────────────────────────────────────────────────────────


    def _get_existing_sound(self, filename: str) -> str | None:
        """Return sound event name if the OGG already exists in the resource pack."""
        stem = Path(filename).stem
        ogg = Path(self.data_folder) / "resourcepack" / "sounds" / "mapdisplays" / f"{stem}.ogg"
        return f"mapdisplays.{stem}" if ogg.exists() else None

    def _prepare_sound(self, video_path: Path) -> str | None:
        """
        Extract audio from the video to an OGG file, register the sound event,
        and return the Bedrock sound event name; or None if extraction failed.
        """
        stem = video_path.stem
        event_name = f"mapdisplays.{stem}"
        ogg_path = (
            Path(self.data_folder)
            / "resourcepack"
            / "sounds"
            / "mapdisplays"
            / f"{stem}.ogg"
        )

        if ogg_path.exists():
            return event_name  # Already prepared from a previous load

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.logger.warning(
                "[MapDisplays] FFmpeg not found on PATH — sound disabled for this video.\n"
                "[MapDisplays] Install FFmpeg (https://ffmpeg.org) and reload the plugin."
            )
            return None

        try:
            self.logger.info(f"[MapDisplays] Extracting audio from '{video_path.name}'…")
            result = subprocess.run(
                [
                    ffmpeg,
                    "-i", str(video_path),
                    "-vn",
                    "-c:a", "libvorbis",
                    "-q:a", "5",
                    str(ogg_path),
                    "-y",
                ],
                capture_output=True,
                timeout=600,
            )
            if result.returncode != 0:
                self.logger.error(
                    f"[MapDisplays] FFmpeg audio extraction failed:\n"
                    f"{result.stderr.decode('utf-8', errors='replace')}"
                )
                return None

            self.logger.info(
                f"[MapDisplays] Audio extracted: {ogg_path.name}"
            )
            self._register_sound_event(stem)
            self._print_resourcepack_notice()
            return event_name

        except subprocess.TimeoutExpired:
            self.logger.error("[MapDisplays] FFmpeg timeout — video may be very long.")
            return None
        except Exception as exc:
            self.logger.error(f"[MapDisplays] Audio extraction error: {exc}")
            return None

    def _register_sound_event(self, stem: str) -> None:
        """Add a sound event entry to resourcepack/sounds.json, then bump pack version."""
        sounds_path = Path(self.data_folder) / "resourcepack" / "sounds.json"
        try:
            sounds: dict = json.loads(sounds_path.read_text()) if sounds_path.exists() else {}
            sounds[f"mapdisplays.{stem}"] = {
                "sounds": [f"sounds/mapdisplays/{stem}"],
                "category": "neutral",
                "min_distance": 0,
                "max_distance": 0,
            }
            sounds_path.write_text(json.dumps(sounds, indent=2))
        except Exception as exc:
            self.logger.error(f"[MapDisplays] Failed to update sounds.json: {exc}")
            return
        # Bump version so clients pick up the new audio
        self._bump_resourcepack_version(reason=f"added audio: {stem}")

    def _unregister_sound_event(self, stem: str) -> None:
        """Remove a sound event entry from sounds.json, delete OGG, then bump pack version."""
        sounds_path = Path(self.data_folder) / "resourcepack" / "sounds.json"
        ogg_path = (
            Path(self.data_folder)
            / "resourcepack" / "sounds" / "mapdisplays" / f"{stem}.ogg"
        )
        try:
            if sounds_path.exists():
                sounds: dict = json.loads(sounds_path.read_text())
                sounds.pop(f"mapdisplays.{stem}", None)
                sounds_path.write_text(json.dumps(sounds, indent=2))
        except Exception as exc:
            self.logger.error(f"[MapDisplays] Failed to update sounds.json on remove: {exc}")
        try:
            if ogg_path.exists():
                ogg_path.unlink()
        except Exception as exc:
            self.logger.error(f"[MapDisplays] Failed to delete OGG '{stem}.ogg': {exc}")
        self._bump_resourcepack_version(reason=f"removed audio: {stem}")

    def _bump_resourcepack_version(self, reason: str = "") -> None:
        """
        Increment the patch component of the resource pack manifest version,
        then push the new version into world_resource_packs.json.
        """
        manifest_path = Path(self.data_folder) / "resourcepack" / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
            ver: list = list(manifest["header"]["version"])
            ver[2] += 1  # bump patch
            manifest["header"]["version"] = ver
            # Keep module version in sync with header version
            for mod in manifest.get("modules", []):
                mod["version"] = ver
            manifest_path.write_text(json.dumps(manifest, indent=2))
            self.logger.info(
                f"[MapDisplays] Resource pack version bumped to "
                f"{ver[0]}.{ver[1]}.{ver[2]}"
                + (f" ({reason})" if reason else "")
            )
            self._update_world_resource_packs(ver)
        except Exception as exc:
            self.logger.error(f"[MapDisplays] Failed to bump resource pack version: {exc}")

    def _update_world_resource_packs(self, version: list | None = None) -> None:
        """
        Register (or update) our resource pack entry in
        <world_folder>/world_resource_packs.json.

        If version is None the current version is read from manifest.json.
        Creates the file if it doesn't exist.
        """
        world = self._get_world_folder()
        if not world:
            return  # world_folder not configured or not found

        if version is None:
            # Read current version from manifest
            try:
                manifest_path = Path(self.data_folder) / "resourcepack" / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                version = list(manifest["header"]["version"])
            except Exception as exc:
                self.logger.error(
                    f"[MapDisplays] Cannot read manifest version for world pack update: {exc}"
                )
                return

        wrp_path = world / "world_resource_packs.json"
        try:
            packs: list = json.loads(wrp_path.read_text()) if wrp_path.exists() else []
        except Exception:
            packs = []

        # Find existing entry for our pack UUID, or create a new one
        entry = next((p for p in packs if p.get("pack_id") == _RP_HEADER_UUID), None)
        if entry is None:
            entry = {"pack_id": _RP_HEADER_UUID, "version": version}
            packs.append(entry)
        else:
            entry["version"] = version

        try:
            wrp_path.write_text(json.dumps(packs, indent=2))
            self.logger.info(
                f"[MapDisplays] world_resource_packs.json updated — "
                f"pack version {version[0]}.{version[1]}.{version[2]} — "
                f"world: {world.name}"
            )
        except Exception as exc:
            self.logger.error(
                f"[MapDisplays] Failed to write world_resource_packs.json: {exc}"
            )

    def _print_resourcepack_notice(self) -> None:
        world = self._get_world_folder()
        rp_path = Path(self.data_folder) / "resourcepack"
        self.logger.info("=" * 64)
        self.logger.info("[MapDisplays] RESOURCE PACK UPDATED")
        self.logger.info(f"  Pack location : {rp_path}")
        if world:
            self.logger.info(f"  World folder  : {world}")
            self.logger.info("  world_resource_packs.json has been updated automatically.")
        else:
            self.logger.info(
                "  world_folder is not set in config.json — "
                "distribute the pack to clients manually."
            )
        self.logger.info("=" * 64)

    def _play_sound_all(self, sound_name: str) -> None:
        """Start playing a sound event for every online player."""
        for player in self.server.online_players:
            try:
                player.play_sound(player.location, sound_name, 1.0, 1.0)
            except Exception:
                pass

    def _stop_sound_for(self, display: MapDisplay) -> None:
        """Stop the current sound event for this display on all online players."""
        sound = display.state.sound_name
        if not sound:
            return
        for player in self.server.online_players:
            try:
                player.stop_sound(sound)
            except Exception:
                pass

    def _bind_sound_loop(self, display: MapDisplay, state: VideoFileState) -> None:
        """
        Bind a callback to VideoFileState.on_loop so that when the video
        restarts, the Bedrock sound event is stopped then replayed for all
        online players, keeping audio and video in sync.
        """
        def _on_loop():
            sound = state.sound_name
            if not sound or not self._running:
                return

            def _restart_sound():
                for player in self.server.online_players:
                    try:
                        player.stop_sound(sound)
                    except Exception:
                        pass
                # Brief delay before replay so the stop packet lands first
                def _replay():
                    for player in self.server.online_players:
                        try:
                            player.play_sound(player.location, sound, 1.0, 1.0)
                        except Exception:
                            pass
                self.server.scheduler.run_task(self, _replay, delay=2)

            self.server.scheduler.run_task(self, _restart_sound)

        state.on_loop = _on_loop