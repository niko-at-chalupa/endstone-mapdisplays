from .mapdisplays_states import PyIdleState, PyDisplayState
from importlib import resources
import contextlib

class StateManager:
    def __init__(self, width: int, height: int, resource_path: str = "resources/mapdisplays_idle.webm"):
        self._exitstack = contextlib.ExitStack()

        self._width = width
        self._height = height
        self._state: PyDisplayState = PyIdleState(width, height, self._get_resource_path(resource_path), 20)
        #self._state: PyDisplayState = PyIdleState(width, height, resource_path)

    def _get_resource_path(self, resource_path: str) -> str:
        ref = resources.files("endstone_mapdisplays").joinpath(resource_path)
        path = self._exitstack.enter_context(resources.as_file(ref))
        return str(path)

    def transition(self, new_state: PyDisplayState):
        old = self._state
        self._state = new_state
        old.stop()

    def get_full_frame(self):
        return self._state.get_full_frame()

    def stop(self):
        self._state.stop()
        self._exitstack.close()