"""When the app ends. Pure logic: the server feeds it one tick per second."""

import time
from pathlib import Path

# A tick that comes this late means that the PC slept, not that the window was closed.
SLEEP_GAP_SECONDS = 15


class Heartbeat:
    """Touch a file only while the monitor ticks and the event loop answers it.

    The launcher watches the file until Start, so neither a dead monitor nor a hung server can
    make Metashape wait forever.
    """

    def __init__(self, path: Path):
        self.path = path
        self._answered = False

    def _answer(self) -> None:
        self._answered = True

    def tick(self, schedule) -> None:
        """``schedule`` runs a callback on the event loop."""
        if self._answered:
            self._answered = False
            self.path.touch()
        schedule(self._answer)


class Lifecycle:
    """Decide from the connected windows and the run state whether the app ends.

    - Before Start a closed window means cancel, after a grace period for reloads.
    - During a run a reload, sleep or lost socket never cancels anything.
    - After the final state the app ends once that state was seen and the window is closed.
    - Without any window it ends as well, so it cannot hang or remain as an orphan process.
    """

    def __init__(
        self,
        connect_timeout: float = 120,
        close_grace: float = 10,
        final_grace: float = 120,
        clock=time.monotonic,
    ):
        self.connect_timeout = connect_timeout
        self.close_grace = close_grace
        self.final_grace = final_grace
        self._clock = clock
        self._last = clock()
        self._absent = 0.0
        self._state = None
        self._connected_once = False
        self._final_seen = False

    def tick(self, state: str, windows: int, standalone: bool = False) -> str | None:
        """Return None, "cancel" (closed before Start), "no_window" or "exit"."""
        now = self._clock()
        elapsed, self._last = now - self._last, now
        if elapsed > SLEEP_GAP_SECONDS:
            self._absent = elapsed = 0.0
        final = state in ("done", "error", "cancelled")
        if state != self._state:
            if self._state is not None:
                self._absent = elapsed = 0.0
            self._state = state
        if windows:
            self._absent = 0.0
            self._connected_once = True
            self._final_seen |= final
            return None
        self._absent += elapsed
        if not standalone:
            if state == "running":
                return None
            if final:
                grace = self.close_grace if self._final_seen else self.final_grace
                return "exit" if self._absent >= grace else None
        if not self._connected_once:
            return "no_window" if self._absent >= self.connect_timeout else None
        if self._absent < self.close_grace:
            return None
        return "exit" if standalone else "cancel"
