"""Queue-based background thread worker."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ThreadWorker:
    """Runs queued callables in a persistent background thread.

    Functions submitted via :meth:`submit` are executed sequentially in the
    order they are received.  The worker thread is a daemon so it will not
    prevent the process from exiting.

    Example::

        worker = ThreadWorker()
        worker.submit(requests.get, "http://example.com")
        worker.join()
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[Callable, tuple, dict]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ThreadWorker")
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        """Queue *fn* for execution in the background thread.

        Args:
            fn:     Callable to invoke.
            *args:  Positional arguments forwarded to *fn*.
            **kwargs: Keyword arguments forwarded to *fn*.
        """
        self._queue.put((fn, args, kwargs))

    def join(self) -> None:
        """Block until all currently queued tasks have completed."""
        self._queue.join()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loop executed in the background thread."""
        while True:
            fn, args, kwargs = self._queue.get()
            try:
                fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.error("ThreadWorker: unhandled exception in submitted task: %s", exc)
            finally:
                self._queue.task_done()
