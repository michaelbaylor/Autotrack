"""Unit tests for ThreadWorker."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from autotrack.utils.thread_worker import ThreadWorker


class TestThreadWorker:
    def test_submit_executes_function(self):
        """A submitted function should be called by the worker thread."""
        results: list[int] = []
        worker = ThreadWorker()

        worker.submit(results.append, 42)
        worker.join()

        assert results == [42]

    def test_join_waits_for_completion(self):
        """join() should block until the submitted work is finished."""
        completed: list[bool] = []

        def slow_task():
            time.sleep(0.05)
            completed.append(True)

        worker = ThreadWorker()
        worker.submit(slow_task)
        worker.join()

        assert completed == [True]

    def test_exception_in_submitted_function_does_not_crash_worker(self):
        """An exception thrown by a submitted function must not kill the worker."""
        results: list[str] = []

        def bad_task():
            raise RuntimeError("intentional error")

        def good_task():
            results.append("ok")

        worker = ThreadWorker()
        worker.submit(bad_task)
        worker.submit(good_task)
        worker.join()

        assert results == ["ok"]

    def test_multiple_sequential_submits_all_execute(self):
        """All submitted tasks should execute in order."""
        results: list[int] = []
        worker = ThreadWorker()

        for i in range(10):
            worker.submit(results.append, i)

        worker.join()

        assert results == list(range(10))

    def test_submit_with_kwargs(self):
        """submit() should forward keyword arguments to the callable."""
        received: dict[str, Any] = {}

        def record_kwargs(**kwargs):
            received.update(kwargs)

        worker = ThreadWorker()
        worker.submit(record_kwargs, key="value", number=99)
        worker.join()

        assert received == {"key": "value", "number": 99}

    def test_worker_thread_is_daemon(self):
        """The background thread should be a daemon so it does not block process exit."""
        worker = ThreadWorker()
        assert worker._thread.daemon is True

    def test_worker_is_alive_after_construction(self):
        """The background thread should be running immediately after creation."""
        worker = ThreadWorker()
        assert worker._thread.is_alive()

    def test_task_done_called_after_exception(self):
        """task_done() must be called even when the submitted function raises,
        so that a subsequent join() does not deadlock."""
        worker = ThreadWorker()

        worker.submit(lambda: (_ for _ in ()).throw(ValueError("oops")))

        # join() would block forever if task_done() was not called on exception
        finished = threading.Event()

        def set_event():
            finished.set()

        worker.submit(set_event)
        worker.join()

        assert finished.is_set()
