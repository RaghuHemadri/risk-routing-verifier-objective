"""
Asynchronous JSONL writer — decouples disk I/O from GPU compute.

A background thread drains a queue and writes JSON lines to disk so the
main (GPU) thread never blocks on ``f.write()`` / ``f.flush()``.

Usage
-----
    writer = AsyncJSONLWriter("output.jsonl", mode="a")
    writer.start()

    for batch in gpu_loop:
        results = model(batch)
        writer.write(results)     # returns instantly

    writer.close()                # flushes remaining items & joins thread

The writer is also a context manager::

    with AsyncJSONLWriter("out.jsonl") as w:
        w.write({"key": "value"})
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel object to signal the writer thread to shut down.
_STOP = object()


class AsyncJSONLWriter:
    """Thread-safe, non-blocking JSONL writer backed by a ``queue.Queue``.

    Parameters
    ----------
    path : str | Path
        Destination JSONL file.
    mode : str
        File open mode — ``"w"`` to overwrite, ``"a"`` to append.
    flush_every : int
        Call ``file.flush()`` after this many writes (1 = every write).
    maxsize : int
        Maximum items buffered in the queue.  ``0`` means unlimited.
        If the queue is full, ``write()`` will block until space is
        available — this provides natural back-pressure if the disk
        can't keep up.
    json_kwargs : dict | None
        Extra keyword arguments forwarded to ``json.dumps``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str = "a",
        flush_every: int = 1,
        maxsize: int = 0,
        json_kwargs: dict | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = mode
        self._flush_every = max(1, flush_every)
        self._json_kwargs = json_kwargs or {"default": str}
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._thread: threading.Thread | None = None
        self._started = False
        self._items_written = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "AsyncJSONLWriter":
        """Launch the background writer thread.  Idempotent."""
        with self._lock:
            if self._started:
                return self
            self._started = True
        self._thread = threading.Thread(
            target=self._drain_loop, name=f"jsonl-writer-{self.path.name}", daemon=True
        )
        self._thread.start()
        logger.debug("AsyncJSONLWriter started → %s", self.path)
        return self

    def write(self, record: dict[str, Any]) -> None:
        """Enqueue a single JSON-serialisable dict (non-blocking)."""
        if not self._started:
            raise RuntimeError("Writer not started — call .start() first")
        self._queue.put(record)

    def write_many(self, records: list[dict[str, Any]]) -> None:
        """Enqueue multiple records at once."""
        for rec in records:
            self.write(rec)

    def write_raw(self, line: str) -> None:
        """Enqueue a pre-serialised line (no json.dumps applied)."""
        if not self._started:
            raise RuntimeError("Writer not started — call .start() first")
        self._queue.put(("__raw__", line))

    @property
    def pending(self) -> int:
        """Approximate number of items still queued."""
        return self._queue.qsize()

    @property
    def items_written(self) -> int:
        """Total items flushed to disk so far."""
        return self._items_written

    def flush(self) -> None:
        """Block until the queue is fully drained to disk."""
        self._queue.join()

    def close(self) -> None:
        """Signal the writer to finish, drain remaining items, and join."""
        if not self._started:
            return
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join()
        self._started = False
        logger.debug(
            "AsyncJSONLWriter closed (%d items written) → %s",
            self._items_written, self.path,
        )

    # Context manager support
    def __enter__(self) -> "AsyncJSONLWriter":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Background writer loop
    # ------------------------------------------------------------------

    def _drain_loop(self) -> None:
        """Runs on the background thread — drains queue → file."""
        writes_since_flush = 0
        with open(self.path, self._mode) as f:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    f.flush()
                    self._queue.task_done()
                    break

                try:
                    if isinstance(item, tuple) and len(item) == 2 and item[0] == "__raw__":
                        line = item[1]
                    else:
                        line = json.dumps(item, **self._json_kwargs)
                    f.write(line + "\n")
                    writes_since_flush += 1
                    self._items_written += 1

                    if writes_since_flush >= self._flush_every:
                        f.flush()
                        writes_since_flush = 0
                except Exception:
                    logger.exception("AsyncJSONLWriter: failed to write record")
                finally:
                    self._queue.task_done()
