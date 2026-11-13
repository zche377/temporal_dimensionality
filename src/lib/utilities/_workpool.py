"""Reusable continuous / dynamic-N work-pool: a global work-stealing queue over independent subprocess
units, admission-gated by live GPU memory.

Replaces the "static epoch-shard + barrier" pattern (launch ``N`` processes over fixed ``epochs[i::N]``
slices, ``wait`` for all ``N``, repeat per ``(model, ROI)``) — whose fast workers idle at every boundary
— with ``max_workers`` persistent threads that each steal the next :class:`WorkUnit` from one shared
queue until it drains. No barriers; effective concurrency floats with free VRAM via the :class:`GpuGate`.

Threads (not processes) are correct here because every worker spends essentially all its time blocked in
``subprocess.wait()``, which releases the GIL — real parallelism across the child processes, with the
parent doing only negligible bookkeeping.

Design notes worth knowing before reusing this:
  * **Admission safety is the reserve, not the stagger.** ``GpuGate.admit`` reads *instantaneous* free
    mem, but a just-launched GPU job takes seconds-to-minutes to ramp to its peak (import torch -> load
    data -> compute burst). A generous ``reserve_gb`` is what absorbs in-flight allocations a freshly
    admitted worker will consume but hasn't yet. The admission lock + ``launch_cooldown_s`` only close
    the *sub-second* race of two workers reading the same snapshot at once; they cannot cover the ramp.
  * **The feature/result cache this typically feeds is non-atomic and unlocked.** Two units that
    concurrently write the *same* file corrupt it. Point the pool only at units whose writes are disjoint
    (e.g. partitioned by epoch) or read-only. This is a *caller* discipline; the engine does not enforce
    it.
  * **A crash/OOM is a clean miss** as long as the unit writes its output atomically-or-not-at-all: the
    failed unit's ``is_done`` stays False and a re-run fills only the gaps. That is the whole resume story.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Callable, Iterable, Mapping, Sequence

from loguru import logger

from lib.utilities._gpu import gpu_free_gb


@dataclass(frozen=True)
class WorkUnit:
    """One schedulable unit: an argv run as a subprocess, ``shell=False``.

    ``key`` must be unique — it names the per-unit log file and keys the report. ``peak_gb`` overrides the
    pool's default gate budget for a heavier/lighter unit. ``depends_on`` is RESERVED (a DAG-ready hook);
    v1 requires it empty.
    """

    key: str
    command: Sequence[str]
    env: Mapping[str, str] | None = None
    peak_gb: float | None = None
    depends_on: tuple[str, ...] = ()


@dataclass
class GpuGate:
    """Admission control on live free VRAM — the "dynamic-N" lever.

    A worker may launch its unit only when ``free_gb() >= peak_gb + reserve_gb``; otherwise it waits and
    polls, so the pool's *effective* concurrency shrinks and grows with real free memory instead of a
    fixed count. ``free_gb`` is an instance method (not a bare call) precisely so tests can monkeypatch it
    with a scripted sequence and exercise the backpressure logic with no CUDA context.
    """

    reserve_gb: float = 4.0
    device: int = 0

    def free_gb(self) -> float | None:
        return gpu_free_gb(self.device)

    def admit(self, peak_gb: float) -> bool:
        free = self.free_gb()
        if free is None:  # unknown free mem -> cannot gate, allow (degrade to pure work-steal)
            return True
        return free >= peak_gb + self.reserve_gb


@dataclass
class PoolReport:
    """Outcome of a :func:`run_pool` call."""

    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # is_done() True at pop-time
    failed: list[tuple[str, int]] = field(default_factory=list)  # (key, final returncode)
    retried: dict[str, int] = field(default_factory=dict)  # key -> retries consumed
    elapsed_s: float = 0.0
    per_unit_s: dict[str, float] = field(default_factory=dict)  # key -> total wall time across attempts

    def summary(self) -> str:
        return (
            f"work-pool: {len(self.done)} done, {len(self.skipped)} skipped, "
            f"{len(self.failed)} failed in {self.elapsed_s:.1f}s"
            + (f" (failed: {[k for k, _ in self.failed]})" if self.failed else "")
        )


def _safe_name(key: str) -> str:
    """A filesystem-safe basename for a unit's log file (keys may contain '/', ':' etc.)."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in key) or "unit"


def _kill_tree(proc: subprocess.Popen, sig: int) -> None:
    """Signal ``proc``'s whole process group (it is a session leader via ``start_new_session=True``)."""
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        pass  # already gone / not ours -> nothing to do


def run_pool(
    units: Iterable[WorkUnit],
    *,
    max_workers: int,
    gpu_gate: GpuGate | None = None,
    default_peak_gb: float = 6.0,
    is_done: Callable[[WorkUnit], bool] | None = None,
    threads_per_worker: int | None = None,
    max_retries: int = 0,
    log_dir: Path | str | None = None,
    poll_seconds: float = 2.0,
    launch_cooldown_s: float = 2.0,
) -> PoolReport:
    """Run ``units`` across ``max_workers`` work-stealing threads until the queue drains.

    Args:
        max_workers: optimistic concurrency ceiling; the ``gpu_gate`` throttles the real number in flight.
        gpu_gate: admission control on free VRAM. ``None`` -> pure work-steal at ``max_workers``.
        default_peak_gb: gate budget for units whose ``peak_gb`` is ``None``.
        is_done: skip a popped unit for which this returns True (resume / cache-hit); re-checked before
            each attempt so a sibling finishing it mid-flight never burns a slot.
        threads_per_worker: value for ``OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`` in each unit's env; when
            ``None`` defaults to ``max(1, cpu_count // max_workers)`` (mirrors the shard launchers).
        max_retries: extra attempts for a unit that exits non-zero (subprocess isolation: a failure never
            kills the pool).
        log_dir: per-unit stdout+stderr -> ``{log_dir}/{safe key}.log``; ``None`` inherits the parent's.
        poll_seconds: gate re-check interval while VRAM is tight.
        launch_cooldown_s: pause held under the admission lock after a gated launch, so the next worker
            measures only after this one has begun allocating (settle window; ignored when no gate).

    Returns:
        A :class:`PoolReport`. Also logs a one-line summary.
    """
    unit_list = list(units)
    if any(u.depends_on for u in unit_list):
        raise ValueError("WorkUnit.depends_on is reserved; the DAG scheduler is not implemented in v1")
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    if threads_per_worker is None:
        threads_per_worker = max(1, (os.cpu_count() or max_workers) // max_workers)
    log_path = Path(log_dir) if log_dir is not None else None
    if log_path is not None:
        log_path.mkdir(parents=True, exist_ok=True)

    q: queue.Queue[WorkUnit] = queue.Queue()
    for u in unit_list:
        q.put(u)

    report = PoolReport()
    report_lock = threading.Lock()
    admit_lock = threading.Lock()  # serialize gated admission (close the sub-second snapshot race)
    live_lock = threading.Lock()
    live: dict[str, subprocess.Popen] = {}
    stop = threading.Event()

    def build_env(unit: WorkUnit) -> dict[str, str]:
        env = os.environ.copy()
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = str(threads_per_worker)
        if unit.env:
            env.update(unit.env)
        return env

    def launch(unit: WorkUnit, env: dict[str, str], out) -> subprocess.Popen | None:
        """Wait for admission (if gated), then spawn. Returns None if we stopped while waiting."""
        peak = unit.peak_gb if unit.peak_gb is not None else default_peak_gb
        while not stop.is_set():
            if gpu_gate is None:
                proc = subprocess.Popen(unit.command, env=env, stdout=out,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                with live_lock:
                    live[unit.key] = proc
                return proc
            with admit_lock:
                if gpu_gate.admit(peak):
                    proc = subprocess.Popen(unit.command, env=env, stdout=out,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    with live_lock:
                        live[unit.key] = proc
                    if launch_cooldown_s > 0:
                        time.sleep(launch_cooldown_s)  # settle window, held under admit_lock
                    return proc
            # not admitted: lock released, back off before re-measuring
            stop.wait(poll_seconds)
        return None

    def process(unit: WorkUnit) -> None:
        if is_done is not None and is_done(unit):
            with report_lock:
                report.skipped.append(unit.key)
            return
        env = build_env(unit)
        attempts = max_retries + 1
        for attempt in range(attempts):
            if stop.is_set():
                return
            if is_done is not None and is_done(unit):  # a sibling may have finished it while queued
                with report_lock:
                    report.skipped.append(unit.key)
                return
            out = None
            if log_path is not None:
                out = open(log_path / f"{_safe_name(unit.key)}.log", "w" if attempt == 0 else "a")
            t0 = monotonic()
            try:
                proc = launch(unit, env, out)
                if proc is None:  # stopped during admission wait
                    return
                rc = proc.wait()
            finally:
                with live_lock:
                    live.pop(unit.key, None)
                if out is not None:
                    out.close()
            with report_lock:
                report.per_unit_s[unit.key] = report.per_unit_s.get(unit.key, 0.0) + (monotonic() - t0)
            if rc == 0:
                with report_lock:
                    report.done.append(unit.key)
                return
            if attempt < attempts - 1:
                with report_lock:
                    report.retried[unit.key] = report.retried.get(unit.key, 0) + 1
                logger.warning(f"unit {unit.key!r} exited {rc}; retry {attempt + 1}/{max_retries}")
            else:
                with report_lock:
                    report.failed.append((unit.key, rc))
                logger.error(f"unit {unit.key!r} failed (exit {rc}) after {attempts} attempt(s)")

    def worker() -> None:
        while not stop.is_set():
            try:
                unit = q.get_nowait()
            except queue.Empty:
                return
            try:
                process(unit)
            except Exception as e:  # noqa: BLE001 — never let one unit's bug kill its worker thread
                logger.exception(f"unit {unit.key!r} raised in the pool: {e}")
                with report_lock:
                    report.failed.append((unit.key, -1))
            finally:
                q.task_done()

    logger.info(
        f"work-pool: {len(unit_list)} units, max_workers={max_workers}, "
        f"{'gated (reserve %.1fGB)' % gpu_gate.reserve_gb if gpu_gate else 'no gate'}, "
        f"threads/worker={threads_per_worker}"
    )
    t_start = monotonic()
    threads = [threading.Thread(target=worker, name=f"pool-{i}") for i in range(max_workers)]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.2)
    except KeyboardInterrupt:
        logger.warning("work-pool interrupted; terminating live subprocesses")
        stop.set()
        with live_lock:
            procs = list(live.values())
        for proc in procs:
            _kill_tree(proc, signal.SIGTERM)
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _kill_tree(proc, signal.SIGKILL)
        for t in threads:
            t.join(timeout=15)
        report.elapsed_s = monotonic() - t_start
        raise
    report.elapsed_s = monotonic() - t_start
    logger.info(report.summary())
    return report
