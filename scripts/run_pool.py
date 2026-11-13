"""General CLI over the reusable GPU work-pool (``lib.utilities.run_pool``).

Reads a stream of work-units and runs them across ``--max-workers`` work-stealing threads, each launched
only when live free VRAM allows (``free >= peak + reserve``). It is "``xargs -P`` / GNU ``parallel`` for a
single shared GPU": continuous work-stealing + per-job VRAM admission + resumable skip + subprocess
isolation. Reach for it when the bottleneck is **GPU memory**; for RAM/CPU-bound fan-out ``parallel
--memfree`` already suffices.

Two input formats (stdin or ``--units-file``):
  * **JSONL** (default): one object per line, e.g.
      {"key": "enc/lolo/V1/ep5", "command": ["python", "scripts/encoding_analysis.py", "..."], "peak_gb": 5}
    (``env`` and ``peak_gb`` optional).
  * **``--shell``**: one shell command per line; each becomes ``bash -c <line>`` keyed ``line{N}``
      (``generate_cmds.sh | python scripts/run_pool.py --shell --max-workers 6 --peak-gb 5``).

Examples:
  printf '%s\\n' '{"key":"a","command":["echo","hi"]}' | python scripts/run_pool.py --max-workers 2 --no-gate
  cat units.jsonl | python scripts/run_pool.py --max-workers 6 --peak-gb 5 --reserve-gb 4 --log-dir artifacts/pool
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from loguru import logger  # noqa: E402

from lib.utilities import GpuGate, WorkUnit, run_pool  # noqa: E402


def _read_units(stream, shell: bool) -> list[WorkUnit]:
    units: list[WorkUnit] = []
    for i, raw in enumerate(stream):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if shell:
            units.append(WorkUnit(key=f"line{i}", command=["bash", "-c", line]))
        else:
            obj = json.loads(line)
            units.append(WorkUnit(
                key=obj["key"], command=list(obj["command"]),
                env=obj.get("env"), peak_gb=obj.get("peak_gb"),
            ))
    return units


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--units-file", default=None, help="read units from this file instead of stdin")
    p.add_argument("--shell", action="store_true", help="input is one shell command per line (not JSONL)")
    p.add_argument("--max-workers", type=int, required=True, help="optimistic concurrency ceiling")
    p.add_argument("--peak-gb", type=float, default=6.0, help="default per-unit GPU peak for the gate")
    p.add_argument("--reserve-gb", type=float, default=4.0, help="free-VRAM headroom held back")
    p.add_argument("--device", type=int, default=0, help="CUDA device index for the gate")
    p.add_argument("--no-gate", action="store_true", help="disable GPU gating (pure work-steal)")
    p.add_argument("--threads-per-worker", type=int, default=None,
                   help="OMP/MKL/OPENBLAS/NUMEXPR threads per unit (default cpu_count//max_workers)")
    p.add_argument("--retries", type=int, default=0, help="extra attempts for a unit that exits non-zero")
    p.add_argument("--log-dir", default=None, help="per-unit stdout+stderr -> {log-dir}/{key}.log")
    p.add_argument("--poll-seconds", type=float, default=2.0, help="gate re-check interval when tight")
    p.add_argument("--launch-cooldown-s", type=float, default=2.0,
                   help="settle pause held under the admission lock after a gated launch")
    p.add_argument("--skip-existing", default=None, metavar="TEMPLATE",
                   help="resume: skip a unit when glob(TEMPLATE.format(key=...)) matches "
                        "(e.g. '/cache/{key}.nc'); TEMPLATE may contain shell wildcards")
    args = p.parse_args()

    stream = open(args.units_file) if args.units_file else sys.stdin
    try:
        units = _read_units(stream, args.shell)
    finally:
        if args.units_file:
            stream.close()
    if not units:
        logger.warning("no work-units on input; nothing to do")
        return

    is_done = None
    if args.skip_existing:
        template = args.skip_existing
        is_done = lambda u: bool(glob.glob(template.format(key=u.key)))  # noqa: E731

    gate = None if args.no_gate else GpuGate(reserve_gb=args.reserve_gb, device=args.device)
    report = run_pool(
        units, max_workers=args.max_workers, gpu_gate=gate, default_peak_gb=args.peak_gb,
        is_done=is_done, threads_per_worker=args.threads_per_worker, max_retries=args.retries,
        log_dir=args.log_dir, poll_seconds=args.poll_seconds, launch_cooldown_s=args.launch_cooldown_s,
    )
    print(report.summary())
    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
