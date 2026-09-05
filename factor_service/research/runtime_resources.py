"""Read the node's effective limits, not the physical host's advertised RAM.

This module stays standard-library-only so the remote supervisor does not load
Qlib, pandas or a second copy of the training runtime into memory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import os
from pathlib import Path
import re
import subprocess
import sys


MIB = 1024 ** 2
GIB = 1024 ** 3
MEMORY_RESERVE_RATIO = 0.15
MEMORY_HARD_FLOOR_BYTES = 256 * MIB
MEMORY_LOW_SAMPLE_LIMIT = 3


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _integer(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def _pairs(path: Path) -> dict[str, int]:
    values = {}
    for line in _read(path).splitlines():
        parts = line.split()
        if len(parts) >= 2:
            values[parts[0].rstrip(":")] = _integer(parts[1])
    return values


def _cgroup_paths(proc_root: Path, cgroup_root: Path, controller: str) -> list[Path]:
    # A container may expose its own root, or a nested host cgroup. Inspect both
    # the visible root and ancestors: a leaf with memory.max=max is still bounded
    # by its parent's limit.
    bases = [cgroup_root, cgroup_root / controller]
    if controller == "cpu":
        bases.append(cgroup_root / "cpu,cpuacct")
    paths = list(bases)
    for line in _read(proc_root / "self/cgroup").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        unified = parts[0] == "0" and not parts[1]
        if not unified and controller not in parts[1].split(","):
            continue
        relative = Path(parts[2].lstrip("/"))
        if ".." in relative.parts:
            continue
        for base in ([cgroup_root] if unified else bases[1:]):
            path = base / relative
            while path != base:
                paths.append(path)
                path = path.parent
    return list(dict.fromkeys(paths))


@dataclass(frozen=True)
class RuntimeResources:
    cpu_cores: int
    memory_limit_bytes: int
    memory_available_bytes: int
    memory_source: str
    process_rss_bytes: int = 0
    oom_kill_count: int = 0

    @property
    def reserve_bytes(self) -> int:
        return max(512 * MIB, int(self.memory_limit_bytes * MEMORY_RESERVE_RATIO))

    @property
    def training_headroom_bytes(self) -> int:
        return max(0, self.memory_available_bytes - self.reserve_bytes)

    @property
    def runtime_reserve_bytes(self) -> int:
        # The 15% planning reserve is advisory, not a measured runtime limit.
        # Keep a bounded emergency margin while allowing a supervised attempt.
        return max(512 * MIB, min(GIB, int(self.memory_limit_bytes * 0.05)))

    def public(self) -> dict[str, int | str | float]:
        return {
            **asdict(self),
            "memory_reserve_bytes": self.reserve_bytes,
            "training_headroom_bytes": self.training_headroom_bytes,
            "runtime_memory_floor_bytes": self.runtime_reserve_bytes,
            "memory_used_bytes": self.memory_limit_bytes - self.memory_available_bytes,
        }


class RuntimeMemoryGuard:
    """Ignore one low sample, but stop persistent or critically low capacity."""

    def __init__(self) -> None:
        self.low_samples = 0

    def observe(self, resources: RuntimeResources) -> bool:
        low = resources.memory_available_bytes < resources.runtime_reserve_bytes
        self.low_samples = self.low_samples + 1 if low else 0
        return (resources.memory_available_bytes <= MEMORY_HARD_FLOOR_BYTES
                or self.low_samples >= MEMORY_LOW_SAMPLE_LIMIT)


def _command_output(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=2, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _darwin_resources(pid: int | None) -> RuntimeResources:
    hardware = _command_output(["/usr/sbin/sysctl", "-n", "hw.memsize", "hw.logicalcpu"]).splitlines()
    total = _integer(hardware[0]) if hardware else 0
    cores = max(1, _integer(hardware[1]) if len(hardware) > 1 else os.cpu_count() or 1)
    vm = _command_output(["/usr/bin/vm_stat"])
    page_size = re.search(r"page size of (\d+) bytes", vm)
    pages = {
        name: int(count)
        for name, count in re.findall(r"^(Pages [^:]+):\s+(\d+)\.", vm, re.MULTILINE)
    }
    required = ("Pages free", "Pages inactive", "Pages speculative")
    if total <= 0 or not page_size or not all(name in pages for name in required):
        # Missing probes must not turn into an invented budget or bypass the guard.
        return RuntimeResources(cores, 0, 0, "unavailable")
    # Apple's vm_stat subtracts speculative pages from its displayed free count.
    # Add them back once, plus inactive reclaimable pages; do not add purgeable
    # pages again or count compressed/swap space as physical training capacity.
    available = sum(pages[name] for name in required) * int(page_size.group(1))
    rss = _integer(_command_output(["/bin/ps", "-o", "rss=", "-p", str(pid or os.getpid())])) * 1024
    return RuntimeResources(
        cpu_cores=cores, memory_limit_bytes=total,
        memory_available_bytes=min(total, max(0, available)),
        memory_source="darwin_vm_stat", process_rss_bytes=max(0, rss),
    )


def read_runtime_resources(
    *, pid: int | None = None, proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> RuntimeResources:
    if sys.platform == "darwin" and proc_root == Path("/proc") and cgroup_root == Path("/sys/fs/cgroup"):
        return _darwin_resources(pid)
    memory = _pairs(proc_root / "meminfo")
    total = memory.get("MemTotal", 0) * 1024
    available = memory.get("MemAvailable", memory.get("MemFree", 0)) * 1024
    source = "host" if total else "unavailable"
    oom_kills = 0
    for path in _cgroup_paths(proc_root, cgroup_root, "memory"):
        for limit_file, usage_file, inactive_key, label in (
            ("memory.max", "memory.current", "inactive_file", "cgroup_v2"),
            ("memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file", "cgroup_v1"),
        ):
            limit = _integer(_read(path / limit_file))
            usage_text = _read(path / usage_file)
            if not 0 < limit < 2 ** 60 or not usage_text:
                continue
            stats = _pairs(path / "memory.stat")
            # Inactive filesystem cache is reclaimable. Counting it as live
            # training RAM would kill jobs just after copying their Parquet files.
            working = max(0, _integer(usage_text) - stats.get(inactive_key, 0))
            remaining = max(0, limit - working)
            available = min(available, remaining) if total else remaining
            if not total or limit <= total:
                total, source = limit, label
        events = _pairs(path / "memory.events")
        oom_kills = max(oom_kills, events.get("oom_kill", 0))

    cores = max(1, os.cpu_count() or 1)
    if hasattr(os, "sched_getaffinity"):
        try:
            cores = min(cores, max(1, len(os.sched_getaffinity(0))))
        except OSError:
            pass
    for path in _cgroup_paths(proc_root, cgroup_root, "cpu"):
        quota = _read(path / "cpu.max").split()
        if len(quota) == 2 and _integer(quota[0]) > 0 and _integer(quota[1]) > 0:
            cores = min(cores, max(1, _integer(quota[0]) // _integer(quota[1])))
        quota_us = _integer(_read(path / "cpu.cfs_quota_us"))
        period_us = _integer(_read(path / "cpu.cfs_period_us"))
        if quota_us > 0 and period_us > 0:
            cores = min(cores, max(1, quota_us // period_us))
    status = _pairs(proc_root / str(pid or os.getpid()) / "status")
    return RuntimeResources(
        cpu_cores=cores,
        memory_limit_bytes=total,
        memory_available_bytes=min(total, max(0, available)),
        memory_source=source,
        process_rss_bytes=status.get("VmRSS", 0) * 1024,
        oom_kill_count=oom_kills,
    )


def snapshot_memory_estimate(row_count: int, feature_count: int) -> int:
    # Float64 features/label plus index overhead; raw + processed snapshots,
    # one training slice and preprocessing/native conversion workspace. This is
    # a preflight estimate, not a guarantee of peak usage; live monitoring remains
    # mandatory. It intentionally does NOT multiply by the number of windows.
    frame = max(0, row_count) * (8 * (max(0, feature_count) + 1) + 32)
    return 4 * frame + 512 * MIB


def release_training_memory() -> None:
    """Release dead window/trial objects and return idle native pages to Linux."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    if sys.platform.startswith("linux"):
        import ctypes

        try:
            trim = ctypes.CDLL(None).malloc_trim
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)
        except (AttributeError, OSError):
            pass
