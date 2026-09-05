from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, TYPE_CHECKING

from factor_service.research.runtime_resources import (
    GIB, MEMORY_HARD_FLOOR_BYTES, RuntimeMemoryGuard, RuntimeResources,
    read_runtime_resources, snapshot_memory_estimate,
)

if TYPE_CHECKING:
    from factor_service.research.trainer import TrainingResult


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one AlphaFactorService model from a transferred immutable dataset",
    )
    parser.add_argument("job_path", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("result_path", type=Path)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("progress_path", type=Path)
    parser.add_argument("--training-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.training_child:
        try:
            _train(args)
        except MemoryError:
            _record_failure(args, "node_out_of_memory", "训练子进程内存分配失败，未发布不完整模型", read_runtime_resources())
            raise SystemExit(78)
    else:
        raise SystemExit(_supervise(args))


def _append_progress(path: Path, event: dict[str, Any]) -> None:
    data = (json.dumps(event, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    # The child and supervisor share this append-only journal. A single append
    # syscall keeps their JSON records separate even while native training blocks.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)


class _ProgressTail:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.event: dict[str, Any] = {"stage": "remote_runtime_initialized", "percent": 1, "details": {}}

    def update(self) -> dict[str, Any]:
        if not self.path.exists():
            return self.event
        with self.path.open("rb") as source:
            source.seek(self.offset)
            for line in source:
                if not line.endswith(b"\n"):
                    break
                self.offset += len(line)
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(value, dict) and value.get("stage") and value["stage"] != "resource_heartbeat":
                    self.event = value
        return self.event


def _record_failure(
    args: argparse.Namespace, code: str, message: str, resources: RuntimeResources,
) -> None:
    payload = {"error_code": code, "message": message, "resources": resources.public()}
    path = args.result_path.with_name("remote_failure.json")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    last = _ProgressTail(args.progress_path).update()
    _append_progress(args.progress_path, {
        "stage": code, "percent": last.get("percent", 1),
        "details": {**last.get("details", {}), **payload},
    })
    print(message, file=sys.stderr, flush=True)


def _stop_training_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        # Data-loader / native-library descendants may outlive the Python
        # leader, especially after an OOM kill. Its dedicated group is still
        # owned by this attempt and must not leak into the next training job.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return
    # This process group was created exclusively for this training child. Never
    # signal the node's other workloads, SSH server or supervisor's own group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _supervise(args: argparse.Namespace) -> int:
    args.progress_path.parent.mkdir(parents=True, exist_ok=True)
    args.progress_path.unlink(missing_ok=True)
    resources = read_runtime_resources()
    if resources.memory_limit_bytes <= 0:
        _record_failure(args, "node_resource_unavailable", "无法读取节点真实内存限制，未启动训练", resources)
        return 78
    if resources.memory_available_bytes <= MEMORY_HARD_FLOOR_BYTES:
        _record_failure(args, "node_memory_budget_exceeded", "节点当前实际可用内存已低于紧急底线，未启动训练", resources)
        return 78
    env = dict(os.environ)
    requested_threads = max(1, int(env.get("ALPHA_EFFECTIVE_NUM_THREADS") or resources.cpu_cores))
    threads = min(32, requested_threads, resources.cpu_cores)
    for key in ("ALPHA_EFFECTIVE_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(threads)
    _append_progress(args.progress_path, {
        "stage": "remote_runtime_resources", "percent": 1,
        "details": {"resources": resources.public(), "effective_num_threads": threads},
    })
    command = [
        sys.executable, "-m", "factor_service.research.remote_runner",
        str(args.job_path), str(args.work_dir), str(args.result_path),
        str(args.artifact_root), str(args.progress_path), "--training-child",
    ]
    process = subprocess.Popen(command, env=env, start_new_session=True)
    tail = _ProgressTail(args.progress_path)
    baseline_oom_kills = resources.oom_kill_count
    previous_handlers = {}

    def stopped(signum, _frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, stopped)
    last_report = 0.0
    peak_rss = 0
    memory_guard = RuntimeMemoryGuard()
    try:
        while True:
            tail.update()
            resources = read_runtime_resources(pid=process.pid)
            peak_rss = max(peak_rss, resources.process_rss_bytes)
            code = process.poll()
            if code is not None:
                _append_progress(args.progress_path, {
                    'stage': 'resource_heartbeat', 'percent': 0,
                    'details': {'resources': {**resources.public(), 'process_peak_rss_bytes': peak_rss},
                                'resource_sampled_at': time.time(), 'effective_num_threads': threads},
                })
                if code in (-signal.SIGKILL, 137) and resources.oom_kill_count > baseline_oom_kills:
                    _record_failure(args, "node_out_of_memory", "训练子进程被节点OOM终止；已保留完成窗口，未发布不完整模型", resources)
                return code if code >= 0 else 128 - code
            if resources.memory_limit_bytes <= 0:
                _stop_training_child(process)
                _record_failure(args, "node_resource_unavailable", "训练期间无法读取节点内存，已安全终止当前训练", resources)
                return 78
            if memory_guard.observe(resources):
                _stop_training_child(process)
                _record_failure(args, "node_memory_budget_exceeded", (
                    f"节点可用内存{resources.memory_available_bytes / GIB:.2f} GiB"
                    f"触发运行期保护（底线{resources.runtime_reserve_bytes / GIB:.2f} GiB，"
                    f"连续低内存采样{memory_guard.low_samples}次；紧急不足立即停止）；"
                    "已终止当前训练并保留完成窗口，未发布不完整模型"
                ), resources)
                return 78
            if memory_guard.low_samples == 1:
                _append_progress(args.progress_path, {
                    "stage": "memory_pressure_warning", "percent": tail.event.get("percent", 1),
                    "details": {**tail.event.get("details", {}), "resources": resources.public(),
                                "message": "实际可用内存偏低，继续采样确认"},
                })
            now = time.monotonic()
            if now - last_report >= 10:
                _append_progress(args.progress_path, {
                    "stage": "resource_heartbeat", "percent": 0,
                    "details": {
                        "resources": {**resources.public(), "process_peak_rss_bytes": peak_rss},
                        "effective_num_threads": threads,
                        "resource_sampled_at": time.time(),
                    },
                })
                last_report = now
            time.sleep(1)
    finally:
        _stop_training_child(process)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def _train(args: argparse.Namespace) -> None:
    # Heavy imports belong only to the supervised child.
    from factor_service.research.config import Settings
    from factor_service.research.job import validate_job
    from factor_service.research.trainer import QlibTrainer

    job = validate_job(json.loads(args.job_path.read_text(encoding="utf-8")))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.progress_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = args.artifact_root / "datasets" / str(job["dataset_hash"]) / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = snapshot_memory_estimate(
        int(manifest["row_count"]), len(manifest["feature_names"]),
    )
    resources = read_runtime_resources()
    _append_progress(args.progress_path, {
        "stage": "memory_preflight", "percent": 1,
        "details": {"estimated_working_set_bytes": required, "resources": resources.public()},
    })
    if required > resources.training_headroom_bytes:
        _append_progress(args.progress_path, {
            "stage": "memory_preflight_warning", "percent": 1,
            "details": {"estimated_working_set_bytes": required, "resources": resources.public(),
                        "message": "预估工作集超过规划预算，仍尝试训练；由运行期实际内存保护决定是否停止"},
        })

    settings = Settings(
        # The immutable snapshot must already exist.  These values are deliberately
        # unusable so a remote node can never silently query a different data source.
        clickhouse_host="127.0.0.1",
        clickhouse_port=1,
        clickhouse_user="remote-snapshot-only",
        clickhouse_password="",
        factor_database="ab_factor",
        model_database="ab_model",
        source_database="starlight",
        work_root=args.work_dir.parent,
        model_artifacts_root=args.artifact_root,
        scheduler_enabled=False,
        scheduler_refresh_seconds=60,
    )

    started_at = time.perf_counter()
    stage_started_at: dict[str, float] = {}

    def progress(stage: str, percent: int, details: dict[str, Any]) -> None:
        now = time.perf_counter()
        stage_started_at.setdefault(str(stage), now)
        payload = {
            "stage": str(stage),
            "percent": max(0, min(int(percent), 100)),
            "details": {
                **dict(details or {}),
                "elapsed_seconds": round(now - started_at, 3),
                "stage_elapsed_seconds": round(
                    now - stage_started_at[str(stage)], 3,
                ),
                "effective_num_threads": int(
                    os.environ.get("ALPHA_EFFECTIVE_NUM_THREADS") or 0
                ),
                "accelerator": str(
                    os.environ.get("ALPHA_MODEL_ACCELERATOR") or "cpu"
                ),
            },
        }
        _append_progress(args.progress_path, payload)

    progress("remote_runtime_initialized", 1, {
        "job_id": str(job.get("job_id") or ""),
        "model_kind": str((job.get("config_json") or {}).get("model", {}).get("kind") or ""),
    })

    result = QlibTrainer(settings).train(job, args.work_dir, progress=progress)
    _write_result(
        args.result_path,
        result,
        work_dir=args.work_dir,
        artifact_root=args.artifact_root,
    )
    progress("remote_packaged", 89, {"artifact_count": len(result.artifacts)})


def _write_result(
    path: Path,
    result: TrainingResult,
    *,
    work_dir: Path,
    artifact_root: Path,
) -> None:
    payload = {
        "schema_version": "alphablocks.remote-training-result.v1",
        "result": result.result,
        "artifacts": [
            {"kind": str(kind), **_scoped_path(artifact, work_dir, artifact_root)}
            for kind, artifact in result.artifacts
        ],
        "predictions": _scoped_path(result.predictions_path, work_dir, artifact_root),
    }
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _scoped_path(path: Path, work_dir: Path, artifact_root: Path) -> dict[str, str]:
    # Normalize ``..`` without dereferencing symlinks. Remote immutable datasets
    # deliberately live in a shared cache and are mounted into artifact_root via
    # a symlink; resolving that link would incorrectly classify them as escaped.
    # The local loader applies its own containment check before consuming paths.
    scoped = Path(os.path.abspath(path))
    work = Path(os.path.abspath(work_dir))
    artifacts = Path(os.path.abspath(artifact_root))
    if scoped == work or work in scoped.parents:
        return {"scope": "work", "path": scoped.relative_to(work).as_posix()}
    if scoped == artifacts or artifacts in scoped.parents:
        return {
            "scope": "artifact_root",
            "path": scoped.relative_to(artifacts).as_posix(),
        }
    raise ValueError(f"远程训练产物路径不属于允许目录: {scoped}")


if __name__ == "__main__":
    main()
