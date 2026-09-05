from __future__ import annotations

from datetime import datetime, timezone
from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

from factor_service.research import __version__
from factor_service.model_artifacts import ModelArtifactStore
from factor_service.model_object_store import ModelObjectStore, ModelObjectStoreConfig
from factor_service.model_research_repository import ModelResearchRepository
from factor_service.research.control import ResearchControl, ResearchControlError
from factor_service.research.config import Settings
from factor_service.research.errors import classify_exception, error_payload
from factor_service.research.inference import DailyInferenceRunner
from factor_service.research.job import (
    MODEL_PARAM_FIELDS,
    CancellationToken,
    safe_job_dir,
    validate_job,
)
from factor_service.research.state import JobStateStore
from factor_service.research.snapshot import prune_stale_dataset_staging
from factor_service.research.dataset_archive import archive_for_settings, DATASET_FILES
from factor_service.research.execution_progress import ExecutionProgress
from factor_service.research.trainer import QlibTrainer, TrainingResult


ACTIVE_STATUSES = {"leased", "running", "uploading"}

# Persist user-facing workflow milestones while keeping high-frequency iteration
# heartbeats event-free. This makes the upload/queue chronology available after
# the user leaves and reopens the training page.
PERSISTED_PROGRESS_STAGES = frozenset({
    "distributed_plan", "distributed_task_restoring", "distributed_task_failed",
    "packaging", "packaged", "rolling_series_ready", "publishing_predictions", "completing",
    "distributed_started", "distributed_task_started", "distributed_task_completed",
    "distributed_node_cleanup_pending", "distributed_task_retrying",
    "distributed_node_memory_failed", "distributed_node_ssh_failed", "distributed_task_reassigned",
    "archiving_dataset",
    "dataset_archived",
    "checking_dataset_snapshot",
    "loading_factors",
    "loading_prices",
    "building_labels",
    "splitting_dataset",
    "dataset_ready",
    "remote_dataset_staged",
    "remote_checking_power",
    "remote_powering_on",
    "remote_waiting_for_ssh",
    "remote_materializing_dataset",
    "remote_preparing",
    "remote_dataset_cache_prepared",
    "remote_snapshot_uploaded",
    "remote_resources_ready",
    "remote.remote_runtime_resources",
    "remote.memory_preflight",
    "remote.memory_preflight_warning", "remote.memory_pressure_warning",
    "remote.node_memory_budget_exceeded",
    "remote.node_out_of_memory",
    "remote.node_resource_unavailable",
    "remote.walk_forward_window_saved",
    "training_metrics_writing",
    "training_metrics_written",
    "remote.training_metrics_writing",
    "remote.training_metrics_written",
    "remote_training",
    "remote_packaged",
    "remote_process_cleanup_warning",
    "remote_downloading",
    "remote_powering_off",
    "remote_power_off_failed",
    "remote_kept_alive",
    "uploading_model_archive",
})


class ResearchWorker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.work_root.mkdir(parents=True, exist_ok=True)
        if not self.settings.work_root.is_dir():
            raise ValueError(f"研究存储根目录无效: {self.settings.work_root}")
        self.artifact_store = ModelArtifactStore(self.settings.model_artifacts_root)
        self.model_object_store = ModelObjectStore(
            getattr(self.settings, "model_object_store", ModelObjectStoreConfig()),
        )
        self.repository = ModelResearchRepository()
        self.dataset_archive = archive_for_settings(
            settings, active_hashes=self.repository.active_dataset_hashes,
        )
        self.control = ResearchControl(
            self.repository,
            self.artifact_store,
            self.model_object_store,
        )
        self.trainer: QlibTrainer | None = None
        self.inference_runner: DailyInferenceRunner | None = None
        self.state_store = JobStateStore(settings.work_root)
        self.stopping = False
        self.recovery_pending = False
        self.active_job_id = ""
        self.active_lease_token = ""
        self.last_job_id = ""
        self.last_job_status = ""
        self.last_error = ""
        self.scheduler_last_error = ""
        self.scheduler_last_tick_at = ""
        self.scheduler_last_result: dict[str, Any] = {}
        self.dataset_cache_last_error = ""
        self.dataset_cache_last_cleanup_at = ""
        self.dataset_cache_last_result: dict[str, Any] = {}
        self.current_progress: dict[str, Any] = {}
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._shutdown_event = threading.Event()
        self._state_lock = threading.Lock()
        self._job_thread: threading.Thread | None = None
        self._recovery_thread: threading.Thread | None = None
        self._scheduler_thread: threading.Thread | None = None
        self._experiment_queue_thread: threading.Thread | None = None
        self._registration_recovery_thread: threading.Thread | None = None
        self._dataset_cache_cleanup_thread: threading.Thread | None = None
        self._active_cancellation: CancellationToken | None = None

    def capabilities(self) -> dict[str, object]:
        versions: dict[str, str] = {}
        for module_name in ("qlib", "lightgbm", "xgboost", "catboost", "torch", "pandas", "pyarrow"):
            try:
                package_name = "pyqlib" if module_name == "qlib" else module_name
                versions[module_name] = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                versions[module_name] = "missing"
        try:
            from factor_service.research.remote import execution_nodes

            nodes = execution_nodes()
        except Exception as exc:
            nodes = [{"id": "local", "type": "local", "available": True}]
            self.last_error = f"远程训练节点配置无效: {exc}"
        return {
            "service_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "models": list(MODEL_PARAM_FIELDS),
            "max_concurrency": 1,
            "packages": versions,
            "dispatch_mode": "local_and_remote_ssh",
            "execution_nodes": nodes,
            "distributed_training": {
                "enabled": True, "max_nodes": 16, "tasks_per_node": 1,
                "optuna_parallelism": "trial", "walk_forward_parallelism": "window",
                "storage": "postgresql_and_minio", "remote_nodes_only": True,
            },
            "service_api_version": "v1",
            "cooperative_cancellation": True,
            "crash_recovery": True,
            "idempotent_prediction_publish": True,
            "daily_inference": True,
        }

    def start(self) -> None:
        """Start crash recovery inside the unified FactorService process."""
        if self.stopping:
            raise RuntimeError("研究调度器已经关闭")
        self._start_recovery_if_needed()
        self._registration_recovery_thread = threading.Thread(
            target=self._registration_recovery_loop,
            name="model-registration-recovery",
            daemon=True,
        )
        self._registration_recovery_thread.start()
        self._dataset_cache_cleanup_thread = threading.Thread(
            target=self._dataset_cache_cleanup_loop,
            name="dataset-cache-cleanup",
            daemon=True,
        )
        self._dataset_cache_cleanup_thread.start()
        if bool(getattr(self.settings, "scheduler_enabled", True)):
            self._scheduler_thread = threading.Thread(
                target=self._schedule_loop,
                name="model-inference-scheduler",
                daemon=True,
            )
            self._scheduler_thread.start()
        if bool(getattr(self.settings, "experiment_worker_enabled", True)):
            self._experiment_queue_thread = threading.Thread(
                target=self._experiment_queue_loop,
                name="model-experiment-queue",
                daemon=True,
            )
            self._experiment_queue_thread.start()
        print(
            "AlphaFactorService内置研究调度器已启动"
            + (
                ""
                if bool(getattr(self.settings, "experiment_worker_enabled", True))
                else "（实验训练Worker已禁用）"
            ),
            flush=True,
        )

    def close(self) -> None:
        """Stop active research work during the unified API shutdown."""
        self._stop()
        job_thread = self._job_thread
        if job_thread is not None and job_thread.is_alive():
            job_thread.join(timeout=30)
        recovery_thread = self._recovery_thread
        if recovery_thread is not None and recovery_thread.is_alive():
            recovery_thread.join(timeout=6)
        scheduler_thread = self._scheduler_thread
        if scheduler_thread is not None and scheduler_thread.is_alive():
            scheduler_thread.join(timeout=6)
        experiment_thread = self._experiment_queue_thread
        if experiment_thread is not None and experiment_thread.is_alive():
            experiment_thread.join(timeout=6)
        registration_thread = self._registration_recovery_thread
        if registration_thread is not None and registration_thread.is_alive():
            registration_thread.join(timeout=6)
        dataset_cleanup_thread = self._dataset_cache_cleanup_thread
        if dataset_cleanup_thread is not None and dataset_cleanup_thread.is_alive():
            dataset_cleanup_thread.join(timeout=6)

    def submit(self, payload: dict[str, Any]) -> dict[str, object]:
        """Validate and accept one centrally leased job without blocking HTTP."""
        incoming_job_id = str(payload.get("job_id") or "").strip() if isinstance(payload, dict) else ""
        incoming_lease_token = str(payload.get("lease_token") or "").strip() if isinstance(payload, dict) else ""
        with self._state_lock:
            if (
                incoming_job_id
                and self._job_thread is not None
                and self._job_thread.is_alive()
                and self.active_job_id == incoming_job_id
                and (
                    not getattr(self, "active_lease_token", "")
                    or self.active_lease_token == incoming_lease_token
                )
            ):
                # Repeated local dispatches are idempotent while the same lease is active.
                return {"accepted": True, "duplicate": True, "job_id": incoming_job_id}
        job = validate_job(payload)
        job_id = str(job["job_id"])
        lease_token = str(job["lease_token"])
        with self._state_lock:
            if self.stopping:
                raise RuntimeError("调度服务正在关闭")
            if self.recovery_pending:
                raise RuntimeError("调度服务正在恢复上一个中断任务")
            if self._job_thread is not None and self._job_thread.is_alive():
                raise RuntimeError(f"调度服务正在执行任务: {self.active_job_id}")
            self.active_job_id = job_id
            self.active_lease_token = lease_token
            self.last_job_id = job_id
            self.last_job_status = "accepted"
            self.last_error = ""
            self.current_progress = {"stage": "accepted", "percent": 1}

        # Persist before starting the task thread so restart recovery can safely
        # requeue the lease instead of executing twice.
        persisted = False
        try:
            self.state_store.save(job, "accepted", self.current_progress)
            persisted = True
            self.control.stage(job_id, lease_token, "running", self.current_progress)
            thread = threading.Thread(
                target=self._run_job,
                args=(job,),
                name=f"training-{job_id[:16]}",
                daemon=False,
            )
            with self._state_lock:
                self._job_thread = thread
            thread.start()
        except Exception:
            self._set_state(
                active_job_id="", last_job_status="dispatch_failed",
                active_lease_token="", recovery_pending=persisted,
            )
            if persisted:
                self._start_recovery_thread()
            raise
        return {"accepted": True, "job_id": job_id}

    def status(self) -> dict[str, object]:
        with self._state_lock:
            busy = self._job_thread is not None and self._job_thread.is_alive()
            scheduler_ready = (
                not bool(getattr(self.settings, "scheduler_enabled", True))
                or not self.scheduler_last_error
            )
            ready = not self.stopping and not self.recovery_pending and scheduler_ready
            return {
                "ok": True,
                "service": "AlphaFactorServiceResearch",
                "research_version": __version__,
                "ready": ready,
                "busy": busy,
                "recovery_pending": self.recovery_pending,
                "active_job_id": self.active_job_id,
                "last_job_id": self.last_job_id,
                "last_job_status": self.last_job_status,
                "last_error": self.last_error,
                "scheduler": {
                    "enabled": bool(getattr(self.settings, "scheduler_enabled", True)),
                    "last_tick_at": self.scheduler_last_tick_at,
                    "last_error": self.scheduler_last_error,
                    "last_result": dict(self.scheduler_last_result),
                },
                "dataset_cache": {
                    "capacity": 1 if self.dataset_archive is not None else None,
                    "retention_policy": "single_dataset" if self.dataset_archive is not None else "ttl",
                    "storage_mode": (
                        "minio_with_temporary_local_files" if self.dataset_archive is not None
                        else "local_cache"
                    ),
                    "retention_hours": float(getattr(
                        self.settings, "dataset_cache_retention_hours", 24.0,
                    )),
                    "last_cleanup_at": self.dataset_cache_last_cleanup_at,
                    "last_error": self.dataset_cache_last_error,
                    "last_result": dict(self.dataset_cache_last_result),
                },
                "progress": dict(self.current_progress),
                "started_at": self.started_at,
                "capabilities": self.capabilities(),
            }

    def _experiment_queue_loop(self) -> None:
        """Serially drain explicitly-created parameter or horizon studies."""
        while not self._shutdown_event.wait(1.0):
            with self._state_lock:
                busy = self._job_thread is not None and self._job_thread.is_alive()
                unavailable = self.stopping or self.recovery_pending
            if busy or unavailable:
                continue
            leased: dict[str, Any] | None = None
            try:
                leased = self.repository.claim_next_experiment_job(lease_seconds=90)
                if leased is not None:
                    self.submit(leased)
            except Exception as exc:
                if leased is not None:
                    try:
                        self.repository.release_dispatch_lease(
                            str(leased["job_id"]),
                            lease_token=str(leased.get("lease_token") or ""),
                            error_message=f"研究实验自动调度失败: {exc}",
                        )
                    except Exception:
                        pass
                print(f"研究实验队列暂不可用: {exc}", file=sys.stderr, flush=True)
                self._shutdown_event.wait(2.0)

    def _registration_recovery_loop(self) -> None:
        """Retry only results explicitly marked for automatic candidate registration."""
        while not self._shutdown_event.is_set():
            try:
                result = self.repository.reconcile_pending_training_results(limit=100)
                if int(result.get("finalized") or 0) > 0:
                    print(
                        "自动候选模型入库恢复: "
                        + json.dumps(result, ensure_ascii=False),
                        flush=True,
                    )
            except Exception as exc:
                print(f"自动候选模型入库恢复暂不可用: {exc}", file=sys.stderr, flush=True)
            self._shutdown_event.wait(30.0)

    def _dataset_cache_cleanup_loop(self) -> None:
        interval = max(
            60.0,
            float(getattr(
                self.settings, "dataset_cache_cleanup_interval_seconds", 3600.0,
            )),
        )
        while not self._shutdown_event.is_set():
            try:
                self._run_dataset_cache_cleanup()
            except Exception as exc:
                with self._state_lock:
                    self.dataset_cache_last_cleanup_at = datetime.now(
                        timezone.utc,
                    ).isoformat()
                    self.dataset_cache_last_error = str(exc)
                print(f"训练数据集缓存清理暂不可用: {exc}", file=sys.stderr, flush=True)
            self._shutdown_event.wait(interval)

    def _run_dataset_cache_cleanup(self) -> dict[str, Any]:
        retention_hours = float(getattr(
            self.settings, "dataset_cache_retention_hours", 24.0,
        ))
        protected_hashes = self.repository.active_dataset_hashes()
        if self.dataset_archive is not None:
            result = self.dataset_archive.cleanup(protected_hashes=protected_hashes)
            # Failed uploads/registrations retain their only local recovery copy.
            result["storage_mode"] = "minio_with_temporary_local_files"
        else:
            result = self.artifact_store.prune_dataset_cache(
                retention_seconds=retention_hours * 60 * 60,
                protected_hashes=protected_hashes,
            )
            result["staging"] = prune_stale_dataset_staging(
                self.settings.work_root,
                retention_seconds=retention_hours * 60 * 60,
            )
        with self._state_lock:
            self.dataset_cache_last_cleanup_at = datetime.now(
                timezone.utc,
            ).isoformat()
            self.dataset_cache_last_error = "; ".join(
                str(item.get("error", "")) for item in result.get("errors", [])
            )
            self.dataset_cache_last_result = dict(result)
        if result.get("deleted"):
            print(
                "已清理旧训练数据集缓存: "
                + json.dumps(result, ensure_ascii=False),
                flush=True,
            )
        return result

    def _schedule_loop(self) -> None:
        from factor_service.research.schedule import run_inference_schedule_tick

        refresh_seconds = max(
            10.0,
            float(getattr(self.settings, "scheduler_refresh_seconds", 60.0)),
        )
        while not self._shutdown_event.is_set():
            try:
                result = run_inference_schedule_tick(self.repository, self)
                with self._state_lock:
                    self.scheduler_last_tick_at = datetime.now(timezone.utc).isoformat()
                    self.scheduler_last_error = ""
                    self.scheduler_last_result = result
                if result.get("submitted"):
                    print(
                        "每日推理调度: " + json.dumps(result, ensure_ascii=False, default=str),
                        flush=True,
                    )
            except Exception as exc:
                with self._state_lock:
                    self.scheduler_last_tick_at = datetime.now(timezone.utc).isoformat()
                    self.scheduler_last_error = str(exc)
                print(f"每日推理调度暂不可用: {exc}", file=sys.stderr, flush=True)
            self._shutdown_event.wait(refresh_seconds)

    def _run_job(self, job: dict[str, Any]) -> None:
        job_id = str(job["job_id"])
        lease_token = str(job["lease_token"])
        attempt = max(1, int(job.get("attempt_count") or 1))
        work_dir = safe_job_dir(self.settings.work_root, job_id) / f"attempt-{attempt:03d}"
        # Every lease attempt owns an independent directory. Remote execution
        # writes its descriptor before launching the isolated runner, so retries
        # must recreate the attempt directory even when a previous failed
        # attempt was already cleaned up.
        work_dir.mkdir(parents=True, exist_ok=True)
        monitor_stop = threading.Event()
        job_kind = str(job.get("kind") or "train")
        execution = (job.get("config_json") or {}).get("execution") or {}
        max_runtime_minutes = int(execution.get("max_runtime_minutes") or 720)
        cancellation = CancellationToken(
            self._shutdown_event,
            timeout_seconds=(max_runtime_minutes * 60 if job_kind == "train" else None),
        )
        monitor_thread = threading.Thread(
            target=self._monitor_lease,
            args=(job_id, lease_token, cancellation, monitor_stop),
            name=f"lease-{job_id[:8]}",
            daemon=True,
        )
        report_pending = False
        action_label = "每日推理" if job_kind == "infer" else "训练"
        print(f"开始{action_label} {job_id}", flush=True)
        self.execution_observation = ExecutionProgress()
        self._set_state(
            active_job_id=job_id, last_job_id=job_id,
            last_job_status="running", _active_cancellation=cancellation,
        )
        monitor_thread.start()
        dataset_use = ExitStack()
        try:
            if job_kind == "train":
                if ((job.get('config_json') or {}).get('walk_forward', {}).get('enabled')
                        and self.model_object_store.enabled
                        and not self.model_object_store.enabled_for('walk_forward_series')):
                    from factor_service.model_object_store import ModelObjectStoreConfigurationError
                    raise ModelObjectStoreConfigurationError('滚动训练需要启用walk_forward_series归档，不能只归档主模型包')
                dataset_use.enter_context(
                    self.dataset_archive.cache_slot(str(job["dataset_hash"]))
                    if self.dataset_archive is not None
                    else self.artifact_store.dataset_usage(str(job["dataset_hash"]))
                )
            self._report_progress(job, cancellation, "validating", 2, {})
            progress_callback = lambda stage, percent, details: self._report_progress(
                job, cancellation, stage, percent, details,
            )
            if job_kind == "infer":
                if self.inference_runner is not None:
                    trained = self.inference_runner.run(
                        job, work_dir, cancellation=cancellation, progress=progress_callback,
                    )
                else:
                    trained = self._run_isolated_model(job, work_dir, cancellation)
            else:
                execution_node_id = str(
                    ((job.get("config_json") or {}).get("execution") or {}).get(
                        "node_id"
                    )
                    or "local"
                )
                if ((job.get("config_json") or {}).get("execution") or {}).get("mode") == "distributed":
                    trained = self._run_isolated_model(job, work_dir, cancellation)
                elif execution_node_id != "local":
                    from factor_service.research.remote import (
                        RemoteResearchExecutor,
                        get_remote_node,
                    )

                    trained = RemoteResearchExecutor(
                        self.settings, get_remote_node(execution_node_id),
                    ).train(
                        job, work_dir,
                        cancellation=cancellation,
                        progress=progress_callback,
                    )
                elif self.trainer is not None:
                    trained = self.trainer.train(
                        job, work_dir, cancellation=cancellation, progress=progress_callback,
                    )
                else:
                    trained = self._run_isolated_model(job, work_dir, cancellation)
            cancellation.checkpoint()
            self._report_progress(job, cancellation, 'uploading', 90, {'artifact_count': len(trained.artifacts)})
            self.control.stage(job_id, lease_token, 'uploading', dict(self.current_progress))
            artifact_count = max(1, len(trained.artifacts))
            remote_artifacts: list[dict[str, object]] = []
            for artifact_index, (kind, path) in enumerate(trained.artifacts):
                cancellation.checkpoint()
                saved = self.artifact_store.publish_file(
                    job_id=job_id,
                    artifact_kind=kind,
                    source_path=path,
                    dataset_hash=str(job.get("dataset_hash") or ""),
                )
                remote = None
                if self.dataset_archive is not None and kind in DATASET_FILES.values():
                    record = self.dataset_archive.record(str(job["dataset_hash"]))
                    if record is None:
                        raise ValueError("训练数据集尚未完成MinIO归档与登记")
                    remote = record["files_json"][path.name]
                    if str(remote["sha256"]) != str(saved["sha256"]):
                        raise ValueError("训练产物与MinIO冻结数据集不一致")
                elif self.model_object_store.enabled_for(kind):
                    self._report_progress(
                        job,
                        cancellation,
                        "uploading_model_archive",
                        min(96, 90 + int(6 * artifact_index / artifact_count)),
                        {"artifact": kind, "bucket": self.model_object_store.config.bucket},
                    )
                    remote = self.model_object_store.publish_file(
                        job_id=job_id,
                        model_id=str(job["model_id"]),
                        model_version=int(
                            job.get("model_version")
                            or (job.get("config_json") or {}).get("planned_model_version")
                            or 0
                        ),
                        artifact_kind=kind,
                        source_path=saved["path"],
                        digest=str(saved["sha256"]),
                        size_bytes=int(saved["size_bytes"]),
                        checkpoint=cancellation.checkpoint,
                        progress=lambda done, total: self._report_progress(
                            job, cancellation, "uploading_model_archive",
                            min(96, 90 + int(6 * (artifact_index + done / max(1, total)) / artifact_count)),
                            {"artifact": kind, "upload_bytes": done, "upload_total_bytes": total},
                        ),
                    )
                    if remote is not None:
                        remote_artifacts.append(dict(remote))
                self.control.record_artifact(
                    job_id,
                    lease_token,
                    kind=kind,
                    file_name=path.name,
                    relative_path=str(saved["relative_path"]),
                    digest=str(saved["sha256"]),
                    size_bytes=int(saved["size_bytes"]),
                    dataset_hash=str(job.get("dataset_hash") or ""),
                    object_store_uri=str((remote or {}).get("object_uri") or ""),
                    object_store_version_id=str((remote or {}).get("version_id") or ""),
                    object_store_sha256=str((remote or {}).get("sha256") or ""),
                )
                self._report_progress(
                    job,
                    cancellation,
                    "uploading",
                    min(96, 90 + int(6 * (artifact_index + 1) / artifact_count)),
                    {
                        "artifact": kind,
                        "artifact_index": artifact_index + 1,
                        "artifact_count": artifact_count,
                    },
                )
            if remote_artifacts:
                trained.result["object_storage"] = {
                    **self.model_object_store.public_config(),
                    "artifacts": remote_artifacts,
                }
            self._report_progress(job, cancellation, "publishing_predictions", 97, {})
            publisher = self.trainer or QlibTrainer(self.settings)
            inserted = publisher.publish_predictions(
                trained.predictions_path, job, cancellation=cancellation,
            )
            expected = int(trained.result["predictions"]["row_count"])
            if inserted != expected:
                raise ValueError(f"预测发布行数不一致: 期望{expected}，实际{inserted}")
            self._report_progress(job, cancellation, "completing", 99, {
                "prediction_rows": inserted,
            })
            self.control.complete(job_id, lease_token, trained.result)
            self.state_store.clear()
            self._set_state(last_job_status="succeeded", last_error="", current_progress={
                "stage": "succeeded", "percent": 100,
            })
            print(f"{action_label}完成 {job_id}", flush=True)
        except Exception as exc:
            retryable, code = classify_exception(exc)
            metadata = error_payload(exc)
            error = f"[{code}] {type(exc).__name__}: {exc}"
            terminal = "retry_queued" if retryable else ("canceled" if code == "canceled" else "failed")
            self._set_state(last_job_status=terminal, last_error=str(exc))
            print(f"{error}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            try:
                self.state_store.save(job, "failure_report_pending", {
                    "stage": "failure_report_pending",
                    "percent": int(self.current_progress.get("percent") or 0),
                    "error_message": error,
                    "error_code": code,
                    "retryable": retryable,
                })
            except Exception as state_error:
                print(f"失败恢复状态保存失败: {state_error}", file=sys.stderr, flush=True)
            try:
                response = self.control.fail(job_id, lease_token, error, retryable=retryable)
                reported_job = dict(response.get("job") or {})
                self.state_store.clear()
                self._set_state(last_job_status=str(reported_job.get("status") or terminal))
            except Exception as report_error:
                if self._job_already_succeeded(job_id, lease_token, report_error):
                    self.state_store.clear()
                    self._set_state(last_job_status="succeeded", last_error="")
                else:
                    report_pending = True
                    self._set_state(recovery_pending=True)
                    print(
                        f"失败状态回传失败: {report_error}; 将由恢复线程重试 "
                        f"({metadata})",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            dataset_use.close()
            monitor_stop.set()
            if monitor_thread.is_alive():
                monitor_thread.join(timeout=6)
            self._set_state(active_job_id="", active_lease_token="", _active_cancellation=None)
            if report_pending:
                self._start_recovery_thread()

    def _run_isolated_model(
        self, job: dict[str, Any], work_dir: Path, cancellation: CancellationToken,
    ) -> TrainingResult:
        """Keep native ML runtimes out of the long-lived scheduler process.

        LightGBM, CatBoost and PyTorch each ship an OpenMP runtime. Loading them in
        one macOS process is crash-prone, while a spawned process also ensures that
        PyTorch initializes its CPU pool on its main thread.
        """
        work_dir.mkdir(parents=True, exist_ok=True)
        descriptor = work_dir / "isolated_job.json"
        result_path = work_dir / "isolated_result.json"
        stdout_path = work_dir / "isolated_stdout.log"
        stderr_path = work_dir / "isolated_stderr.log"
        progress_path = work_dir / "isolated_progress.jsonl"
        progress_path.unlink(missing_ok=True)
        progress_offset = 0
        descriptor.write_text(
            json.dumps(job, ensure_ascii=False, sort_keys=True), encoding="utf-8",
        )
        result_path.unlink(missing_ok=True)
        environment = os.environ.copy()
        params = dict((job.get("config_json") or {}).get("model", {}).get("params") or {})
        thread_count = str(max(1, int(params.get("num_threads") or 4)))
        environment.setdefault("OMP_NUM_THREADS", thread_count)
        environment.setdefault("MKL_NUM_THREADS", thread_count)
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [
                    sys.executable, "-m", "factor_service.research.isolated_runner",
                    str(job.get("kind") or "train"), str(descriptor),
                    str(work_dir), str(result_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment, stdout=stdout, stderr=stderr,
            )
            try:
                while process.poll() is None:
                    cancellation.checkpoint()
                    if progress_path.exists():
                        with progress_path.open('rb') as events:
                            events.seek(progress_offset)
                            for line in events:
                                if not line.endswith(b'\n'):
                                    break
                                progress_offset += len(line)
                                event = json.loads(line)
                                self._report_progress(job, cancellation, event['stage'],
                                                      event['percent'], event['details'])
                    time.sleep(0.5)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=60 if ((job.get('config_json') or {}).get('execution') or {}).get('mode') == 'distributed' else 5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise
        if process.returncode != 0:
            error = stderr_path.read_text(encoding="utf-8", errors="replace")[-20_000:]
            if result_path.is_file():
                payload = json.loads(result_path.read_text(encoding='utf-8'))
                from factor_service.research.errors import restore_job_error
                restored = restore_job_error(payload)
                if restored is not None:
                    raise restored
            raise RuntimeError(
                f"隔离模型进程失败(returncode={process.returncode}):\n{error}"
            )
        if not result_path.is_file():
            raise RuntimeError("隔离模型进程未生成结果描述文件")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return TrainingResult(
            result=dict(payload["result"]),
            artifacts=[(str(kind), Path(path)) for kind, path in payload["artifacts"]],
            predictions_path=Path(payload["predictions_path"]),
        )

    def _report_progress(
        self,
        job: dict[str, Any],
        cancellation: CancellationToken,
        stage: str,
        percent: int,
        details: dict[str, Any],
    ) -> None:
        cancellation.checkpoint()
        if not hasattr(self, 'execution_observation'):
            self.execution_observation = ExecutionProgress()
        payload = self.execution_observation.update(stage, percent, details)
        with self._state_lock:
            previous_stage = str(self.current_progress.get("stage") or "")
        self.state_store.save(job, stage, payload)
        self._set_state(current_progress=payload)
        record_event = stage != previous_stage and stage in PERSISTED_PROGRESS_STAGES
        # Iteration heartbeats remain event-free; only workflow transitions are
        # written to model_job_events for the persistent training log.
        try:
            self.control.renew(
                str(job["job_id"]), str(job["lease_token"]), payload,
                record_event=record_event,
            )
        except ResearchControlError as exc:
            if not exc.retryable:
                cancellation.cancel("任务已取消或租约已失效")
                cancellation.checkpoint()
            print(f"进度回传暂时失败: {exc}", file=sys.stderr, flush=True)

    def _monitor_lease(
        self,
        job_id: str,
        lease_token: str,
        cancellation: CancellationToken,
        stop: threading.Event,
    ) -> None:
        """Poll cooperative cancellation and keep the 90-second lease alive."""
        next_renewal = time.monotonic()
        while not stop.wait(5):
            try:
                control = self.control.control(job_id, lease_token)
                if bool(control.get("cancel_requested")):
                    cancellation.cancel("用户已请求取消任务")
                    return
                status = str(control.get("status") or "")
                if status not in ACTIVE_STATUSES:
                    cancellation.cancel(f"任务状态已变为{status or '未知'}")
                    return
                if time.monotonic() >= next_renewal:
                    with self._state_lock:
                        progress = dict(self.current_progress)
                    self.control.renew(job_id, lease_token, progress)
                    next_renewal = time.monotonic() + 30
            except ResearchControlError as exc:
                if not exc.retryable:
                    cancellation.cancel("任务租约已失效")
                    return
                print(f"任务控制检查暂时失败 {job_id}: {exc}", file=sys.stderr, flush=True)
            except Exception as exc:
                print(f"任务控制检查失败 {job_id}: {exc}", file=sys.stderr, flush=True)

    def _start_recovery_if_needed(self) -> None:
        try:
            state = self.state_store.load()
        except Exception as exc:
            self._set_state(recovery_pending=True, last_error=f"恢复状态文件无效: {exc}")
            print(self.last_error, file=sys.stderr, flush=True)
            return
        if state:
            self._set_state(recovery_pending=True, last_job_status="recovering")
            self._start_recovery_thread()

    def _start_recovery_thread(self) -> None:
        with self._state_lock:
            if self._recovery_thread is not None and self._recovery_thread.is_alive():
                return
            thread = threading.Thread(
                target=self._recover_interrupted_job,
                name="job-recovery",
                daemon=True,
            )
            self._recovery_thread = thread
        thread.start()

    def _recover_interrupted_job(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                state = self.state_store.load()
                if not state:
                    self._set_state(recovery_pending=False)
                    return
                job = validate_job(dict(state.get("job") or {}))
                job_id = str(job["job_id"])
                lease_token = str(job["lease_token"])
                control = self.control.control(job_id, lease_token)
                status = str(control.get("status") or "")
                if status in ACTIVE_STATUSES:
                    canceled = bool(control.get("cancel_requested"))
                    progress = dict(state.get("progress") or {})
                    report_pending = str(state.get("phase") or "") == "failure_report_pending"
                    error_message = str(progress.get("error_message") or "").strip()
                    retryable = bool(progress.get("retryable", True))
                    response = self.control.fail(
                        job_id,
                        lease_token,
                        error_message if report_pending and error_message else
                        "AlphaFactorService研究进程在任务执行期间重启，任务已安全重新排队",
                        retryable=(retryable if report_pending else True) and not canceled,
                    )
                    recovered_status = str((response.get("job") or {}).get("status") or "queued")
                else:
                    recovered_status = status or "released"
                self.state_store.clear()
                self._set_state(
                    recovery_pending=False,
                    last_job_id=job_id,
                    last_job_status=recovered_status,
                    last_error="",
                )
                print(f"已恢复中断任务 {job_id}: {recovered_status}", flush=True)
                return
            except ResearchControlError as exc:
                if not exc.retryable:
                    # The lease was already released or finished in the control database.
                    self.state_store.clear()
                    self._set_state(recovery_pending=False, last_job_status="released", last_error="")
                    return
                self._set_state(last_error=f"恢复任务暂时失败: {exc}")
            except Exception as exc:
                retryable, _ = classify_exception(exc)
                self._set_state(last_error=f"恢复任务失败: {exc}")
                if not retryable:
                    print(self.last_error, file=sys.stderr, flush=True)
                    return
            self._shutdown_event.wait(5)

    def _job_already_succeeded(
        self, job_id: str, lease_token: str, report_error: Exception,
    ) -> bool:
        # Handles a completion that was committed before the caller observed it.
        if not isinstance(report_error, ResearchControlError):
            return False
        try:
            return str(self.control.control(job_id, lease_token).get("status") or "") == "succeeded"
        except Exception:
            return False

    def _stop(self, *_args: object) -> None:
        self.stopping = True
        self._shutdown_event.set()

    def _set_state(self, **values: object) -> None:
        with self._state_lock:
            for name, value in values.items():
                setattr(self, name, value)


__all__ = ["ResearchWorker"]
