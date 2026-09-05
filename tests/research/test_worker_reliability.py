from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from factor_service.research.errors import JobCanceled, WorkerShutdown
from factor_service.research.job import CancellationToken
from factor_service.research.trainer import TrainingResult
from factor_service.research.worker import ResearchWorker
from tests.research.utils import valid_job


@pytest.mark.parametrize('code,retryable,error_name', [
    ('retryable_error', True, 'RetryableJobError'),
    ('canceled', False, 'JobCanceled'),
    ('node_memory_budget_exceeded', False, 'NodeMemoryBudgetExceeded'),
    ('training_timeout', False, 'TrainingTimeout'),
    ('permanent_error', False, 'PermanentJobError'),
    ('node_ssh_authentication_failed', False, 'NodeSSHAuthenticationError'),
    ('node_ssh_connection_failed', False, 'NodeSSHConnectionError'),
    ('node_execution_unavailable', False, 'NodeExecutionUnavailable'),
])
def test_isolated_runner_preserves_error_semantics(tmp_path, monkeypatch, code, retryable, error_name):
    import json
    from factor_service.research import errors
    def spawn(command, **kwargs):
        Path(command[-1]).write_text(json.dumps(dict(error='synthetic failure', error_code=code, retryable=retryable)))
        return SimpleNamespace(returncode=1, poll=lambda: 1)
    monkeypatch.setattr('factor_service.research.worker.subprocess.Popen', spawn)
    worker = ResearchWorker.__new__(ResearchWorker)
    with pytest.raises(getattr(errors, error_name), match='synthetic failure'):
        worker._run_isolated_model(valid_job(), tmp_path, CancellationToken())


def _settings(tmp_path: Path):
    return SimpleNamespace(
        work_root=tmp_path,
        model_artifacts_root=tmp_path / "artifacts",
    )


class _Api:
    def __init__(self) -> None:
        self.failed: list[tuple[str, bool]] = []
        self.failure_messages: list[str] = []
        self.completed: list[str] = []
        self.completed_results: list[dict] = []
        self.artifacts: list[dict] = []
        self.renewals: list[dict] = []

    def renew(self, *_args, **_kwargs):
        self.renewals.append(dict(_kwargs))
        return {"ok": True}

    def control(self, *_args, **_kwargs):
        return {"status": "running", "cancel_requested": False}

    def stage(self, *_args, **_kwargs):
        return {"ok": True}

    def record_artifact(self, *_args, **kwargs):
        self.artifacts.append(dict(kwargs))
        return {"ok": True}

    def complete(self, job_id, _lease_token, _result):
        self.completed.append(job_id)
        self.completed_results.append(dict(_result))
        return {"ok": True}

    def fail(self, job_id, _lease_token, _error, retryable=True):
        self.failed.append((job_id, retryable))
        self.failure_messages.append(_error)
        return {"ok": True, "job": {"status": "queued" if retryable else "failed"}}


class _FailingTrainer:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def train(self, *_args, **_kwargs):
        raise self.error


class _SuccessfulTrainer:
    def __init__(self, artifact: Path, predictions: Path) -> None:
        self.artifact = artifact
        self.predictions = predictions

    def train(self, *_args, **_kwargs):
        return TrainingResult(
            result={"predictions": {"row_count": 0}},
            artifacts=[("bundle", self.artifact)],
            predictions_path=self.predictions,
        )

    def publish_predictions(self, *_args, **_kwargs):
        return 0


def test_successful_job_publishes_artifact_locally_then_records_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "qlib_experiment.tar.gz"
    predictions = tmp_path / "predictions.parquet"
    artifact.write_bytes(b"formal model bundle")
    predictions.write_bytes(b"")
    worker = ResearchWorker(_settings(tmp_path / "work"))
    api = _Api()
    worker.control = api
    worker.trainer = _SuccessfulTrainer(artifact, predictions)

    worker._run_job(valid_job())

    assert api.completed == ["model_job_test"]
    assert len(api.artifacts) == 1
    registered = api.artifacts[0]
    assert registered["kind"] == "bundle"
    assert registered["file_name"] == "qlib_experiment.tar.gz"
    assert registered["relative_path"].endswith("bundle/qlib_experiment.tar.gz")
    assert worker.artifact_store.resolve(registered["relative_path"]).read_bytes() == b"formal model bundle"


def test_worker_cleanup_protects_dataset_reserved_by_active_job(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "work")
    settings.dataset_cache_retention_hours = 24
    worker = ResearchWorker(settings)
    protected_hash = "5" * 64
    expired_hash = "6" * 64
    source = tmp_path / "dataset.parquet"
    source.write_bytes(b"factor training data")
    for dataset_hash in (protected_hash, expired_hash):
        worker.artifact_store.publish_file(
            job_id="job-cleanup", artifact_kind="dataset",
            source_path=source, dataset_hash=dataset_hash,
        )
        worker.artifact_store.touch_dataset(dataset_hash, used_at=100)
    worker.repository = SimpleNamespace(
        active_dataset_hashes=lambda: {protected_hash},
    )

    result = worker._run_dataset_cache_cleanup()

    assert result["deleted"] == [expired_hash]
    assert result["protected"] == 1
    assert worker.dataset_cache_last_error == ""
    assert worker.dataset_cache_last_cleanup_at
    assert (worker.artifact_store.root / "datasets" / protected_hash).is_dir()


def test_successful_job_archives_final_model_before_registering_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "qlib_experiment.tar.gz"
    predictions = tmp_path / "predictions.parquet"
    artifact.write_bytes(b"formal model bundle")
    predictions.write_bytes(b"")
    worker = ResearchWorker(_settings(tmp_path / "work"))
    api = _Api()
    worker.control = api
    worker.trainer = _SuccessfulTrainer(artifact, predictions)

    class _ObjectStore:
        config = SimpleNamespace(bucket="alphablocks-models")
        calls: list[dict] = []

        @staticmethod
        def enabled_for(kind):
            return kind == "bundle"

        def publish_file(self, **kwargs):
            self.calls.append(dict(kwargs))
            return {
                "object_uri": "s3://alphablocks-models/models/test_model/bundle.tgz",
                "version_id": "minio-version-1",
                "sha256": kwargs["digest"],
            }

        @staticmethod
        def public_config():
            return {"provider": "s3", "bucket": "alphablocks-models"}

    object_store = _ObjectStore()
    worker.model_object_store = object_store

    worker._run_job(valid_job())

    assert api.completed == ["model_job_test"]
    assert len(object_store.calls) == 1
    assert object_store.calls[0]["model_version"] == 1
    assert api.artifacts[0]["object_store_uri"].startswith("s3://alphablocks-models/")
    assert api.artifacts[0]["object_store_version_id"] == "minio-version-1"
    assert api.artifacts[0]["object_store_sha256"] == object_store.calls[0]["digest"]
    assert api.completed_results[0]["object_storage"]["bucket"] == "alphablocks-models"


def test_retry_attempt_creates_fresh_work_directory(tmp_path: Path) -> None:
    artifact = tmp_path / "qlib_experiment.tar.gz"
    predictions = tmp_path / "predictions.parquet"
    artifact.write_bytes(b"formal model bundle")
    predictions.write_bytes(b"")

    class _WorkDirCheckingTrainer(_SuccessfulTrainer):
        def train(self, _job, work_dir, **_kwargs):
            assert work_dir.name == "attempt-002"
            assert work_dir.is_dir()
            return super().train()

    worker = ResearchWorker(_settings(tmp_path / "work"))
    api = _Api()
    worker.control = api
    worker.trainer = _WorkDirCheckingTrainer(artifact, predictions)
    job = valid_job()
    job["attempt_count"] = 2

    worker._run_job(job)

    assert api.completed == ["model_job_test"]


def test_invalid_data_failure_is_not_retried(tmp_path: Path) -> None:
    worker = ResearchWorker(_settings(tmp_path))
    api = _Api()
    worker.control = api
    worker.trainer = _FailingTrainer(ValueError("bad frozen data"))

    worker._run_job(valid_job())

    assert api.failed == [("model_job_test", False)]
    assert worker.state_store.load() is None


def test_shutdown_failure_is_requeued(tmp_path: Path) -> None:
    worker = ResearchWorker(_settings(tmp_path))
    api = _Api()
    worker.control = api
    worker.trainer = _FailingTrainer(WorkerShutdown("restart"))

    worker._run_job(valid_job())

    assert api.failed == [("model_job_test", True)]


def test_lease_monitor_observes_remote_cancel(tmp_path: Path) -> None:
    worker = ResearchWorker(_settings(tmp_path))

    class _CancelApi(_Api):
        def control(self, *_args, **_kwargs):
            return {"status": "running", "cancel_requested": True}

    class _OneTick:
        calls = 0

        def wait(self, _seconds):
            self.calls += 1
            return self.calls > 1

    worker.control = _CancelApi()
    cancellation = CancellationToken()
    worker._monitor_lease("model_job_test", "lease", cancellation, _OneTick())

    with pytest.raises(JobCanceled, match="用户"):
        cancellation.checkpoint()


def test_restart_recovery_requeues_active_job(tmp_path: Path) -> None:
    worker = ResearchWorker(_settings(tmp_path))
    api = _Api()
    worker.control = api
    job = valid_job()
    worker.state_store.save(job, "training", {"percent": 60})
    worker.recovery_pending = True

    worker._recover_interrupted_job()

    assert api.failed == [("model_job_test", True)]
    assert worker.state_store.load() is None
    assert worker.recovery_pending is False
    assert worker.last_job_status == "queued"


def test_recovery_reports_original_pending_failure(tmp_path: Path) -> None:
    worker = ResearchWorker(_settings(tmp_path))
    api = _Api()
    worker.control = api
    job = valid_job()
    worker.state_store.save(job, "failure_report_pending", {
        "error_message": "[control_database_transient] PostgreSQL unavailable",
        "retryable": True,
    })
    worker.recovery_pending = True

    worker._recover_interrupted_job()

    assert api.failed == [("model_job_test", True)]
    assert api.failure_messages == ["[control_database_transient] PostgreSQL unavailable"]


def test_progress_state_survives_process_boundary(tmp_path: Path) -> None:
    worker = ResearchWorker(_settings(tmp_path))
    worker.control = _Api()
    job = valid_job()
    cancellation = CancellationToken()

    worker._report_progress(job, cancellation, "loading_factors", 22, {"factor_index": 1})

    saved = worker.state_store.load()
    assert saved is not None
    assert saved["phase"] == "loading_factors"
    assert saved["progress"]["percent"] == 22


def test_progress_log_persists_upload_milestones_without_heartbeat_noise(
    tmp_path: Path,
) -> None:
    worker = ResearchWorker(_settings(tmp_path))
    api = _Api()
    worker.control = api
    job = valid_job()
    cancellation = CancellationToken()

    worker._report_progress(job, cancellation, "remote_dataset_staged", 56, {})
    worker._report_progress(job, cancellation, "remote_checking_power", 57, {})
    worker._report_progress(job, cancellation, "remote_preparing", 57, {})
    worker._report_progress(job, cancellation, "remote_preparing", 58, {})
    worker._report_progress(job, cancellation, "training_final_model", 70, {})
    worker._report_progress(
        job, cancellation, "remote_snapshot_uploaded", 60,
        {"dataset_cache_hit": False},
    )

    assert [item["record_event"] for item in api.renewals] == [
        True, True, True, False, False, True,
    ]


def test_acceptance_state_write_failure_does_not_leave_worker_busy(tmp_path: Path) -> None:
    worker = ResearchWorker(_settings(tmp_path))

    class _BrokenState:
        @staticmethod
        def save(*_args, **_kwargs):
            raise OSError("disk full")

    worker.state_store = _BrokenState()

    with pytest.raises(OSError, match="disk full"):
        worker.submit(valid_job())

    assert worker.active_job_id == ""
    assert worker.active_lease_token == ""
    assert worker.recovery_pending is False
