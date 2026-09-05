from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from factor_service.api import model_research
from factor_service.schemas import ModelBacktestJobOut


class _Repository:
    def __init__(self) -> None:
        self.jobs = {
            "job-1": {"job_id": "job-1", "status": "queued"},
            "job-done": {"job_id": "job-done", "status": "succeeded"},
        }
        self.inference_payload = None
        self.validation_payload = None
        self.registry_payload = None
        self.deleted_payload = None
        self.architectures = {}

    def list_jobs(
        self, *, status, experiment_id="", kind="", model_id="",
        model_version=None, trade_date="", limit,
    ):
        jobs = list(self.jobs.values())
        if experiment_id:
            jobs = [
                job for job in jobs
                if job.get("config_json", {}).get("experiment", {}).get("experiment_id")
                == experiment_id
            ]
        if kind:
            jobs = [job for job in jobs if job.get("kind") == kind]
        if model_id:
            jobs = [job for job in jobs if job.get("model_id") == model_id]
        if model_version is not None:
            jobs = [job for job in jobs if job.get("model_version") == model_version]
        if trade_date:
            jobs = [
                job for job in jobs
                if job.get("config_json", {}).get("inference", {}).get("trade_date")
                == trade_date
            ]
        return jobs[:limit]

    def resolve_model_reference(self, reference, *, version=None):
        return {
            "model_id": "model-a",
            "model_version": int(version or 3),
            "name": str(reference),
            "state": "candidate",
            "resolved_from": "name",
        }

    def create_training_job(self, payload):
        self.created_payload = payload
        job = {"job_id": "job-created", "status": "queued", "title": payload["title"]}
        self.jobs[job["job_id"]] = job
        return job

    def incremental_training_precheck(self, model_id, version, payload):
        return {
            "status": "ready",
            "passed": True,
            "can_submit": True,
            "checks": [{
                "key": "feature_identity",
                "label": "冻结特征完全一致",
                "passed": True,
            }],
            "contract": {
                "source_model_id": model_id,
                "source_model_version": version,
                "candidate_date_end": payload["dataset"].get("date_end"),
            },
        }

    def create_model_architecture(self, payload):
        architecture = {
            **payload,
            "architecture_id": "architecture-created",
            "state": "draft",
            "revision": 1,
            "readiness": {"ready": False},
        }
        self.architectures[architecture["architecture_id"]] = architecture
        return dict(architecture)

    def list_model_architectures(self, *, limit):
        return list(self.architectures.values())[:limit]

    def get_model_architecture(self, architecture_id):
        return dict(self.architectures[architecture_id])

    def update_model_architecture(self, architecture_id, payload):
        architecture = {
            **self.architectures[architecture_id], **payload,
            "architecture_id": architecture_id,
            "revision": int(self.architectures[architecture_id]["revision"]) + 1,
        }
        self.architectures[architecture_id] = architecture
        return dict(architecture)

    def activate_model_architecture(self, architecture_id):
        self.architectures[architecture_id]["state"] = "active"
        return dict(self.architectures[architecture_id])

    def archive_model_architecture(self, architecture_id):
        self.architectures[architecture_id]["state"] = "archived"
        return dict(self.architectures[architecture_id])

    def delete_model_architecture(self, architecture_id):
        architecture = self.architectures.pop(architecture_id)
        return {
            "architecture_id": architecture_id,
            "name": architecture["name"],
            "state": architecture["state"],
            "revision": architecture["revision"],
            "engines": architecture.get("engines") or [],
        }

    def create_training_experiment(self, payload):
        experiment_id = "model_experiment_created"
        jobs = [
            {
                "job_id": f"job-trial-{index}",
                "status": "queued",
                "config_json": {"experiment": {
                    "experiment_id": experiment_id,
                    "trial_index": index,
                    "trial_count": 2,
                    "search_params": {"learning_rate": value},
                }},
            }
            for index, value in enumerate([0.03, 0.05], start=1)
        ]
        self.jobs.update({job["job_id"]: job for job in jobs})
        return {
            "experiment_id": experiment_id,
            "title": payload["title"],
            "trial_count": 2,
            "statuses": {"queued": 2},
            "jobs": jobs,
        }

    def get_training_experiment(self, experiment_id):
        jobs = self.list_jobs(experiment_id=experiment_id, status="", limit=24)
        return {
            "experiment_id": experiment_id,
            "trial_count": len(jobs),
            "jobs": jobs,
        }

    def restart_training_experiment(self, experiment_id):
        return {
            "experiment_id": "model_experiment_retried",
            "restarted_from_experiment_id": experiment_id,
            "trial_count": 2,
            "statuses": {"queued": 2},
            "jobs": [],
        }

    def get_job(self, job_id):
        return dict(self.jobs[job_id])

    def retry_job(self, job_id, *, idempotency_key):
        job = {
            **self.jobs[job_id],
            "status": "queued",
            "retryable": False,
            "idempotency_key": idempotency_key,
        }
        self.jobs[job_id] = job
        return dict(job)

    def register_training_result(self, job_id):
        job = {
            **self.jobs[job_id],
            "status": "succeeded",
            "model_version": 1,
            "registration_status": "registered",
        }
        self.jobs[job_id] = job
        return dict(job)

    def decline_training_result(self, job_id):
        job = {
            **self.jobs[job_id],
            "status": "succeeded",
            "model_version": None,
            "registration_status": "declined",
        }
        self.jobs[job_id] = job
        return dict(job)

    def claim_specific_job(self, job_id, *, lease_seconds):
        job = {
            **self.jobs[job_id],
            "status": "leased",
            "lease_owner": "alpha-factor-service",
            "lease_token": "lease-1",
            "attempt_count": 1,
        }
        self.jobs[job_id] = job
        return dict(job)

    def release_dispatch_lease(self, *_args, **_kwargs):
        raise AssertionError("successful dispatch must not release the lease")

    def get_model(self, model_id, version):
        return {
            "model_id": model_id,
            "version": version,
            "job_id": "job-done",
            "dataset_id": "dataset-1",
            "dataset_hash": "dataset-hash",
            "name": "测试模型",
            "model_kind": "lightgbm",
            "dataset_spec": {
                "research_target": "stock_selection",
                "prediction_scope": "stock",
                "universe_id": "csi500",
                "date_start": "2022-01-01",
                "date_end": "2025-12-31",
                "label": {
                    "kind": "future_5d_cross_sectional_rank",
                    "horizon_trading_days": 5,
                },
                "split": {"embargo_days": 5},
                "factors": [{
                    "factor_id": "mom_20", "factor_version": 1,
                    "params_hash": "params-hash", "params": {},
                }],
            },
            "job_config_json": {"model": {"kind": "lightgbm", "params": {}}},
            "metrics_json": {
                "test_days": 80, "rank_ic": 0.04, "ic_ir": 0.6,
                "validation": {"days": 80, "rank_ic": 0.04, "ic_ir": 0.6},
            },
            "manifest_json": {"future_function_guards": ["PIT"]},
            "state": "validated",
            "is_default": False,
        }

    def list_models(self, *, limit):
        return [self.get_model("model-1", 1)][:limit]

    def update_model_registry(
        self, model_id, version, *, action, validation_approved, note,
    ):
        self.registry_payload = {
            "model_id": model_id, "version": version, "action": action,
            "validation_approved": validation_approved, "note": note,
        }
        model = self.get_model(model_id, version)
        if action == "set_default":
            model["is_default"] = True
        if action == "archive":
            model["state"] = "archived"
        return model

    def delete_model(self, model_id, version):
        self.deleted_payload = {"model_id": model_id, "version": version}
        return {
            "model_id": model_id,
            "version": version,
            "name": "测试模型",
            "job_id": "job-done",
            "dataset_id": "dataset-1",
            "dataset_hash": "d" * 64,
            "dataset_deleted": False,
        }

    def delete_strategy_deployment(self, model_id, version, *, mode):
        self.deleted_deployment_payload = {
            "model_id": model_id, "version": version, "mode": mode,
        }
        return {
            "deployment_id": "deployment-1",
            "model_id": model_id,
            "model_version": version,
            "mode": mode,
        }

    def get_strategy_deployment(self, model_id, version, *, mode):
        return {
            "deployment_id": "deployment-1",
            "model_id": model_id,
            "model_version": version,
            "mode": mode,
            "strategy_snapshot_json": {"live_id": "live-gone"},
        }

    def mark_validated(self, model_id, version, backtest_job_id, *, validation):
        self.validation_payload = dict(validation)
        return {"model_id": model_id, "version": version, "state": "validated"}

    def record_validation_result(
        self, model_id, version, backtest_job_id, *, approved, validation,
    ):
        self.validation_payload = {**validation, "approved_by_repository": approved}
        return {"model_id": model_id, "version": version}

    def create_inference_job(self, model_id, version, payload):
        self.inference_payload = dict(payload)
        return {
            "job_id": "infer-done",
            "model_id": model_id,
            "model_version": version,
            "status": "succeeded",
        }

    def list_artifacts(self, _job_id):
        return [{"artifact_kind": "bundle", "artifact_id": "artifact-1"}]

    def list_inference_runs(
        self, *, status="", model_id="", model_version=None,
        trade_date="", limit=100,
    ):
        rows = [{
            "job_id": "infer-history-1",
            "model_id": "model-1",
            "model_version": 1,
            "model_kind": "lightgbm",
            "model_name": "测试模型",
            "status": "succeeded",
            "trade_date": "2026-08-13",
            "trigger": "manual",
            "prediction_rows": 500,
            "requested_at": "2026-08-13T16:10:00+08:00",
        }]
        if status:
            rows = [item for item in rows if item["status"] == status]
        if model_id:
            rows = [item for item in rows if item["model_id"] == model_id]
        if model_version is not None:
            rows = [item for item in rows if item["model_version"] == model_version]
        if trade_date:
            rows = [item for item in rows if item["trade_date"] == trade_date]
        return rows[:limit]

    def list_inference_schedules(self):
        return [{
            "model_id": "model-1", "model_version": 1,
            "enabled": True, "run_after_local": "16:30",
            "manifest_json": {"large": "internal-only"},
            "dataset_spec": {"large": "internal-only"},
        }]

    def update_inference_schedule(self, model_id, version, payload):
        return {
            "model_id": model_id, "model_version": version,
            "enabled": bool(payload.get("enabled", True)),
            "run_after_local": payload.get("run_after_local", "16:30"),
        }


class _Scheduler:
    def __init__(self) -> None:
        self.submitted = []

    def submit(self, job):
        self.submitted.append(dict(job))
        return {"accepted": True, "job_id": job["job_id"]}

    def status(self):
        return {"ready": True, "busy": False, "active_job_id": "", "progress": {}}


def _client(monkeypatch, repository, scheduler) -> TestClient:
    app = FastAPI()
    app.state.research_worker = scheduler
    app.include_router(model_research.router)
    monkeypatch.setattr(model_research, "repository", repository)
    monkeypatch.setattr(
        model_research,
        "get_training_resource_settings",
        lambda: {
            "schema_version": "alphablocks.model-training-resources.v1",
            "revision": 1,
            "entity_assets": {
                "selection_mode": "all_authorized_stock_assets",
                "bindings": [],
            },
            "local_node": {
                "id": "local", "type": "local", "name": "本机训练",
                "description": "test", "enabled": True,
                "max_runtime_minutes": 1440,
            },
        },
    )
    monkeypatch.setattr(
        model_research.model_repository,
        "latest_model_backtest_jobs",
        lambda _identities: {},
    )
    return TestClient(app)


def test_training_job_is_created_and_dispatched_locally(monkeypatch) -> None:
    repository = _Repository()
    scheduler = _Scheduler()
    client = _client(monkeypatch, repository, scheduler)

    created = client.post("/model-research/jobs", json={"title": "smoke"})
    dispatched = client.post("/model-research/jobs/job-1/dispatch", json={})

    assert created.status_code == 201
    assert created.json()["job"]["job_id"] == "job-created"
    assert dispatched.status_code == 202
    assert dispatched.json()["service"]["accepted"] is True
    assert scheduler.submitted[0]["lease_owner"] == "alpha-factor-service"


def test_model_score_query_freezes_resolved_version_and_exact_date(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    captured = {}

    def snapshot(**kwargs):
        captured.update(kwargs)
        return {
            "model_id": kwargs["model_id"],
            "model_version": kwargs["model_version"],
            "trade_date": kwargs["trade_date"].isoformat(),
            "rows": [{"entity_code": "000001.SZ", "score": 0.9}],
        }

    monkeypatch.setattr(
        model_research.model_repository, "model_score_snapshot", snapshot,
    )

    response = client.post(
        "/model-research/model-scores/query",
        json={"model": "大小盘选股", "date": "2026-08-28", "topk": 20},
    )

    assert response.status_code == 200
    assert response.json()["resolved_model"]["model_version"] == 3
    assert captured == {
        "model_id": "model-a",
        "model_version": 3,
        "trade_date": date(2026, 8, 28),
        "topk": 20,
    }


def test_model_score_range_query_freezes_one_revision_and_bounded_dates(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    captured = {}

    def snapshots(**kwargs):
        captured.update(kwargs)
        return [{"trade_date": kwargs["date_start"].isoformat(), "rows": []}]

    monkeypatch.setattr(
        model_research.model_repository, "model_score_snapshots", snapshots,
    )
    response = client.post(
        "/model-research/model-scores/query-range",
        json={
            "model": "大小盘选股",
            "date_start": "2026-08-01",
            "date_end": "2026-08-28",
            "topk": 20,
        },
    )

    assert response.status_code == 200
    assert response.json()["resolved_model"]["model_version"] == 3
    assert captured == {
        "model_id": "model-a",
        "model_version": 3,
        "date_start": date(2026, 8, 1),
        "date_end": date(2026, 8, 28),
        "topk": 20,
    }


@pytest.mark.parametrize("endpoint", ["jobs", "experiments"])
@pytest.mark.parametrize("second_limit,expected_status", [(720, 201), (240, 409)])
def test_distributed_submission_validates_each_real_node(monkeypatch, endpoint, second_limit, expected_status):
    from factor_service.research import remote

    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    checked = []

    def get_node(node_id):
        assert node_id in {"cpu-a", "cpu-b"}
        checked.append(node_id)
        return SimpleNamespace(name=node_id, max_runtime_minutes=720 if node_id == "cpu-a" else second_limit)

    monkeypatch.setattr(remote, "get_remote_node", get_node)
    execution = {"mode": "distributed", "node_id": "distributed", "node_ids": ["cpu-a", "cpu-b"], "max_runtime_minutes": 720}
    response = client.post(f"/model-research/{endpoint}", json={"title": "distributed", "execution": execution})
    assert response.status_code == expected_status, response.text
    assert checked == ["cpu-a", "cpu-b"]
    if expected_status == 201 and endpoint == "jobs":
        assert repository.created_payload["execution"] == execution
    if expected_status == 409:
        assert "cpu-b" in response.json()["detail"]
        assert not hasattr(repository, "created_payload")


@pytest.mark.parametrize("nodes", [["cpu-a", "local"], ["cpu-a", "cpu-a"], ["distributed"]])
def test_distributed_submission_rejects_invalid_nodes_before_lookup(monkeypatch, nodes):
    from factor_service.research import remote

    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    monkeypatch.setattr(remote, "get_remote_node", lambda node_id: pytest.fail("invalid nodes must not be resolved"))
    response = client.post("/model-research/jobs", json={"title": "invalid", "execution": {"node_ids": nodes}})
    assert response.status_code == 400
    assert not hasattr(repository, "created_payload")


def test_training_job_freezes_configured_database_binding(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    monkeypatch.setattr(
        model_research,
        "get_training_resource_settings",
        lambda: {
            "schema_version": "alphablocks.model-training-data-bindings.v1",
            "revision": 2,
            "bindings": {
                "stock_industry_one_hot": {
                    "enabled": True,
                    "source_type": "entity_asset",
                    "source_id": "asset_industry_membership",
                    "source_label": "行业成分与权重",
                    "provider_node_id": "industry_membership_weight_real",
                    "field_bindings": {
                        "trade_date": "trade_date",
                        "instrument": "con_code",
                        "industry_code": "index_code",
                        "industry_name": "level1_name",
                        "industry_level": "level_type",
                        "weight": "weight",
                    },
                    "industry_level_value": "1",
                    "catalog_updated_at": "",
                },
            },
        },
    )
    payload = {
        "title": "resource-policy",
        "execution": {"node_id": "local", "max_runtime_minutes": 180},
        "dataset": {
            "industry_feature": {"enabled": True},
            "factors": [],
        },
    }

    accepted = client.post("/model-research/jobs", json=payload)

    assert accepted.status_code == 201
    frozen = repository.created_payload["dataset"]["industry_feature"]
    assert frozen["schema_version"] == "alphablocks.stock-industry-one-hot.v2"
    assert frozen["data_binding"]["source_type"] == "node"
    assert frozen["data_binding"]["source_id"] == "industry_membership_weight_real"
    assert frozen["data_binding"]["provider_node_id"] == "industry_membership_weight_real"
    assert frozen["data_binding"]["settings_revision"] == 2


def test_completed_training_requires_explicit_registration(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.post("/model-research/jobs/job-done/register", json={})

    assert response.status_code == 200
    assert response.json()["job"]["registration_status"] == "registered"
    assert response.json()["job"]["model_version"] == 1


def test_completed_training_can_be_kept_out_of_registry(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.post("/model-research/jobs/job-done/decline-registration", json={})

    assert response.status_code == 200
    assert response.json()["job"]["registration_status"] == "declined"
    assert response.json()["job"]["model_version"] is None


def test_training_targets_expose_real_availability(monkeypatch) -> None:
    class _DatasetBuilder:
        def __init__(self, _settings):
            pass

        @staticmethod
        def target_capabilities():
            return [
                {"target": "stock_selection", "ready": True},
                {
                    "target": "industry_rotation", "ready": True,
                    "minimum_date": "2021-12-13",
                    "missing_fields": [],
                },
            ]

    monkeypatch.setattr(model_research, "DatasetBuilder", _DatasetBuilder)
    monkeypatch.setattr(model_research, "load_research_settings", lambda: object())
    client = _client(monkeypatch, _Repository(), _Scheduler())

    response = client.get("/model-research/training-targets")

    assert response.status_code == 200
    assert response.json()["targets"][1]["minimum_date"] == "2021-12-13"


def test_execution_nodes_and_local_status_are_exposed_without_secrets(monkeypatch) -> None:
    from factor_service.research import remote

    monkeypatch.setattr(remote, "execution_nodes", lambda: [
        {"id": "local", "type": "local", "available": True},
        {
            "id": "autodl-gpu-01", "type": "remote", "available": True,
            "credential_type": "password",
        },
    ])
    client = _client(monkeypatch, _Repository(), _Scheduler())

    nodes = client.get("/model-research/execution-nodes")
    status = client.get("/model-research/execution-nodes/local/status")

    assert nodes.status_code == 200
    assert nodes.json()["storage"] == "postgresql"
    assert nodes.json()["nodes"][1]["id"] == "autodl-gpu-01"
    assert "private-value" not in nodes.text
    assert "ssh_password" not in nodes.text
    assert status.status_code == 200
    assert status.json()["status"]["online"] is True


def test_training_resource_settings_are_read_and_updated_with_revision(monkeypatch) -> None:
    current = {
        "schema_version": "alphablocks.model-training-data-bindings.v1",
        "revision": 4,
        "updated_at": "2026-08-27T00:00:00+00:00",
        "bindings": {
            "stock_industry_one_hot": {
                "enabled": False,
                "source_type": "node",
                "source_id": "",
                "source_label": "",
                "provider_node_id": "",
                "field_bindings": {},
                "industry_level_value": "1",
                "catalog_updated_at": "",
            },
        },
    }
    captured = {}

    class _SettingsRepository:
        def update(self, source, *, expected_revision):
            captured.update({
                "source": source, "expected_revision": expected_revision,
            })
            return {**current, "revision": 5}

    client = _client(monkeypatch, _Repository(), _Scheduler())
    monkeypatch.setattr(
        model_research, "get_training_resource_settings", lambda: current,
    )
    monkeypatch.setattr(
        model_research, "TrainingResourceSettingsRepository",
        _SettingsRepository,
    )

    loaded = client.get("/model-research/training-data-bindings")
    updated = client.put(
        "/model-research/training-data-bindings",
        json={"expected_revision": 4, "settings": {
            "schema_version": current["schema_version"],
            "bindings": current["bindings"],
        }},
    )

    assert loaded.status_code == 200
    assert loaded.json()["storage"] == "postgresql"
    assert loaded.json()["settings"]["revision"] == 4
    assert updated.status_code == 200
    assert updated.json()["settings"]["revision"] == 5
    assert captured["expected_revision"] == 4


def test_execution_node_settings_crud_api_never_returns_password_values(monkeypatch) -> None:
    from factor_service.research import remote_settings

    node = {
        "id": "autodl-gpu-01",
        "type": "remote",
        "name": "AutoDL GPU",
        "host": "gpu.example.test",
        "credential_type": "password",
        "ssh_password_configured": True,
    }
    monkeypatch.setattr(remote_settings, "list_remote_node_settings", lambda: [node])
    monkeypatch.setattr(
        remote_settings, "create_remote_node_setting", lambda payload: {**node, **payload},
    )
    monkeypatch.setattr(
        remote_settings, "update_remote_node_setting",
        lambda node_id, payload: {**node, **payload, "id": node_id},
    )
    monkeypatch.setattr(
        remote_settings, "delete_remote_node_setting",
        lambda node_id: {"id": node_id, "deleted": True},
    )
    client = _client(monkeypatch, _Repository(), _Scheduler())

    listed = client.get("/model-research/execution-node-settings")
    created = client.post(
        "/model-research/execution-node-settings", json={"id": "autodl-gpu-01"},
    )
    updated = client.put(
        "/model-research/execution-node-settings/autodl-gpu-01",
        json={"name": "AutoDL A100"},
    )
    deleted = client.delete(
        "/model-research/execution-node-settings/autodl-gpu-01",
    )

    assert listed.status_code == 200
    assert listed.json()["storage"] == "postgresql"
    assert listed.json()["security"]["password_values_returned"] is False
    assert listed.json()["security"]["secret_storage"] == "postgresql_aes_gcm"
    assert listed.json()["security"]["master_key_storage"] == "process_environment"
    assert created.status_code == 201
    assert updated.json()["node"]["name"] == "AutoDL A100"
    assert deleted.json()["deleted"] is True
    assert "private-password" not in "".join(
        response.text for response in (listed, created, updated, deleted)
    )


def test_model_registry_action_uses_validation_gate_and_returns_pool_state(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    monkeypatch.setattr(
        model_research.model_repository, "latest_model_backtests", lambda _keys: {},
    )
    monkeypatch.setattr(
        model_research, "_model_validation_view",
        lambda _model, _backtest: {"approved": True, "passed": True},
    )

    response = client.post(
        "/model-research/models/model-1/versions/1/registry",
        json={"action": "set_default", "note": "首个正式主模型"},
    )

    assert response.status_code == 200
    assert repository.registry_payload == {
        "model_id": "model-1",
        "version": 1,
        "action": "set_default",
        "validation_approved": True,
        "note": "首个正式主模型",
    }
    assert response.json()["model"]["registry"]["is_default"] is True


def test_model_version_delete_cleans_evidence_and_artifacts(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    cleanup_calls = []

    class _Store:
        def __init__(self, _root):
            pass

        @staticmethod
        def delete_job_artifacts(job_id):
            cleanup_calls.append(("artifacts", job_id))
            return {"job_artifacts": True, "pending_uploads": False}

    monkeypatch.setattr(
        model_research.model_repository,
        "active_model_backtest_jobs",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        model_research.model_repository,
        "delete_model_data",
        lambda **kwargs: cleanup_calls.append(("clickhouse", kwargs)) or {
            "backtest_job_count": 2,
            "predictions_deleted": True,
        },
    )
    monkeypatch.setattr(model_research, "ModelArtifactStore", _Store)
    monkeypatch.setattr(
        model_research,
        "load_research_settings",
        lambda: SimpleNamespace(model_artifacts_root="/tmp/model-artifacts"),
    )

    response = client.delete("/model-research/models/model-1/versions/1")

    assert response.status_code == 200
    assert response.json()["deleted"]["job_id"] == "job-done"
    assert response.json()["warnings"] == []
    assert repository.deleted_payload == {"model_id": "model-1", "version": 1}
    assert cleanup_calls == [
        ("clickhouse", {"model_id": "model-1", "model_version": 1}),
        ("artifacts", "job-done"),
    ]


def test_model_version_delete_rejects_active_backtest(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    monkeypatch.setattr(
        model_research.model_repository,
        "active_model_backtest_jobs",
        lambda **_kwargs: [{
            "backtest_job_id": "model-backtest-running",
            "status": "running",
        }],
    )

    response = client.delete("/model-research/models/model-1/versions/1")

    assert response.status_code == 409
    assert "model-backtest-running" in response.json()["detail"]
    assert repository.deleted_payload is None


def test_strategy_deployment_delete_is_idempotent_and_model_scoped(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.request(
        "DELETE",
        "/model-research/models/model-1/versions/1/strategy-deployments/paper",
        json={
            "live_id": "live-gone",
            "reason": "orphaned_paper_strategy",
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["deleted"]["deployment_id"] == "deployment-1"
    assert repository.deleted_deployment_payload == {
        "model_id": "model-1", "version": 1, "mode": "paper",
    }


def test_model_research_report_routes(monkeypatch) -> None:
    client = _client(monkeypatch, _Repository(), _Scheduler())
    monkeypatch.setattr(
        model_research.model_repository, "latest_model_backtests", lambda _keys: {},
    )

    report = client.get(
        "/model-research/models/model-1/versions/1/research-report",
    )
    markdown = client.get(
        "/model-research/models/model-1/versions/1/research-report.md",
    )

    assert report.status_code == 200
    assert report.json()["report"]["identity"]["model_id"] == "model-1"
    assert "Dataset Hash" in report.json()["markdown"]
    assert markdown.status_code == 200
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert "attachment;" in markdown.headers["content-disposition"]


def test_grid_experiment_creates_a_queryable_job_batch(monkeypatch) -> None:
    client = _client(monkeypatch, _Repository(), _Scheduler())

    created = client.post(
        "/model-research/experiments",
        json={"title": "LGBM参数实验", "search": {"parameters": {}}},
    )
    experiment_id = created.json()["experiment"]["experiment_id"]
    loaded = client.get(f"/model-research/experiments/{experiment_id}")
    jobs = client.get(
        "/model-research/jobs", params={"experiment_id": experiment_id},
    )

    assert created.status_code == 201
    assert loaded.status_code == 200
    assert loaded.json()["experiment"]["trial_count"] == 2
    assert len(jobs.json()["jobs"]) == 2


def test_failed_experiment_can_be_restarted_as_a_new_run(monkeypatch) -> None:
    repository = _Repository()
    repository.jobs["failed-trial"] = {
        "job_id": "failed-trial",
        "status": "failed",
        "config_json": {
            "execution": {"node_id": "local"},
            "experiment": {
                "experiment_id": "model_experiment_failed",
                "trial_index": 1,
                "trial_count": 1,
                "search_params": {"model_kind": "lightgbm"},
            },
        },
    }
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.post(
        "/model-research/experiments/model_experiment_failed/retry",
    )

    assert response.status_code == 201
    experiment = response.json()["experiment"]
    assert experiment["experiment_id"] == "model_experiment_retried"
    assert (
        experiment["restarted_from_experiment_id"]
        == "model_experiment_failed"
    )


def test_failed_job_can_be_retried_in_place_with_idempotency_key(monkeypatch) -> None:
    repository = _Repository()
    repository.jobs["failed-trial"] = {
        "job_id": "failed-trial",
        "kind": "train",
        "status": "failed",
        "retryable": True,
        "config_json": {"execution": {"node_id": "local"}},
    }
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.post(
        "/model-research/jobs/failed-trial/retry",
        json={"idempotency_key": "retry-once"},
    )

    assert response.status_code == 200
    assert response.json()["job"]["status"] == "queued"
    assert response.json()["job"]["idempotency_key"] == "retry-once"


def test_experiment_lineage_conflict_is_returned_as_http_409(monkeypatch) -> None:
    repository = _Repository()

    def reject_experiment(_payload):
        raise model_research.ModelResearchConflict(
            "下一轮实验必须完整继承父任务的冻结因子"
        )

    repository.create_training_experiment = reject_experiment
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.post(
        "/model-research/experiments",
        json={"title": "篡改的下一轮实验", "search": {"strategy": "factor_ablation"}},
    )

    assert response.status_code == 409
    assert "冻结因子" in response.json()["detail"]


def test_dataset_preflight_endpoint_freezes_before_calendar_validation(
    monkeypatch,
) -> None:
    client = _client(monkeypatch, _Repository(), _Scheduler())
    calls = []

    def freeze(payload):
        payload["dataset"]["data_bindings"] = {"frozen": True}
        calls.append("freeze")

    class Preflight:
        def validate(self, payload):
            assert payload["dataset"]["data_bindings"] == {"frozen": True}
            calls.append("validate")
            return {
                "schema_version": "alphablocks.dataset-preflight.v1",
                "dataset": payload["dataset"],
                "dataset_hash": "a" * 64,
                "segments": {},
                "calendar": {},
            }

    monkeypatch.setattr(model_research, "_freeze_training_data_bindings", freeze)
    monkeypatch.setattr(model_research, "DatasetPreflightService", Preflight)

    response = client.post(
        "/model-research/dataset-preflight",
        json={"dataset": {"name": "preflight"}},
    )

    assert response.status_code == 200
    assert response.json()["preflight"]["dataset_hash"] == "a" * 64
    assert calls == ["freeze", "validate"]


def test_removed_page_only_experiment_and_scheduler_endpoints_return_404(
    monkeypatch,
) -> None:
    client = _client(monkeypatch, _Repository(), _Scheduler())

    assert client.get("/model-research/experiments").status_code == 405
    assert client.post(
        "/model-research/inference-scheduler/tick", json={"force": True},
    ).status_code == 404


def test_removed_research_template_routes_return_404(monkeypatch) -> None:
    client = _client(monkeypatch, _Repository(), _Scheduler())

    assert client.get("/model-research/research-templates").status_code == 404
    assert client.post(
        "/model-research/research-templates", json={"name": "已删除"},
    ).status_code == 404
    assert client.get(
        "/model-research/research-templates/template-1",
    ).status_code == 404
    assert client.put(
        "/model-research/research-templates/template-1", json={"name": "已删除"},
    ).status_code == 404
    assert client.post(
        "/model-research/research-templates/template-1/archive",
    ).status_code == 404


def test_incremental_training_precheck_route_is_server_authoritative(monkeypatch) -> None:
    client = _client(monkeypatch, _Repository(), _Scheduler())

    response = client.post(
        "/model-research/models/model-1/versions/1/incremental-training-precheck",
        json={
            "dataset": {"date_end": "2026-08-14"},
            "model": {"kind": "lightgbm", "params": {}},
            "walk_forward": {"enabled": False},
        },
    )

    assert response.status_code == 200
    precheck = response.json()["precheck"]
    assert precheck["passed"] is True
    assert precheck["contract"]["source_model_version"] == 1


def test_model_architecture_crud_and_activation_routes(monkeypatch) -> None:
    repository = _Repository()
    monkeypatch.setattr(
        model_research.model_repository,
        "active_model_backtest_jobs",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        model_research.model_repository,
        "delete_model_data",
        lambda **_kwargs: {
            "backtest_job_count": 0,
            "predictions_deleted": True,
        },
    )
    client = _client(monkeypatch, repository, _Scheduler())
    payload = {
        "name": "三引擎选股架构",
        "merge_method": "priority",
        "top_n": 20,
        "rebalance_every": 5,
        "engines": [{
            "engine_key": "ambush",
            "model_id": "model-a",
            "model_version": 1,
            "priority": 1,
        }],
    }

    created = client.post("/model-research/architectures", json=payload)
    assert created.status_code == 201
    architecture_id = created.json()["architecture"]["architecture_id"]

    listed = client.get("/model-research/architectures")
    assert listed.status_code == 200
    assert listed.json()["architectures"][0]["architecture_id"] == architecture_id

    updated = client.put(
        f"/model-research/architectures/{architecture_id}",
        json={**payload, "name": "三引擎选股架构 v2", "revision": 1},
    )
    assert updated.status_code == 200
    assert updated.json()["architecture"]["revision"] == 2

    activated = client.post(
        f"/model-research/architectures/{architecture_id}/activate",
    )
    assert activated.status_code == 200
    assert activated.json()["architecture"]["state"] == "active"

    archived = client.post(
        f"/model-research/architectures/{architecture_id}/archive",
    )
    assert archived.status_code == 200
    assert archived.json()["architecture"]["state"] == "archived"

    deleted = client.delete(
        f"/model-research/architectures/{architecture_id}",
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted"]["architecture_id"] == architecture_id
    assert deleted.json()["cleanup"]["clickhouse"]["predictions_deleted"] is True
    assert architecture_id not in repository.architectures


def test_model_architecture_can_start_a_shared_engine_backtest(monkeypatch) -> None:
    repository = _Repository()
    architecture = repository.create_model_architecture({
        "name": "组合架构",
        "universe_id": "csi500",
        "top_n": 20,
        "rebalance_every": 5,
        "engines": [{
            "engine_key": "ranker", "enabled": True,
            "model_id": "model-a", "model_version": 1,
        }],
    })
    architecture["readiness"] = {
        "ready": False, "research_backtest_ready": True,
    }
    repository.architectures[architecture["architecture_id"]] = architecture
    captured = {}
    created = ModelBacktestJobOut(
        backtest_job_id="architecture-backtest-1",
        model_id=architecture["architecture_id"], model_version=1,
        universe_id="csi500", benchmark_code="000905.SH",
        date_preset="1y", status="pending",
        configuration={"signal_source": "model_architecture"},
    )

    def create_backtest(source, **kwargs):
        captured.update({"architecture": source, **kwargs})
        return created

    monkeypatch.setattr(
        model_research.model_repository,
        "create_architecture_backtest_job",
        create_backtest,
    )
    monkeypatch.setattr(model_research, "run_model_backtest_job", lambda _job_id: None)
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.post(
        f"/model-research/architectures/{architecture['architecture_id']}/backtests",
        json={"date_preset": "1y"},
    )

    assert response.status_code == 201
    assert response.json()["backtest"]["backtest_job_id"] == "architecture-backtest-1"
    assert captured["date_preset"] == "1y"
    assert captured["ablation_profile"] == "full"
    assert captured["architecture"]["architecture_id"] == architecture["architecture_id"]


def test_model_architecture_can_start_and_list_ablation_backtests(monkeypatch) -> None:
    repository = _Repository()
    architecture = {
        "architecture_id": "architecture-hierarchical",
        "revision": 3,
        "state": "draft",
        "pipeline_mode": "hierarchical",
        "readiness": {"ready": False, "research_backtest_ready": True},
    }
    repository.architectures[architecture["architecture_id"]] = architecture
    captured = []

    def create_backtest(source, **kwargs):
        profile = kwargs["ablation_profile"]
        captured.append(profile)
        return ModelBacktestJobOut(
            backtest_job_id=f"architecture-backtest-{profile}",
            model_id=source["architecture_id"], model_version=source["revision"],
            universe_id="csi500", benchmark_code="000905.SH",
            date_preset=kwargs["date_preset"], status="pending",
            configuration={
                "signal_source": "model_architecture",
                "ablation_profile": profile,
            },
        )

    profiles = [
        {"key": "stock_only", "label": "仅个股"},
        {"key": "industry_stock", "label": "行业 + 个股"},
        {"key": "full", "label": "分层全开"},
    ]
    monkeypatch.setattr(
        model_research.model_repository, "architecture_ablation_profiles",
        lambda: profiles,
    )
    monkeypatch.setattr(
        model_research.model_repository, "create_architecture_backtest_job",
        create_backtest,
    )
    monkeypatch.setattr(
        model_research.model_repository, "list_architecture_backtest_jobs",
        lambda *_args, **_kwargs: [
            create_backtest(architecture, date_preset="1y", ablation_profile="full")
        ],
    )
    monkeypatch.setattr(model_research, "run_model_backtest_job", lambda _job_id: None)
    client = _client(monkeypatch, repository, _Scheduler())

    created = client.post(
        "/model-research/architectures/architecture-hierarchical/ablation-backtests",
        json={"date_preset": "1y"},
    )
    assert created.status_code == 201
    assert [item["configuration"]["ablation_profile"] for item in created.json()["backtests"]] == [
        "stock_only", "industry_stock", "full",
    ]

    listed = client.get(
        "/model-research/architectures/architecture-hierarchical/backtests",
    )
    assert listed.status_code == 200
    assert listed.json()["backtests"][0]["configuration"]["ablation_profile"] == "full"


def test_architecture_backtest_result_does_not_mutate_model_validation(monkeypatch) -> None:
    result = ModelBacktestJobOut(
        backtest_job_id="architecture-backtest-1",
        model_id="architecture-a", model_version=2,
        universe_id="csi500", benchmark_code="000905.SH",
        date_preset="1y", status="success",
        configuration={"signal_source": "model_architecture"},
    )
    monkeypatch.setattr(
        model_research.model_repository,
        "get_model_backtest_job",
        lambda _job_id: result,
    )
    repository = _Repository()
    repository.get_model = lambda *_args: (_ for _ in ()).throw(
        AssertionError("架构回测不能按模型版本执行验证写入")
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get("/model-research/model-backtests/architecture-backtest-1")

    assert response.status_code == 200
    assert response.json()["backtest"]["backtest_kind"] == "model_architecture"
    assert response.json()["backtest"]["validation"] is None


def test_prediction_overview_endpoint_forwards_research_scope(monkeypatch) -> None:
    captured = {}

    def overview(**kwargs):
        captured.update(kwargs)
        return {
            "selected_date": date(2026, 8, 14),
            "cross_section": {"row_count": 500, "pit_violation_count": 0},
        }

    monkeypatch.setattr(
        model_research.model_repository, "model_prediction_overview", overview,
    )
    client = _client(monkeypatch, _Repository(), _Scheduler())

    response = client.get(
        "/model-research/models/model-a/versions/1/prediction-overview",
        params={"trade_date": "2026-08-14", "top_n": 50, "history_days": 60},
    )

    assert response.status_code == 200
    assert response.json()["overview"]["selected_date"] == "2026-08-14"
    assert captured == {
        "model_id": "model-a", "model_version": 1,
        "trade_date": date(2026, 8, 14), "top_n": 50, "history_days": 60,
    }


def test_feature_drift_endpoint_uses_frozen_dataset_artifact(monkeypatch) -> None:
    repository = _Repository()
    dataset_hash = "b" * 64
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "model_kind": "lightgbm",
        "dataset_hash": dataset_hash,
        "manifest_json": {},
    }
    captured = {}

    def drift(requested_hash, artifact_root):
        captured.update({
            "dataset_hash": requested_hash,
            "artifact_root": str(artifact_root),
        })
        return {
            "dataset_hash": requested_hash,
            "status": "stable",
            "counts": {"stable": 3, "medium": 0, "severe": 0},
            "features": [],
        }

    monkeypatch.setattr(model_research, "dataset_feature_drift", drift)
    monkeypatch.setattr(
        model_research, "load_service_settings",
        lambda: SimpleNamespace(model_artifacts_root="/tmp/model-artifacts"),
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get(
        "/model-research/models/model-a/versions/1/feature-drift",
    )

    assert response.status_code == 200
    assert response.json()["drift"]["status"] == "stable"
    assert response.json()["drift"]["sources"] == [{
        "model_id": "model-a", "model_version": 1, "model_kind": "lightgbm",
    }]
    assert captured == {
        "dataset_hash": dataset_hash,
        "artifact_root": "/tmp/model-artifacts",
    }


def test_walk_forward_attribution_endpoint_uses_frozen_manifest(monkeypatch) -> None:
    repository = _Repository()
    dataset_hash = "f" * 64
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "model_kind": "lightgbm",
        "dataset_hash": dataset_hash,
        "dataset_spec": {"factors": [{
            "factor_id": "momentum", "factor_version": 1,
            "label": "动量因子",
        }]},
        "manifest_json": {
            "feature_names": ["momentum__v1__abcd1234"],
            "walk_forward": {"enabled": True, "windows": [{
                "window": 1,
                "segments": {
                    "train": ["2024-01-01", "2024-12-31"],
                    "test": ["2025-01-01", "2025-12-31"],
                },
                "metrics": {"rank_ic": 0.01},
            }]},
        },
    }
    captured = {}

    def attribution(requested_hash, artifact_root, walk_forward):
        captured.update({
            "dataset_hash": requested_hash,
            "artifact_root": str(artifact_root),
            "enabled": walk_forward["enabled"],
        })
        window = {"window": 1, "features": [{
            "factor": "momentum__v1__abcd1234", "status": "decayed",
        }]}
        return {
            "eligible": True, "weak_window": window, "windows": [window],
        }

    monkeypatch.setattr(model_research, "dataset_walk_forward_attribution", attribution)
    monkeypatch.setattr(
        model_research, "load_service_settings",
        lambda: SimpleNamespace(model_artifacts_root="/tmp/model-artifacts"),
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get(
        "/model-research/models/model-a/versions/1/walk-forward-attribution",
    )

    assert response.status_code == 200
    factor = response.json()["diagnostics"]["weak_window"]["features"][0]
    assert factor["label"] == "动量因子"
    assert captured == {
        "dataset_hash": dataset_hash,
        "artifact_root": "/tmp/model-artifacts",
        "enabled": True,
    }


def test_architecture_walk_forward_attribution_combines_engines(monkeypatch) -> None:
    repository = _Repository()
    architecture_id = "architecture-wfa"
    repository.architectures[architecture_id] = {
        "architecture_id": architecture_id,
        "revision": 1,
        "engines": [{
            "engine_key": "stock", "display_name": "个股模型",
            "stage": "stock_rank", "role": "stock_selection",
            "model_id": "model-stock", "model_version": 1, "enabled": True,
        }],
    }
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "name": "个股模型",
        "model_kind": "lightgbm",
        "dataset_hash": "e" * 64,
        "manifest_json": {"walk_forward": {"enabled": True, "windows": [{}]}},
    }
    job = SimpleNamespace(model_dump=lambda mode: {
        "status": "success", "configuration": {}, "payload": {},
    })
    monkeypatch.setattr(
        model_research.model_repository,
        "list_architecture_backtest_jobs",
        lambda *args, **kwargs: [job],
    )
    monkeypatch.setattr(
        model_research, "architecture_walk_forward_attribution",
        lambda jobs: {
            "eligible": True,
            "weak_window": {
                "window": 3, "all_profiles_negative": True,
                "market_regime": "strong_bull",
                "industry_gate_delta": 0.02,
            },
        },
    )
    monkeypatch.setattr(
        model_research, "dataset_walk_forward_attribution",
        lambda *args: {
            "primary_cause": "factor_signal_decay",
            "conclusion": "W3最弱",
            "weak_window": {
                "window": 3, "model_rank_ic": 0.001,
                "counts": {"reversed": 0, "decayed": 1}, "features": [],
            },
            "windows": [],
        },
    )
    monkeypatch.setattr(
        model_research, "load_service_settings",
        lambda: SimpleNamespace(model_artifacts_root="/tmp/model-artifacts"),
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get(
        f"/model-research/architectures/{architecture_id}/walk-forward-attribution",
    )

    assert response.status_code == 200
    diagnostics = response.json()["diagnostics"]
    assert diagnostics["engines"][0]["weak_window"]["model_rank_ic"] == 0.001
    assert any(item["key"] == "stock_signal_decay" for item in diagnostics["findings"])


def test_permutation_importance_endpoint_freezes_top_features(monkeypatch) -> None:
    repository = _Repository()
    dataset_hash = "c" * 64
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "job_id": "job-model",
        "model_kind": "lightgbm",
        "dataset_hash": dataset_hash,
        "feature_importance_json": [
            {"factor": "f2", "importance": 9},
            {"factor": "f1", "importance": 4},
        ],
        "manifest_json": {
            "feature_names": ["f1", "f2", "f3"],
            "segments": {"test": ["2024-01-01", "2024-01-31"]},
            "model_params": {"num_leaves": 31},
        },
    }
    repository.list_artifacts = lambda job_id: [{
        "artifact_kind": "bundle", "relative_path": "job/bundle/model.tar.gz",
    }]

    class _Store:
        def __init__(self, _root):
            pass

        @staticmethod
        def resolve(relative_path):
            return f"/resolved/{relative_path}"

    captured = {}

    def permutation(bundle_path, dataset_path, **kwargs):
        captured.update({
            "bundle_path": str(bundle_path),
            "dataset_path": str(dataset_path),
            **kwargs,
        })
        return {
            "baseline": {"rank_ic": 0.05},
            "features": [],
            "feature_count": len(kwargs["feature_names"]),
        }

    monkeypatch.setattr(model_research, "ModelArtifactStore", _Store)
    monkeypatch.setattr(
        model_research, "load_service_settings",
        lambda: SimpleNamespace(model_artifacts_root="/tmp/model-artifacts"),
    )
    monkeypatch.setattr(
        model_research, "artifact_model_permutation_importance", permutation,
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get(
        "/model-research/models/model-a/versions/1/permutation-importance",
        params={"limit": 2},
    )

    assert response.status_code == 200
    assert response.json()["diagnostics"]["truncated"] is True
    assert captured["feature_names"] == ["f2", "f1"]
    assert captured["model_params"] == {"num_leaves": 31}
    assert captured["dataset_path"] == (
        f"/resolved/datasets/{dataset_hash}/dataset.parquet"
    )


def test_feature_redundancy_endpoint_uses_model_dataset_hash(monkeypatch) -> None:
    repository = _Repository()
    dataset_hash = "f" * 64
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "model_kind": "xgboost",
        "dataset_hash": dataset_hash,
        "manifest_json": {},
    }
    captured = {}

    def redundancy(requested_hash, artifact_root, *, threshold):
        captured.update({
            "dataset_hash": requested_hash,
            "artifact_root": str(artifact_root),
            "threshold": threshold,
        })
        return {
            "dataset_hash": requested_hash,
            "threshold": threshold,
            "groups": [],
        }

    monkeypatch.setattr(model_research, "dataset_feature_redundancy", redundancy)
    monkeypatch.setattr(
        model_research, "load_service_settings",
        lambda: SimpleNamespace(model_artifacts_root="/tmp/model-artifacts"),
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get(
        "/model-research/models/model-a/versions/1/feature-redundancy",
        params={"threshold": 0.9},
    )

    assert response.status_code == 200
    assert response.json()["diagnostics"]["threshold"] == 0.9
    assert captured == {
        "dataset_hash": dataset_hash,
        "artifact_root": "/tmp/model-artifacts",
        "threshold": 0.9,
    }


def test_factor_validation_audit_endpoint_uses_frozen_dataset(monkeypatch) -> None:
    repository = _Repository()
    dataset_hash = "2" * 64
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "model_kind": "lightgbm",
        "dataset_hash": dataset_hash,
        "manifest_json": {},
    }
    captured = {}

    def audit(requested_hash, artifact_root):
        captured.update({
            "dataset_hash": requested_hash,
            "artifact_root": str(artifact_root),
        })
        return {
            "dataset_hash": requested_hash,
            "method": {"test_segment_read": False},
            "factors": [],
        }

    monkeypatch.setattr(model_research, "dataset_factor_validation_audit", audit)
    monkeypatch.setattr(
        model_research, "load_service_settings",
        lambda: SimpleNamespace(model_artifacts_root="/tmp/model-artifacts"),
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get(
        "/model-research/models/model-a/versions/1/factor-validation-audit",
    )

    assert response.status_code == 200
    assert response.json()["diagnostics"]["method"]["test_segment_read"] is False
    assert captured == {
        "dataset_hash": dataset_hash,
        "artifact_root": "/tmp/model-artifacts",
    }


def test_shap_summary_endpoint_uses_frozen_bundle_and_validation_split(
    monkeypatch,
) -> None:
    repository = _Repository()
    dataset_hash = "1" * 64
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "job_id": "job-tree",
        "model_kind": "lightgbm",
        "dataset_hash": dataset_hash,
        "manifest_json": {
            "feature_names": ["f1", "f2"],
            "segments": {"valid": ["2024-03-01", "2024-03-31"]},
        },
    }
    repository.list_artifacts = lambda job_id: [{
        "artifact_kind": "bundle", "relative_path": "job-tree/bundle/model.tar.gz",
    }]

    class _Store:
        def __init__(self, _root):
            pass

        @staticmethod
        def resolve(relative_path):
            return f"/resolved/{relative_path}"

    captured = {}

    def shap(bundle_path, dataset_path, **kwargs):
        captured.update({
            "bundle_path": str(bundle_path),
            "dataset_path": str(dataset_path),
            **kwargs,
        })
        return {"split": kwargs["split"], "rows_used": 3000, "features": []}

    monkeypatch.setattr(model_research, "ModelArtifactStore", _Store)
    monkeypatch.setattr(
        model_research, "load_service_settings",
        lambda: SimpleNamespace(model_artifacts_root="/tmp/model-artifacts"),
    )
    monkeypatch.setattr(model_research, "artifact_model_shap_summary", shap)
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get(
        "/model-research/models/model-a/versions/1/shap-summary",
        params={"split": "valid", "sample_rows": 3000},
    )

    assert response.status_code == 200
    assert response.json()["diagnostics"]["dataset_hash"] == dataset_hash
    assert captured["bundle_path"] == "/resolved/job-tree/bundle/model.tar.gz"
    assert captured["dataset_path"] == f"/resolved/datasets/{dataset_hash}/dataset.parquet"
    assert captured["feature_names"] == ["f1", "f2"]
    assert captured["split"] == "valid"
    assert captured["sample_rows"] == 3000


def test_training_diagnostics_endpoint_recovers_bundle_history(monkeypatch) -> None:
    repository = _Repository()
    dataset_hash = "2" * 64
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "job_id": "job-history",
        "model_kind": "xgboost",
        "dataset_hash": dataset_hash,
        "manifest_json": {
            "model_kind": "xgboost",
            "model_params": {
                "num_boost_round": 300,
                "early_stopping_rounds": 30,
            },
        },
    }
    repository.list_artifacts = lambda job_id: [{
        "artifact_kind": "bundle", "relative_path": "job-history/bundle/model.tar.gz",
    }]

    class _Store:
        def __init__(self, _root):
            pass

        @staticmethod
        def resolve(relative_path):
            return f"/resolved/{relative_path}"

    captured = {}

    def diagnostics(bundle_path, **kwargs):
        captured.update({"bundle_path": str(bundle_path), **kwargs})
        return {"available": True, "best_iteration": 42, "history": []}

    monkeypatch.setattr(model_research, "ModelArtifactStore", _Store)
    monkeypatch.setattr(
        model_research, "load_service_settings",
        lambda: SimpleNamespace(model_artifacts_root="/tmp/model-artifacts"),
    )
    monkeypatch.setattr(
        model_research, "artifact_model_training_diagnostics", diagnostics,
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get(
        "/model-research/models/model-a/versions/1/training-diagnostics",
    )

    assert response.status_code == 200
    assert response.json()["diagnostics"]["best_iteration"] == 42
    assert response.json()["diagnostics"]["dataset_hash"] == dataset_hash
    assert captured == {
        "bundle_path": "/resolved/job-history/bundle/model.tar.gz",
        "model_kind": "xgboost",
        "model_params": {
            "num_boost_round": 300,
            "early_stopping_rounds": 30,
        },
    }


def test_quantile_diagnostics_endpoint_uses_frozen_label_horizon(monkeypatch) -> None:
    repository = _Repository()
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "dataset_hash": "3" * 64,
        "dataset_spec": {"label": {"horizon_trading_days": 10}},
        "manifest_json": {},
    }
    captured = {}

    def diagnostics(**kwargs):
        captured.update(kwargs)
        return {"status": "strong", "groups": [], "horizon_trading_days": 10}

    monkeypatch.setattr(
        model_research.model_repository,
        "model_prediction_quantile_diagnostics",
        diagnostics,
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get(
        "/model-research/models/model-a/versions/4/quantile-diagnostics",
        params={"quantiles": 8, "sample_interval": 10},
    )

    assert response.status_code == 200
    payload = response.json()["diagnostics"]
    assert payload["dataset_hash"] == "3" * 64
    assert payload["frozen_label_horizon_trading_days"] == 10
    assert payload["is_frozen_label_horizon"] is True
    assert captured == {
        "model_id": "model-a",
        "model_version": 4,
        "horizon": 10,
        "quantiles": 8,
        "sample_interval": 10,
    }

    captured.clear()
    response = client.get(
        "/model-research/models/model-a/versions/4/quantile-diagnostics",
        params={"quantiles": 5, "horizon": 3},
    )

    assert response.status_code == 200
    payload = response.json()["diagnostics"]
    assert payload["frozen_label_horizon_trading_days"] == 10
    assert payload["is_frozen_label_horizon"] is False
    assert captured == {
        "model_id": "model-a",
        "model_version": 4,
        "horizon": 3,
        "quantiles": 5,
        "sample_interval": 1,
    }


def test_stability_diagnostics_endpoint_uses_frozen_label_horizon(monkeypatch) -> None:
    repository = _Repository()
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "dataset_hash": "6" * 64,
        "dataset_spec": {"label": {"horizon_trading_days": 10}},
        "manifest_json": {},
    }
    captured = {}

    def diagnostics(**kwargs):
        captured.update(kwargs)
        return {"status": "stable", "windows": [], "daily": []}

    monkeypatch.setattr(
        model_research.model_repository,
        "model_prediction_stability_diagnostics",
        diagnostics,
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get(
        "/model-research/models/model-a/versions/4/stability-diagnostics",
        params={"rolling_window": 15},
    )

    assert response.status_code == 200
    payload = response.json()["diagnostics"]
    assert payload["dataset_hash"] == "6" * 64
    assert payload["frozen_label_horizon_trading_days"] == 10
    assert captured == {
        "model_id": "model-a",
        "model_version": 4,
        "horizon": 10,
        "rolling_window": 15,
        "quantiles": 5,
    }


def test_exposure_diagnostics_endpoint_uses_frozen_label_horizon(monkeypatch) -> None:
    repository = _Repository()
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "dataset_hash": "4" * 64,
        "dataset_spec": {"label": {"horizon_trading_days": 10}},
        "manifest_json": {},
    }
    captured = {}

    def diagnostics(**kwargs):
        captured.update(kwargs)
        return {"status": "stable", "horizon_trading_days": 10}

    monkeypatch.setattr(
        model_research.model_repository,
        "model_prediction_exposure_diagnostics",
        diagnostics,
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get(
        "/model-research/models/model-a/versions/4/exposure-diagnostics",
        params={"score_quantiles": 5},
    )

    assert response.status_code == 200
    payload = response.json()["diagnostics"]
    assert payload["dataset_hash"] == "4" * 64
    assert payload["frozen_label_horizon_trading_days"] == 10
    assert captured == {
        "model_id": "model-a",
        "model_version": 4,
        "horizon": 10,
        "score_quantiles": 5,
    }


def test_prediction_distribution_diagnostics_endpoint(monkeypatch) -> None:
    repository = _Repository()
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "dataset_hash": "5" * 64,
        "dataset_spec": {},
        "manifest_json": {},
    }
    captured = {}

    def diagnostics(**kwargs):
        captured.update(kwargs)
        return {"status": "stable", "windows": []}

    monkeypatch.setattr(
        model_research.model_repository,
        "model_prediction_distribution_diagnostics",
        diagnostics,
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get(
        "/model-research/models/model-a/versions/4/prediction-distribution-diagnostics",
        params={"bins": 12},
    )

    assert response.status_code == 200
    payload = response.json()["diagnostics"]
    assert payload["dataset_hash"] == "5" * 64
    assert captured == {
        "model_id": "model-a",
        "model_version": 4,
        "bins": 12,
    }


def test_model_comparison_rechecks_common_oos_metrics_before_ranking(monkeypatch) -> None:
    repository = _Repository()

    def get_model(model_id, version):
        return {
            "model_id": model_id,
            "version": version,
            "name": "LightGBM" if model_id == "lgbm" else "XGBoost",
            "model_kind": model_id,
            "dataset_hash": "same-dataset",
            "dataset_spec": {
                "universe_id": "csi500",
                "label": {
                    "kind": "future_5d_cross_sectional_rank",
                    "horizon_trading_days": 5,
                    "range": [-1, 1],
                },
            },
            "metrics_json": {"rank_ic": 0.01, "ic_ir": 0.1, "test_days": 40},
            "manifest_json": {},
            "state": "candidate",
        }

    repository.get_model = get_model
    backtests = {
        ("lgbm", 1): ModelBacktestJobOut(
            backtest_job_id="bt-lgbm", model_id="lgbm", model_version=1,
            universe_id="csi500", benchmark_code="000905.SH", date_preset="custom",
            date_start=date(2026, 1, 2), date_end=date(2026, 3, 31),
            top_n=20, rebalance_every=5, status="success",
            excess_annual_return=0.10, sharpe_ratio=0.8, max_drawdown=-0.12,
        ),
        ("xgb", 1): ModelBacktestJobOut(
            backtest_job_id="bt-xgb", model_id="xgb", model_version=1,
            universe_id="csi500", benchmark_code="000905.SH", date_preset="custom",
            date_start=date(2026, 1, 2), date_end=date(2026, 3, 31),
            top_n=20, rebalance_every=5, status="success",
            excess_annual_return=0.08, sharpe_ratio=0.7, max_drawdown=-0.10,
        ),
    }
    monkeypatch.setattr(
        model_research.model_repository,
        "latest_model_backtests",
        lambda keys: {key: backtests[key] for key in keys},
    )
    monkeypatch.setattr(
        model_research.model_repository,
        "model_prediction_comparison",
        lambda **_kwargs: {
            "common_rows": 1000, "common_days": 40,
            "evaluation_rows": 1000, "evaluation_days": 40,
            "date_start": "2026-01-02", "date_end": "2026-03-31",
            "sources": [], "correlation_matrix": [[1.0, 0.4], [0.4, 1.0]],
            "metrics": [
                {"source_key": "lgbm::v1", "rank_ic": 0.05, "ic_ir": 0.5, "rmse": 0.7},
                {"source_key": "xgb::v1", "rank_ic": 0.03, "ic_ir": 0.4, "rmse": 0.8},
            ],
        },
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.post("/model-research/model-comparisons", json={"models": [
        {"model_id": "lgbm", "model_version": 1},
        {"model_id": "xgb", "model_version": 1},
    ]})

    assert response.status_code == 200
    comparison = response.json()["comparison"]
    assert comparison["compatibility"]["research_comparable"] is True
    assert comparison["compatibility"]["backtest_comparable"] is True
    assert comparison["models"][0]["comparison_metrics"]["rank_ic"] == 0.05
    rank_ic = next(item for item in comparison["leaders"] if item["metric"] == "rank_ic")
    assert rank_ic["winner"]["model_id"] == "lgbm"
    assert rank_ic["advantage"] == pytest.approx(0.02)


def test_create_ensemble_materializes_and_evaluates_real_predictions(monkeypatch) -> None:
    repository = _Repository()
    draft = {
        "job_id": "ensemble-job",
        "model_id": "ensemble-demo",
        "model_version": 1,
        "dataset_hash": "a" * 64,
        "dataset_spec": {"label": {"horizon_trading_days": 5}},
        "config_json": {"ensemble": {
            "fingerprint": "b" * 64,
            "sources": [
                {"model_id": "lgbm", "model_version": 1, "weight": 0.5},
                {"model_id": "xgb", "model_version": 1, "weight": 0.5},
            ],
        }},
    }
    repository.reserve_ensemble_model = lambda _payload: draft
    repository.fail_ensemble_model = lambda *_args: None
    repository.complete_ensemble_model = lambda _job_id, predictions, evaluation: {
        "model_id": "ensemble-demo", "version": 1, "model_kind": "ensemble",
        "state": "candidate", "metrics_json": evaluation,
        "manifest_json": draft["config_json"], "prediction_json": predictions,
        "dataset_spec": draft["dataset_spec"],
    }
    monkeypatch.setattr(
        model_research.model_repository, "ensemble_prediction_availability",
        lambda **_kwargs: {"trade_date": "2026-08-13", "row_count": 500},
    )
    monkeypatch.setattr(
        model_research.model_repository, "materialize_ensemble_predictions",
        lambda **_kwargs: {"row_count": 500, "date_end": "2026-08-13"},
    )
    monkeypatch.setattr(
        model_research.model_repository, "evaluate_model_predictions",
        lambda **_kwargs: {"test_days": 60, "rank_ic": 0.04, "ic_ir": 0.5},
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.post("/model-research/ensembles", json={
        "sources": [
            {"model_id": "lgbm", "model_version": 1},
            {"model_id": "xgb", "model_version": 1},
        ],
    })

    assert response.status_code == 201
    assert response.json()["model"]["model_kind"] == "ensemble"
    assert response.json()["model"]["metrics_json"]["rank_ic"] == 0.04


def _experiment_model(job_id: str = "job-trial-1") -> dict:
    return {
        "job_id": job_id,
        "model_id": "demo-grid",
        "version": 1 if job_id.endswith("1") else 2,
        "dataset_spec": {"factors": [{"factor_id": "mom_20"}]},
        "metrics_json": {
            "test_days": 80, "rank_ic": 0.08, "ic_ir": 0.8,
            "validation": {"days": 60, "rank_ic": 0.04, "ic_ir": 0.5},
        },
        "manifest_json": {},
        "state": "candidate",
        "job_config_json": {"experiment": {
            "experiment_id": "model_experiment_gate",
            "trial_index": 1 if job_id.endswith("1") else 2,
            "trial_count": 2,
            "search_params": {"num_leaves": 31},
        }},
    }


def _experiment_selection(selected_job_id: str, status: str = "selected") -> dict:
    return {
        "experiment_id": "model_experiment_gate",
        "selection": {
            "policy": "alphablocks.parameter-selection.v1",
            "status": status,
            "selected_job_id": selected_job_id,
            "selected_model_id": "demo-grid" if selected_job_id else "",
            "selected_model_version": 1 if selected_job_id else 0,
            "trial_assessments": [{
                "job_id": "job-trial-1", "passed": bool(selected_job_id),
                "failed_checks": [] if selected_job_id else ["rank_ic"],
            }],
        },
    }


def test_parameter_experiment_blocks_non_finalist_backtest(monkeypatch) -> None:
    repository = _Repository()
    repository.get_model = lambda _model_id, _version: _experiment_model("job-trial-2")
    repository.get_training_experiment = lambda _experiment_id: _experiment_selection(
        "job-trial-1",
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.post(
        "/model-research/models/demo-grid/versions/2/backtests", json={},
    )

    assert response.status_code == 409
    assert "仅验证集排名最高" in response.json()["detail"]


def test_parameter_experiment_allows_selected_qualified_backtest(monkeypatch) -> None:
    repository = _Repository()
    repository.get_model = lambda _model_id, _version: _experiment_model("job-trial-1")
    repository.get_training_experiment = lambda _experiment_id: _experiment_selection(
        "job-trial-1",
    )
    client = _client(monkeypatch, repository, _Scheduler())
    created = ModelBacktestJobOut(
        backtest_job_id="model_backtest_selected",
        model_id="demo-grid",
        model_version=1,
        universe_id="csi500",
        benchmark_code="000905.SH",
        date_preset="3y",
        status="pending",
    )
    monkeypatch.setattr(
        model_research.model_repository, "create_model_backtest_job", lambda _payload: created,
    )
    monkeypatch.setattr(model_research, "run_model_backtest_job", lambda _job_id: None)

    response = client.post(
        "/model-research/models/demo-grid/versions/1/backtests", json={},
    )

    assert response.status_code == 201
    assert response.json()["backtest"]["backtest_job_id"] == "model_backtest_selected"


def test_formal_top20_rejects_strong_test_when_validation_failed(monkeypatch) -> None:
    repository = _Repository()
    source = repository.get_model("demo", 1)
    source["metrics_json"] = {
        "test_days": 80,
        "rank_ic": 0.09,
        "ic_ir": 0.8,
        "validation": {"days": 60, "rank_ic": -0.01, "ic_ir": -0.1},
    }
    repository.get_model = lambda _model_id, _version: source
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.post(
        "/model-research/models/demo/versions/1/backtests", json={},
    )

    assert response.status_code == 409
    assert "验证RankIC" in response.json()["detail"]
    assert "验证ICIR" in response.json()["detail"]


def test_formal_top20_rejects_enabled_walk_forward_when_stability_failed(
    monkeypatch,
) -> None:
    repository = _Repository()
    source = repository.get_model("demo", 1)
    source["manifest_json"] = {
        "future_function_guards": ["PIT"],
        "walk_forward": {
            "enabled": True,
            "window_count": 3,
            "aggregate": {
                "window_ic_mean": 0.05,
                "window_ic_std": 0.01,
                "positive_ic_window_ratio": 1.0,
                "ic_ir": 0.28,
            },
            "stability": {
                "status": "mixed",
                "passed": False,
                "failed_checks": ["ic_ir"],
            },
        },
    }
    repository.get_model = lambda _model_id, _version: source
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.post(
        "/model-research/models/demo/versions/1/backtests", json={},
    )

    assert response.status_code == 409
    assert "Walk-Forward跨期稳定性" in response.json()["detail"]


def test_formal_top20_allows_enabled_walk_forward_when_stable(monkeypatch) -> None:
    repository = _Repository()
    source = repository.get_model("demo", 1)
    source["manifest_json"] = {
        "future_function_guards": ["PIT"],
        "walk_forward": {
            "enabled": True,
            "window_count": 3,
            "stability": {"status": "stable", "passed": True},
        },
    }
    repository.get_model = lambda _model_id, _version: source
    client = _client(monkeypatch, repository, _Scheduler())
    created = ModelBacktestJobOut(
        backtest_job_id="model_backtest_wfa_stable",
        model_id="demo",
        model_version=1,
        universe_id="csi500",
        benchmark_code="000905.SH",
        date_preset="3y",
        status="pending",
    )
    monkeypatch.setattr(
        model_research.model_repository,
        "create_model_backtest_job",
        lambda _payload: created,
    )
    monkeypatch.setattr(model_research, "run_model_backtest_job", lambda _job_id: None)

    response = client.post(
        "/model-research/models/demo/versions/1/backtests", json={},
    )

    assert response.status_code == 201
    assert response.json()["backtest"]["backtest_job_id"] == "model_backtest_wfa_stable"


def test_portfolio_sensitivity_creates_research_only_topn_jobs(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    captured = []

    def create(payload):
        captured.append(payload)
        return ModelBacktestJobOut(
            backtest_job_id=f"sensitivity-{payload.top_n}",
            model_id=payload.model_id,
            model_version=payload.model_version,
            universe_id=payload.universe_id,
            benchmark_code="000905.SH",
            date_preset=payload.date_preset,
            top_n=payload.top_n,
            rebalance_every=payload.rebalance_every,
            configuration={"research_only": payload.research_only},
            status="pending",
        )

    monkeypatch.setattr(
        model_research.model_repository, "create_model_backtest_job", create,
    )
    monkeypatch.setattr(model_research, "run_model_backtest_job", lambda _job_id: None)

    response = client.post(
        "/model-research/models/demo/versions/1/portfolio-sensitivity",
        json={"top_ns": [100, 20, 50], "rebalance_every": 5},
    )

    assert response.status_code == 201
    assert [item["top_n"] for item in response.json()["jobs"]] == [20, 50, 100]
    assert all(item.research_only is True for item in captured)
    assert all(item.rebalance_every == 5 for item in captured)


def test_research_only_backtest_never_updates_validation_state(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    result = _negative_backtest().model_copy(update={
        "backtest_job_id": "sensitivity-20",
        "configuration": {"research_only": True},
    })
    monkeypatch.setattr(
        model_research.model_repository,
        "get_model_backtest_job",
        lambda _job_id: result,
    )

    response = client.get("/model-research/model-backtests/sensitivity-20")

    assert response.status_code == 200
    assert response.json()["backtest"]["validation"] is None
    assert response.json()["backtest"]["backtest_kind"] == "portfolio_sensitivity"
    assert repository.validation_payload is None


def test_succeeded_job_dispatch_is_idempotent(monkeypatch) -> None:
    scheduler = _Scheduler()
    client = _client(monkeypatch, _Repository(), scheduler)

    response = client.post("/model-research/jobs/job-done/dispatch", json={})

    assert response.status_code == 200
    assert response.json()["service"]["accepted"] is False
    assert scheduler.submitted == []


def test_default_inference_date_is_rechecked_as_an_explicit_date(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    calls = []

    def availability(_model, *, trade_date="", data_cutoff=""):
        calls.append((trade_date, data_cutoff))
        return {
            "trade_date": "2026-08-13",
            "requested_trade_date_available": True if trade_date else None,
        }

    monkeypatch.setattr(model_research, "_inference_availability", availability)
    response = client.post(
        "/model-research/models/demo/versions/1/inferences",
        json={"data_cutoff": "2026-08-13T16:00:00+08:00"},
    )

    assert response.status_code == 200
    assert calls == [
        ("", "2026-08-13T16:00:00+08:00"),
        ("2026-08-13", "2026-08-13T16:00:00+08:00"),
    ]
    assert repository.inference_payload["trade_date"] == "2026-08-13"


def test_inference_precheck_reports_production_readiness(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    monkeypatch.setattr(
        model_research,
        "_inference_availability",
        lambda _model, **_kwargs: {
            "trade_date": "2026-08-14",
            "market_latest_date": "2026-08-14",
            "factor_latest_date": "2026-08-14",
            "factor_count": 1,
            "requested_trade_date_available": True,
        },
    )
    monkeypatch.setattr(
        model_research.model_repository, "latest_model_backtests", lambda _keys: {},
    )
    monkeypatch.setattr(
        model_research, "_model_validation_view",
        lambda _model, _backtest: {"approved": True, "passed": True},
    )

    response = client.get(
        "/model-research/models/model-1/versions/1/inference-precheck",
        params={
            "trade_date": "2026-08-14",
            "data_cutoff": "2026-08-14T16:00:00+08:00",
        },
    )

    assert response.status_code == 200
    precheck = response.json()["precheck"]
    assert precheck["status"] == "ready"
    assert precheck["passed"] is True
    assert precheck["can_submit"] is True
    assert {item["key"] for item in precheck["items"]} == {
        "model_active", "validation", "artifact", "frozen_inputs",
        "trade_date", "signal_close", "schedule",
    }


def test_inference_history_is_compact_and_filterable(monkeypatch) -> None:
    client = _client(monkeypatch, _Repository(), _Scheduler())

    response = client.get(
        "/model-research/inference-runs",
        params={
            "model_id": "model-1", "model_version": 1,
            "status": "succeeded", "limit": 20,
        },
    )

    assert response.status_code == 200
    assert response.json()["status_counts"] == {"succeeded": 1}
    assert response.json()["runs"][0] == {
        "job_id": "infer-history-1",
        "model_id": "model-1",
        "model_version": 1,
        "model_kind": "lightgbm",
        "model_name": "测试模型",
        "status": "succeeded",
        "trade_date": "2026-08-13",
        "trigger": "manual",
        "prediction_rows": 500,
        "requested_at": "2026-08-13T16:10:00+08:00",
    }


def test_inference_schedule_browser_payload_excludes_internal_manifests(monkeypatch) -> None:
    client = _client(monkeypatch, _Repository(), _Scheduler())

    response = client.get("/model-research/inference-schedules")

    assert response.status_code == 200
    schedule = response.json()["schedules"][0]
    assert schedule["model_id"] == "model-1"
    assert "manifest_json" not in schedule
    assert "dataset_spec" not in schedule


def test_candidate_model_cannot_enable_production_inference(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    monkeypatch.setattr(
        model_research.model_repository, "latest_model_backtests", lambda _keys: {},
    )
    monkeypatch.setattr(
        model_research, "_model_validation_view",
        lambda _model, _backtest: {"approved": False, "passed": False},
    )

    response = client.put(
        "/model-research/models/model-1/versions/1/inference-schedule",
        json={"enabled": True, "run_after_local": "16:30"},
    )

    assert response.status_code == 409
    assert "只有通过研究门槛" in response.json()["detail"]


def test_ensemble_diagnostics_endpoint_uses_frozen_sources(monkeypatch) -> None:
    repository = _Repository()
    frozen_sources = [
        {"model_id": "lgbm", "model_version": 1, "weight": 0.4},
        {"model_id": "xgb", "model_version": 2, "weight": 0.6},
    ]
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "model_kind": "ensemble",
        "dataset_spec": {"label": {"horizon_trading_days": 10}},
        "manifest_json": {"ensemble": {"sources": frozen_sources}},
    }
    captured = {}

    def diagnostics(*, sources, horizon):
        captured.update({"sources": sources, "horizon": horizon})
        return {"test_days": 20, "average_pairwise_correlation": 0.35}

    monkeypatch.setattr(
        model_research.model_repository, "ensemble_model_diagnostics", diagnostics,
    )
    client = _client(monkeypatch, repository, _Scheduler())

    response = client.get(
        "/model-research/models/ensemble-demo/versions/3/ensemble-diagnostics",
    )

    assert response.status_code == 200
    assert response.json()["diagnostics"]["test_days"] == 20
    assert captured == {"sources": frozen_sources, "horizon": 10}


def _negative_backtest() -> ModelBacktestJobOut:
    return ModelBacktestJobOut(
        backtest_job_id="model_backtest_negative",
        model_id="demo",
        model_version=1,
        universe_id="csi500",
        benchmark_code="000905.SH",
        date_preset="3y",
        status="success",
        annual_return=-0.02,
        excess_annual_return=-0.01,
        sharpe_ratio=0.1,
        turnover_rate=0.08,
        max_drawdown=-0.12,
        trading_days=80,
    )


def test_successful_backtest_does_not_auto_validate_negative_excess(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    monkeypatch.setattr(
        model_research.model_repository,
        "get_model_backtest_job",
        lambda _job_id: _negative_backtest(),
    )

    response = client.get(
        "/model-research/model-backtests/model_backtest_negative"
    )

    assert response.status_code == 200
    assert response.json()["backtest"]["validation"]["passed"] is False
    assert repository.validation_payload["approved_by_repository"] is False
    assert repository.validation_payload["policy"] == "alphablocks.research-gate.v2"


def test_model_response_uses_effective_gate_state() -> None:
    model = _Repository().get_model("demo", 1)

    response = model_research._model_response(model, _negative_backtest())

    assert response["registry_state"] == "validated"
    assert response["state"] == "candidate"
    assert response["validation"]["failed_checks"] == ["excess_annual_return"]


def test_failed_gate_requires_reasoned_manual_override(monkeypatch) -> None:
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())
    monkeypatch.setattr(
        model_research.model_repository,
        "get_model_backtest_job",
        lambda _job_id: _negative_backtest(),
    )
    path = "/model-research/models/demo/versions/1/backtests/model_backtest_negative/validate"

    rejected = client.post(path, json={})
    missing_reason = client.post(path, json={"override": True})
    approved = client.post(
        path,
        json={"override": True, "reason": "仅用于模拟盘观察"},
    )

    assert rejected.status_code == 409
    assert missing_reason.status_code == 400
    assert approved.status_code == 200
    assert repository.validation_payload["manual_override"] is True
    assert repository.validation_payload["override_reason"] == "仅用于模拟盘观察"


def test_exact_replay_exposes_reproducibility_audit(monkeypatch) -> None:
    class _ReplayRepository(_Repository):
        def get_model(self, model_id, version):
            model = super().get_model(model_id, version)
            model.update({
                "model_id": model_id,
                "version": version,
                "job_id": "source_job" if model_id == "source-model" else "replay_job",
                "dataset_hash": "frozen-dataset",
                "metrics_json": {"validation": {"rank_ic": 0.04, "days": 80}},
            })
            if model_id == "replay-model":
                model["job_config_json"] = {
                    "research_origin": {
                        "mode": "exact_replay",
                        "source_job_id": "source_job",
                        "source_model_id": "source-model",
                        "source_model_version": 1,
                        "source_dataset_hash": "frozen-dataset",
                        "source_config_hash": "source-config-hash",
                    },
                }
            return model

    monkeypatch.setattr(
        model_research.model_repository,
        "model_prediction_reproducibility_audit",
        lambda **_kwargs: {
            "status": "exact", "passed": True, "key_set_equal": True,
            "common_rows": 40000, "common_days": 80,
        },
    )
    client = _client(monkeypatch, _ReplayRepository(), _Scheduler())

    response = client.get(
        "/model-research/models/replay-model/versions/1/reproducibility-audit"
    )

    assert response.status_code == 200
    assert response.json()["audit"]["status"] == "exact"
    assert response.json()["audit"]["source"]["model_id"] == "source-model"


def test_return_calibration_uses_the_model_frozen_horizon(monkeypatch) -> None:
    captured = {}
    repository = _Repository()
    client = _client(monkeypatch, repository, _Scheduler())

    def _calibration(**kwargs):
        captured.update(kwargs)
        return {
            "status": "insufficient_history",
            "available": False,
            "message": "有效样本不足",
        }

    monkeypatch.setattr(
        model_research.model_repository,
        "model_stock_return_calibration",
        _calibration,
    )

    response = client.get(
        "/model-research/models/model-1/versions/1/return-calibration",
        params={"entity_code": "600036.SH", "trade_date": "2025-12-31"},
    )

    assert response.status_code == 200
    assert response.json()["forecast"]["available"] is False
    assert captured["horizon"] == 5
    assert captured["entity_code"] == "600036.SH"
