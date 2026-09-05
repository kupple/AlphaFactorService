from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from types import SimpleNamespace

import pytest

import factor_service.model_research_repository as repository_module
from factor_service.model_research_repository import (
    ModelResearchRepository,
    ModelResearchConflict,
    ModelResearchError,
    _architecture_readiness,
    _architecture_spec,
    _attempt_identity,
    _attempt_row,
    _canonical_json,
    _dataset_spec,
    _ensemble_spec,
    _experiment_summary,
    _factor_ablation_trials,
    _grid_search_trials,
    _horizon_search_values,
    _historical_dataset_spec,
    _incremental_training_assessment,
    _job_row,
    _model_spec,
    _model_payload_references,
    _research_origin_spec,
    _optuna_spec,
    _registration_payloads,
    _training_dataset_source,
    _walk_forward_spec,
)
from factor_service.research.industry_feature import normalize_industry_feature
from factor_service.research.size_rotation_feature import (
    normalize_size_rotation_feature,
)


def _source() -> dict:
    return {
        "name": "smoke",
        "date_start": "2020-01-01",
        "date_end": "2024-01-01",
        "data_cutoff": "2024-01-01T15:00:00+08:00",
        "factors": [{
            "factor_id": "mom_20",
            "factor_version": 2,
            "params_hash": "a" * 64,
            "params": {"window": 20},
        }],
    }


@pytest.mark.parametrize("has_heartbeat", [False, True])
def test_resumed_distributed_job_is_worker_json_ready(monkeypatch, tmp_path, has_heartbeat):
    from factor_service.research.state import JobStateStore
    from factor_service.training_subtask_repository import TrainingSubtaskRepository

    stamp = datetime(2026, 9, 4, 9, 30, tzinfo=timezone.utc)
    row = {
        "job_id": "job-resume", "kind": "train", "status": "queued",
        "config_json": {"execution": {"mode": "distributed"}},
        "created_at": stamp,
    }
    task = {
        "task_key": "trial_00000", "kind": "optuna_trial",
        "node_id": "lan-cpu-05", "state": "failed", "attempt_count": 1,
        "error": "connection lost", "heartbeat_at": stamp if has_heartbeat else None,
        "created_at": stamp, "updated_at": stamp, "owner_token": "private-owner",
    }

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _statement, params):
            assert params == ("job-resume",)
            return SimpleNamespace(fetchone=lambda: row)

    repository = ModelResearchRepository.__new__(ModelResearchRepository)
    repository.database = SimpleNamespace(connection=Connection)
    monkeypatch.setattr(repository_module, "_require_attempt_audit_schema", lambda _conn: None)
    monkeypatch.setattr(TrainingSubtaskRepository, "list", lambda _self, _job_id: [task])

    job = repository.get_job("job-resume")
    store = JobStateStore(tmp_path)
    store.save(job, "accepted")
    assert store.load()["job"] == job
    subtask = job["subtasks"][0]
    assert subtask["created_at"] == stamp.isoformat()
    assert subtask["updated_at"] == stamp.isoformat()
    assert subtask["heartbeat_at"] == (stamp.isoformat() if has_heartbeat else None)
    assert "owner_token" not in subtask
    assert task["created_at"] is stamp


def test_create_inference_job_rejects_date_outside_rolling_revision() -> None:
    repository = ModelResearchRepository.__new__(ModelResearchRepository)
    repository.get_model = lambda _model_id, _version: {
        "manifest_json": {
            "walk_forward": {
                "enabled": True,
                "prediction_date_start": "2024-01-02",
                "prediction_date_end": "2024-01-31",
                "windows": [{
                    "effective_date_start": "2024-01-02",
                    "effective_date_end": "2024-01-31",
                }],
            },
        },
    }

    with pytest.raises(ModelResearchConflict, match="必须先训练并发布"):
        repository.create_inference_job(
            "model-1", 1,
            {
                "trade_date": "2024-02-01",
                "data_cutoff": "2024-02-01T16:00:00+08:00",
            },
        )


def test_attach_artifact_object_storage_updates_completed_attempt(monkeypatch) -> None:
    artifact = {
        "artifact_id": "artifact-1",
        "job_id": "job-1",
        "artifact_kind": "bundle",
        "file_name": "model.tar.gz",
        "relative_path": "job-1/bundle/model.tar.gz",
        "sha256": "a" * 64,
        "size_bytes": 10,
        "dataset_hash": "b" * 64,
        "object_store_uri": "",
        "object_store_version_id": "",
        "object_store_sha256": "",
    }

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def transaction(self):
            return self

        def execute(self, statement, params):
            sql = " ".join(str(statement).split())
            if sql.startswith("SELECT * FROM model_artifacts"):
                return SimpleNamespace(fetchone=lambda: dict(artifact))
            if sql.startswith("UPDATE model_artifacts"):
                return SimpleNamespace(fetchone=lambda: {
                    **artifact,
                    "object_store_uri": params[0],
                    "object_store_version_id": params[1],
                    "object_store_sha256": params[2],
                })
            if sql.startswith("SELECT attempt_count FROM model_jobs"):
                return SimpleNamespace(fetchone=lambda: {"attempt_count": 1})
            raise AssertionError(sql)

    database = SimpleNamespace(connection=lambda: _Connection())
    audited = []
    monkeypatch.setattr(
        repository_module,
        "_update_attempt_audit_row",
        lambda *_args, **kwargs: audited.append(kwargs) or True,
    )

    result = ModelResearchRepository(database).attach_artifact_object_storage(
        "artifact-1",
        object_store_uri="s3://alphablocks-models/models/model.tar.gz",
        object_store_version_id="version-1",
        object_store_sha256="a" * 64,
    )

    assert result["object_store_version_id"] == "version-1"
    assert audited[0]["ordinal"] == 1
    assert audited[0]["artifact"]["object_store_sha256"] == "a" * 64


def test_dataset_spec_preserves_versioned_capability_identities() -> None:
    spec = _dataset_spec({
        **_source(),
        "research_target": "stock_selection",
        "target_ref": {"id": "stock_selection", "version": 1},
        "transform_refs": [{
            "id": "sw2021_industry_one_hot",
            "version": 2,
            "implementation_hash": "factor-service:industry-one-hot.v2",
            "fit_scope": "pit_membership_dictionary",
        }],
        "universe_rule_refs": [{
            "id": "exclude_st", "version": 1, "params": {},
        }],
    })

    assert spec["target_ref"] == {"id": "stock_selection", "version": 1}
    assert spec["transform_refs"][0]["version"] == 2
    assert spec["universe_rule_refs"] == [{
        "id": "exclude_st", "version": 1, "params": {},
    }]


def test_dataset_spec_rejects_target_identity_mismatch() -> None:
    with pytest.raises(ModelResearchError, match="target_ref.id"):
        _dataset_spec({
            **_source(),
            "research_target": "stock_selection",
            "target_ref": {"id": "industry_rotation", "version": 1},
        })


def test_model_spec_preserves_v1_identity_and_rejects_unknown_version() -> None:
    assert _model_spec({
        "kind": "lightgbm", "version": 1, "params": {},
    })["version"] == 1
    with pytest.raises(ModelResearchError, match="model.version=1"):
        _model_spec({"kind": "lightgbm", "version": 2, "params": {}})


def _trained_model(
    model_id: str, version: int, *, validation_icir: float,
    validation_rank_ic: float = 0.03, test_icir: float = 99.0,
    horizon: int = 5, model_kind: str | None = None,
) -> dict:
    return {
        "model_id": model_id,
        "version": version,
        "name": model_id,
        "model_kind": model_kind or ("lightgbm" if version == 1 else "xgboost"),
        "dataset_hash": (model_id[-1:] or "a") * 64,
        "dataset_spec": {
            **_dataset_spec({
                **_source(), "label_horizon_trading_days": horizon,
            }),
            "date_start": "2020-01-01",
            "date_end": "2024-01-01",
        },
        "metrics_json": {
            "ic_ir": test_icir,
            "validation": {
                "days": 80,
                "rank_ic": validation_rank_ic,
                "ic_ir": validation_icir,
            },
        },
        "job_config_json": {"model": {"params": {"seed": 42}}},
    }


def test_model_reference_detection_covers_ensemble_and_incremental_lineage() -> None:
    ensemble = {
        "manifest_json": {
            "ensemble": {
                "sources": [{"model_id": "source-model", "model_version": 2}],
            },
        },
        "config_json": {},
    }
    incremental = {
        "manifest_json": {},
        "config_json": {
            "incremental_training": {
                "source_model_id": "source-model",
                "source_model_version": 2,
            },
        },
    }

    assert _model_payload_references(
        ensemble, model_id="source-model", model_version=2,
    ) is True
    assert _model_payload_references(
        incremental, model_id="source-model", model_version=2,
    ) is True
    assert _model_payload_references(
        ensemble, model_id="source-model", model_version=3,
    ) is False


def test_job_row_only_marks_terminal_infrastructure_failure_retryable() -> None:
    base = {
        "job_id": "job-failed",
        "kind": "train",
        "status": "failed",
        "result_json": {"failure": {"retryable": True}},
    }

    assert _job_row(base)["retryable"] is True
    assert _job_row({
        **base, "result_json": {"failure": {"retryable": False}},
    })["retryable"] is False
    assert _job_row({**base, "status": "queued"})["retryable"] is False


@pytest.mark.parametrize("failure", [
    {"retryable": False, "error_code": "node_memory_budget_exceeded"},
    {"retryable": False, "code": "node_out_of_memory"},
    {"retryable": False, "message": "[node_memory_budget_exceeded] insufficient memory"},
])
def test_memory_failure_allows_explicit_retry_only_for_terminal_training(failure):
    base = {"job_id": "job-memory", "kind": "train", "status": "failed",
            "result_json": {"failure": failure}}
    assert _job_row(base)["retryable"] is True
    assert failure["retryable"] is False  # Audit/automatic retry policy is unchanged.
    assert _job_row({**base, "status": "running"})["retryable"] is False
    assert _job_row({**base, "kind": "inference"})["retryable"] is False


@pytest.mark.parametrize("failure", [None, {},
    {"retryable": False, "code": "node_resource_unavailable"},
    {"retryable": False, "message": "invalid data mentioning node_out_of_memory"},
])
def test_unrelated_permanent_failure_is_not_made_retryable(failure):
    assert _job_row({"job_id": "job-invalid", "kind": "train", "status": "failed",
                     "result_json": {"failure": failure}})["retryable"] is False


def test_legacy_truncated_failure_uses_longer_error_without_changing_audit():
    from copy import deepcopy
    from factor_service.model_research_repository import _failure_allows_manual_retry
    prefix = '[unexpected_error] RuntimeError: 隔离模型进程失败(returncode=1):\n'
    message = (prefix + 'INFO initialized\n' * 70
               + 'An exception has been raised[PermanentJobError: Received disconnect from 10.0.0.5 '
               'port 22:2: Too many authentication failures].\nTraceback')
    raw = {'job_id': 'old-job', 'kind': 'train', 'status': 'failed', 'error_message': message,
           'result_json': {'failure': {'retryable': False, 'message': message[:1000]}}}
    original = deepcopy(raw)
    result = _job_row(raw)
    assert result['retryable'] is True
    assert result['error_message'].startswith('[node_ssh_authentication_failed]')
    assert 'INFO' not in result['error_message']
    assert raw == original and result['result_json'] == original['result_json']
    assert _failure_allows_manual_retry(raw['result_json']['failure'], error_message=message)
    assert not _failure_allows_manual_retry(raw['result_json']['failure'], error_message=prefix + 'ValueError: bad dataset')


class _Cursor:
    def __init__(self, row=None, rows=None) -> None:
        self.row = row
        self.rows = rows

    def fetchone(self):
        return self.row

    def fetchall(self):
        if self.rows is not None:
            return self.rows
        return [] if self.row is None else [self.row]


class _RecordingConnection:
    def __init__(self, job_row: dict, *, existing_version=None) -> None:
        self.job_row = job_row
        self.existing_version = existing_version
        self.queries: list[str] = []
        self.executions: list[tuple[str, tuple]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def transaction(self):
        return self

    def execute(self, query, _params=()):
        normalized = " ".join(str(query).split())
        self.queries.append(normalized)
        self.executions.append((normalized, tuple(_params)))
        if normalized.startswith("SELECT to_regclass"):
            return _Cursor({"relation": "model_job_attempts"})
        if normalized.startswith("INSERT INTO model_job_events"):
            return _Cursor({"event_id": len(self.executions)})
        if normalized.startswith("SELECT attempt_count FROM model_jobs"):
            return _Cursor({"attempt_count": int(self.job_row.get("attempt_count") or 0)})
        if normalized.startswith("UPDATE model_job_attempts"):
            return _Cursor({"attempt_id": "model-attempt-recorded"})
        if "WHERE status IN ('leased', 'running', 'uploading')" in normalized:
            return _Cursor(rows=[])
        if normalized.startswith("SELECT * FROM model_jobs"):
            return _Cursor(self.job_row)
        if normalized.startswith("SELECT job_id FROM model_versions"):
            return _Cursor(self.existing_version)
        if normalized.startswith("SELECT COALESCE(max(version), 0) + 1"):
            return _Cursor({"version": 7})
        if normalized.startswith("SELECT GREATEST"):
            return _Cursor({"version": 7})
        return _Cursor()


class _RecordingDatabase:
    def __init__(self, connection: _RecordingConnection) -> None:
        self.recording_connection = connection

    def connection(self):
        return self.recording_connection


def test_active_dataset_hashes_include_queued_and_executing_jobs() -> None:
    class _ActiveDatasetConnection(_RecordingConnection):
        def execute(self, query, _params=()):
            normalized = " ".join(str(query).split())
            self.queries.append(normalized)
            self.executions.append((normalized, tuple(_params)))
            if normalized.startswith("SELECT DISTINCT specs.spec_hash"):
                return _Cursor(rows=[
                    {"dataset_hash": "a" * 64},
                    {"dataset_hash": "b" * 64},
                ])
            return super().execute(query, _params)

    connection = _ActiveDatasetConnection({})
    repository = ModelResearchRepository(_RecordingDatabase(connection))

    hashes = repository.active_dataset_hashes()

    assert hashes == {"a" * 64, "b" * 64}
    assert "'queued', 'leased', 'running', 'uploading'" in connection.queries[0]


def test_dataset_contract_locks_versions_and_lookahead_guards() -> None:
    spec = _dataset_spec(_source())

    assert spec["universe_id"] == "csi500"
    assert spec["factors"][0]["factor_version"] == 2
    assert spec["split"]["embargo_days"] == 5
    assert spec["availability"]["event_available_at_lte_signal_close"] is True
    assert spec["availability"]["source_available_at_lte_data_cutoff"] is True
    assert spec["pipeline_version"] == "alphablocks.dataset-pipeline.v9"
    assert spec["data_bindings"]["settings_revision"] == 0
    assert spec["sample_filters"] == {
        "minimum_listing_trading_days": 60,
        "exclude_st": True,
        "exclude_delisting": True,
        "custom_formulas": [],
    }
    assert spec["preprocessing"] == {
        "schema_version": "alphablocks.cross-sectional-feature-preprocessing.v1",
        "enabled": True,
        "missing": {
            "method": "cross_sectional_median",
            "all_missing_value": 0.0,
        },
        "winsorize": {
            "method": "quantile",
            "lower": 0.01,
            "upper": 0.99,
            "minimum_observations": 10,
        },
        "standardize": {
            "method": "zscore",
            "ddof": 0,
            "constant_value": 0.0,
        },
    }
    assert spec["industry_feature"] == normalize_industry_feature(
        {"enabled": False}, default_enabled=True,
    )
    assert spec["materialization"] == {
        "mode": "on_demand", "format": "parquet", "persist_factor_values": False,
    }


def test_dataset_contract_preserves_legacy_unfiltered_replay() -> None:
    spec = _dataset_spec({
        **_source(),
        "pipeline_version": "alphablocks.dataset-pipeline.v2",
    })

    assert spec["sample_filters"] == {
        "minimum_listing_trading_days": 0,
        "exclude_st": False,
        "exclude_delisting": False,
        "custom_formulas": [],
    }
    assert spec["preprocessing"]["enabled"] is False
    assert spec["industry_feature"]["enabled"] is False


def test_dataset_contract_freezes_preprocessing_and_changes_identity() -> None:
    enabled = _dataset_spec({
        **_source(),
        "preprocessing": {"enabled": True},
    })
    disabled = _dataset_spec({
        **_source(),
        "preprocessing": {"enabled": False},
    })

    assert enabled["preprocessing"]["enabled"] is True
    assert disabled["preprocessing"]["enabled"] is False
    assert enabled != disabled

    with pytest.raises(ModelResearchError, match="enabled必须是布尔值"):
        _dataset_spec({
            **_source(),
            "preprocessing": {"enabled": 1},
        })


def test_legacy_context_preprocessing_switch_is_moved_into_dataset() -> None:
    source = _training_dataset_source({
        "dataset": _source(),
        "context": {"preprocessing_enabled": False},
    })

    assert source["preprocessing"] == {"enabled": False}
    assert "preprocessing" not in _source()

    explicit = _training_dataset_source({
        "dataset": {**_source(), "preprocessing": {"enabled": True}},
        "context": {"preprocessing_enabled": False},
    })
    assert explicit["preprocessing"] == {"enabled": True}


def test_persisted_dataset_without_preprocessing_uses_legacy_disabled_mode() -> None:
    historical = _historical_dataset_spec(_source())

    assert historical["preprocessing"]["enabled"] is False
    assert historical["industry_feature"]["enabled"] is False


def test_dataset_contract_freezes_industry_one_hot_and_changes_hash() -> None:
    base = {**_source(), "date_start": "2022-01-04"}
    enabled = _dataset_spec({
        **base,
        "industry_feature": {"enabled": True},
    })
    disabled = _dataset_spec({
        **base,
        "industry_feature": {"enabled": False},
    })

    assert enabled["industry_feature"] == normalize_industry_feature(
        {"enabled": True}, default_enabled=False,
    )
    assert disabled["industry_feature"]["enabled"] is False
    assert sha256(_canonical_json(enabled).encode("utf-8")).hexdigest() != (
        sha256(_canonical_json(disabled).encode("utf-8")).hexdigest()
    )
    with pytest.raises(ModelResearchError, match="2021-12-13"):
        _dataset_spec({
            **_source(),
            "industry_feature": {"enabled": True},
        })
    with pytest.raises(ModelResearchError, match="仅支持个股选股"):
        _dataset_spec({
            **base,
            "research_target": "industry_rotation",
            "industry_feature": {"enabled": True},
        })


def test_dataset_contract_freezes_size_rotation_and_changes_hash() -> None:
    def pool(source_id: str, selector_value: str) -> dict:
        return {
            "schema_version": "alphablocks.configured-stock-pool-source.v1",
            "source_id": source_id,
            "source_kind": "configured_stock_pool",
            "label": source_id,
            "version": 10,
            "available": True,
            "pit": True,
            "settings_revision": 10,
            "binding_id": "index_membership",
            "binding_fingerprint": "a" * 64,
            "selector": {
                "field_role": "index_code",
                "operator": "eq",
                "value": selector_value,
            },
            "benchmark_code": selector_value,
            "config_fingerprint": "b" * 64,
        }

    source = {
        "enabled": True,
        "large_pool": pool("large", "large-selector"),
        "small_pool": pool("small", "small-selector"),
        "return_window": 10,
        "basket_size": 20,
        "regime_window": 60,
    }
    enabled = _dataset_spec({**_source(), "size_rotation_feature": source})
    disabled = _dataset_spec({
        **_source(), "size_rotation_feature": {"enabled": False},
    })

    assert enabled["size_rotation_feature"] == normalize_size_rotation_feature(
        source, default_enabled=False,
    )
    assert disabled["size_rotation_feature"]["enabled"] is False
    assert sha256(_canonical_json(enabled).encode("utf-8")).hexdigest() != (
        sha256(_canonical_json(disabled).encode("utf-8")).hexdigest()
    )
    with pytest.raises(ModelResearchError, match="仅支持个股选股"):
        _dataset_spec({
            **_source(),
            "research_target": "industry_rotation",
            "date_start": "2022-01-04",
            "size_rotation_feature": source,
        })


def test_legacy_context_industry_switch_is_moved_only_for_new_requests() -> None:
    source = _training_dataset_source({
        "dataset": {**_source(), "date_start": "2022-01-04"},
        "context": {"industry_as_feature": True},
    })

    assert source["industry_feature"] == {"enabled": True}
    assert "industry_feature" not in _source()

    explicit = _training_dataset_source({
        "dataset": {
            **_source(),
            "industry_feature": {"enabled": False},
        },
        "context": {"industry_as_feature": True},
    })
    assert explicit["industry_feature"] == {"enabled": False}

    # Historical jobs never executed the former UI-only context switch.
    historical = _historical_dataset_spec({
        **_source(),
        "pipeline_version": "alphablocks.dataset-pipeline.v6",
    })
    assert historical["industry_feature"]["enabled"] is False


def test_dataset_contract_validates_sample_filters() -> None:
    with pytest.raises(ModelResearchError, match="0至5000"):
        _dataset_spec({
            **_source(),
            "sample_filters": {
                "minimum_listing_trading_days": 5001,
                "exclude_st": True,
                "exclude_delisting": True,
            },
        })
    with pytest.raises(ModelResearchError, match="exclude_st必须是布尔值"):
        _dataset_spec({
            **_source(),
            "sample_filters": {
                "minimum_listing_trading_days": 60,
                "exclude_st": "yes",
                "exclude_delisting": True,
            },
        })


def test_dataset_contract_freezes_custom_sample_filter_formula() -> None:
    spec = _dataset_spec({
        **_source(),
        "sample_filters": {
            "minimum_listing_trading_days": 60,
            "exclude_st": True,
            "exclude_delisting": True,
            "custom_formulas": [{
                "name": "价格与流动性",
                "expression": "$close >= 5 && $amount >= 100000000",
            }],
        },
    })

    formula = spec["sample_filters"]["custom_formulas"][0]
    assert formula["formula_id"].startswith("sample_filter_")
    assert formula["name"] == "价格与流动性"
    assert formula["required_fields"] == ["amount", "close"]
    assert formula["max_window"] == 1


def test_dataset_contract_freezes_stock_entity_asset_formula_fields() -> None:
    spec = _dataset_spec({
        **_source(),
        "sample_filters": {
            "minimum_listing_trading_days": 60,
            "exclude_st": True,
            "exclude_delisting": True,
            "custom_formulas": [{
                "name": "实体资产质量过滤",
                "expression": "$quality_score >= 80 && $close > 5",
                "available_fields": [
                    {
                        "field": "quality_score",
                        "label": "质量分",
                        "data_type": "float64",
                        "entity_id": "stock",
                        "asset_id": "stock_quality_daily",
                        "asset_name": "股票质量日频",
                        "asset_updated_at": "2026-08-25T10:00:00Z",
                        "provider_node": "quality-provider",
                    },
                    {
                        "field": "close",
                        "label": "收盘价",
                        "data_type": "decimal",
                        "entity_id": "stock",
                        "asset_id": "stock_market_daily",
                        "asset_name": "股票日行情",
                        "asset_updated_at": "2026-08-24T10:00:00Z",
                        "provider_node": "market-provider",
                    },
                ],
            }],
        },
    })

    formula = spec["sample_filters"]["custom_formulas"][0]
    assert formula["required_fields"] == ["close", "quality_score"]
    assert [item["field"] for item in formula["field_bindings"]] == [
        "close", "quality_score",
    ]
    assert formula["field_bindings"][1]["asset_id"] == "stock_quality_daily"
    assert "available_fields" not in formula


def test_dataset_contract_rejects_unsafe_custom_sample_filter_field() -> None:
    with pytest.raises(ModelResearchError, match="不支持的字段"):
        _dataset_spec({
            **_source(),
            "sample_filters": {
                "minimum_listing_trading_days": 60,
                "exclude_st": True,
                "exclude_delisting": True,
                "custom_formulas": [{
                    "name": "非法字段",
                    "expression": "$future_return > 0",
                }],
            },
        })


def test_classification_target_freezes_binary_label_and_model_loss() -> None:
    dataset = _dataset_spec({**_source(), "target_mode": "classification"})
    model = _model_spec(
        {"kind": "lightgbm", "params": {"loss": "mse"}},
        target_mode=dataset["target_mode"],
    )

    assert dataset["target_mode"] == "classification"
    assert dataset["label"] == {
        "kind": "future_5d_direction",
        "mode": "classification",
        "horizon_trading_days": 5,
        "range": [0.0, 1.0],
        "classes": [0, 1],
        "positive_class": "future_return_gt_zero",
        "formula": "1[future_return(T,T+5) > 0]",
    }
    assert model["params"]["loss"] == "binary"
    assert model["params"]["objective"] == "binary"
    assert model["params"]["metric"] == "auc"


def test_model_spec_accepts_quantmind_objective_metrics() -> None:
    regression = _model_spec({
        "kind": "lightgbm",
        "params": {"objective": "regression", "metric": "mae"},
    }, target_mode="return")
    classification = _model_spec({
        "kind": "lightgbm",
        "params": {"objective": "binary", "metric": "binary_logloss"},
    }, target_mode="classification")

    assert regression["params"]["metric"] == "mae"
    assert classification["params"]["metric"] == "binary_logloss"

    with pytest.raises(ModelResearchError, match="Metric"):
        _model_spec({
            "kind": "lightgbm",
            "params": {"objective": "binary", "metric": "rmse"},
        }, target_mode="classification")


def test_completed_job_row_exposes_pending_and_registered_states() -> None:
    pending = _job_row({
        "job_id": "job-pending",
        "kind": "train",
        "status": "succeeded",
        "model_version": None,
        "config_json": {"planned_model_version": 3},
    })
    registered = _job_row({**pending, "model_version": 3})
    declined = _job_row({
        **pending,
        "result_json": {
            "metrics": {"ic": 0.03},
            "registration": {"status": "declined"},
        },
    })
    automatic = _job_row({
        **pending,
        "result_json": {
            "metrics": {"ic": 0.03},
            "registration": {"status": "automatic_pending"},
        },
    })

    assert pending["planned_model_version"] == 3
    assert pending["registration_status"] == "legacy_pending_confirmation"
    assert automatic["registration_status"] == "automatic_pending"
    assert registered["registration_status"] == "registered"
    assert declined["registration_status"] == "declined"


def test_training_completion_persists_automatic_registration_intent() -> None:
    row = {
        "job_id": "job-pending",
        "kind": "train",
        "status": "running",
        "lease_owner": "alpha-factor-service",
        "lease_token": "lease-token",
        "cancel_requested": False,
        "model_id": "model-pending",
        "dataset_id": "dataset-pending",
        "title": "待确认模型",
        "model_kind": "lightgbm",
        "config_json": {"planned_model_version": 3},
        "attempt_count": 1,
    }
    connection = _RecordingConnection(row)
    repository = ModelResearchRepository(_RecordingDatabase(connection))
    repository.get_job = lambda _job_id: {"kind": "train", "status": "running"}

    repository.complete_job(
        "job-pending", lease_token="lease-token",
        result={"metrics": {"ic": 0.03}},
    )

    sql = "\n".join(connection.queries)
    assert "INSERT INTO model_versions" not in sql
    assert "model_version = NULL" in sql
    assert "INSERT INTO model_job_events" in sql
    result_payload = next(
        value.obj
        for query, params in connection.executions
        if query.startswith("UPDATE model_jobs")
        for value in params
        if hasattr(value, "obj") and "metrics" in value.obj
    )
    assert result_payload["registration"]["status"] == "automatic_pending"


def _claimable_job(*, attempt_count: int = 0, status: str = "queued") -> dict:
    return {
        "job_id": "job-attempt-audit",
        "kind": "train",
        "status": status,
        "lease_owner": "alpha-factor-service" if status == "leased" else "",
        "lease_token": "lease-token" if status == "leased" else "",
        "lease_expires_at": datetime(2030, 1, 1, tzinfo=timezone.utc),
        "cancel_requested": False,
        "attempt_count": attempt_count,
        "max_attempts": max(3, attempt_count + 1),
        "model_id": "model-attempt-audit",
        "dataset_id": "dataset-attempt-audit",
        "config_json": {"execution": {"node_id": "gpu-node-a"}},
    }


def test_claim_persists_real_attempt_identity_ordinal_and_node() -> None:
    connection = _RecordingConnection(_claimable_job())
    repository = ModelResearchRepository(_RecordingDatabase(connection))
    repository.get_job = lambda _job_id: {"job_id": _job_id, "attempts": []}

    repository.claim_specific_job("job-attempt-audit")

    inserts = [
        params for query, params in connection.executions
        if query.startswith("INSERT INTO model_job_attempts")
    ]
    assert len(inserts) == 1
    params = inserts[0]
    assert params[:4] == (
        _attempt_identity("job-attempt-audit", 1),
        "job-attempt-audit",
        1,
        "gpu-node-a",
    )


def test_dispatch_failure_closes_attempt_without_deleting_or_reusing_ordinal() -> None:
    connection = _RecordingConnection(
        _claimable_job(attempt_count=1, status="leased"),
    )
    repository = ModelResearchRepository(_RecordingDatabase(connection))
    repository.get_job = lambda _job_id: {"job_id": _job_id, "status": "queued"}

    repository.release_dispatch_lease(
        "job-attempt-audit",
        lease_token="lease-token",
        error_message="worker endpoint unavailable",
    )

    sql = "\n".join(connection.queries)
    assert "DELETE FROM model_job_attempts" not in sql
    assert "attempt_count = GREATEST(attempt_count - 1, 0)" not in sql
    assert "max_attempts = GREATEST(max_attempts, attempt_count + 1)" in sql
    terminal_updates = [
        (query, params) for query, params in connection.executions
        if query.startswith("UPDATE model_job_attempts")
        and "finished_at = %s" in query
    ]
    assert len(terminal_updates) == 1
    _, params = terminal_updates[0]
    assert "failed" in params
    error = next(value.obj for value in params if hasattr(value, "obj"))
    assert error == {
        "code": "dispatch_failed",
        "message": "worker endpoint unavailable",
        "retryable": True,
    }


@pytest.mark.parametrize('failure', [
    {'retryable': True},
    {'retryable': False, 'message': '[node_memory_budget_exceeded] NodeMemoryBudgetExceeded: low memory'},
    {'retryable': False, 'message': '[node_out_of_memory] NodeOutOfMemory: allocation failed'},
    {'retryable': False, 'error_code': 'node_ssh_authentication_failed'},
    {'retryable': False, 'error_code': 'node_execution_unavailable'},
    {'retryable': False, 'message': '[unexpected_error] RuntimeError: 隔离模型进程失败(returncode=1):\n'
     'An exception has been raised[PermanentJobError: Too many authentication failures].\nTraceback'},
])
def test_manual_retry_preserves_prior_attempt_and_next_claim_appends_ordinal(failure) -> None:
    failed = _claimable_job(attempt_count=1, status="failed")
    failed.update({
        "lease_owner": "",
        "lease_token": "",
        "result_json": {"failure": failure},
    })
    retry_connection = _RecordingConnection(failed)
    repository = ModelResearchRepository(_RecordingDatabase(retry_connection))
    repository.get_job = lambda _job_id: {"job_id": _job_id, "status": "queued"}

    repository.retry_job("job-attempt-audit", idempotency_key="retry-1")

    retry_sql = "\n".join(retry_connection.queries)
    assert "INSERT INTO model_job_attempts" not in retry_sql
    assert "DELETE FROM model_job_attempts" not in retry_sql

    queued = _claimable_job(attempt_count=1, status="queued")
    claim_connection = _RecordingConnection(queued)
    repository = ModelResearchRepository(_RecordingDatabase(claim_connection))
    repository.get_job = lambda _job_id: {"job_id": _job_id, "attempts": []}

    repository.claim_specific_job("job-attempt-audit")

    insert_params = next(
        params for query, params in claim_connection.executions
        if query.startswith("INSERT INTO model_job_attempts")
    )
    assert insert_params[:3] == (
        _attempt_identity("job-attempt-audit", 2),
        "job-attempt-audit",
        2,
    )


def test_artifact_recovery_claim_creates_local_uploading_attempt() -> None:
    canceled = _claimable_job(attempt_count=2, status="canceled")
    canceled.update({
        "lease_owner": "",
        "lease_token": "",
        "cancel_requested": True,
        "result_json": {"failure": {"retryable": False}},
    })

    class _RecoveryConnection(_RecordingConnection):
        def execute(self, query, _params=()):
            normalized = " ".join(str(query).split())
            if normalized.startswith("SELECT status FROM model_job_attempts"):
                self.queries.append(normalized)
                self.executions.append((normalized, tuple(_params)))
                return _Cursor({"status": "canceled"})
            return super().execute(query, _params)

    connection = _RecoveryConnection(canceled)
    repository = ModelResearchRepository(_RecordingDatabase(connection))
    repository.get_job = lambda _job_id: {
        "job_id": _job_id,
        "status": "uploading",
        "attempt_count": 3,
    }

    claimed = repository.claim_artifact_recovery(
        "job-attempt-audit",
        source_attempt=2,
    )

    insert_params = next(
        params for query, params in connection.executions
        if query.startswith("INSERT INTO model_job_attempts")
    )
    assert insert_params[:4] == (
        _attempt_identity("job-attempt-audit", 3),
        "job-attempt-audit",
        3,
        "local",
    )
    job_update = next(
        query for query, _params in connection.executions
        if query.startswith("UPDATE model_jobs SET status = 'uploading'")
    )
    assert "cancel_requested = FALSE" in job_update
    assert "attempt_count = attempt_count + 1" in job_update
    assert claimed["artifact_recovery_source_attempt"] == 2
    assert claimed["lease_token"]


def test_artifact_recovery_rejects_nonterminal_source_attempt() -> None:
    failed = _claimable_job(attempt_count=2, status="failed")
    failed.update({"lease_owner": "", "lease_token": ""})

    class _RecoveryConnection(_RecordingConnection):
        def execute(self, query, _params=()):
            normalized = " ".join(str(query).split())
            if normalized.startswith("SELECT status FROM model_job_attempts"):
                return _Cursor({"status": "running"})
            return super().execute(query, _params)

    repository = ModelResearchRepository(
        _RecordingDatabase(_RecoveryConnection(failed)),
    )

    with pytest.raises(ModelResearchConflict, match="尚未终止"):
        repository.claim_artifact_recovery(
            "job-attempt-audit",
            source_attempt=2,
        )


def test_attempt_row_exposes_error_log_cursor_and_artifact_identities() -> None:
    started = datetime(2026, 8, 28, 1, 2, tzinfo=timezone.utc)
    finished = datetime(2026, 8, 28, 1, 4, tzinfo=timezone.utc)
    attempt = _attempt_row({
        "attempt_id": "model-attempt-2",
        "job_id": "job-attempt-audit",
        "ordinal": 2,
        "status": "failed",
        "execution_node_id": "gpu-node-a",
        "started_at": started,
        "finished_at": finished,
        "error_json": {"code": "oom", "retryable": True},
        "log_start_event_id": 12,
        "log_end_event_id": 18,
        "artifact_refs_json": {
            "artifact-b": {"artifact_id": "artifact-b", "sha256": "b" * 64},
            "artifact-a": {"artifact_id": "artifact-a", "sha256": "a" * 64},
        },
    })

    assert attempt["ordinal"] == 2
    assert attempt["retryable"] is True
    assert attempt["error"] == {"code": "oom", "retryable": True}
    assert attempt["logs"] == [{
        "kind": "event_stream",
        "job_id": "job-attempt-audit",
        "attempt_ordinal": 2,
        "start_cursor": 12,
        "end_cursor": 18,
    }]
    assert [item["artifact_id"] for item in attempt["artifacts"]] == [
        "artifact-a", "artifact-b",
    ]


def test_explicit_registration_inserts_model_version_and_links_artifacts() -> None:
    row = {
        "job_id": "job-pending",
        "kind": "train",
        "status": "succeeded",
        "model_version": None,
        "model_id": "model-pending",
        "dataset_id": "dataset-pending",
        "title": "待确认模型",
        "model_kind": "lightgbm",
        "config_json": {"planned_model_version": 3},
        "result_json": {
            "metrics": {"ic": 0.03},
            "feature_importance": [],
            "predictions": {"row_count": 100},
            "manifest": {"schema_version": "test"},
        },
    }
    connection = _RecordingConnection(row)
    repository = ModelResearchRepository(_RecordingDatabase(connection))
    repository.get_job = lambda _job_id: dict(row)

    repository.register_training_result("job-pending")

    sql = "\n".join(connection.queries)
    assert "INSERT INTO model_versions" in sql
    assert "UPDATE model_jobs SET model_id" in sql
    assert "UPDATE model_artifacts SET model_id" in sql


def test_explicit_registration_assigns_public_model_identity_at_registration() -> None:
    row = {
        "job_id": "job-public-id",
        "kind": "train",
        "status": "succeeded",
        "model_version": None,
        "model_id": "temporary-study-model",
        "dataset_id": "dataset-public-id",
        "title": "公开模型",
        "model_kind": "lightgbm",
        "config_json": {"planned_model_version": 1},
        "result_json": {
            "metrics": {"ic": 0.03},
            "feature_importance": [],
            "predictions": {"row_count": 100},
            "manifest": {"schema_version": "test"},
        },
    }
    connection = _RecordingConnection(row)
    repository = ModelResearchRepository(_RecordingDatabase(connection))
    repository.get_job = lambda _job_id: dict(row)
    repository._prediction_alias_version = lambda **kwargs: (
        int(kwargs["minimum_version"])
    )
    copy_calls: list[dict] = []
    repository._copy_training_prediction_identity = lambda **kwargs: (
        copy_calls.append(kwargs)
        or {"copied": True, "source_rows": 100, "target_rows": 100}
    )

    repository.register_training_result(
        "job-public-id", model_id="csi500-stock-selector",
    )

    sql = "\n".join(connection.queries)
    assert "FROM model_versions WHERE model_id" in sql
    assert "FROM model_jobs" in sql
    assert "status NOT IN ('failed', 'canceled')" in sql
    assert "UPDATE model_jobs SET model_id" in sql
    assert copy_calls == [{
        "job_id": "job-public-id",
        "source_model_id": "temporary-study-model",
        "source_model_version": 1,
        "target_model_id": "csi500-stock-selector",
        "target_model_version": 7,
        "expected_source_rows": 100,
    }]


def test_registration_payloads_preserve_training_and_registration_identities() -> None:
    source_config = {"planned_model_version": 1, "model": {"kind": "lightgbm"}}
    source_result = {
        "predictions": {"row_count": 12, "model_version": 1},
        "manifest": {
            "model_id": "temporary-model",
            "model_version": 1,
            "schema_version": "test",
        },
    }

    config, result = _registration_payloads(
        job_id="job-registration",
        config=source_config,
        result=source_result,
        training_model_id="temporary-model",
        training_model_version=1,
        registered_model_id="public-model",
        registered_model_version=7,
        prediction_alias={"copied": True, "source_rows": 12, "target_rows": 12},
        registered_at="2026-08-28T00:00:00+00:00",
    )

    training_identity = {
        "model_id": "temporary-model", "model_version": 1,
        "job_id": "job-registration",
    }
    registration_identity = {
        "model_id": "public-model", "model_version": 7,
        "job_id": "job-registration",
    }
    assert config["planned_model_version"] == 7
    assert config["training_identity"] == training_identity
    assert config["registration_identity"] == registration_identity
    assert config["bundle_identity"] == training_identity
    assert result["training_identity"] == training_identity
    assert result["registration_identity"] == registration_identity
    assert result["manifest"]["model_id"] == "public-model"
    assert result["manifest"]["model_version"] == 7
    assert result["manifest"]["bundle_identity"] == training_identity
    assert result["predictions"]["model_id"] == "public-model"
    assert result["predictions"]["model_version"] == 7
    assert result["registration"]["status"] == "registered"
    assert result["prediction_identity_alias"]["source_retained"] is True
    assert source_config["planned_model_version"] == 1
    assert source_result["manifest"]["model_id"] == "temporary-model"


def test_prediction_identity_copy_is_non_destructive_and_count_verified(
    monkeypatch,
) -> None:
    class QueryResult:
        def __init__(self, count: int) -> None:
            self.result_rows = [[count]]

    class FakeClient:
        def __init__(self) -> None:
            self.results = iter((
                [[3]],
                [[0, []]],
                [[3]],
            ))
            self.commands: list[str] = []

        def query(self, _query, *, parameters):
            assert parameters["job_id"] == "job-copy"
            result = next(self.results)
            response = QueryResult(0)
            response.result_rows = result
            return response

        def command(self, query, *, parameters):
            assert parameters["source_model_id"] == "internal-model"
            assert parameters["target_model_id"] == "public-model"
            self.commands.append(" ".join(str(query).split()))

    client = FakeClient()
    import clickhouse_connect
    from factor_service.research import config as research_config

    monkeypatch.setattr(clickhouse_connect, "get_client", lambda **_kwargs: client)
    monkeypatch.setattr(research_config, "load_settings", lambda: SimpleNamespace(
        model_database="factor_model",
        clickhouse_host="localhost",
        clickhouse_port=8123,
        clickhouse_user="default",
        clickhouse_password="",
    ))

    report = ModelResearchRepository._copy_training_prediction_identity(
        job_id="job-copy",
        source_model_id="internal-model",
        source_model_version=1,
        target_model_id="public-model",
        target_model_version=7,
        expected_source_rows=3,
    )

    assert report == {"copied": True, "source_rows": 3, "target_rows": 3}
    assert len(client.commands) == 1
    assert client.commands[0].startswith("INSERT INTO")
    assert "SELECT" in client.commands[0]
    assert "DELETE" not in client.commands[0]


def test_prediction_alias_version_reuses_own_orphan_and_skips_foreign_orphan(
    monkeypatch,
) -> None:
    class QueryResult:
        def __init__(self, rows) -> None:
            self.result_rows = rows

    class FakeClient:
        def __init__(self, run_ids) -> None:
            self.run_ids = run_ids

        def query(self, _query, *, parameters):
            assert parameters == {"target_model_id": "public-model"}
            return QueryResult([[7, self.run_ids]])

    clients = iter((FakeClient(["job-copy"]), FakeClient(["another-job"])))
    import clickhouse_connect
    from factor_service.research import config as research_config

    monkeypatch.setattr(
        clickhouse_connect, "get_client", lambda **_kwargs: next(clients),
    )
    monkeypatch.setattr(research_config, "load_settings", lambda: SimpleNamespace(
        model_database="factor_model",
        clickhouse_host="localhost",
        clickhouse_port=8123,
        clickhouse_user="default",
        clickhouse_password="",
    ))

    assert ModelResearchRepository._prediction_alias_version(
        target_model_id="public-model",
        job_id="job-copy",
        minimum_version=7,
    ) == 7
    assert ModelResearchRepository._prediction_alias_version(
        target_model_id="public-model",
        job_id="job-copy",
        minimum_version=7,
    ) == 8


def test_prediction_identity_copy_reuses_verified_existing_alias(monkeypatch) -> None:
    class QueryResult:
        def __init__(self, rows) -> None:
            self.result_rows = rows

    class FakeClient:
        def __init__(self) -> None:
            self.results = iter(([[3]], [[3, ["job-copy"]]]))
            self.commands: list[str] = []

        def query(self, _query, *, parameters):
            return QueryResult(next(self.results))

        def command(self, query, *, parameters):
            self.commands.append(str(query))

    client = FakeClient()
    import clickhouse_connect
    from factor_service.research import config as research_config

    monkeypatch.setattr(clickhouse_connect, "get_client", lambda **_kwargs: client)
    monkeypatch.setattr(research_config, "load_settings", lambda: SimpleNamespace(
        model_database="factor_model",
        clickhouse_host="localhost",
        clickhouse_port=8123,
        clickhouse_user="default",
        clickhouse_password="",
    ))

    report = ModelResearchRepository._copy_training_prediction_identity(
        job_id="job-copy",
        source_model_id="internal-model",
        source_model_version=1,
        target_model_id="public-model",
        target_model_version=7,
        expected_source_rows=3,
    )

    assert report == {"copied": False, "source_rows": 3, "target_rows": 3}
    assert client.commands == []


def test_declining_registration_preserves_result_without_creating_version() -> None:
    row = {
        "job_id": "job-pending",
        "kind": "train",
        "status": "succeeded",
        "model_version": None,
        "result_json": {"metrics": {"ic": 0.03}},
    }
    connection = _RecordingConnection(row)
    repository = ModelResearchRepository(_RecordingDatabase(connection))
    repository.get_job = lambda _job_id: {
        **row,
        "registration_status": "declined",
    }

    repository.decline_training_result("job-pending")

    sql = "\n".join(connection.queries)
    assert "UPDATE model_jobs SET result_json" in sql
    assert "INSERT INTO model_versions" not in sql
    assert "job.registration_declined" not in sql
    assert "INSERT INTO model_job_events" in sql


def test_declined_training_result_cannot_be_registered() -> None:
    repository = ModelResearchRepository.__new__(ModelResearchRepository)
    repository.get_job = lambda _job_id: {
        "job_id": "job-declined",
        "kind": "train",
        "status": "succeeded",
        "result_json": {"registration": {"status": "declined"}},
    }

    with pytest.raises(ModelResearchConflict, match="已被排除入库"):
        repository.register_training_result("job-declined")


def test_parameter_experiment_only_registers_validation_selected_job() -> None:
    repository = ModelResearchRepository.__new__(ModelResearchRepository)
    repository.get_job = lambda _job_id: {
        "job_id": "job-runner-up",
        "kind": "train",
        "status": "succeeded",
        "config_json": {"experiment": {"experiment_id": "experiment-1"}},
    }
    repository.get_training_experiment = lambda _experiment_id: {
        "selection": {
            "status": "selected",
            "selected_job_id": "job-winner",
        },
    }

    with pytest.raises(ModelResearchConflict, match="验证集入选版本"):
        repository.register_training_result("job-runner-up")


def test_completed_standalone_training_is_automatically_registered() -> None:
    repository = ModelResearchRepository.__new__(ModelResearchRepository)
    current = {
        "job_id": "job-standalone",
        "kind": "train",
        "status": "succeeded",
        "model_version": None,
        "config_json": {},
        "result_json": {"registration": {"status": "automatic_pending"}},
    }
    repository.get_job = lambda _job_id: dict(current)
    registered: list[str] = []
    repository.register_training_result = lambda job_id: (
        registered.append(job_id) or {**current, "model_version": 4}
    )

    result = repository.finalize_training_result("job-standalone")

    assert result["model_version"] == 4
    assert registered == ["job-standalone"]


def test_completed_experiment_registers_only_selected_trial() -> None:
    repository = ModelResearchRepository.__new__(ModelResearchRepository)
    current = {
        "job_id": "job-final-trial",
        "kind": "train",
        "status": "succeeded",
        "model_version": None,
        "config_json": {"experiment": {"experiment_id": "experiment-1"}},
        "result_json": {"registration": {"status": "automatic_pending"}},
    }
    repository.get_job = lambda _job_id: dict(current)
    summary = {
        "selection": {
            "complete": True,
            "status": "selected",
            "selected_job_id": "job-winner",
        },
        "jobs": [current],
    }
    repository.get_training_experiment = lambda _experiment_id: summary
    registered: list[str] = []
    closed: list[tuple[dict, str]] = []
    repository.register_training_result = lambda job_id: registered.append(job_id) or {}
    repository._close_experiment_registration = lambda source, *, selected_job_id: (
        closed.append((source, selected_job_id))
    )

    repository.finalize_training_result("job-final-trial")

    assert registered == ["job-winner"]
    assert closed == [(summary, "job-winner")]


def test_registration_recovery_queries_only_durable_automatic_pending_jobs() -> None:
    queries: list[str] = []

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, params):
            queries.append(" ".join(str(statement).split()))
            assert params == (25,)
            return SimpleNamespace(fetchall=lambda: [
                {"job_id": "job-pending-1"},
                {"job_id": "job-pending-2"},
            ])

    repository = ModelResearchRepository(
        SimpleNamespace(connection=lambda: _Connection()),
    )
    finalized: list[str] = []
    repository.finalize_training_result = lambda job_id: (
        finalized.append(job_id)
        or {
            "model_version": 1 if job_id.endswith("1") else None,
            "result_json": {
                "registration": {
                    "status": (
                        "registered" if job_id.endswith("1")
                        else "experiment_not_selected"
                    ),
                },
            },
        }
    )

    result = repository.reconcile_pending_training_results(limit=25)

    assert finalized == ["job-pending-1", "job-pending-2"]
    assert result == {"pending": 2, "finalized": 2}
    assert "result_json -> 'registration' ->> 'status' = 'automatic_pending'" in queries[0]


def test_dataset_contract_freezes_stock_entity_asset_fields_without_factor_definition() -> None:
    spec = _dataset_spec({
        **_source(),
        "factors": [{
            "feature_kind": "entity_field",
            "factor_id": "entity_stock_daily_close_1234abcd",
            "entity_id": "stock",
            "asset_id": "asset_stock_daily_stock_daily_real",
            "asset_name": "股票日线数据",
            "asset_updated_at": "2026-08-12T16:14:50+00:00",
            "provider_node": "stock_daily_real",
            "field": "close",
            "label": "收盘价",
            "data_type": "number",
        }],
    })

    feature = spec["factors"][0]
    assert feature["feature_kind"] == "entity_field"
    assert feature["asset_id"] == "asset_stock_daily_stock_daily_real"
    assert feature["field"] == "close"
    assert feature["factor_version"] == 1
    assert len(feature["params_hash"]) == 64
    assert feature["params"] == {}


def test_dataset_spec_accepts_custom_split_ratios() -> None:
    spec = _dataset_spec({
        **_source(),
        "split": {"valid": 0.15, "test": 0.1},
    })

    assert spec["split"]["train"] == pytest.approx(0.75)
    assert spec["split"]["valid"] == pytest.approx(0.15)
    assert spec["split"]["test"] == pytest.approx(0.1)
    assert spec["split"]["embargo_days"] == 5
    assert spec["split"]["mode"] == "ratio"


def test_dataset_spec_accepts_explicit_train_validation_and_test_dates() -> None:
    spec = _dataset_spec({
        **_source(),
        "split": {
            "mode": "dates",
            "train": ["2020-01-01", "2022-12-30"],
            "validation": ["2023-01-09", "2023-06-30"],
            "test": ["2023-07-10", "2024-01-01"],
            "embargo_days": 5,
        },
    })

    assert spec["split"] == {
        "mode": "dates",
        "train": ["2020-01-01", "2022-12-30"],
        "valid": ["2023-01-09", "2023-06-30"],
        "test": ["2023-07-10", "2024-01-01"],
        "embargo_days": 5,
    }


def test_dataset_spec_rejects_mixed_or_overlapping_date_split() -> None:
    with pytest.raises(ModelResearchError, match="严格有序"):
        _dataset_spec({
            **_source(),
            "split": {
                "mode": "dates",
                "train": ["2020-01-01", "2023-01-15"],
                "validation": ["2023-01-09", "2023-06-30"],
                "test": ["2023-07-10", "2024-01-01"],
            },
        })


def test_dataset_spec_accepts_custom_universe() -> None:
    spec = _dataset_spec({**_source(), "universe_id": "all_a"})

    assert spec["universe_id"] == "all_a"
    assert spec["index_code"] == "000985.SH"
    assert spec["benchmark_code"] == "000985.SH"

    csi300 = _dataset_spec({**_source(), "universe_id": "csi300"})
    assert csi300["universe_id"] == "csi300"
    assert csi300["index_code"] == "000300.SH"


def test_dataset_spec_rejects_unknown_universe() -> None:
    with pytest.raises(ModelResearchError, match="不支持的股票池"):
        _dataset_spec({**_source(), "universe_id": "csi2000"})


def test_dataset_spec_rejects_invalid_split_ratios() -> None:
    with pytest.raises(ModelResearchError, match="训练集不低于30%"):
        _dataset_spec({**_source(), "split": {"valid": 0.5, "test": 0.4}})
    with pytest.raises(ModelResearchError, match="不低于5%"):
        _dataset_spec({**_source(), "split": {"valid": 0.0, "test": 0.1}})
    with pytest.raises(ModelResearchError, match="必须是数字"):
        _dataset_spec({**_source(), "split": {"valid": "high", "test": 0.2}})


def test_incremental_training_requires_exact_lightgbm_feature_contract() -> None:
    source = _trained_model(
        "stock-model", 1, validation_icir=0.5, model_kind="lightgbm",
    )
    source.update({
        "job_id": "model_job_source",
        "state": "validated",
        "dataset_hash": "d" * 64,
    })
    candidate_dataset = _dataset_spec({
        **_source(),
        "date_end": "2025-01-02",
        "data_cutoff": "2025-01-02T15:00:00+08:00",
    })
    candidate_model = _model_spec({"kind": "lightgbm", "params": {}})
    bundle = {
        "artifact_id": "artifact-bundle",
        "relative_path": "jobs/source/bundle.tar.gz",
        "sha256": "b" * 64,
        "file_name": "bundle.tar.gz",
    }

    ready = _incremental_training_assessment(
        source, bundle,
        dataset=candidate_dataset,
        model=candidate_model,
        walk_forward=_walk_forward_spec({}),
    )

    assert ready["passed"] is True
    assert ready["contract"]["mode"] == "lightgbm_append_trees_new_data_only"
    assert ready["contract"]["minimum_new_trading_sessions"] == 60

    changed = {
        **candidate_dataset,
        "factors": [{
            **candidate_dataset["factors"][0],
            "params_hash": "c" * 64,
        }],
    }
    blocked = _incremental_training_assessment(
        source, bundle,
        dataset=changed,
        model=candidate_model,
        walk_forward=_walk_forward_spec({}),
    )
    assert blocked["passed"] is False
    assert "feature_identity" in blocked["failed_checks"]

    changed_industry = {
        **candidate_dataset,
        "industry_feature": normalize_industry_feature(
            {"enabled": True}, default_enabled=False,
        ),
    }
    industry_blocked = _incremental_training_assessment(
        source, bundle,
        dataset=changed_industry,
        model=candidate_model,
        walk_forward=_walk_forward_spec({}),
    )
    assert industry_blocked["passed"] is False
    assert "target_contract" in industry_blocked["failed_checks"]


def test_incremental_contract_records_immutable_bundle_identity() -> None:
    source = _trained_model(
        "public-model", 7, validation_icir=0.5, model_kind="lightgbm",
    )
    source.update({
        "job_id": "model_job_source",
        "state": "validated",
        "dataset_hash": "d" * 64,
        "manifest_json": {
            "model_id": "public-model",
            "model_version": 7,
            "bundle_identity": {
                "model_id": "temporary-model",
                "model_version": 1,
                "job_id": "model_job_source",
            },
        },
    })
    candidate_dataset = _dataset_spec({
        **_source(),
        "date_end": "2025-01-02",
        "data_cutoff": "2025-01-02T15:00:00+08:00",
    })
    assessment = _incremental_training_assessment(
        source,
        {
            "artifact_id": "artifact-bundle",
            "relative_path": "jobs/source/bundle.tar.gz",
            "sha256": "b" * 64,
            "file_name": "bundle.tar.gz",
        },
        dataset=candidate_dataset,
        model=_model_spec({"kind": "lightgbm", "params": {}}),
        walk_forward=_walk_forward_spec({}),
    )

    assert assessment["passed"] is True
    assert assessment["contract"]["source_model_id"] == "public-model"
    assert assessment["contract"]["source_model_version"] == 7
    assert assessment["contract"]["source_bundle_identity"] == {
        "model_id": "temporary-model",
        "model_version": 1,
        "job_id": "model_job_source",
    }


def test_incremental_training_accepts_legacy_v5_disabled_preprocessing() -> None:
    source = _trained_model(
        "legacy-stock-model", 1, validation_icir=0.5, model_kind="lightgbm",
    )
    source.update({
        "job_id": "model_job_legacy_source",
        "state": "validated",
        "dataset_hash": "e" * 64,
    })
    source["dataset_spec"] = {
        **source["dataset_spec"],
        "pipeline_version": "alphablocks.dataset-pipeline.v5",
    }
    source["dataset_spec"].pop("preprocessing")
    source["dataset_spec"].pop("industry_feature")
    candidate_dataset = _dataset_spec({
        **_source(),
        "date_end": "2025-01-02",
        "data_cutoff": "2025-01-02T15:00:00+08:00",
        "preprocessing": {"enabled": False},
    })
    assessment = _incremental_training_assessment(
        source,
        {
            "artifact_id": "artifact-bundle",
            "relative_path": "jobs/source/bundle.tar.gz",
            "sha256": "b" * 64,
            "file_name": "bundle.tar.gz",
        },
        dataset=candidate_dataset,
        model=_model_spec({"kind": "lightgbm", "params": {}}),
        walk_forward=_walk_forward_spec({}),
    )

    assert assessment["passed"] is True


def test_dataset_contract_freezes_label_horizon_and_matching_embargo() -> None:
    spec = _dataset_spec({
        **_source(),
        "label_horizon_trading_days": 10,
    })

    assert spec["label"] == {
        "kind": "future_10d_cross_sectional_rank",
        "horizon_trading_days": 10,
        "range": [-1.0, 1.0],
    }
    assert spec["split"]["embargo_days"] == 10


@pytest.mark.parametrize("horizon", [0, 31, "invalid"])
def test_dataset_contract_rejects_unsupported_label_horizon(horizon) -> None:
    with pytest.raises(ModelResearchError, match=r"T\+1至T\+30"):
        _dataset_spec({
            **_source(),
            "label_horizon_trading_days": horizon,
        })


@pytest.mark.parametrize("horizon", [1, 2, 5, 20, 30])
def test_dataset_contract_accepts_single_horizon_range(horizon) -> None:
    spec = _dataset_spec({
        **_source(),
        "label_horizon_trading_days": horizon,
    })
    assert spec["label"]["horizon_trading_days"] == horizon
    assert spec["split"]["embargo_days"] == horizon


def test_training_job_freezes_remote_execution_node() -> None:
    # The normalizer is exercised separately from database writes through the
    # exported job spec helpers used by repository tests.
    from factor_service.model_research_repository import _execution_spec

    assert _execution_spec({
        "node_id": "autodl-gpu-01", "max_runtime_minutes": 240,
    }) == {
        "node_id": "autodl-gpu-01",
        "mode": "remote_ssh_docker",
        "max_runtime_minutes": 240,
    }
    assert _execution_spec({}) == {
        "node_id": "local", "mode": "local", "max_runtime_minutes": 720,
    }
    with pytest.raises(ModelResearchError, match="max_runtime_minutes"):
        _execution_spec({"max_runtime_minutes": 30})


def test_removed_market_style_dataset_target_is_rejected() -> None:
    with pytest.raises(ModelResearchError, match="训练目标只支持"):
        _dataset_spec({**_source(), "research_target": "market_style"})


def test_industry_rotation_dataset_rejects_pre_sw2021_history() -> None:
    source = {**_source(), "research_target": "industry_rotation"}

    with pytest.raises(ModelResearchError, match="2021-12-13"):
        _dataset_spec(source)


def test_industry_rotation_dataset_freezes_sw2021_daily_snapshot_contract() -> None:
    source = {
        **_source(),
        "research_target": "industry_rotation",
        "date_start": "2022-01-04",
    }

    spec = _dataset_spec(source)

    assert spec["research_target"] == "industry_rotation"
    assert spec["prediction_scope"] == "industry"
    assert spec["label"] == {
        "kind": "future_5d_industry_rank",
        "horizon_trading_days": 5,
        "range": [-1.0, 1.0],
        "classification": "sw2021_l1",
        "safe_start": "2021-12-13",
    }
    assert spec["availability"]["industry_snapshot_date_eq_signal_date"] is True


def test_dataset_contract_rejects_unlocked_factor() -> None:
    source = _source()
    del source["factors"][0]["params_hash"]

    with pytest.raises(ModelResearchError, match="params_hash"):
        _dataset_spec(source)


def _origin_source_job() -> dict:
    dataset = _dataset_spec(_source())
    model = _model_spec({
        "kind": "lightgbm",
        "params": {"learning_rate": 0.03, "num_leaves": 15},
    })
    walk_forward = _walk_forward_spec({
        "enabled": True,
        "train_sessions": 756,
        "valid_sessions": 60,
        "test_sessions": 20,
        "step_sessions": 20,
        "embargo_sessions": 5,
        "oos_date_start": "2023-01-03",
        "oos_date_end": "2024-12-31",
    })
    return {
        "job_id": "source-job",
        "status": "succeeded",
        "model_id": "source-model",
        "model_version": 2,
        "dataset_hash": "f" * 64,
        "dataset_spec": dataset,
        "config_json": {"model": model, "walk_forward": walk_forward},
    }


def test_research_origin_marks_identical_configuration_as_exact_replay() -> None:
    source_job = _origin_source_job()
    result = _research_origin_spec(
        {"requested_mode": "exact_replay"},
        source_type="model_version",
        source_id="source-model.v2",
        source_job=source_job,
        source_model_id="source-model",
        source_model_version=2,
        dataset=source_job["dataset_spec"],
        model=source_job["config_json"]["model"],
        walk_forward=source_job["config_json"]["walk_forward"],
    )

    assert result["mode"] == "exact_replay"
    assert result["changed_sections"] == []
    assert result["source_dataset_hash"] == "f" * 64
    assert len(result["source_config_hash"]) == 64


def test_research_origin_does_not_call_legacy_pipeline_an_exact_replay() -> None:
    source_job = _origin_source_job()
    source_job["dataset_spec"] = {
        **source_job["dataset_spec"],
        "pipeline_version": "alphablocks.dataset-pipeline.v5",
    }
    source_job["dataset_spec"].pop("preprocessing")
    candidate = _dataset_spec({
        **_source(),
        "preprocessing": {"enabled": False},
    })

    with pytest.raises(ModelResearchConflict, match="dataset"):
        _research_origin_spec(
            {"requested_mode": "exact_replay"},
            source_type="model_version",
            source_id="source-model.v2",
            source_job=source_job,
            source_model_id="source-model",
            source_model_version=2,
            dataset=candidate,
            model=source_job["config_json"]["model"],
            walk_forward=source_job["config_json"]["walk_forward"],
        )


def test_research_origin_rejects_changed_configuration_claiming_exact_replay() -> None:
    source_job = _origin_source_job()
    changed_model = _model_spec({
        "kind": "lightgbm",
        "params": {"learning_rate": 0.08, "num_leaves": 15},
    })

    with pytest.raises(ModelResearchConflict, match="model"):
        _research_origin_spec(
            {"requested_mode": "exact_replay"},
            source_type="experiment",
            source_id="experiment-a",
            source_job=source_job,
            source_model_id="source-model",
            source_model_version=2,
            dataset=source_job["dataset_spec"],
            model=changed_model,
            walk_forward=source_job["config_json"]["walk_forward"],
        )


def test_research_origin_records_derived_study_and_declared_design_change() -> None:
    source_job = _origin_source_job()
    result = _research_origin_spec(
        {
            "requested_mode": "derived",
            "declared_changes": ["research_design"],
        },
        source_type="model_version",
        source_id="source-model.v2",
        source_job=source_job,
        source_model_id="source-model",
        source_model_version=2,
        dataset=source_job["dataset_spec"],
        model=source_job["config_json"]["model"],
        walk_forward=source_job["config_json"]["walk_forward"],
    )

    assert result["mode"] == "derived"
    assert result["changed_sections"] == ["research_design"]


def test_research_origin_resolver_verifies_model_version_owns_source_job() -> None:
    source_job = _origin_source_job()
    repository = ModelResearchRepository.__new__(ModelResearchRepository)
    repository.get_model = lambda model_id, version: {
        "model_id": model_id,
        "version": version,
        "job_id": "different-job",
    }

    with pytest.raises(ModelResearchConflict, match="模型版本与来源训练任务不匹配"):
        repository._resolve_research_origin(
            {
                "source_type": "model_version",
                "source_id": "source-model.v2",
                "source_job_id": source_job["job_id"],
                "source_model_id": "source-model",
                "source_model_version": 2,
                "requested_mode": "exact_replay",
            },
            dataset=source_job["dataset_spec"],
            model=source_job["config_json"]["model"],
            walk_forward=source_job["config_json"]["walk_forward"],
        )


def test_research_origin_resolver_only_accepts_selected_experiment_job() -> None:
    source_job = _origin_source_job()
    repository = ModelResearchRepository.__new__(ModelResearchRepository)
    repository.get_training_experiment = lambda _experiment_id: {
        "selection": {
            "status": "selected",
            "selected_job_id": "another-job",
            "selected_model_id": "source-model",
            "selected_model_version": 2,
        },
    }

    with pytest.raises(ModelResearchConflict, match="验证集选出的入选任务"):
        repository._resolve_research_origin(
            {
                "source_type": "experiment",
                "source_id": "experiment-a",
                "source_job_id": source_job["job_id"],
                "requested_mode": "exact_replay",
            },
            dataset=source_job["dataset_spec"],
            model=source_job["config_json"]["model"],
            walk_forward=source_job["config_json"]["walk_forward"],
        )


def test_ensemble_contract_freezes_sources_and_equal_weights() -> None:
    sources = [_trained_model("model-a", 1, validation_icir=0.4), _trained_model("model-b", 2, validation_icir=0.8)]

    spec = _ensemble_spec({
        "name": "LGBM + XGB",
        "sources": [
            {"model_id": "model-a", "model_version": 1},
            {"model_id": "model-b", "model_version": 2},
        ],
        "weight_strategy": "equal",
    }, sources)

    assert spec["fusion_method"] == "linear_score"
    assert [item["weight"] for item in spec["sources"]] == [0.5, 0.5]
    assert spec["dataset"]["materialization"]["mode"] == "model_prediction_fusion"
    assert spec["dataset"]["ensemble_sources"][0]["dataset_hash"] == sources[0]["dataset_hash"]


def test_ensemble_icir_weights_only_use_validation_metrics() -> None:
    sources = [
        _trained_model("model-a", 1, validation_icir=0.2, test_icir=999),
        _trained_model("model-b", 2, validation_icir=0.8, test_icir=-999),
    ]

    spec = _ensemble_spec({
        "sources": [
            {"model_id": "model-a", "model_version": 1},
            {"model_id": "model-b", "model_version": 2},
        ],
        "weight_strategy": "validation_icir",
    }, sources)

    assert [item["weight"] for item in spec["sources"]] == pytest.approx([0.2, 0.8])
    assert spec["weight_metric"] == "validation.ic_ir"


def test_multi_horizon_ensemble_freezes_primary_evaluation_horizon() -> None:
    sources = [
        _trained_model(
            "horizon-a", 1, validation_icir=0.2, horizon=1,
            model_kind="lightgbm",
        ),
        _trained_model(
            "horizon-b", 2, validation_icir=0.8, horizon=5,
            model_kind="lightgbm",
        ),
    ]

    spec = _ensemble_spec({
        "ensemble_mode": "multi_horizon",
        "evaluation_horizon_trading_days": 5,
        "weight_strategy": "validation_icir",
        "sources": [
            {"model_id": "horizon-a", "model_version": 1},
            {"model_id": "horizon-b", "model_version": 2},
        ],
    }, sources)

    assert spec["ensemble_mode"] == "multi_horizon"
    assert spec["source_horizons"] == [1, 5]
    assert spec["evaluation_horizon_trading_days"] == 5
    assert spec["dataset"]["label"]["horizon_trading_days"] == 5
    assert spec["dataset"]["split"]["embargo_days"] == 5
    assert spec["dataset"]["materialization"]["mode"] == (
        "multi_horizon_model_prediction_fusion"
    )
    assert [item["weight"] for item in spec["sources"]] == pytest.approx([0.2, 0.8])


def test_multi_horizon_ensemble_rejects_different_factors() -> None:
    first = _trained_model(
        "horizon-a", 1, validation_icir=0.2, horizon=1,
        model_kind="lightgbm",
    )
    second = _trained_model(
        "horizon-b", 2, validation_icir=0.8, horizon=5,
        model_kind="lightgbm",
    )
    second["dataset_spec"] = {
        **second["dataset_spec"],
        "factors": [{
            **second["dataset_spec"]["factors"][0],
            "factor_id": "different_factor",
        }],
    }

    with pytest.raises(ModelResearchError, match="完全相同的冻结因子"):
        _ensemble_spec({
            "ensemble_mode": "multi_horizon",
            "sources": [
                {"model_id": "horizon-a", "model_version": 1},
                {"model_id": "horizon-b", "model_version": 2},
            ],
        }, [first, second])


def test_ensemble_rejects_duplicate_source_or_nonpositive_manual_weight() -> None:
    source = _trained_model("model-a", 1, validation_icir=0.4)
    with pytest.raises(ModelResearchError, match="重复"):
        _ensemble_spec({"sources": [
            {"model_id": "model-a", "model_version": 1},
            {"model_id": "model-a", "model_version": 1},
        ]}, [source, source])
    with pytest.raises(ModelResearchError, match="大于0"):
        _ensemble_spec({
            "weight_strategy": "manual",
            "sources": [
                {"model_id": "model-a", "model_version": 1, "weight": 1},
                {"model_id": "model-b", "model_version": 2, "weight": 0},
            ],
        }, [source, _trained_model("model-b", 2, validation_icir=0.5)])


def test_model_architecture_locks_engine_models_and_frozen_feature_groups() -> None:
    models = [
        {
            **_trained_model("model-a", 1, validation_icir=0.4),
            "state": "validated",
            "prediction_json": {
                "row_count": 1000, "date_start": "2024-01-02",
                "date_end": "2024-06-28",
            },
            "manifest_json": {"model_params": {"num_leaves": 64}},
        },
        {
            **_trained_model("model-b", 2, validation_icir=0.6),
            "state": "validated",
            "prediction_json": {
                "row_count": 1000, "date_start": "2024-03-01",
                "date_end": "2024-09-30",
            },
            "manifest_json": {"model_params": {"max_depth": 6}},
        },
    ]
    spec = _architecture_spec({
        "name": "大小盘双引擎",
        "merge_method": "weighted_score",
        "top_n": 20,
        "rebalance_every": 5,
        "engines": [
            {
                "engine_key": "large_cap", "display_name": "大盘引擎",
                "role": "large_cap_selection", "model_id": "model-a",
                "model_version": 1, "weight": 2, "score_threshold": 0.4,
                "priority": 1,
            },
            {
                "engine_key": "small_cap", "display_name": "小盘引擎",
                "role": "small_cap_selection", "model_id": "model-b",
                "model_version": 2, "weight": 1, "score_threshold": 0.6,
                "priority": 2,
            },
        ],
    }, models)

    assert spec["schema_version"] == "alphablocks.model-architecture.v2"
    assert spec["pipeline_mode"] == "flat"
    assert spec["universe_id"] == "csi500"
    assert spec["engine_count"] == 2
    assert spec["engines"][0]["factor_ids"] == ["mom_20"]
    assert spec["engines"][0]["architecture"] == {"num_leaves": 64}
    assert [item["normalized_weight"] for item in spec["engines"]] == pytest.approx([
        2 / 3, 1 / 3,
    ])
    assert spec["execution_contract"]["engine_features"] == "locked_by_model_version"

    readiness = _architecture_readiness(spec, models)
    assert readiness["ready"] is True
    assert readiness["definition_only"] is False
    assert readiness["research_backtest_ready"] is True
    common = next(
        item for item in readiness["checks"]
        if item["key"] == "common_prediction_range"
    )
    assert common["actual"] == "2024-03-01 至 2024-06-28"

    no_overlap = _architecture_readiness(spec, [
        models[0],
        {**models[1], "prediction_json": {
            "row_count": 1000, "date_start": "2025-01-02",
            "date_end": "2025-06-30",
        }},
    ])
    assert no_overlap["ready"] is False
    assert no_overlap["research_backtest_ready"] is False
    assert "common_prediction_range" in no_overlap["failed_checks"]


def test_hierarchical_architecture_freezes_complete_decision_chain() -> None:
    models = [
        {
            **_trained_model(model_id, version, validation_icir=0.4),
            "state": "validated",
            "prediction_json": {
                "row_count": 1000, "date_start": "2024-01-02",
                "date_end": "2024-06-28",
            },
        }
        for model_id, version in (
            ("model-industry", 1), ("model-stock", 2),
        )
    ]
    models[0]["dataset_spec"] = {
        **models[0]["dataset_spec"],
        "research_target": "industry_rotation",
        "prediction_scope": "industry",
    }
    spec = _architecture_spec({
        "name": "行业个股分层架构",
        "pipeline_mode": "hierarchical",
        "merge_method": "weighted_score",
        "engines": [
            {
                "engine_key": "industry", "role": "industry_rotation",
                "model_id": "model-industry", "model_version": 1,
            },
            {
                "engine_key": "stock", "role": "stock_selection",
                "model_id": "model-stock", "model_version": 2,
            },
        ],
    }, models)

    assert spec["pipeline_mode"] == "hierarchical"
    assert [item["stage"] for item in spec["engines"]] == [
        "industry_gate", "stock_rank",
    ]
    assert spec["execution_contract"]["stage_order"] == [
        "industry_gate", "risk_gate", "stock_rank",
    ]
    readiness = _architecture_readiness(spec, models)
    assert readiness["research_backtest_ready"] is True
    assert readiness["stage_counts"] == {
        "industry_gate": 1, "stock_rank": 1,
    }


def test_hierarchical_architecture_rejects_missing_industry_stage() -> None:
    models = [_trained_model("model-stock", 1, validation_icir=0.4)]
    with pytest.raises(ModelResearchError, match="行业轮动"):
        _architecture_spec({
            "name": "不完整分层架构",
            "pipeline_mode": "hierarchical",
            "merge_method": "weighted_score",
            "engines": [
                {
                    "role": "stock_selection", "model_id": "model-stock",
                    "model_version": 1,
                },
            ],
        }, models)


def test_model_architecture_rejects_duplicate_priority_and_unvalidated_activation() -> None:
    models = [
        {**_trained_model("model-a", 1, validation_icir=0.4), "state": "validated"},
        {**_trained_model("model-b", 2, validation_icir=0.6), "state": "candidate"},
    ]
    with pytest.raises(ModelResearchError, match="不同优先级"):
        _architecture_spec({
            "name": "冲突架构", "merge_method": "priority",
            "engines": [
                {"model_id": "model-a", "model_version": 1, "priority": 1},
                {"model_id": "model-b", "model_version": 2, "priority": 1},
            ],
        }, models)

    spec = _architecture_spec({
        "name": "候选架构", "merge_method": "union",
        "engines": [
            {"model_id": "model-a", "model_version": 1, "priority": 1},
            {"model_id": "model-b", "model_version": 2, "priority": 2},
        ],
    }, models)
    readiness = _architecture_readiness(spec, [
        {**models[0], "prediction_json": {
            "row_count": 100, "date_start": "2024-01-02",
            "date_end": "2024-06-28",
        }},
        {**models[1], "prediction_json": {
            "row_count": 100, "date_start": "2024-03-01",
            "date_end": "2024-09-30",
        }},
    ])
    assert readiness["ready"] is False
    assert readiness["research_backtest_ready"] is True
    assert "models_validated" in readiness["failed_checks"]
    assert readiness["research_failed_checks"] == []


def test_model_architecture_freezes_aligned_walk_forward_windows() -> None:
    windows = [
        {"window": 1, "segments": {"test": ["2021-05-11", "2022-05-24"]},
         "metrics": {"rank_ic": 0.03, "ic_ir": 0.4, "test_days": 252}},
        {"window": 2, "segments": {"test": ["2022-05-25", "2023-06-05"]},
         "metrics": {"rank_ic": 0.04, "ic_ir": 0.5, "test_days": 252}},
        {"window": 3, "segments": {"test": ["2023-06-06", "2024-06-20"]},
         "metrics": {"rank_ic": 0.05, "ic_ir": 0.6, "test_days": 252}},
    ]
    models = [
        {
            **_trained_model(model_id, version, validation_icir=0.4),
            "state": "candidate",
            "manifest_json": {"walk_forward": {
                "enabled": True, "strategy": "rolling", "windows": windows,
            }},
        }
        for model_id, version in [("model-a", 1), ("model-b", 2)]
    ]

    spec = _architecture_spec({
        "name": "WFA双引擎", "merge_method": "weighted_score",
        "engines": [
            {"model_id": "model-a", "model_version": 1, "priority": 1},
            {"model_id": "model-b", "model_version": 2, "priority": 2},
        ],
    }, models)

    assert spec["walk_forward"]["eligible"] is True
    assert spec["walk_forward"]["window_count"] == 3
    assert spec["engines"][0]["walk_forward"]["windows"][0] == {
        "window": 1,
        "test_start": "2021-05-11",
        "test_end": "2022-05-24",
        "rank_ic": 0.03,
        "ic_ir": 0.4,
        "test_days": 252,
    }


def test_model_architecture_rejects_misaligned_walk_forward_contract() -> None:
    models = [
        {
            **_trained_model("model-a", 1, validation_icir=0.4),
            "manifest_json": {"walk_forward": {
                "enabled": True, "windows": [{
                    "window": 1,
                    "segments": {"test": ["2021-01-01", "2021-12-31"]},
                }],
            }},
        },
        {
            **_trained_model("model-b", 2, validation_icir=0.4),
            "manifest_json": {"walk_forward": {
                "enabled": True, "windows": [{
                    "window": 1,
                    "segments": {"test": ["2022-01-01", "2022-12-31"]},
                }],
            }},
        },
    ]
    spec = _architecture_spec({
        "name": "错位WFA", "engines": [
            {"model_id": "model-a", "model_version": 1, "priority": 1},
            {"model_id": "model-b", "model_version": 2, "priority": 2},
        ],
    }, models)

    assert spec["walk_forward"]["eligible"] is False
    assert "窗口不一致" in spec["walk_forward"]["reason"]


def test_model_contract_uses_whitelist_and_deterministic_seed() -> None:
    spec = _model_spec({"params": {"learning_rate": 0.03, "unsafe": "ignored"}})

    assert spec["params"]["learning_rate"] == 0.03
    assert spec["params"]["seed"] == 42
    assert "unsafe" not in spec["params"]


def test_all_supported_models_have_distinct_training_implementations() -> None:
    expected = {
        "lightgbm": "qlib.contrib.model.gbdt.LGBModel",
        "xgboost": "qlib.contrib.model.xgboost.XGBModel",
        "catboost": "qlib.contrib.model.catboost_model.CatBoostModel",
        "mlp": "factor_service.research.models.QlibTorchMLPModel",
        "lstm": "factor_service.research.models.QlibTorchLSTMModel",
        "transformer_lstm": "factor_service.research.models.QlibTorchTransformerLSTMModel",
    }

    for kind, implementation in expected.items():
        spec = _model_spec({"kind": kind, "params": {}})
        assert spec["qlib_model"] == implementation
        assert spec["params"]["seed"] == 42


def test_mlp_contract_freezes_explicit_hidden_layer_widths() -> None:
    spec = _model_spec({
        "kind": "mlp",
        "params": {"hidden_layers": [64, 128, 256]},
    })

    assert spec["params"]["hidden_layers"] == [64, 128, 256]
    assert "hidden_size" not in spec["params"]
    assert "layer_count" not in spec["params"]


def test_mlp_contract_rejects_invalid_hidden_layer_widths() -> None:
    with pytest.raises(ModelResearchError, match="hidden_layers"):
        _model_spec({"kind": "mlp", "params": {"hidden_layers": [64, 5000]}})


def test_lstm_contract_freezes_sequence_architecture() -> None:
    spec = _model_spec({
        "kind": "lstm",
        "params": {"lookback_window": 40, "hidden_size": 96, "num_layers": 3},
    })

    assert spec["params"]["lookback_window"] == 40
    assert spec["params"]["hidden_size"] == 96
    assert spec["params"]["num_layers"] == 3
    assert spec["params"]["learning_rate"] == 0.001


def test_transformer_lstm_contract_freezes_both_encoders() -> None:
    spec = _model_spec({
        "kind": "transformer_lstm",
        "params": {
            "lookback_window": 40, "d_model": 48, "nhead": 4,
            "transformer_layers": 2, "lstm_hidden_size": 96,
        },
    })

    assert spec["params"]["lookback_window"] == 40
    assert spec["params"]["d_model"] == 48
    assert spec["params"]["nhead"] == 4
    assert spec["params"]["lstm_hidden_size"] == 96


def test_transformer_lstm_contract_rejects_incompatible_heads() -> None:
    with pytest.raises(ModelResearchError, match="d_model.*nhead"):
        _model_spec({
            "kind": "transformer_lstm", "params": {"d_model": 30, "nhead": 8},
        })


def test_walk_forward_contract_has_strict_independent_test_defaults() -> None:
    spec = _walk_forward_spec({
        "enabled": True,
        "oos_date_start": "2023-01-03",
        "oos_date_end": "2024-12-31",
    })

    assert spec["strategy"] == "rolling"
    assert spec["train_sessions"] == 756
    assert spec["valid_sessions"] == 60
    assert spec["test_sessions"] == 20
    assert spec["step_sessions"] == 20
    assert spec["embargo_sessions"] == 5


def test_walk_forward_contract_accepts_explicit_zero_validation() -> None:
    spec = _walk_forward_spec({
        "enabled": True,
        "valid_sessions": 0,
        "oos_date_start": "2023-01-03",
        "oos_date_end": "2024-12-31",
    })

    assert spec["valid_sessions"] == 0


def test_walk_forward_contract_rejects_overlapping_test_windows() -> None:
    with pytest.raises(ModelResearchError, match="步长必须等于测试窗口"):
        _walk_forward_spec({
            "enabled": True,
            "test_sessions": 20,
            "step_sessions": 10,
            "oos_date_start": "2023-01-03",
            "oos_date_end": "2024-12-31",
        })


def test_job_transport_serializes_database_timestamps() -> None:
    row = {
        "job_id": "model_job_test",
        "lease_token": "secret-token",
        "requested_at": datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc),
    }

    result = _job_row(row)

    assert "lease_token" not in result
    assert result["requested_at"] == "2026-08-14T12:00:00+00:00"


def test_grid_search_builds_deterministic_cartesian_trials() -> None:
    trials = _grid_search_trials(
        {"kind": "lightgbm", "params": {"n_estimators": 200}},
        {
            "strategy": "grid",
            "parameters": {
                "num_leaves": [31, 63],
                "learning_rate": [0.03, 0.05],
            },
            "max_trials": 4,
        },
    )

    assert len(trials) == 4
    assert [
        (trial["learning_rate"], trial["num_leaves"]) for trial in trials
    ] == [(0.03, 31), (0.03, 63), (0.05, 31), (0.05, 63)]
    assert all(trial["n_estimators"] == 200 for trial in trials)
    assert all(trial["seed"] == 42 for trial in trials)


def test_grid_search_supports_mlp_architecture_candidates() -> None:
    trials = _grid_search_trials(
        {"kind": "mlp", "params": {}},
        {
            "parameters": {
                "hidden_layers": [[64, 128], [64, 128, 256]],
            },
        },
    )

    assert [trial["hidden_layers"] for trial in trials] == [
        [64, 128], [64, 128, 256],
    ]


def test_optuna_spec_freezes_robust_tree_search_contract() -> None:
    spec = _optuna_spec(
        {"enabled": True, "n_trials": 25},
        model_kind="lightgbm",
        walk_forward={"enabled": True, "valid_sessions": 60},
        incremental=False,
    )

    assert spec == {
        "enabled": True,
        "backend": "optuna",
        "n_trials": 25,
        "objective": "validation_rank_icir",
        "direction": "maximize",
        "sampler": "tpe",
        "seed": 42,
        "validation_windows": 3,
        "seed_count": 3,
        "stability_penalty": 0.5,
        "minimum_positive_window_ratio": 0.6,
        "validation_mode": "walk_forward_folds",
        "search_space_version": "alphablocks.tree-optuna.v3",
    }

    fixed = _optuna_spec(
        {"enabled": True}, model_kind="lightgbm",
        walk_forward={"enabled": False}, incremental=False,
    )
    assert fixed["validation_mode"] == "fixed_subwindows"
    assert fixed["search_space_version"] == "alphablocks.tree-optuna.v3"


def test_optuna_spec_rejects_unsupported_or_validationless_training() -> None:
    with pytest.raises(ModelResearchError, match="只支持LightGBM"):
        _optuna_spec(
            {"enabled": True}, model_kind="mlp",
            walk_forward={}, incremental=False,
        )
    with pytest.raises(ModelResearchError, match="验证长度为0"):
        _optuna_spec(
            {"enabled": True}, model_kind="xgboost",
            walk_forward={"enabled": True, "valid_sessions": 0},
            incremental=False,
        )


def test_horizon_search_normalizes_supported_unique_values() -> None:
    assert _horizon_search_values({"horizons": [10, 3, 5, 3]}) == [3, 5, 10]

    with pytest.raises(ModelResearchError, match="至少需要两个不同"):
        _horizon_search_values({"horizons": [5, 5]})
    with pytest.raises(ModelResearchError, match="只支持"):
        _horizon_search_values({"horizons": [1, 20]})


def test_canceled_experiment_restart_clones_frozen_trials_into_new_experiment() -> None:
    repository = object.__new__(ModelResearchRepository)
    source_experiment_id = "model_experiment_failed"
    source_jobs = [
        {
            "job_id": f"failed-job-{index}",
            "model_id": "shared-model",
            "model_kind": kind,
            "title": f"模型对比 · {index}/2 · {kind}",
            "status": status,
            "dataset_hash": "a" * 64,
            "dataset_spec": _source(),
            "requested_at": f"2026-08-25T01:0{index}:00+00:00",
            "updated_at": f"2026-08-25T01:0{index}:00+00:00",
            "config_json": {
                "model": {"kind": kind, "params": {"seed": 42}},
                "walk_forward": {"enabled": True},
                "execution": {"node_id": "autodl-pro-test-01"},
                "experiment": {
                    "experiment_id": source_experiment_id,
                    "title": "模型对比",
                    "strategy": "model_ensemble",
                    "trial_index": index,
                    "trial_count": 2,
                    "search_params": {"model_kind": kind},
                    "auto_dispatch": True,
                },
            },
        }
        for index, (kind, status) in enumerate(
            (("lightgbm", "canceled"), ("lstm", "canceled")), start=1,
        )
    ]
    repository.get_training_experiment = lambda experiment_id: {
        "experiment_id": experiment_id,
        "title": "模型对比",
        "model_id": "shared-model",
        "jobs": source_jobs,
    }
    captured = []

    def create_job(payload):
        captured.append(payload)
        index = len(captured)
        return {
            "job_id": f"retried-job-{index}",
            "model_id": payload["model_id"],
            "model_kind": payload["model"]["kind"],
            "title": payload["title"],
            "status": "queued",
            "dataset_hash": "a" * 64,
            "dataset_spec": payload["dataset"],
            "requested_at": f"2026-08-25T02:0{index}:00+00:00",
            "updated_at": f"2026-08-25T02:0{index}:00+00:00",
            "config_json": {
                "model": payload["model"],
                "execution": payload["execution"],
                "experiment": payload["experiment"],
            },
        }

    repository.create_training_job = create_job

    restarted = repository.restart_training_experiment(source_experiment_id)

    assert restarted["experiment_id"] != source_experiment_id
    assert restarted["restarted_from_experiment_id"] == source_experiment_id
    assert restarted["statuses"] == {"queued": 2}
    assert [item["model"]["kind"] for item in captured] == ["lightgbm", "lstm"]
    assert all(
        item["experiment"]["experiment_id"] == restarted["experiment_id"]
        and item["experiment"]["auto_dispatch"] is True
        for item in captured
    )
    assert all(item["dataset"] == _source() for item in captured)


def test_horizon_experiment_creates_separate_frozen_datasets() -> None:
    repository = object.__new__(ModelResearchRepository)
    captured = []

    def create_job(payload):
        captured.append(payload)
        index = len(captured)
        dataset = payload["dataset"]
        return {
            "job_id": f"horizon-job-{index}",
            "model_id": payload["model_id"],
            "model_kind": payload["model"]["kind"],
            "status": "queued",
            "dataset_hash": str(index) * 64,
            "dataset_spec": dataset,
            "requested_at": f"2026-08-15T01:0{index}:00+00:00",
            "updated_at": f"2026-08-15T01:0{index}:00+00:00",
            "config_json": {"experiment": payload["experiment"]},
        }

    repository.create_training_job = create_job
    result = repository.create_training_experiment({
        "title": "多周期研究",
        "model_id": "horizon-model",
        "dataset": _source(),
        "model": {"kind": "lightgbm", "params": {}},
        "search": {"strategy": "horizon_grid", "horizons": [1, 5, 10]},
    })

    assert result["strategy"] == "horizon_grid"
    assert result["shared_dataset"] is False
    assert result["dataset_count"] == 3
    assert result["label_horizons"] == [1, 5, 10]
    assert result["search_parameters"] == ["label_horizon_trading_days"]
    assert [
        item["dataset"]["label"]["horizon_trading_days"] for item in captured
    ] == [1, 5, 10]
    assert [item["dataset"]["split"]["embargo_days"] for item in captured] == [1, 5, 10]
    assert [item["walk_forward"]["embargo_sessions"] for item in captured] == [1, 5, 10]
    assert [
        item["experiment"]["search_params"]["label_horizon_trading_days"]
        for item in captured
    ] == [1, 5, 10]


def test_factor_ablation_builds_baseline_and_single_removal_datasets() -> None:
    source = {
        **_source(),
        "factors": [
            {
                "factor_id": factor_id,
                "factor_version": 1,
                "params_hash": factor_id * 8,
                "params": {},
            }
            for factor_id in ("mom", "value", "risk")
        ],
    }

    trials = _factor_ablation_trials(
        source,
        {"factor_ids": ["value", "mom", "value"], "max_trials": 3},
    )

    assert [
        item["search_params"]["removed_factor_id"] for item in trials
    ] == ["__baseline__", "value", "mom"]
    assert [
        [factor["factor_id"] for factor in item["dataset"]["factors"]]
        for item in trials
    ] == [
        ["mom", "value", "risk"],
        ["mom", "risk"],
        ["value", "risk"],
    ]


def test_factor_ablation_rejects_unknown_or_oversized_selection() -> None:
    source = {
        **_source(),
        "factors": [
            {
                "factor_id": "mom", "factor_version": 1,
                "params_hash": "a" * 16, "params": {},
            },
            {
                "factor_id": "risk", "factor_version": 1,
                "params_hash": "b" * 16, "params": {},
            },
        ],
    }

    with pytest.raises(ModelResearchError, match="不存在"):
        _factor_ablation_trials(source, {"factor_ids": ["unknown"]})
    with pytest.raises(ModelResearchError, match="超过本次上限"):
        _factor_ablation_trials(
            source, {"factor_ids": ["mom", "risk"], "max_trials": 2},
        )


def test_factor_ablation_experiment_freezes_variants_and_selection_policy() -> None:
    repository = object.__new__(ModelResearchRepository)
    captured = []
    source = {
        **_source(),
        "factors": [
            {
                "factor_id": factor_id,
                "factor_version": 1,
                "params_hash": (factor_id * 16)[:16],
                "params": {},
            }
            for factor_id in ("mom", "value", "risk")
        ],
    }

    def create_job(payload):
        captured.append(payload)
        index = len(captured)
        return {
            "job_id": f"ablation-job-{index}",
            "model_id": payload["model_id"],
            "model_kind": payload["model"]["kind"],
            "status": "queued",
            "dataset_hash": str(index) * 64,
            "dataset_spec": payload["dataset"],
            "requested_at": f"2026-08-15T02:0{index}:00+00:00",
            "updated_at": f"2026-08-15T02:0{index}:00+00:00",
            "config_json": {"experiment": payload["experiment"]},
        }

    repository.create_training_job = create_job
    result = repository.create_training_experiment({
        "title": "因子消融",
        "model_id": "ablation-model",
        "dataset": source,
        "model": {"kind": "lightgbm", "params": {}},
        "search": {
            "strategy": "factor_ablation",
            "factor_ids": ["value", "mom"],
            "max_trials": 3,
        },
    })

    assert result["strategy"] == "factor_ablation"
    assert result["parent_experiment_id"] == ""
    assert result["iteration"] == 1
    assert result["dataset_count"] == 3
    assert result["shared_dataset"] is False
    assert result["search_parameters"] == ["removed_factor_id"]
    assert result["selection"]["policy"] == "alphablocks.factor-ablation-selection.v1"
    assert [
        item["experiment"]["search_params"]["removed_factor_id"]
        for item in captured
    ] == ["__baseline__", "value", "mom"]
    assert [len(item["dataset"]["factors"]) for item in captured] == [3, 2, 2]
    assert result["lineage_trial_count"] == 3
    assert result["lineage_trial_remaining"] == 21


def test_model_ensemble_experiment_trains_each_selected_kind_once() -> None:
    repository = object.__new__(ModelResearchRepository)
    captured = []

    def create_job(payload):
        captured.append(payload)
        index = len(captured)
        return {
            "job_id": f"ensemble-job-{index}",
            "model_id": payload["model_id"],
            "model_kind": payload["model"]["kind"],
            "status": "queued",
            "dataset_hash": "0" * 64,
            "dataset_spec": payload["dataset"],
            "requested_at": f"2026-08-15T04:0{index}:00+00:00",
            "updated_at": f"2026-08-15T04:0{index}:00+00:00",
            "config_json": {"experiment": payload["experiment"]},
        }

    repository.create_training_job = create_job
    result = repository.create_training_experiment({
        "title": "多模型对比",
        "model_id": "ensemble-model",
        "dataset": _source(),
        "model": {"kind": "lightgbm", "params": {}},
        "search": {
            "strategy": "model_ensemble",
            "model_kinds": ["lightgbm", "xgboost", "mlp"],
            "model_params_by_kind": {
                "lightgbm": {"num_leaves": 15},
                "xgboost": {"max_depth": 4},
                "mlp": {"hidden_layers": [32, 64]},
            },
            "model_versions_by_kind": {
                "lightgbm": 1, "xgboost": 1, "mlp": 1,
            },
        },
    })

    assert result["strategy"] == "model_ensemble"
    assert result["shared_dataset"] is True
    assert result["dataset_count"] == 1
    assert result["trial_count"] == 3
    assert result["search_parameters"] == ["model_kind"]
    assert result["selection"]["policy"] == "alphablocks.model-ensemble-selection.v2"
    assert result["selection"]["selection_unit"] == "model_kind"
    assert [
        item["model"]["kind"] for item in captured
    ] == ["lightgbm", "xgboost", "mlp"]
    assert [item["model"]["version"] for item in captured] == [1, 1, 1]
    assert [
        item["experiment"]["search_params"]["model_kind"] for item in captured
    ] == ["lightgbm", "xgboost", "mlp"]
    assert captured[0]["model"]["params"]["num_leaves"] == 15
    assert captured[1]["model"]["params"]["max_depth"] == 4
    assert captured[2]["model"]["params"]["hidden_layers"] == [32, 64]
    assert all(
        item["dataset"] == _dataset_spec(_source()) for item in captured
    )


def test_model_ensemble_summary_selects_best_qualified_absolute_validation_icir() -> None:
    def job(index: int, kind: str, *, rank_ic: float, ic_ir: float) -> dict:
        return {
            "job_id": f"ensemble-job-{index}",
            "model_id": "ensemble-model",
            "model_version": index,
            "model_kind": kind,
            "status": "succeeded",
            "dataset_hash": "0" * 64,
            "dataset_spec": _dataset_spec(_source()),
            "requested_at": f"2026-08-15T04:0{index}:00+00:00",
            "updated_at": f"2026-08-15T05:0{index}:00+00:00",
            "config_json": {"experiment": {
                "experiment_id": "model_experiment_ensemble",
                "title": "多模型对比",
                "strategy": "model_ensemble",
                "trial_index": index,
                "trial_count": 3,
                "search_params": {"model_kind": kind},
            }},
            "result_json": {"metrics": {"validation": {
                "days": 60,
                "rank_ic": rank_ic,
                "ic_ir": ic_ir,
                "rmse": 0.7,
            }}},
        }

    result = _experiment_summary("model_experiment_ensemble", [
        job(1, "lightgbm", rank_ic=0.07, ic_ir=0.25),
        job(2, "xgboost", rank_ic=0.03, ic_ir=-0.72),
        job(3, "mlp", rank_ic=0.05, ic_ir=0.61),
    ])

    assert result["selection"]["selected_job_id"] == "ensemble-job-3"
    assert result["selection"]["best_observed_job_id"] == "ensemble-job-2"
    assert result["selection"]["selected_model_kind"] == "mlp"
    assert result["comparison"]["selection_metric"] == "abs(validation.ic_ir)"
    assert [
        item["job_id"] for item in result["comparison"]["trials"]
        if item["is_selected"]
    ] == ["ensemble-job-3"]


def test_model_ensemble_rejects_misaligned_version_identity_map() -> None:
    repository = object.__new__(ModelResearchRepository)
    repository.create_training_job = lambda _payload: pytest.fail(
        "invalid version map must fail before creating jobs"
    )

    with pytest.raises(ModelResearchError, match="逐项对齐"):
        repository.create_training_experiment({
            "dataset": _source(),
            "model": {"kind": "lightgbm", "version": 1, "params": {}},
            "search": {
                "strategy": "model_ensemble",
                "model_kinds": ["lightgbm", "xgboost"],
                "model_params_by_kind": {"lightgbm": {}, "xgboost": {}},
                "model_versions_by_kind": {"lightgbm": 1},
            },
        })


def test_model_ensemble_stacking_creates_one_composite_trial() -> None:
    repository = object.__new__(ModelResearchRepository)
    captured = []

    def create_job(payload):
        captured.append(payload)
        return {
            "job_id": "stacking-job-1",
            "model_id": payload["model_id"],
            "model_kind": payload["model"]["kind"],
            "status": "queued",
            "dataset_hash": "0" * 64,
            "dataset_spec": payload["dataset"],
            "requested_at": "2026-08-15T04:01:00+00:00",
            "updated_at": "2026-08-15T04:01:00+00:00",
            "config_json": {"experiment": payload["experiment"]},
        }

    repository.create_training_job = create_job
    result = repository.create_training_experiment({
        "title": "树模型 Stacking",
        "model_id": "stacking-model",
        "dataset": _source(),
        "model": {"kind": "lightgbm", "params": {}},
        "search": {
            "strategy": "model_ensemble",
            "model_kinds": ["lightgbm", "xgboost", "linear"],
            "model_params_by_kind": {
                "lightgbm": {"num_leaves": 15},
                "xgboost": {"max_depth": 4},
                "linear": {"alpha": 3.0},
            },
            "ensemble_method": "stacking",
            "n_folds": 4,
            "meta_alpha": 2.5,
        },
    })

    assert result["trial_count"] == 1
    assert len(captured) == 1
    model = captured[0]["model"]
    assert model["kind"] == "stacking"
    assert model["params"]["n_folds"] == 4
    assert model["params"]["meta_alpha"] == 2.5
    assert model["ensemble"]["family"] == "classical"
    assert [item["kind"] for item in model["base_models"]] == [
        "lightgbm", "xgboost", "linear",
    ]
    assert captured[0]["experiment"]["search_params"]["ensemble_method"] == "stacking"


def test_model_ensemble_stacking_inherits_classification_target() -> None:
    repository = object.__new__(ModelResearchRepository)
    captured = []

    def create_job(payload):
        captured.append(payload)
        return {
            "job_id": "classification-stacking-job",
            "model_id": payload["model_id"],
            "model_kind": payload["model"]["kind"],
            "status": "queued",
            "dataset_hash": "0" * 64,
            "dataset_spec": payload["dataset"],
            "requested_at": "2026-08-15T04:01:00+00:00",
            "updated_at": "2026-08-15T04:01:00+00:00",
            "config_json": {"experiment": payload["experiment"]},
        }

    repository.create_training_job = create_job
    repository.create_training_experiment({
        "dataset": {**_source(), "target_mode": "classification"},
        "search": {
            "strategy": "model_ensemble",
            "model_kinds": ["lightgbm", "xgboost"],
            "model_params_by_kind": {"lightgbm": {}, "xgboost": {}},
            "ensemble_method": "stacking",
        },
    })

    model = captured[0]["model"]
    assert model["params"]["loss"] == "binary"
    assert model["params"]["objective"] == "binary"
    assert model["params"]["metric"] == "auc"
    assert all(
        item["params"]["loss"] == "binary" for item in model["base_models"]
    )


def test_model_ensemble_stacking_rejects_cross_family_selection() -> None:
    repository = object.__new__(ModelResearchRepository)
    repository.create_training_job = lambda _payload: None

    with pytest.raises(ModelResearchError, match="同一模型族"):
        repository.create_training_experiment({
            "dataset": _source(),
            "search": {
                "strategy": "model_ensemble",
                "model_kinds": ["lightgbm", "lstm"],
                "model_params_by_kind": {"lightgbm": {}, "lstm": {}},
                "ensemble_method": "stacking",
            },
        })


def test_model_ensemble_rejects_too_few_or_unknown_kinds() -> None:
    repository = object.__new__(ModelResearchRepository)
    repository.create_training_job = lambda _payload: None

    with pytest.raises(ModelResearchError, match="2到8"):
        repository.create_training_experiment({
            "dataset": _source(),
            "search": {
                "strategy": "model_ensemble",
                "model_kinds": ["lightgbm"],
                "model_params_by_kind": {"lightgbm": {}},
            },
        })
    with pytest.raises(ModelResearchError, match="只允许"):
        repository.create_training_experiment({
            "dataset": _source(),
            "search": {
                "strategy": "model_ensemble",
                "model_kinds": ["lightgbm", "unknown_model"],
                "model_params_by_kind": {"lightgbm": {}, "unknown_model": {}},
            },
        })


def test_model_spec_supports_all_available_model_kinds() -> None:
    expected = {
        "lightgbm": "qlib.contrib.model.gbdt.LGBModel",
        "xgboost": "qlib.contrib.model.xgboost.XGBModel",
        "catboost": "qlib.contrib.model.catboost_model.CatBoostModel",
        "random_forest": "factor_service.research.models.QlibSklearnRandomForestModel",
        "linear": "factor_service.research.models.QlibSklearnRidgeModel",
        "mlp": "factor_service.research.models.QlibTorchMLPModel",
        "gru": "factor_service.research.models.QlibTorchGRUModel",
        "lstm": "factor_service.research.models.QlibTorchLSTMModel",
        "alstm": "factor_service.research.models.QlibTorchALSTMModel",
        "transformer": "factor_service.research.models.QlibTorchTransformerModel",
        "tabnet": "factor_service.research.models.QlibNativeTabNetAdapter",
        "tcn": "factor_service.research.models.QlibTorchTCNModel",
        "nativetft": "factor_service.research.models.QlibTorchNativeTFTModel",
        "transformer_lstm": "factor_service.research.models.QlibTorchTransformerLSTMModel",
    }
    for kind, qlib_model in expected.items():
        spec = _model_spec({"kind": kind, "params": {}})
        assert spec["kind"] == kind
        assert spec["qlib_model"] == qlib_model
        assert spec["params"]["seed"] == 42
        assert spec["params"]["num_threads"] == 4
    with pytest.raises(ModelResearchError, match="只允许"):
        _model_spec({"kind": "unknown_model", "params": {}})


def test_model_spec_matches_quantmind_hyperparameter_contract() -> None:
    lightgbm = _model_spec({"kind": "lightgbm", "params": {}})["params"]
    assert lightgbm["learning_rate"] == 0.02
    assert lightgbm["n_estimators"] == 2000
    assert lightgbm["min_data_in_leaf"] == 300
    assert lightgbm["min_child_samples"] == 150
    assert lightgbm["path_smooth"] == 1.0
    assert lightgbm["bagging_freq"] == 5
    assert lightgbm["lambda_l1"] == 0.5
    assert lightgbm["lambda_l2"] == 1.0
    assert lightgbm["feature_fraction"] == 0.7
    assert lightgbm["bagging_fraction"] == 0.8

    xgboost = _model_spec({"kind": "xgboost", "params": {}})["params"]
    assert xgboost["max_depth"] == 4
    assert xgboost["subsample"] == 0.7
    assert xgboost["colsample_bytree"] == 0.65
    assert xgboost["reg_alpha"] == 0.5
    assert xgboost["reg_lambda"] == 2.0
    assert xgboost["min_child_weight"] == 100.0

    catboost = _model_spec({"kind": "catboost", "params": {}})["params"]
    assert catboost["random_strength"] == 1.5
    assert catboost["bagging_temperature"] == 0.8
    assert catboost["od_wait"] == 100

    random_forest = _model_spec({"kind": "random_forest", "params": {}})["params"]
    assert random_forest["n_estimators"] == 300
    assert random_forest["max_depth"] == 0
    assert random_forest["max_features"] == "sqrt"

    linear = _model_spec({"kind": "linear", "params": {}})["params"]
    assert linear["alpha"] == 3.0
    assert linear["fit_intercept"] is True
    mlp = _model_spec({
        "kind": "mlp", "params": {"hidden_size": 128, "layer_count": 3},
    })["params"]
    assert mlp["hidden_size"] == 128
    assert mlp["layer_count"] == 3

    gru = _model_spec({"kind": "gru", "params": {}})["params"]
    expected_gru = {
        "learning_rate": 0.001,
        "lookback_window": 20,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "max_steps": 200,
        "batch_size": 4000,
    }
    assert {key: gru[key] for key in expected_gru} == expected_gru
    assert gru["early_stopping_rounds"] == 20
    transformer = _model_spec({"kind": "transformer", "params": {}})["params"]
    assert transformer["learning_rate"] == 0.0001
    assert transformer["lookback_window"] == 20
    assert transformer["batch_size"] == 4000
    assert _model_spec({"kind": "tabnet", "params": {}})["params"][
        "learning_rate"
    ] == 0.005
    tcn = _model_spec({"kind": "tcn", "params": {}})["params"]
    assert tcn["hidden_size"] == 128
    assert tcn["num_layers"] == 2
    assert tcn["dropout"] == 0.2
    assert _model_spec({"kind": "nativetft", "params": {}})["params"][
        "learning_rate"
    ] == 0.0005


def test_next_ablation_inherits_selected_parent_and_freezes_lineage_budget() -> None:
    repository = object.__new__(ModelResearchRepository)
    captured = []
    source = {
        **_source(),
        "factors": [
            {
                "factor_id": factor_id,
                "factor_version": 1,
                "params_hash": (factor_id * 16)[:16],
                "params": {},
            }
            for factor_id in ("mom", "value", "risk")
        ],
    }
    normalized_dataset = _dataset_spec(source)
    normalized_model = _model_spec({"kind": "lightgbm", "params": {}})
    normalized_walk_forward = _walk_forward_spec({})
    parent_job = {
        "job_id": "model_job_parent",
        "model_id": "ablation-model",
        "model_kind": "lightgbm",
        "status": "succeeded",
        "dataset_spec": normalized_dataset,
        "config_json": {
            "model": normalized_model,
            "walk_forward": normalized_walk_forward,
        },
    }
    parent_summary = {
        "experiment_id": "model_experiment_parent",
        "strategy": "factor_ablation",
        "iteration": 1,
        "trial_count": 3,
        "parent_experiment_id": "",
        "selection": {
            "status": "selected",
            "selected_job_id": "model_job_parent",
        },
        "jobs": [parent_job],
    }
    repository.get_training_experiment = lambda _experiment_id: parent_summary

    def create_job(payload):
        captured.append(payload)
        index = len(captured)
        return {
            "job_id": f"next-ablation-job-{index}",
            "model_id": payload["model_id"],
            "model_kind": payload["model"]["kind"],
            "status": "queued",
            "dataset_hash": str(index) * 64,
            "dataset_spec": payload["dataset"],
            "requested_at": f"2026-08-15T03:0{index}:00+00:00",
            "updated_at": f"2026-08-15T03:0{index}:00+00:00",
            "config_json": {"experiment": payload["experiment"]},
        }

    repository.create_training_job = create_job
    result = repository.create_training_experiment({
        "title": "第二轮因子消融",
        "model_id": "ablation-model",
        "parent_experiment_id": "model_experiment_parent",
        "parent_job_id": "model_job_parent",
        "iteration": 2,
        "dataset": normalized_dataset,
        "model": normalized_model,
        "walk_forward": normalized_walk_forward,
        "search": {
            "strategy": "factor_ablation",
            "factor_ids": ["value", "risk"],
            "max_trials": 3,
        },
    })

    assert result["parent_experiment_id"] == "model_experiment_parent"
    assert result["parent_job_id"] == "model_job_parent"
    assert result["iteration"] == 2
    assert result["lineage_prior_trial_count"] == 3
    assert result["lineage_trial_count"] == 6
    assert result["lineage_trial_remaining"] == 18
    assert all(item["experiment"]["iteration"] == 2 for item in captured)

    with pytest.raises(ModelResearchConflict, match="parent_job_id"):
        repository.create_training_experiment({
            "title": "错误父任务",
            "model_id": "ablation-model",
            "parent_experiment_id": "model_experiment_parent",
            "parent_job_id": "model_job_not_selected",
            "iteration": 2,
            "dataset": normalized_dataset,
            "model": normalized_model,
            "walk_forward": normalized_walk_forward,
            "search": {
                "strategy": "factor_ablation",
                "factor_ids": ["risk"],
            },
        })

    with pytest.raises(ModelResearchConflict, match="冻结因子"):
        repository.create_training_experiment({
            "title": "篡改继承",
            "model_id": "ablation-model",
            "parent_experiment_id": "model_experiment_parent",
            "parent_job_id": "model_job_parent",
            "iteration": 2,
            "dataset": {**normalized_dataset, "date_end": "2023-12-31"},
            "model": normalized_model,
            "walk_forward": normalized_walk_forward,
            "search": {
                "strategy": "factor_ablation",
                "factor_ids": ["risk"],
            },
        })


def test_next_ablation_rejects_trials_beyond_the_lineage_budget() -> None:
    repository = object.__new__(ModelResearchRepository)
    source = {
        **_source(),
        "factors": [
            {
                "factor_id": factor_id,
                "factor_version": 1,
                "params_hash": (factor_id * 16)[:16],
                "params": {},
            }
            for factor_id in ("mom", "value")
        ],
    }
    dataset = _dataset_spec(source)
    model = _model_spec({"kind": "lightgbm", "params": {}})
    walk_forward = _walk_forward_spec({})
    parent_job = {
        "job_id": "model_job_parent",
        "model_id": "ablation-model",
        "status": "succeeded",
        "dataset_spec": dataset,
        "config_json": {"model": model, "walk_forward": walk_forward},
    }
    summaries = {
        "model_experiment_parent": {
            "strategy": "factor_ablation",
            "iteration": 2,
            "trial_count": 11,
            "parent_experiment_id": "model_experiment_root",
            "selection": {
                "status": "selected",
                "selected_job_id": "model_job_parent",
            },
            "jobs": [parent_job],
        },
        "model_experiment_root": {
            "strategy": "factor_ablation",
            "iteration": 1,
            "trial_count": 12,
            "parent_experiment_id": "",
            "selection": {"status": "selected"},
            "jobs": [],
        },
    }
    repository.get_training_experiment = lambda experiment_id: summaries[experiment_id]
    repository.create_training_job = lambda _payload: pytest.fail(
        "超预算实验不应创建任何任务"
    )

    with pytest.raises(ModelResearchConflict, match="累计最多允许24组"):
        repository.create_training_experiment({
            "title": "第三轮超预算消融",
            "model_id": "ablation-model",
            "parent_experiment_id": "model_experiment_parent",
            "parent_job_id": "model_job_parent",
            "iteration": 3,
            "dataset": dataset,
            "model": model,
            "walk_forward": walk_forward,
            "search": {
                "strategy": "factor_ablation",
                "factor_ids": ["value"],
                "max_trials": 2,
            },
        })


def test_experiment_summary_contains_history_card_metadata() -> None:
    jobs = [{
        "job_id": "job-1",
        "model_id": "grid-model",
        "model_version": 1,
        "model_kind": "lightgbm",
        "dataset_hash": "a" * 64,
        "dataset_spec": {
            "date_start": "2024-01-02", "date_end": "2024-12-31",
            "factors": [{"factor_id": "mom_20"}],
        },
        "requested_at": "2026-08-15T01:00:00+00:00",
        "updated_at": "2026-08-15T01:05:00+00:00",
        "status": "succeeded",
        "config_json": {"experiment": {
            "experiment_id": "model_experiment_test",
            "title": "参数历史",
            "trial_index": 1,
            "search_params": {"num_leaves": 31, "learning_rate": 0.05},
        }},
        "result_json": {"metrics": {
            "test_days": 80, "test_rows": 30000, "rank_ic": 0.03,
            "ic": 0.03, "ic_ir": 0.4, "rmse": 0.45,
            "validation": {
                "days": 60, "rows": 24000, "rank_ic": 0.04,
                "ic": 0.04, "ic_ir": 0.5, "rmse": 0.4,
            },
        }},
    }]

    result = _experiment_summary("model_experiment_test", jobs)

    assert result["model_kind"] == "lightgbm"
    assert result["factor_count"] == 1
    assert result["search_parameters"] == ["learning_rate", "num_leaves"]
    assert result["selection"]["status"] == "selected"
    assert result["selection"]["selected_model_version"] == 1
    comparison = result["comparison"]
    assert comparison["selection_split"] == "validation"
    assert comparison["test_metrics_role"] == "report_only"
    assert comparison["test_metrics_disclosed"] is True
    assert comparison["trials"][0]["validation"]["rank_ic"] == 0.04
    assert comparison["trials"][0]["test"]["rank_ic"] == 0.03
    assert comparison["trials"][0]["gate_passed"] is True
    assert comparison["trials"][0]["is_selected"] is True


def test_experiment_comparison_hides_test_metrics_until_all_trials_finish() -> None:
    jobs = [
        {
            "job_id": "job-1", "model_id": "grid-model", "model_version": 1,
            "status": "succeeded", "requested_at": "2026-08-15T01:00:00+00:00",
            "updated_at": "2026-08-15T01:01:00+00:00",
            "config_json": {"experiment": {"trial_index": 1}},
            "result_json": {"metrics": {
                "rank_ic": 0.08,
                "validation": {"days": 60, "rank_ic": 0.04, "ic_ir": 0.5},
            }},
        },
        {
            "job_id": "job-2", "model_id": "grid-model", "status": "running",
            "requested_at": "2026-08-15T01:02:00+00:00",
            "updated_at": "2026-08-15T01:03:00+00:00",
            "config_json": {"experiment": {"trial_index": 2}},
            "result_json": {},
        },
    ]

    result = _experiment_summary("model_experiment_running", jobs)

    assert result["selection"]["status"] == "evaluating"
    assert result["comparison"]["test_metrics_disclosed"] is False
    assert result["comparison"]["trials"][0]["validation"]["rank_ic"] == 0.04
    assert result["comparison"]["trials"][0]["test"] is None


def test_experiment_parameter_effects_are_ranked_only_by_validation_metrics() -> None:
    jobs = []
    combinations = [
        (0.03, 15, 0.02, 0.80),
        (0.03, 31, 0.04, 0.70),
        (0.05, 15, 0.06, -0.40),
        (0.05, 31, 0.08, -0.30),
    ]
    for index, (learning_rate, num_leaves, validation_ic, test_ic) in enumerate(
        combinations, start=1,
    ):
        jobs.append({
            "job_id": f"job-{index}",
            "model_id": "grid-model",
            "model_version": index,
            "model_kind": "lightgbm",
            "status": "succeeded",
            "requested_at": f"2026-08-15T01:0{index}:00+00:00",
            "updated_at": f"2026-08-15T01:1{index}:00+00:00",
            "config_json": {"experiment": {
                "trial_index": index,
                "search_params": {
                    "learning_rate": learning_rate,
                    "num_leaves": num_leaves,
                },
            }},
            "result_json": {"metrics": {
                "rank_ic": test_ic,
                "ic_ir": test_ic,
                "validation": {
                    "days": 60,
                    "rank_ic": validation_ic,
                    "ic_ir": 0.5,
                },
            }},
        })

    comparison = _experiment_summary(
        "model_experiment_parameter_effects", jobs,
    )["comparison"]

    assert comparison["summary"]["qualified_count"] == 4
    assert comparison["summary"]["qualified_ratio"] == 1.0
    effects = {
        item["parameter"]: item for item in comparison["parameter_effects"]
    }
    learning_rate = effects["learning_rate"]
    assert learning_rate["best_value"] == 0.05
    assert learning_rate["validation_rank_ic_spread"] == pytest.approx(0.04)
    best_value = next(
        item for item in learning_rate["values"] if item["is_best_validation"]
    )
    assert best_value["validation_rank_ic"]["mean"] == pytest.approx(0.07)
    # 测试集表现只做报告；即便方向与验证集相反，也不能改变最佳参数值。
    assert best_value["test_rank_ic"]["mean"] == pytest.approx(-0.35)
    assert effects["num_leaves"]["best_value"] == 31


def test_grid_search_rejects_oversized_or_unsupported_search() -> None:
    with pytest.raises(ModelResearchError, match="超过本次上限"):
        _grid_search_trials(
            {"kind": "lightgbm", "params": {}},
            {
                "parameters": {
                    "learning_rate": [0.01, 0.03, 0.05],
                    "num_leaves": [15, 31],
                },
                "max_trials": 4,
            },
        )
    with pytest.raises(ModelResearchError, match="不允许搜索参数"):
        _grid_search_trials(
            {"kind": "lightgbm", "params": {}},
            {"parameters": {"hidden_layers": [[64, 128]]}},
        )
