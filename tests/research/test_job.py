from __future__ import annotations

from hashlib import sha256
import json
from copy import deepcopy
import os
from pathlib import Path
import threading

import pytest

from factor_service.research.errors import (
    JobCanceled,
    PermanentJobError,
    TrainingTimeout,
    WorkerShutdown,
)
from factor_service.research.job import (
    CancellationToken, _canonical_json, safe_job_dir, validate_job,
)
from factor_service.research.industry_feature import normalize_industry_feature
from factor_service.research.sample_filter_formula import (
    normalize_custom_sample_filters,
)
from factor_service.research.preprocessing import normalize_feature_preprocessing
from factor_service.research.training_resource_settings import (
    BINDING_DEFINITIONS,
    INDEX_MEMBERSHIP_BINDING_ID,
    SECURITY_MASTER_BINDING_ID,
    STOCK_DAILY_BINDING_ID,
    STOCK_STATUS_BINDING_ID,
    TRADING_CALENDAR_BINDING_ID,
    frozen_training_data_bindings,
    normalize_training_resource_settings,
)
from factor_service.research.state import JobStateStore
from tests.research.utils import valid_inference_job, valid_job


def test_job_validation_accepts_frozen_contract() -> None:
    job = validate_job(valid_job())

    assert job["model_id"] == "test_model"
    assert job["dataset_spec"]["universe_id"] == "csi500"


def test_job_validation_requires_parallel_work_for_multi_node():
    from factor_service.research.distributed_config import execution_spec
    source = valid_job()
    source['config_json']['execution'] = execution_spec({'node_ids': ['cpu-a', 'cpu-b']})
    with pytest.raises(PermanentJobError, match='需要开启Optuna'):
        validate_job(source)
    source['config_json']['optuna'] = {'enabled': True, 'n_trials': 10}
    assert validate_job(source)['config_json']['execution']['node_ids'] == ['cpu-a', 'cpu-b']


def test_job_validation_accepts_bounded_training_runtime() -> None:
    source = valid_job()
    source["config_json"]["execution"] = {
        "node_id": "local", "mode": "local", "max_runtime_minutes": 720,
    }

    job = validate_job(source)

    assert job["config_json"]["execution"]["max_runtime_minutes"] == 720

    source["config_json"]["execution"]["max_runtime_minutes"] = 30
    with pytest.raises(PermanentJobError, match="max_runtime_minutes"):
        validate_job(source)


def test_cancellation_token_stops_training_at_deadline(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("factor_service.research.job.time.monotonic", lambda: clock[0])
    token = CancellationToken(timeout_seconds=60)

    token.checkpoint()
    clock[0] = 160.0

    with pytest.raises(TrainingTimeout, match="最长运行时长"):
        token.checkpoint()


def test_job_validation_accepts_and_validates_sample_filters() -> None:
    source = valid_job()
    filters = {
        "minimum_listing_trading_days": 60,
        "exclude_st": True,
        "exclude_delisting": True,
    }
    source["dataset_spec"]["sample_filters"] = filters
    source["config_json"]["dataset"]["sample_filters"] = deepcopy(filters)
    source["dataset_hash"] = sha256(
        _canonical_json(source["dataset_spec"]).encode("utf-8")
    ).hexdigest()

    job = validate_job(source)

    assert job["dataset_spec"]["sample_filters"] == filters

    source["dataset_spec"]["sample_filters"]["exclude_st"] = "yes"
    source["config_json"]["dataset"]["sample_filters"]["exclude_st"] = "yes"
    source["dataset_hash"] = sha256(
        _canonical_json(source["dataset_spec"]).encode("utf-8")
    ).hexdigest()
    with pytest.raises(PermanentJobError, match="exclude_st必须是布尔值"):
        validate_job(source)


def _frozen_core_bindings() -> dict:
    binding_ids = [
        STOCK_DAILY_BINDING_ID,
        SECURITY_MASTER_BINDING_ID,
        INDEX_MEMBERSHIP_BINDING_ID,
        TRADING_CALENDAR_BINDING_ID,
        STOCK_STATUS_BINDING_ID,
    ]
    bindings = {}
    for binding_id in binding_ids:
        roles = BINDING_DEFINITIONS[binding_id]["roles"]
        bindings[binding_id] = {
            "enabled": True,
            "source_type": "node",
            "source_id": f"{binding_id}_node",
            "source_label": binding_id,
            "provider_node_id": f"{binding_id}_node",
            "field_bindings": {
                role["id"]: role["id"] for role in roles
            },
            "catalog_updated_at": "",
        }
    settings = normalize_training_resource_settings({"bindings": bindings})
    settings["revision"] = 1
    return frozen_training_data_bindings(settings, binding_ids)


def test_job_validation_requires_normalized_dataset_contracts_for_v8() -> None:
    source = valid_job()
    preprocessing = normalize_feature_preprocessing(
        {"enabled": True}, default_enabled=False,
    )
    industry_feature = normalize_industry_feature(
        {"enabled": False}, default_enabled=False,
    )
    for target in (source["dataset_spec"], source["config_json"]["dataset"]):
        target["pipeline_version"] = "alphablocks.dataset-pipeline.v8"
        target["preprocessing"] = deepcopy(preprocessing)
        target["industry_feature"] = deepcopy(industry_feature)
        target["data_bindings"] = deepcopy(_frozen_core_bindings())
    source["dataset_hash"] = sha256(
        _canonical_json(source["dataset_spec"]).encode("utf-8")
    ).hexdigest()

    validated = validate_job(source)

    assert validated["dataset_spec"]["preprocessing"] == preprocessing

    for target in (source["dataset_spec"], source["config_json"]["dataset"]):
        target.pop("preprocessing")
    source["dataset_hash"] = sha256(
        _canonical_json(source["dataset_spec"]).encode("utf-8")
    ).hexdigest()
    with pytest.raises(PermanentJobError, match="v8数据集缺少"):
        validate_job(source)

    for target in (source["dataset_spec"], source["config_json"]["dataset"]):
        target["preprocessing"] = deepcopy(preprocessing)
        target.pop("industry_feature")
    source["dataset_hash"] = sha256(
        _canonical_json(source["dataset_spec"]).encode("utf-8")
    ).hexdigest()
    with pytest.raises(PermanentJobError, match="v8数据集缺少.*industry_feature"):
        validate_job(source)


def test_job_validation_rejects_partial_preprocessing_contract() -> None:
    source = valid_job()
    partial = {"enabled": True}
    industry_feature = normalize_industry_feature(
        {"enabled": False}, default_enabled=False,
    )
    for target in (source["dataset_spec"], source["config_json"]["dataset"]):
        target["pipeline_version"] = "alphablocks.dataset-pipeline.v7"
        target["preprocessing"] = deepcopy(partial)
        target["industry_feature"] = deepcopy(industry_feature)
    source["dataset_hash"] = sha256(
        _canonical_json(source["dataset_spec"]).encode("utf-8")
    ).hexdigest()

    with pytest.raises(PermanentJobError, match="不是规范化冻结规格"):
        validate_job(source)


def test_job_validation_freezes_industry_contract_and_safe_start() -> None:
    source = valid_job()
    preprocessing = normalize_feature_preprocessing(
        {"enabled": True}, default_enabled=False,
    )
    enabled = normalize_industry_feature(
        {"enabled": True}, default_enabled=False,
    )
    for target in (source["dataset_spec"], source["config_json"]["dataset"]):
        target.update({
            "pipeline_version": "alphablocks.dataset-pipeline.v7",
            "date_start": "2022-01-04",
            "preprocessing": deepcopy(preprocessing),
            "industry_feature": deepcopy(enabled),
        })
    source["dataset_hash"] = sha256(
        _canonical_json(source["dataset_spec"]).encode("utf-8")
    ).hexdigest()

    validated = validate_job(source)

    assert validated["dataset_spec"]["industry_feature"] == enabled

    partial = deepcopy(source)
    for target in (
        partial["dataset_spec"], partial["config_json"]["dataset"],
    ):
        target["industry_feature"] = {"enabled": True}
    partial["dataset_hash"] = sha256(
        _canonical_json(partial["dataset_spec"]).encode("utf-8")
    ).hexdigest()
    with pytest.raises(PermanentJobError, match="industry_feature不是规范化"):
        validate_job(partial)

    unsafe = deepcopy(source)
    for target in (
        unsafe["dataset_spec"], unsafe["config_json"]["dataset"],
    ):
        target["date_start"] = "2021-12-10"
    unsafe["dataset_hash"] = sha256(
        _canonical_json(unsafe["dataset_spec"]).encode("utf-8")
    ).hexdigest()
    with pytest.raises(PermanentJobError, match="2021-12-13"):
        validate_job(unsafe)

    wrong_target = deepcopy(source)
    for target in (
        wrong_target["dataset_spec"],
        wrong_target["config_json"]["dataset"],
    ):
        target["research_target"] = "industry_rotation"
    wrong_target["dataset_hash"] = sha256(
        _canonical_json(wrong_target["dataset_spec"]).encode("utf-8")
    ).hexdigest()
    with pytest.raises(PermanentJobError, match="仅支持个股选股"):
        validate_job(wrong_target)


def test_job_validation_accepts_frozen_custom_sample_filter_formula() -> None:
    source = valid_job()
    formulas = normalize_custom_sample_filters([{
        "name": "站上20日均线",
        "expression": "$close > Mean($close, 20)",
    }])
    for target in (source["dataset_spec"], source["config_json"]["dataset"]):
        target["sample_filters"] = {
            "minimum_listing_trading_days": 60,
            "exclude_st": True,
            "exclude_delisting": True,
            "custom_formulas": deepcopy(formulas),
        }
    source["dataset_hash"] = sha256(
        _canonical_json(source["dataset_spec"]).encode("utf-8")
    ).hexdigest()

    job = validate_job(source)

    assert job["dataset_spec"]["sample_filters"]["custom_formulas"] == formulas


def test_job_validation_accepts_classification_target_with_binary_loss() -> None:
    source = valid_job()
    for target in (source["dataset_spec"], source["config_json"]["dataset"]):
        target["target_mode"] = "classification"
        target["label"] = {
            "kind": "future_5d_direction",
            "mode": "classification",
            "horizon_trading_days": 5,
            "range": [0.0, 1.0],
            "classes": [0, 1],
        }
    source["config_json"]["model"]["params"].update({
        "loss": "binary", "objective": "binary", "metric": "auc",
    })
    source["dataset_hash"] = sha256(
        _canonical_json(source["dataset_spec"]).encode("utf-8")
    ).hexdigest()

    job = validate_job(source)

    assert job["dataset_spec"]["target_mode"] == "classification"
    assert job["config_json"]["model"]["params"]["loss"] == "binary"
    assert job["config_json"]["model"]["params"]["metric"] == "auc"


def test_job_validation_names_unknown_model_parameters() -> None:
    source = valid_job()
    source["config_json"]["model"]["params"]["foreign_parameter"] = 1

    with pytest.raises(PermanentJobError, match="foreign_parameter"):
        validate_job(source)


def test_job_validation_accepts_quantmind_tree_hyperparameters() -> None:
    lightgbm = valid_job()
    lightgbm["config_json"]["model"]["params"].update({
        "min_data_in_leaf": 300,
        "min_child_samples": 150,
        "path_smooth": 1.0,
        "bagging_freq": 5,
        "lambda_l1": 0.5,
        "lambda_l2": 1.0,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
    })
    assert validate_job(lightgbm)["config_json"]["model"]["params"][
        "min_data_in_leaf"
    ] == 300

    catboost = valid_job()
    catboost["config_json"]["model"] = {
        "kind": "catboost",
        "params": {"bagging_temperature": 0.8, "od_wait": 100},
    }
    assert validate_job(catboost)["config_json"]["model"]["params"][
        "bagging_temperature"
    ] == 0.8


def test_job_validation_rejects_classification_target_with_regression_loss() -> None:
    source = valid_job()
    for target in (source["dataset_spec"], source["config_json"]["dataset"]):
        target["target_mode"] = "classification"
        target["label"]["mode"] = "classification"
    source["dataset_hash"] = sha256(
        _canonical_json(source["dataset_spec"]).encode("utf-8")
    ).hexdigest()

    with pytest.raises(PermanentJobError, match="binary"):
        validate_job(source)


def test_job_validation_accepts_all_a_universe() -> None:
    source = valid_job()
    for target in (source["dataset_spec"], source["config_json"]["dataset"]):
        target["universe_id"] = "all_a"
        target["index_code"] = "000985.SH"
    source["dataset_hash"] = sha256(
        _canonical_json(source["dataset_spec"]).encode("utf-8")
    ).hexdigest()

    job = validate_job(source)

    assert job["dataset_spec"]["universe_id"] == "all_a"


def test_job_validation_rejects_unknown_or_mismatched_universe() -> None:
    source = valid_job()
    source["dataset_spec"]["universe_id"] = "csi2000"
    source["config_json"]["dataset"]["universe_id"] = "csi2000"
    source["dataset_hash"] = sha256(
        _canonical_json(source["dataset_spec"]).encode("utf-8")
    ).hexdigest()
    with pytest.raises(PermanentJobError, match="不支持的股票池"):
        validate_job(source)

    candidate = valid_job()
    candidate["dataset_spec"]["universe_id"] = "all_a"
    candidate["config_json"]["dataset"]["universe_id"] = "all_a"
    candidate["dataset_hash"] = sha256(
        _canonical_json(candidate["dataset_spec"]).encode("utf-8")
    ).hexdigest()
    with pytest.raises(PermanentJobError, match="index_code"):
        validate_job(candidate)


def test_job_validation_accepts_all_phase2_model_kinds() -> None:
    source = valid_job()
    cases = {
        "xgboost": {"max_depth": 4, "n_estimators": 20},
        "catboost": {"depth": 4, "n_estimators": 20},
        "random_forest": {"max_depth": 4, "n_estimators": 20},
        "linear": {"alpha": 1.0, "max_iter": 100},
        "mlp": {"hidden_layers": [16, 32, 64], "max_steps": 20, "batch_size": 32},
        "gru": {
            "lookback_window": 20, "hidden_size": 32, "num_layers": 2,
            "dropout": 0.1, "max_steps": 20, "batch_size": 32,
        },
        "lstm": {
            "lookback_window": 20, "hidden_size": 32, "num_layers": 2,
            "dropout": 0.1, "max_steps": 20, "batch_size": 32,
        },
        "alstm": {
            "lookback_window": 20, "hidden_size": 32, "num_layers": 2,
            "dropout": 0.1, "max_steps": 20, "batch_size": 32,
        },
        "transformer": {
            "lookback_window": 20, "d_model": 32, "nhead": 4,
            "transformer_layers": 1, "dim_feedforward": 64,
            "dropout": 0.1, "max_steps": 20, "batch_size": 32,
        },
        "tabnet": {
            "n_d": 16, "n_a": 16, "n_steps": 3,
            "max_steps": 20, "batch_size": 32,
        },
        "tcn": {
            "lookback_window": 20, "hidden_size": 32, "kernel_size": 3,
            "num_layers": 2, "dropout": 0.1,
            "max_steps": 20, "batch_size": 32,
        },
        "nativetft": {
            "lookback_window": 20, "d_model": 32, "nhead": 4,
            "gru_hidden_size": 32, "num_layers": 1, "dim_feedforward": 64,
            "dropout": 0.1, "max_steps": 20, "batch_size": 32,
        },
        "transformer_lstm": {
            "lookback_window": 20, "d_model": 32, "nhead": 4,
            "transformer_layers": 1, "dim_feedforward": 64,
            "lstm_hidden_size": 32, "lstm_layers": 1,
            "dropout": 0.1, "max_steps": 20, "batch_size": 32,
        },
    }
    for kind, params in cases.items():
        candidate = deepcopy(source)
        candidate["config_json"]["model"] = {"kind": kind, "params": params}
        assert validate_job(candidate)["config_json"]["model"]["kind"] == kind


def test_job_validation_accepts_same_family_stacking_and_rejects_cross_family() -> None:
    source = valid_job()
    source["config_json"]["model"] = {
        "kind": "stacking",
        "params": {
            "n_folds": 3,
            "meta_alpha": 1.0,
            "loss": "mse",
            "objective": "regression",
            "metric": "rmse",
        },
        "base_models": [
            {"kind": "lightgbm", "params": {
                "loss": "mse", "objective": "regression", "metric": "rmse",
            }},
            {"kind": "linear", "params": {
                "loss": "mse", "objective": "regression", "metric": "rmse",
            }},
        ],
    }
    validated = validate_job(source)
    assert validated["config_json"]["model"]["kind"] == "stacking"

    cross_family = deepcopy(source)
    cross_family["config_json"]["model"]["base_models"][1] = {
        "kind": "lstm",
        "params": {
            "loss": "mse", "objective": "regression", "metric": "rmse",
        },
    }
    with pytest.raises(PermanentJobError, match="同一模型族"):
        validate_job(cross_family)


def test_job_validation_accepts_walk_forward_contract() -> None:
    source = valid_job()
    source["config_json"]["walk_forward"] = {
        "enabled": True,
        "strategy": "rolling",
        "train_sessions": 756,
        "valid_sessions": 60,
        "test_sessions": 20,
        "step_sessions": 20,
        "embargo_sessions": 5,
        "oos_date_start": "2023-01-03",
        "oos_date_end": "2024-12-31",
    }

    job = validate_job(source)

    assert job["config_json"]["walk_forward"]["train_sessions"] == 756


def test_job_validation_accepts_zero_validation_and_disabled_early_stopping() -> None:
    source = valid_job()
    source["config_json"]["model"]["params"]["early_stopping_rounds"] = 0
    source["config_json"]["walk_forward"] = {
        "enabled": True,
        "strategy": "rolling",
        "train_sessions": 756,
        "valid_sessions": 0,
        "test_sessions": 20,
        "step_sessions": 20,
        "embargo_sessions": 5,
        "oos_date_start": "2023-01-03",
        "oos_date_end": "2024-12-31",
    }

    job = validate_job(source)

    assert job["config_json"]["walk_forward"]["valid_sessions"] == 0
    assert job["config_json"]["model"]["params"]["early_stopping_rounds"] == 0


def test_job_validation_accepts_frozen_optuna_contract() -> None:
    source = valid_job()
    source["config_json"]["optuna"] = {
        "enabled": True,
        "backend": "optuna",
        "n_trials": 20,
        "objective": "validation_rank_icir",
        "direction": "maximize",
        "sampler": "tpe",
        "seed": 42,
        "search_space_version": "alphablocks.tree-optuna.v1",
    }

    job = validate_job(source)

    assert job["config_json"]["optuna"]["n_trials"] == 20


def test_job_validation_accepts_frozen_robust_optuna_contract() -> None:
    source = valid_job()
    source["config_json"]["optuna"] = {
        "enabled": True,
        "backend": "optuna",
        "n_trials": 20,
        "objective": "validation_rank_icir",
        "direction": "maximize",
        "sampler": "tpe",
        "seed": 42,
        "validation_windows": 3,
        "seed_count": 3,
        "stability_penalty": 0.5,
        "minimum_positive_window_ratio": 0.6,
        "search_space_version": "alphablocks.tree-optuna.v2",
    }

    job = validate_job(source)

    assert job["config_json"]["optuna"]["validation_windows"] == 3
    assert job["config_json"]["optuna"]["seed_count"] == 3


def test_job_validation_accepts_walk_forward_optuna_v3_contract() -> None:
    source = valid_job()
    source["config_json"]["walk_forward"] = {
        "enabled": True,
        "strategy": "rolling",
        "train_sessions": 756,
        "valid_sessions": 60,
        "test_sessions": 20,
        "step_sessions": 20,
        "embargo_sessions": 5,
        "oos_date_start": "2023-01-03",
        "oos_date_end": "2024-12-31",
    }
    source["config_json"]["optuna"] = {
        "enabled": True,
        "backend": "optuna",
        "n_trials": 20,
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

    job = validate_job(source)

    assert job["config_json"]["optuna"]["validation_mode"] == (
        "walk_forward_folds"
    )


def test_job_validation_rejects_optuna_without_validation() -> None:
    source = valid_job()
    source["config_json"]["walk_forward"] = {
        "enabled": True,
        "strategy": "rolling",
        "train_sessions": 756,
        "valid_sessions": 0,
        "test_sessions": 20,
        "step_sessions": 20,
        "embargo_sessions": 5,
        "oos_date_start": "2023-01-03",
        "oos_date_end": "2024-12-31",
    }
    source["config_json"]["optuna"] = {
        "enabled": True, "n_trials": 20,
    }

    with pytest.raises(PermanentJobError, match="验证长度为0"):
        validate_job(source)


def test_job_validation_accepts_strict_lightgbm_incremental_contract() -> None:
    source = valid_job()
    source["config_json"]["planned_model_version"] = 2
    source["config_json"]["incremental_training"] = {
        "schema_version": "alphablocks.incremental-training.v1",
        "mode": "lightgbm_append_trees_new_data_only",
        "source_model_id": "test_model",
        "source_model_version": 1,
        "source_job_id": "model_job_source",
        "source_dataset_hash": "c" * 64,
        "source_date_end": "2024-06-28",
        "candidate_date_end": "2024-12-31",
        "minimum_new_trading_sessions": 60,
        "source_artifact": {
            "artifact_id": "artifact_model_bundle",
            "relative_path": "model_job_source/bundle/source.tar.gz",
            "sha256": "b" * 64,
            "file_name": "source.tar.gz",
        },
        "allowed_parameter_changes": ["n_estimators", "early_stopping_rounds"],
    }

    job = validate_job(source)

    assert job["config_json"]["incremental_training"]["source_model_version"] == 1


def test_job_validation_rejects_incremental_path_traversal_or_deep_model() -> None:
    source = valid_job()
    source["config_json"]["planned_model_version"] = 2
    contract = {
        "schema_version": "alphablocks.incremental-training.v1",
        "mode": "lightgbm_append_trees_new_data_only",
        "source_model_id": "test_model",
        "source_model_version": 1,
        "source_job_id": "model_job_source",
        "source_date_end": "2024-06-28",
        "minimum_new_trading_sessions": 60,
        "source_artifact": {
            "artifact_id": "artifact_model_bundle",
            "relative_path": "../source.tar.gz",
            "sha256": "b" * 64,
        },
    }
    source["config_json"]["incremental_training"] = contract
    with pytest.raises(PermanentJobError, match="路径无效"):
        validate_job(source)

    source = valid_job()
    source["config_json"]["planned_model_version"] = 2
    source["config_json"]["model"] = {
        "kind": "lstm", "params": {"max_steps": 20, "batch_size": 32},
    }
    source["config_json"]["incremental_training"] = {
        **contract,
        "source_artifact": {
            **contract["source_artifact"],
            "relative_path": "models/source.tar.gz",
        },
    }
    with pytest.raises(PermanentJobError, match="只支持LightGBM"):
        validate_job(source)


def test_job_validation_rejects_overlapping_walk_forward_tests() -> None:
    source = valid_job()
    source["config_json"]["walk_forward"] = {
        "enabled": True,
        "test_sessions": 20,
        "step_sessions": 10,
        "oos_date_start": "2023-01-03",
        "oos_date_end": "2024-12-31",
    }

    with pytest.raises(PermanentJobError, match="步长必须等于测试窗口"):
        validate_job(source)


def test_job_validation_rejects_tampered_dataset() -> None:
    job = valid_job()
    job["dataset_spec"]["date_end"] = "2026-01-01"

    with pytest.raises(PermanentJobError, match="dataset_hash"):
        validate_job(job)


def test_job_validation_rejects_divergent_config_dataset() -> None:
    job = valid_job()
    job["config_json"]["dataset"]["date_end"] = "2026-01-01"

    with pytest.raises(PermanentJobError, match="config_json.dataset"):
        validate_job(job)


def test_job_validation_rejects_unbounded_model_parameter() -> None:
    job = valid_job()
    job["config_json"]["model"]["params"]["num_threads"] = 1000

    with pytest.raises(PermanentJobError, match="num_threads"):
        validate_job(job)


def test_job_validation_accepts_quantmind_random_forest_feature_sampling() -> None:
    job = valid_job()
    job["config_json"]["model"] = {
        "kind": "random_forest",
        "params": {"n_estimators": 300, "max_depth": 0, "max_features": "sqrt"},
    }

    assert validate_job(job)["config_json"]["model"]["params"]["max_features"] == "sqrt"


def test_job_validation_rejects_unknown_random_forest_feature_sampling() -> None:
    job = valid_job()
    job["config_json"]["model"] = {
        "kind": "random_forest",
        "params": {"max_features": "all"},
    }

    with pytest.raises(PermanentJobError, match="max_features"):
        validate_job(job)


@pytest.mark.parametrize("layers", [[], [2, 64], [64] * 9, "64,128"])
def test_job_validation_rejects_invalid_mlp_hidden_layers(layers) -> None:
    job = valid_job()
    job["config_json"]["model"] = {
        "kind": "mlp",
        "params": {"hidden_layers": layers, "max_steps": 20, "batch_size": 32},
    }

    with pytest.raises(PermanentJobError, match="hidden_layers"):
        validate_job(job)


@pytest.mark.parametrize(
    ("field", "value"), [("lookback_window", 1), ("num_layers", 9), ("dropout", 1.0)],
)
def test_job_validation_rejects_invalid_lstm_architecture(field, value) -> None:
    job = valid_job()
    job["config_json"]["model"] = {
        "kind": "lstm",
        "params": {field: value, "max_steps": 20, "batch_size": 32},
    }

    with pytest.raises(PermanentJobError, match=field):
        validate_job(job)


def test_job_validation_rejects_incompatible_transformer_attention_heads() -> None:
    job = valid_job()
    job["config_json"]["model"] = {
        "kind": "transformer_lstm",
        "params": {"d_model": 30, "nhead": 8, "max_steps": 20, "batch_size": 32},
    }

    with pytest.raises(PermanentJobError, match="d_model.*nhead"):
        validate_job(job)


def test_job_validation_rejects_non_object_config() -> None:
    job = valid_job()
    job["config_json"] = []

    with pytest.raises(PermanentJobError, match="config_json"):
        validate_job(job)


def test_job_validation_accepts_daily_inference_contract() -> None:
    job = validate_job(valid_inference_job())

    assert job["kind"] == "infer"
    assert job["config_json"]["inference"]["trade_date"] == "2024-12-31"


def test_job_validation_rejects_inference_cutoff_before_signal_close() -> None:
    job = valid_inference_job()
    job["config_json"]["inference"]["data_cutoff"] = "2024-12-31T06:00:00+00:00"

    with pytest.raises(PermanentJobError, match="data_cutoff"):
        validate_job(job)


def test_safe_job_dir_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(PermanentJobError):
        safe_job_dir(tmp_path, "../outside")


def test_cancellation_and_shutdown_are_distinct() -> None:
    token = CancellationToken()
    token.cancel("manual cancel")
    with pytest.raises(JobCanceled, match="manual cancel"):
        token.checkpoint()

    shutdown = threading.Event()
    shutdown.set()
    with pytest.raises(WorkerShutdown):
        CancellationToken(shutdown).checkpoint()


def test_job_state_is_atomic_private_and_round_trips(tmp_path: Path) -> None:
    store = JobStateStore(tmp_path)
    job = valid_job()
    store.save(job, "training", {"percent": 60})

    state = store.load()
    assert state is not None
    assert state["job"]["lease_token"] == job["lease_token"]
    assert state["phase"] == "training"
    assert os.stat(store.path).st_mode & 0o777 == 0o600
    json.loads(store.path.read_text(encoding="utf-8"))
    store.clear()
    assert store.load() is None
