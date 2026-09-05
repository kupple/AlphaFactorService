from __future__ import annotations

from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable

from factor_service.factor_backtest import UNIVERSES
from factor_service.entity_field_feature import (
    is_entity_field_feature,
    validate_entity_field_feature_identity,
)
from factor_service.research.errors import (
    JobCanceled,
    PermanentJobError,
    TrainingTimeout,
    WorkerShutdown,
)
from factor_service.research.industry_feature import (
    INDUSTRY_FEATURE_SAFE_START,
    normalize_industry_feature,
)
from factor_service.research.preprocessing import (
    DATASET_PIPELINE_VERSION,
    FROZEN_DATA_BINDING_PIPELINE_VERSIONS,
    REQUIRED_PREPROCESSING_PIPELINE_VERSIONS,
    normalize_feature_preprocessing,
)
from factor_service.research.sample_filter_formula import (
    normalize_custom_sample_filters,
)
from factor_service.research.size_rotation_feature import (
    normalize_size_rotation_feature,
)
from factor_service.research.training_resource_settings import (
    INDEX_MEMBERSHIP_BINDING_ID,
    STOCK_DAILY_BINDING_ID,
    frozen_data_binding,
    normalize_frozen_training_data_bindings,
    required_training_data_binding_ids,
)
from factor_service.research.universe_source import (
    normalize_universe_source,
)
from factor_service.research.universe_field_filter import (
    normalize_universe_field_filters,
)


IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HASH = re.compile(r"^[0-9a-f]{16,64}$")
MODEL_PARAM_FIELDS = {
    "stacking": {
        "n_folds", "meta_alpha", "loss", "objective", "metric",
    },
    "lightgbm": {
        "learning_rate", "num_leaves", "max_depth", "n_estimators", "subsample",
        "colsample_bytree", "reg_alpha", "reg_lambda", "min_child_samples",
        "min_data_in_leaf", "path_smooth", "bagging_freq", "lambda_l1",
        "lambda_l2", "feature_fraction", "bagging_fraction",
        "early_stopping_rounds", "num_threads", "loss", "seed",
        "feature_fraction_seed", "bagging_seed", "data_random_seed", "deterministic",
        "verbosity",
    },
    "xgboost": {
        "learning_rate", "max_depth", "n_estimators", "subsample",
        "colsample_bytree", "reg_alpha", "reg_lambda", "min_child_weight",
        "early_stopping_rounds", "num_threads", "loss", "seed", "deterministic",
        "verbosity",
    },
    "catboost": {
        "learning_rate", "depth", "n_estimators", "l2_leaf_reg", "random_strength",
        "bagging_temperature", "od_wait",
        "early_stopping_rounds", "num_threads", "loss", "seed", "deterministic",
        "verbosity",
    },
    "random_forest": {
        "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
        "max_features", "num_threads", "loss", "seed", "deterministic", "verbosity",
    },
    "linear": {
        "alpha", "fit_intercept", "solver", "max_iter", "num_threads", "loss",
        "seed", "deterministic", "verbosity",
    },
    "mlp": {
        "learning_rate", "hidden_layers", "hidden_size", "layer_count",
        "max_steps", "batch_size",
        "early_stopping_rounds", "eval_steps", "weight_decay", "num_threads",
        "loss", "seed", "deterministic", "verbosity",
    },
    "lstm": {
        "learning_rate", "lookback_window", "hidden_size", "num_layers", "dropout",
        "max_steps", "batch_size", "early_stopping_rounds", "eval_steps",
        "weight_decay", "num_threads", "loss", "seed", "deterministic", "verbosity",
    },
    "gru": {
        "learning_rate", "lookback_window", "hidden_size", "num_layers", "dropout",
        "max_steps", "batch_size", "early_stopping_rounds", "eval_steps",
        "weight_decay", "num_threads", "loss", "seed", "deterministic", "verbosity",
    },
    "alstm": {
        "learning_rate", "lookback_window", "hidden_size", "num_layers", "dropout",
        "max_steps", "batch_size", "early_stopping_rounds", "eval_steps",
        "weight_decay", "num_threads", "loss", "seed", "deterministic", "verbosity",
    },
    "transformer": {
        "learning_rate", "lookback_window", "d_model", "nhead",
        "transformer_layers", "dim_feedforward", "dropout", "max_steps",
        "batch_size", "early_stopping_rounds", "eval_steps", "weight_decay",
        "num_threads", "loss", "seed", "deterministic", "verbosity",
    },
    "tabnet": {
        "learning_rate", "n_d", "n_a", "n_steps", "n_shared", "n_ind",
        "batch_size", "max_steps", "early_stopping_rounds", "pretrain",
        "num_threads", "loss", "seed", "deterministic", "verbosity",
    },
    "tcn": {
        "learning_rate", "lookback_window", "hidden_size", "kernel_size",
        "num_layers", "dropout", "max_steps", "batch_size",
        "early_stopping_rounds", "eval_steps", "weight_decay", "num_threads",
        "loss", "seed", "deterministic", "verbosity",
    },
    "nativetft": {
        "learning_rate", "lookback_window", "d_model", "nhead",
        "gru_hidden_size", "num_layers", "dim_feedforward", "dropout",
        "max_steps", "batch_size", "early_stopping_rounds", "eval_steps",
        "weight_decay", "num_threads", "loss", "seed", "deterministic", "verbosity",
    },
    "transformer_lstm": {
        "learning_rate", "lookback_window", "d_model", "nhead",
        "transformer_layers", "dim_feedforward", "lstm_hidden_size", "lstm_layers",
        "dropout", "max_steps", "batch_size", "early_stopping_rounds", "eval_steps",
        "weight_decay", "num_threads", "loss", "seed", "deterministic", "verbosity",
    },
}
for _fields in MODEL_PARAM_FIELDS.values():
    _fields.update({"objective", "metric"})

CLASSICAL_STACKING_KINDS = frozenset({
    "lightgbm", "xgboost", "catboost", "random_forest", "linear",
})
DEEP_STACKING_KINDS = frozenset({
    "mlp", "gru", "lstm", "alstm", "transformer", "tabnet", "tcn",
    "nativetft", "transformer_lstm",
})
OPTUNA_MODEL_KINDS = frozenset({"lightgbm", "xgboost", "catboost"})


def validate_job(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PermanentJobError("任务载荷必须是JSON对象")
    job_id = _identifier(payload.get("job_id"), "job_id")
    model_id = _identifier(payload.get("model_id"), "model_id")
    lease_token = str(payload.get("lease_token") or "").strip()
    if len(lease_token) < 16 or len(lease_token) > 512:
        raise PermanentJobError("lease_token长度无效")
    if str(payload.get("lease_owner") or "alpha-factor-service") != "alpha-factor-service":
        raise PermanentJobError("任务租约不属于AlphaFactorService研究调度进程")
    dataset_hash = str(payload.get("dataset_hash") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", dataset_hash):
        raise PermanentJobError("dataset_hash必须是64位SHA256")
    config = payload.get("config_json")
    if not isinstance(config, dict):
        raise PermanentJobError("任务缺少config_json")
    spec = payload.get("dataset_spec") or config.get("dataset")
    if not isinstance(spec, dict):
        raise PermanentJobError("任务缺少dataset_spec")
    canonical_hash = sha256(_canonical_json(spec).encode("utf-8")).hexdigest()
    if canonical_hash != dataset_hash:
        raise PermanentJobError("dataset_hash与冻结数据规格不一致")
    configured_spec = config.get("dataset")
    if not isinstance(configured_spec, dict) or _canonical_json(configured_spec) != _canonical_json(spec):
        raise PermanentJobError("config_json.dataset与冻结dataset_spec不一致")
    universe_id = str(spec.get("universe_id") or "").strip()
    index_code = str(spec.get("index_code") or "").strip()
    try:
        universe_source = normalize_universe_source(
            spec.get("universe_source"), allow_empty=True,
        )
    except ValueError as exc:
        raise PermanentJobError(str(exc)) from exc
    if universe_source and universe_source != spec.get("universe_source"):
        raise PermanentJobError(
            "dataset_spec.universe_source不是规范化冻结规格"
        )
    source_kind = str(universe_source.get("source_kind") or "")
    if source_kind == "configured_stock_pool":
        if universe_id != universe_source["source_id"]:
            raise PermanentJobError(
                "dataset_spec的universe_id与冻结配置股票池不一致"
            )
        selector_value = str(
            dict(universe_source.get("selector") or {}).get("value") or ""
        ).strip()
        if index_code != selector_value:
            raise PermanentJobError(
                "dataset_spec的index_code与配置股票池选择值不一致"
            )
    elif universe_source:
        if universe_id != universe_source["source_id"]:
            raise PermanentJobError(
                "dataset_spec的universe_id与冻结成员来源不一致"
            )
        if index_code != UNIVERSES["csi500"]["index_code"]:
            raise PermanentJobError("自定义股票池的报告基准不受支持")
    else:
        if universe_id not in UNIVERSES:
            raise PermanentJobError(f"不支持的股票池: {universe_id}")
        if index_code != UNIVERSES[universe_id]["index_code"]:
            raise PermanentJobError("dataset_spec的股票池与index_code不一致")
    sample_filters = spec.get("sample_filters")
    if sample_filters is not None:
        if not isinstance(sample_filters, dict):
            raise PermanentJobError("dataset_spec.sample_filters必须是对象")
        legacy_boolean_filters = any(
            field in sample_filters
            for field in ("exclude_st", "exclude_delisting")
        )
        required_filter_fields = {"minimum_listing_trading_days"}
        if legacy_boolean_filters:
            required_filter_fields.update({"exclude_st", "exclude_delisting"})
        allowed_filter_fields = required_filter_fields | {"custom_formulas"}
        if (
            not required_filter_fields.issubset(sample_filters)
            or not set(sample_filters).issubset(allowed_filter_fields)
        ):
            raise PermanentJobError(
                "dataset_spec.sample_filters字段不完整或包含未知字段"
            )
        minimum_listing_days = sample_filters["minimum_listing_trading_days"]
        if type(minimum_listing_days) is not int or not 0 <= minimum_listing_days <= 5000:
            raise PermanentJobError("最少上市交易日必须是0至5000的整数")
        if legacy_boolean_filters:
            for field in ("exclude_st", "exclude_delisting"):
                if type(sample_filters[field]) is not bool:
                    raise PermanentJobError(
                        f"sample_filters.{field}必须是布尔值"
                    )
        if "custom_formulas" in sample_filters:
            try:
                normalized_formulas = normalize_custom_sample_filters(
                    sample_filters["custom_formulas"],
                )
            except ValueError as exc:
                raise PermanentJobError(str(exc)) from exc
            if normalized_formulas != sample_filters["custom_formulas"]:
                raise PermanentJobError(
                    "sample_filters.custom_formulas不是规范化冻结规格"
                )
    try:
        universe_field_filters = normalize_universe_field_filters(
            spec.get("universe_field_filters")
        )
    except ValueError as exc:
        raise PermanentJobError(str(exc)) from exc
    if universe_field_filters != list(spec.get("universe_field_filters") or []):
        raise PermanentJobError(
            "dataset_spec.universe_field_filters不是规范化冻结规格"
        )
    preprocessing = spec.get("preprocessing")
    pipeline_version = str(spec.get("pipeline_version") or "")
    if preprocessing is None:
        if pipeline_version in REQUIRED_PREPROCESSING_PIPELINE_VERSIONS:
            version_label = pipeline_version.rsplit(".", 1)[-1]
            raise PermanentJobError(
                f"{version_label}数据集缺少冻结的preprocessing规格"
            )
    else:
        if not isinstance(preprocessing, dict):
            raise PermanentJobError("dataset_spec.preprocessing必须是对象")
        try:
            normalized_preprocessing = normalize_feature_preprocessing(
                preprocessing,
                default_enabled=False,
            )
        except ValueError as exc:
            raise PermanentJobError(str(exc)) from exc
        if normalized_preprocessing != preprocessing:
            raise PermanentJobError("dataset_spec.preprocessing不是规范化冻结规格")
    industry_feature = spec.get("industry_feature")
    if industry_feature is None:
        if pipeline_version in FROZEN_DATA_BINDING_PIPELINE_VERSIONS:
            version_label = pipeline_version.rsplit(".", 1)[-1]
            raise PermanentJobError(
                f"{version_label}数据集缺少冻结的industry_feature规格"
            )
        normalized_industry_feature = normalize_industry_feature(
            None, default_enabled=False,
        )
    else:
        if not isinstance(industry_feature, dict):
            raise PermanentJobError("dataset_spec.industry_feature必须是对象")
        try:
            normalized_industry_feature = normalize_industry_feature(
                industry_feature, default_enabled=False,
            )
        except ValueError as exc:
            raise PermanentJobError(str(exc)) from exc
        if normalized_industry_feature != industry_feature:
            raise PermanentJobError(
                "dataset_spec.industry_feature不是规范化冻结规格"
            )
    size_rotation_feature = spec.get("size_rotation_feature")
    if size_rotation_feature is None:
        normalized_size_rotation_feature = normalize_size_rotation_feature(
            None, default_enabled=False,
        )
    else:
        if not isinstance(size_rotation_feature, dict):
            raise PermanentJobError(
                "dataset_spec.size_rotation_feature必须是对象"
            )
        try:
            normalized_size_rotation_feature = normalize_size_rotation_feature(
                size_rotation_feature, default_enabled=False,
            )
        except ValueError as exc:
            raise PermanentJobError(str(exc)) from exc
        if normalized_size_rotation_feature != size_rotation_feature:
            raise PermanentJobError(
                "dataset_spec.size_rotation_feature不是规范化冻结规格"
            )
    kind = str(payload.get("kind") or "train")
    if kind not in {"train", "infer"}:
        raise PermanentJobError("任务kind只允许train或infer")
    start = _date(spec.get("date_start"), "date_start")
    end = _date(spec.get("date_end"), "date_end")
    if start >= end:
        raise PermanentJobError("训练开始日期必须早于结束日期")
    if normalized_industry_feature["enabled"]:
        if str(spec.get("research_target") or "stock_selection") != "stock_selection":
            raise PermanentJobError("行业编码特征仅支持个股选股训练目标")
        if start < _date(INDUSTRY_FEATURE_SAFE_START, "industry_feature.safe_start"):
            raise PermanentJobError(
                "行业编码特征仅支持2021-12-13及以后；"
                "更早历史包含申万2021版回溯重分类"
            )
    if normalized_size_rotation_feature["enabled"] and str(
        spec.get("research_target") or "stock_selection"
    ) != "stock_selection":
        raise PermanentJobError("大小盘轮动特征仅支持个股选股训练目标")
    if pipeline_version in FROZEN_DATA_BINDING_PIPELINE_VERSIONS:
        try:
            data_bindings = normalize_frozen_training_data_bindings(
                spec.get("data_bindings"),
            )
        except ValueError as exc:
            raise PermanentJobError(str(exc)) from exc
        missing_bindings = [
            binding_id
            for binding_id in required_training_data_binding_ids(spec)
            if frozen_data_binding(data_bindings, binding_id) is None
        ]
        if missing_bindings:
            raise PermanentJobError(
                "dataset_spec.data_bindings缺少冻结绑定: "
                + ", ".join(missing_bindings)
            )
        if normalized_size_rotation_feature["enabled"]:
            daily_binding = frozen_data_binding(
                data_bindings, STOCK_DAILY_BINDING_ID,
            )
            daily_fields = dict(
                (daily_binding or {}).get("field_bindings") or {}
            )
            if not str(daily_fields.get("float_market_cap") or "").strip():
                raise PermanentJobError(
                    "大小盘轮动特征要求训练基础行情绑定float_market_cap"
                )
            membership_binding = frozen_data_binding(
                data_bindings, INDEX_MEMBERSHIP_BINDING_ID,
            )
            binding_fingerprint = str(
                (membership_binding or {}).get("fingerprint") or ""
            )
            for pool_name in ("large_pool", "small_pool"):
                pool = normalized_size_rotation_feature[pool_name]
                if str(pool["binding_fingerprint"]) != binding_fingerprint:
                    raise PermanentJobError(
                        "大小盘轮动股票池与冻结指数成分绑定版本不一致"
                    )
    cutoff = _datetime(spec.get("data_cutoff"), "data_cutoff")
    factors = spec.get("factors")
    if not isinstance(factors, list) or not 1 <= len(factors) <= 100:
        raise PermanentJobError("因子数量必须在1到100之间")
    seen: set[tuple[str, int, str]] = set()
    for factor in factors:
        if not isinstance(factor, dict):
            raise PermanentJobError("因子规格必须是对象")
        factor_id = _identifier(factor.get("factor_id"), "factor_id")
        version = _integer(factor.get("factor_version"), "factor_version", 1, 1_000_000)
        params_hash = str(factor.get("params_hash") or "").strip().lower()
        if not HASH.fullmatch(params_hash):
            raise PermanentJobError(f"因子{factor_id}的params_hash无效")
        if not isinstance(factor.get("params"), dict):
            raise PermanentJobError(f"因子{factor_id}缺少冻结params")
        if is_entity_field_feature(factor):
            try:
                validate_entity_field_feature_identity(factor)
            except ValueError as exc:
                raise PermanentJobError(str(exc)) from exc
        key = (factor_id, version, params_hash)
        if key in seen:
            raise PermanentJobError(f"因子{factor_id}重复")
        seen.add(key)
    materialization = spec.get("materialization")
    if not isinstance(materialization, dict) or materialization != {
        "mode": "on_demand", "format": "parquet", "persist_factor_values": False,
    }:
        raise PermanentJobError("数据集必须使用按需计算的不可变Parquet物化方式")
    coverage = float(spec.get("minimum_factor_coverage", 0.8))
    if not 0.5 <= coverage <= 1.0:
        raise PermanentJobError("minimum_factor_coverage必须在0.5到1之间")
    label_spec = spec.get("label") or {}
    if not isinstance(label_spec, dict):
        raise PermanentJobError("dataset_spec.label必须是对象")
    target_mode = str(
        spec.get("target_mode") or label_spec.get("mode") or "return"
    ).strip().lower()
    if target_mode not in {"return", "classification"}:
        raise PermanentJobError("目标类型只允许return或classification")
    version = _integer(config.get("planned_model_version"), "planned_model_version", 1, 1_000_000)
    if kind == "infer":
        _validate_inference_config(config, model_id=model_id, version=version)
        result = dict(payload)
        result.update({
            "job_id": job_id, "model_id": model_id, "lease_token": lease_token,
            "dataset_hash": dataset_hash, "dataset_spec": dict(spec), "kind": kind,
        })
        result["config_json"] = dict(config)
        result["config_json"]["planned_model_version"] = version
        return result
    model = config.get("model")
    model_kind = str(model.get("kind") or "") if isinstance(model, dict) else ""
    if model_kind not in MODEL_PARAM_FIELDS:
        raise PermanentJobError(
            "模型只允许stacking、lightgbm、xgboost、catboost、random_forest、linear、"
            "mlp、gru、lstm、alstm、transformer、tabnet、tcn、nativetft或transformer_lstm"
        )
    _integer(model.get("version") or 1, "model.version", 1, 1)
    params = model.get("params") or {}
    if not isinstance(params, dict):
        raise PermanentJobError(f"{model_kind}参数必须是对象")
    unknown_params = sorted(set(params) - MODEL_PARAM_FIELDS[model_kind])
    if unknown_params:
        raise PermanentJobError(
            f"{model_kind}参数包含未允许字段: {', '.join(unknown_params)}"
        )
    walk_forward = config.get("walk_forward") or {}
    _validate_walk_forward(walk_forward)
    validationless_walk_forward = (
        walk_forward.get("enabled") is True
        and int(walk_forward.get("valid_sessions", 60)) == 0
    )
    _validate_optuna(
        config.get("optuna") or {},
        model_kind=model_kind,
        walk_forward_enabled=walk_forward.get("enabled") is True,
        validationless_walk_forward=validationless_walk_forward,
        incremental=bool(config.get("incremental_training")),
    )
    _validate_model_params(
        model_kind, params,
        allow_disabled_early_stopping=validationless_walk_forward,
    )
    if model_kind == "stacking":
        _validate_stacking_model(model, target_mode=target_mode)
    execution = config.get("execution") or {"node_id": "local"}
    if not isinstance(execution, dict):
        raise PermanentJobError("execution必须是对象")
    from factor_service.research.distributed_config import execution_spec
    try:
        normalized_execution = execution_spec(execution)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PermanentJobError(str(exc)) from exc
    if execution.get("mode", normalized_execution["mode"]) != normalized_execution["mode"]:
        raise PermanentJobError("execution.mode与node_id不一致")
    if normalized_execution["mode"] == "distributed":
        if model_kind == "stacking" or config.get("incremental_training"):
            raise PermanentJobError("多节点暂不支持Stacking或有前序依赖的增量续训")
        if not (config.get("optuna") or {}).get("enabled") and not walk_forward.get("enabled"):
            raise PermanentJobError("多节点训练需要开启Optuna或独立Walk-Forward")
    _integer(
        execution.get("max_runtime_minutes", 720),
        "execution.max_runtime_minutes",
        60,
        1440,
    )
    _integer(params.get("num_threads", 4), "num_threads", 1, 32)
    expected_loss = "binary" if target_mode == "classification" else "mse"
    if str(params.get("loss", "mse")).strip().lower() != expected_loss:
        raise PermanentJobError(
            f"{target_mode}目标必须使用{expected_loss}损失"
        )
    expected_objective = "binary" if target_mode == "classification" else "regression"
    if str(params.get("objective", expected_objective)).strip().lower() != expected_objective:
        raise PermanentJobError("Objective必须与目标类型一致")
    metric = str(
        params.get("metric") or ("auc" if expected_objective == "binary" else "rmse")
    ).strip().lower()
    supported_metrics = (
        {"auc", "binary_logloss"}
        if expected_objective == "binary"
        else {"l2", "rmse", "mae"}
    )
    if metric not in supported_metrics:
        raise PermanentJobError("Metric必须与Objective一致")
    for field in ("seed", "feature_fraction_seed", "bagging_seed", "data_random_seed"):
        if field not in params and field != "seed":
            continue
        _integer(params.get(field, 42), field, 0, 2_147_483_647)
    if params.get("deterministic", True) is not True:
        raise PermanentJobError("训练必须启用deterministic")
    if config.get("incremental_training"):
        _validate_incremental_training(
            config["incremental_training"],
            model_id=model_id,
            planned_version=version,
            model_kind=model_kind,
            dataset_spec=spec,
            walk_forward=config.get("walk_forward") or {},
        )
    result = dict(payload)
    result.update({
        "job_id": job_id, "model_id": model_id, "lease_token": lease_token,
        "dataset_hash": dataset_hash, "dataset_spec": dict(spec), "kind": kind,
    })
    result["config_json"] = dict(config)
    result["config_json"]["planned_model_version"] = version
    result["dataset_spec"]["data_cutoff"] = cutoff.isoformat()
    return result


def _validate_optuna(
    source: Any,
    *,
    model_kind: str,
    walk_forward_enabled: bool,
    validationless_walk_forward: bool,
    incremental: bool,
) -> None:
    if not source:
        return
    if not isinstance(source, dict):
        raise PermanentJobError("optuna必须是对象")
    unknown = sorted(set(source) - {
        "enabled", "backend", "n_trials", "objective", "direction",
        "sampler", "seed", "search_space_version",
        "validation_windows", "seed_count", "stability_penalty",
        "minimum_positive_window_ratio", "validation_mode",
    })
    if unknown:
        raise PermanentJobError(
            "optuna包含未允许字段: " + ", ".join(unknown)
        )
    if source.get("enabled") is not True:
        raise PermanentJobError("optuna.enabled必须为true或省略整个配置")
    if model_kind not in OPTUNA_MODEL_KINDS:
        raise PermanentJobError("Optuna只支持LightGBM、XGBoost或CatBoost")
    if validationless_walk_forward:
        raise PermanentJobError("验证长度为0时不能开启Optuna")
    if incremental:
        raise PermanentJobError("增量续训不能同时开启Optuna")
    _integer(source.get("n_trials", 20), "optuna.n_trials", 10, 100)
    _integer(source.get("seed", 42), "optuna.seed", 0, 2_147_483_647)
    version = str(
        source.get("search_space_version") or "alphablocks.tree-optuna.v1"
    )
    if version in {
        "alphablocks.tree-optuna.v2", "alphablocks.tree-optuna.v3",
    }:
        _integer(
            source.get("validation_windows", 3),
            "optuna.validation_windows", 2, 8,
        )
        _integer(source.get("seed_count", 3), "optuna.seed_count", 1, 5)
        _number(
            source.get("stability_penalty", 0.5),
            "optuna.stability_penalty", 0.0, 2.0,
        )
        _number(
            source.get("minimum_positive_window_ratio", 0.6),
            "optuna.minimum_positive_window_ratio", 0.0, 1.0,
        )
    if version == "alphablocks.tree-optuna.v3":
        expected_mode = (
            "walk_forward_folds"
            if walk_forward_enabled else "fixed_subwindows"
        )
        if str(source.get("validation_mode") or "") != expected_mode:
            raise PermanentJobError("Optuna验证模式与Walk-Forward配置不一致")
    if str(source.get("backend") or "optuna").strip().lower() != "optuna":
        raise PermanentJobError("optuna.backend必须为optuna")
    if str(source.get("objective") or "validation_rank_icir").strip().lower() != "validation_rank_icir":
        raise PermanentJobError("Optuna目标只支持validation_rank_icir")
    if str(source.get("direction") or "maximize").strip().lower() != "maximize":
        raise PermanentJobError("Optuna方向必须为maximize")
    if str(source.get("sampler") or "tpe").strip().lower() != "tpe":
        raise PermanentJobError("Optuna采样器只支持tpe")
    if version not in {
        "alphablocks.tree-optuna.v1", "alphablocks.tree-optuna.v2",
        "alphablocks.tree-optuna.v3",
    }:
        raise PermanentJobError("Optuna搜索空间版本无效")


def _validate_stacking_model(model: dict[str, Any], *, target_mode: str) -> None:
    base_models = model.get("base_models")
    if (
        not isinstance(base_models, list)
        or not 2 <= len(base_models) <= 8
        or any(not isinstance(item, dict) for item in base_models)
    ):
        raise PermanentJobError("Stacking必须配置2到8个基模型")
    base_kinds = [str(item.get("kind") or "").strip().lower() for item in base_models]
    if len(set(base_kinds)) != len(base_kinds):
        raise PermanentJobError("Stacking不能重复配置同一个基模型")
    kinds = set(base_kinds)
    if not (
        kinds and kinds <= CLASSICAL_STACKING_KINDS
        or kinds and kinds <= DEEP_STACKING_KINDS
    ):
        raise PermanentJobError(
            "Stacking只支持同一模型族；传统模型与深度学习模型不能混合集成"
        )
    expected_loss = "binary" if target_mode == "classification" else "mse"
    expected_objective = "binary" if target_mode == "classification" else "regression"
    for base_model in base_models:
        kind = str(base_model.get("kind") or "").strip().lower()
        if kind not in MODEL_PARAM_FIELDS or kind == "stacking":
            raise PermanentJobError(f"Stacking基模型{kind or '--'}无效")
        _integer(base_model.get("version") or 1, f"{kind}.version", 1, 1)
        params = base_model.get("params") or {}
        if not isinstance(params, dict):
            raise PermanentJobError(f"Stacking基模型{kind}参数必须是对象")
        unknown = sorted(set(params) - MODEL_PARAM_FIELDS[kind])
        if unknown:
            raise PermanentJobError(
                f"Stacking基模型{kind}参数包含未允许字段: {', '.join(unknown)}"
            )
        _validate_model_params(kind, params)
        if str(params.get("loss") or "mse").strip().lower() != expected_loss:
            raise PermanentJobError(f"Stacking基模型{kind}损失与目标类型不一致")
        if str(params.get("objective") or expected_objective).strip().lower() != expected_objective:
            raise PermanentJobError(f"Stacking基模型{kind}Objective与目标类型不一致")


def _validate_model_params(
    kind: str,
    params: dict[str, Any],
    *,
    allow_disabled_early_stopping: bool = False,
) -> None:
    early_stopping_minimum = 0 if allow_disabled_early_stopping else 1
    if kind == "stacking":
        _integer(params.get("n_folds", 3), "n_folds", 2, 10)
        _number(params.get("meta_alpha", 1.0), "meta_alpha", 0.01, 100.0)
        return
    deep_kinds = {
        "mlp", "gru", "lstm", "alstm", "transformer", "tabnet", "tcn",
        "nativetft", "transformer_lstm",
    }
    default_lr = 0.001 if kind in deep_kinds else 0.02
    _number(params.get("learning_rate", default_lr), "learning_rate", 0.000001, 1.0)
    if kind == "lightgbm":
        _integer(params.get("num_leaves", 31), "num_leaves", 2, 65536)
        _integer(params.get("max_depth", -1), "max_depth", -1, 128)
        _integer(params.get("min_data_in_leaf", 300), "min_data_in_leaf", 1, 1_000_000)
        _integer(params.get("min_child_samples", 150), "min_child_samples", 1, 1_000_000)
        _number(params.get("path_smooth", 1.0), "path_smooth", 0.0, 10.0)
        _integer(params.get("bagging_freq", 5), "bagging_freq", 0, 100)
        _number(params.get("lambda_l1", 0.5), "lambda_l1", 0.0, 1_000_000.0)
        _number(params.get("lambda_l2", 1.0), "lambda_l2", 0.0, 1_000_000.0)
        _number(params.get("feature_fraction", 0.7), "feature_fraction", 0.01, 1.0)
        _number(params.get("bagging_fraction", 0.8), "bagging_fraction", 0.01, 1.0)
    elif kind == "xgboost":
        _integer(params.get("max_depth", 6), "max_depth", 1, 128)
        _number(params.get("min_child_weight", 1.0), "min_child_weight", 0.0, 1_000_000.0)
    elif kind == "catboost":
        _integer(params.get("depth", 6), "depth", 1, 16)
        _number(params.get("l2_leaf_reg", 3.0), "l2_leaf_reg", 0.0, 1_000_000.0)
        _number(params.get("random_strength", 1.5), "random_strength", 0.0, 1_000_000.0)
        _number(params.get("bagging_temperature", 0.8), "bagging_temperature", 0.0, 10.0)
        _integer(
            params.get("od_wait", 100), "od_wait",
            early_stopping_minimum, 1000,
        )
    elif kind == "random_forest":
        _integer(params.get("n_estimators", 300), "n_estimators", 1, 100_000)
        _integer(params.get("max_depth", 0), "max_depth", 0, 128)
        _integer(params.get("min_samples_split", 2), "min_samples_split", 2, 1_000_000)
        _integer(params.get("min_samples_leaf", 1), "min_samples_leaf", 1, 1_000_000)
        max_features = params.get("max_features", "sqrt")
        if isinstance(max_features, str):
            if max_features not in {"sqrt", "log2"}:
                raise PermanentJobError("max_features只支持sqrt、log2或(0, 1]数值")
        else:
            _number(max_features, "max_features", 0.000001, 1.0)
        return
    elif kind == "linear":
        _number(params.get("alpha", 1.0), "alpha", 0.0, 1_000_000.0)
        _integer(params.get("max_iter", 1000), "max_iter", 1, 1_000_000)
        if not isinstance(params.get("fit_intercept", True), bool):
            raise PermanentJobError("fit_intercept必须是布尔值")
        if str(params.get("solver") or "auto") not in {
            "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs",
        }:
            raise PermanentJobError("linear.solver无效")
        return
    elif kind == "mlp":
        layers = params.get("hidden_layers")
        if layers is None:
            _integer(params.get("hidden_size", 64), "hidden_size", 4, 4096)
            _integer(params.get("layer_count", 2), "layer_count", 1, 8)
        else:
            if not isinstance(layers, list) or not 1 <= len(layers) <= 8:
                raise PermanentJobError("hidden_layers必须是包含1到8层的数组")
            for index, width in enumerate(layers, start=1):
                _integer(width, f"hidden_layers[{index}]", 4, 4096)
        _integer(params.get("max_steps", 300), "max_steps", 1, 100_000)
        _integer(params.get("batch_size", 2048), "batch_size", 16, 1_000_000)
        _integer(params.get("eval_steps", 10), "eval_steps", 1, 10_000)
        _number(params.get("weight_decay", 0.0001), "weight_decay", 0.0, 1_000_000.0)
        _integer(params.get("early_stopping_rounds", 20), "early_stopping_rounds", early_stopping_minimum, 10_000)
        return
    elif kind in {"gru", "lstm", "alstm"}:
        _integer(params.get("lookback_window", 60), "lookback_window", 2, 252)
        _integer(params.get("hidden_size", 128), "hidden_size", 4, 4096)
        _integer(params.get("num_layers", 2), "num_layers", 1, 8)
        _number(params.get("dropout", 0.2), "dropout", 0.0, 0.9)
        _integer(params.get("max_steps", 300), "max_steps", 1, 100_000)
        _integer(params.get("batch_size", 512), "batch_size", 16, 1_000_000)
        _integer(params.get("eval_steps", 10), "eval_steps", 1, 10_000)
        _number(params.get("weight_decay", 0.0001), "weight_decay", 0.0, 1_000_000.0)
        _integer(params.get("early_stopping_rounds", 20), "early_stopping_rounds", early_stopping_minimum, 10_000)
        return
    elif kind in {"transformer", "nativetft"}:
        _integer(params.get("lookback_window", 60), "lookback_window", 2, 252)
        d_model = _integer(params.get("d_model", 64), "d_model", 8, 1024)
        nhead = _integer(params.get("nhead", 4), "nhead", 1, 32)
        if d_model % nhead != 0:
            raise PermanentJobError("d_model必须能被nhead整除")
        if kind == "transformer":
            _integer(params.get("transformer_layers", 2), "transformer_layers", 1, 8)
        else:
            _integer(params.get("gru_hidden_size", 64), "gru_hidden_size", 4, 4096)
            _integer(params.get("num_layers", 1), "num_layers", 1, 8)
        _integer(params.get("dim_feedforward", 256), "dim_feedforward", 16, 16_384)
        _number(params.get("dropout", 0.2), "dropout", 0.0, 0.9)
        _validate_deep_training_loop(
            params, default_batch_size=256,
            allow_disabled_early_stopping=allow_disabled_early_stopping,
        )
        return
    elif kind == "tcn":
        _integer(params.get("lookback_window", 60), "lookback_window", 2, 252)
        _integer(params.get("hidden_size", 128), "hidden_size", 4, 4096)
        _integer(params.get("kernel_size", 5), "kernel_size", 2, 64)
        _integer(params.get("num_layers", 5), "num_layers", 1, 16)
        _number(params.get("dropout", 0.5), "dropout", 0.0, 0.9)
        _validate_deep_training_loop(
            params, default_batch_size=256,
            allow_disabled_early_stopping=allow_disabled_early_stopping,
        )
        return
    elif kind == "tabnet":
        _integer(params.get("n_d", 64), "n_d", 4, 4096)
        _integer(params.get("n_a", 64), "n_a", 4, 4096)
        _integer(params.get("n_steps", 5), "n_steps", 1, 64)
        _integer(params.get("n_shared", 2), "n_shared", 0, 16)
        _integer(params.get("n_ind", 2), "n_ind", 0, 16)
        _integer(params.get("max_steps", 100), "max_steps", 1, 100_000)
        _integer(params.get("batch_size", 4096), "batch_size", 16, 1_000_000)
        _integer(params.get("early_stopping_rounds", 20), "early_stopping_rounds", early_stopping_minimum, 10_000)
        if not isinstance(params.get("pretrain", False), bool):
            raise PermanentJobError("tabnet.pretrain必须是布尔值")
        return
    elif kind == "transformer_lstm":
        _integer(params.get("lookback_window", 60), "lookback_window", 2, 252)
        d_model = _integer(params.get("d_model", 64), "d_model", 8, 1024)
        nhead = _integer(params.get("nhead", 4), "nhead", 1, 32)
        if d_model % nhead != 0:
            raise PermanentJobError("d_model必须能被nhead整除")
        _integer(params.get("transformer_layers", 2), "transformer_layers", 1, 8)
        _integer(params.get("dim_feedforward", 256), "dim_feedforward", 16, 16_384)
        _integer(params.get("lstm_hidden_size", 128), "lstm_hidden_size", 4, 4096)
        _integer(params.get("lstm_layers", 1), "lstm_layers", 1, 8)
        _number(params.get("dropout", 0.2), "dropout", 0.0, 0.9)
        _integer(params.get("max_steps", 300), "max_steps", 1, 100_000)
        _integer(params.get("batch_size", 256), "batch_size", 16, 1_000_000)
        _integer(params.get("eval_steps", 10), "eval_steps", 1, 10_000)
        _number(params.get("weight_decay", 0.0001), "weight_decay", 0.0, 1_000_000.0)
        _integer(params.get("early_stopping_rounds", 20), "early_stopping_rounds", early_stopping_minimum, 10_000)
        return
    _integer(params.get("n_estimators", 1000), "n_estimators", 1, 100_000)
    _integer(params.get("early_stopping_rounds", 50), "early_stopping_rounds", early_stopping_minimum, 10_000)
    if kind in {"lightgbm", "xgboost"}:
        _number(params.get("subsample", 0.9), "subsample", 0.01, 1.0)
        _number(params.get("colsample_bytree", 0.9), "colsample_bytree", 0.01, 1.0)
        _number(params.get("reg_alpha", 0.0), "reg_alpha", 0.0, 1_000_000.0)
        _number(params.get("reg_lambda", 1.0 if kind == "xgboost" else 0.0), "reg_lambda", 0.0, 1_000_000.0)


def _validate_deep_training_loop(
    params: dict[str, Any], *, default_batch_size: int,
    allow_disabled_early_stopping: bool = False,
) -> None:
    _integer(params.get("max_steps", 300), "max_steps", 1, 100_000)
    _integer(params.get("batch_size", default_batch_size), "batch_size", 16, 1_000_000)
    _integer(params.get("eval_steps", 10), "eval_steps", 1, 10_000)
    _number(params.get("weight_decay", 0.0001), "weight_decay", 0.0, 1_000_000.0)
    _integer(
        params.get("early_stopping_rounds", 20), "early_stopping_rounds",
        0 if allow_disabled_early_stopping else 1, 10_000,
    )


def _validate_walk_forward(source: Any) -> None:
    if not isinstance(source, dict):
        raise PermanentJobError("walk_forward必须是对象")
    if source.get("enabled", False) is not True:
        return
    strategy = str(source.get("strategy") or "rolling")
    if strategy not in {"rolling", "expanding"}:
        raise PermanentJobError("Walk-Forward策略只允许rolling或expanding")
    _integer(
        source.get("train_sessions", 756),
        "walk_forward.train_sessions", 252, 2520,
    )
    _integer(
        source.get("valid_sessions", 60),
        "walk_forward.valid_sessions", 0, 504,
    )
    test_sessions = _integer(
        source.get("test_sessions", 20),
        "walk_forward.test_sessions", 1, 252,
    )
    step_sessions = _integer(
        source.get("step_sessions", 20),
        "walk_forward.step_sessions", 1, 252,
    )
    _integer(
        source.get("embargo_sessions", 5),
        "walk_forward.embargo_sessions", 1, 252,
    )
    try:
        oos_start = date.fromisoformat(str(source.get("oos_date_start") or ""))
        oos_end = date.fromisoformat(str(source.get("oos_date_end") or ""))
    except ValueError as exc:
        raise PermanentJobError("Walk-Forward样本外起止日期必须是ISO日期") from exc
    if oos_start > oos_end:
        raise PermanentJobError("Walk-Forward样本外开始日期不得晚于结束日期")
    if step_sessions != test_sessions:
        raise PermanentJobError(
            "Walk-Forward步长必须等于测试窗口，确保样本外日期完整且不重叠"
        )


def _validate_incremental_training(
    source: Any,
    *,
    model_id: str,
    planned_version: int,
    model_kind: str,
    dataset_spec: dict[str, Any],
    walk_forward: dict[str, Any],
) -> None:
    if not isinstance(source, dict):
        raise PermanentJobError("incremental_training必须是对象")
    if str(source.get("schema_version") or "") != "alphablocks.incremental-training.v1":
        raise PermanentJobError("增量训练schema_version无效")
    if str(source.get("mode") or "") != "lightgbm_append_trees_new_data_only":
        raise PermanentJobError("增量训练mode无效")
    if model_kind != "lightgbm":
        raise PermanentJobError("首版增量续训只支持LightGBM")
    if walk_forward.get("enabled") is True:
        raise PermanentJobError("增量续训暂不支持Walk-Forward")
    if _identifier(source.get("source_model_id"), "source_model_id") != model_id:
        raise PermanentJobError("增量训练来源模型ID不一致")
    source_version = _integer(
        source.get("source_model_version"), "source_model_version", 1, 1_000_000,
    )
    if source_version >= planned_version:
        raise PermanentJobError("增量训练来源版本必须早于目标版本")
    _identifier(source.get("source_job_id"), "source_job_id")
    bundle_identity = source.get("source_bundle_identity")
    if bundle_identity is not None:
        if not isinstance(bundle_identity, dict):
            raise PermanentJobError("source_bundle_identity必须是对象")
        _identifier(
            bundle_identity.get("model_id"),
            "source_bundle_identity.model_id",
        )
        _integer(
            bundle_identity.get("model_version"),
            "source_bundle_identity.model_version", 1, 1_000_000,
        )
        _identifier(
            bundle_identity.get("job_id"),
            "source_bundle_identity.job_id",
        )
    source_end = _date(source.get("source_date_end"), "source_date_end")
    if source_end >= _date(dataset_spec.get("date_end"), "date_end"):
        raise PermanentJobError("增量训练数据集没有新增日期")
    _integer(
        source.get("minimum_new_trading_sessions", 60),
        "minimum_new_trading_sessions", 60, 504,
    )
    artifact = source.get("source_artifact")
    if not isinstance(artifact, dict):
        raise PermanentJobError("增量训练缺少来源模型产物")
    _identifier(artifact.get("artifact_id"), "source_artifact.artifact_id")
    digest = str(artifact.get("sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise PermanentJobError("增量训练来源模型产物SHA256无效")
    relative = Path(str(artifact.get("relative_path") or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise PermanentJobError("增量训练来源模型产物路径无效")


def _validate_inference_config(config: dict[str, Any], *, model_id: str, version: int) -> None:
    if str(config.get("schema_version") or "") != "alphablocks.model-inference.v1":
        raise PermanentJobError("每日推理schema_version无效")
    source = config.get("source_model")
    inference = config.get("inference")
    if not isinstance(source, dict) or not isinstance(inference, dict):
        raise PermanentJobError("每日推理缺少source_model或inference规格")
    if _identifier(source.get("model_id"), "source_model.model_id") != model_id:
        raise PermanentJobError("每日推理源模型ID不一致")
    if _integer(source.get("model_version"), "source_model.model_version", 1, 1_000_000) != version:
        raise PermanentJobError("每日推理源模型版本不一致")
    _identifier(source.get("training_job_id"), "source_model.training_job_id")
    _identifier(source.get("artifact_id"), "source_model.artifact_id")
    digest = str(source.get("artifact_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise PermanentJobError("每日推理模型产物SHA256无效")
    series = source.get('walk_forward_artifact')
    if series is not None:
        if not isinstance(series, dict):
            raise PermanentJobError('每日推理滚动序列产物无效')
        _identifier(series.get('artifact_id'), 'walk_forward_artifact.artifact_id')
        if not re.fullmatch(r'[0-9a-f]{64}', str(series.get('sha256') or '')):
            raise PermanentJobError('每日推理滚动序列SHA256无效')
        if not isinstance(series.get('size_bytes'), int) or series['size_bytes'] <= 0:
            raise PermanentJobError('每日推理滚动序列大小无效')
    trade_date = _date(inference.get("trade_date"), "inference.trade_date")
    cutoff = _datetime(inference.get("data_cutoff"), "inference.data_cutoff")
    feature_cutoff = _datetime(inference.get("feature_cutoff_at"), "inference.feature_cutoff_at")
    if feature_cutoff.date() != trade_date:
        raise PermanentJobError("feature_cutoff_at必须属于目标交易日")
    if cutoff < feature_cutoff:
        raise PermanentJobError("data_cutoff不能早于feature_cutoff_at")


def safe_job_dir(work_root: Path, job_id: str) -> Path:
    clean = _identifier(job_id, "job_id")
    jobs_root = (Path(work_root) / "jobs").resolve()
    candidate = (jobs_root / clean).resolve()
    if candidate.parent != jobs_root:
        raise PermanentJobError("任务工作目录越界")
    return candidate


class CancellationToken:
    def __init__(
        self,
        shutdown_event: threading.Event | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        self._cancel = threading.Event()
        self._shutdown = shutdown_event or threading.Event()
        self._deadline = (
            time.monotonic() + max(0.001, float(timeout_seconds))
            if timeout_seconds is not None
            else None
        )
        self.reason = ""

    def cancel(self, reason: str = "任务已请求取消") -> None:
        self.reason = reason
        self._cancel.set()

    @property
    def canceled(self) -> bool:
        return self._cancel.is_set()

    def checkpoint(self) -> None:
        if self._cancel.is_set():
            raise JobCanceled(self.reason or "任务已请求取消")
        if self._shutdown.is_set():
            raise WorkerShutdown("调度服务正在关闭，任务将重新排队")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TrainingTimeout("训练任务已达到最长运行时长")


ProgressCallback = Callable[[str, int, dict[str, Any]], None]


def _identifier(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not IDENTIFIER.fullmatch(text):
        raise PermanentJobError(f"{field}格式无效")
    return text


def _date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise PermanentJobError(f"{field}不是有效日期") from exc


def _datetime(value: Any, field: str) -> datetime:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise PermanentJobError(f"{field}不是有效时间") from exc
    if result.tzinfo is None:
        raise PermanentJobError(f"{field}必须包含时区")
    return result


def _integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PermanentJobError(f"{field}必须是整数") from exc
    if not minimum <= result <= maximum:
        raise PermanentJobError(f"{field}必须在{minimum}到{maximum}之间")
    return result


def _number(value: Any, field: str, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PermanentJobError(f"{field}必须是数字") from exc
    if not minimum <= result <= maximum:
        raise PermanentJobError(f"{field}必须在{minimum}到{maximum}之间")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = ["CancellationToken", "ProgressCallback", "safe_job_dir", "validate_job"]
