from __future__ import annotations

from datetime import datetime
from contextlib import ExitStack
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from factor_service import model_repository
from factor_service.config import load_settings as load_service_settings
from factor_service.model_artifacts import ArtifactError, ModelArtifactStore
from factor_service.research.dataset_archive import dataset_files, DATASET_FILES, DatasetCacheBusy
from factor_service.model_backtest import run_model_backtest_job
from factor_service.model_diagnostics import (
    architecture_walk_forward_attribution,
    artifact_model_feature_importance,
    artifact_model_permutation_importance,
    artifact_model_shap_summary,
    artifact_model_training_diagnostics,
    dataset_factor_validation_audit,
    dataset_feature_drift,
    dataset_feature_redundancy,
    dataset_walk_forward_attribution,
    isolated_artifact_model_permutation_importance,
)
from factor_service.model_validation import (
    assess_model_validation,
    assess_parameter_trial,
    assess_walk_forward_stability,
)
from factor_service.model_registry import (
    build_model_research_report,
    render_model_research_report_markdown,
)
from factor_service.model_reproducibility import build_model_reproducibility_audit
from factor_service.model_selection import (
    daily_selection as run_daily_selection,
    negative_selection as run_negative_selection,
)
from factor_service.research.config import load_settings as load_research_settings
from factor_service.research.dataset import DatasetBuilder
from factor_service.research.dataset_preflight import DatasetPreflightService
from factor_service.research.dataset_preview import DatasetPreviewService
from factor_service.model_research_repository import (
    ModelResearchConflict,
    ModelResearchError,
    ModelResearchNotFound,
    ModelResearchRepository,
    _execution_spec,
)
from factor_service.research.worker import ResearchWorker
from factor_service.research.schedule import dispatch_job as dispatch_model_job
from factor_service.research.sample_filter_formula import (
    MAX_CUSTOM_SAMPLE_FILTERS,
    MAX_SAMPLE_FILTER_WINDOW,
    normalize_custom_sample_filters,
)
from factor_service.research.size_rotation_feature import (
    normalize_size_rotation_feature,
)
from factor_service.research.trainer import SEQUENCE_MODEL_KINDS
from factor_service.research.training_resource_settings import (
    TrainingResourceRevisionConflict,
    TrainingResourceSettingsRepository,
    configured_stock_pool_sources,
    frozen_data_binding,
    frozen_training_data_bindings,
    get_training_resource_settings,
    normalize_frozen_training_data_bindings,
    required_training_data_binding_ids,
    training_data_binding_catalog,
)
from factor_service.schemas import ModelBacktestJobCreate


router = APIRouter(prefix="/model-research", tags=["model-research"])
repository = ModelResearchRepository()


def _worker(request: Request) -> ResearchWorker:
    worker = getattr(request.app.state, "research_worker", None)
    if worker is None:
        raise ModelResearchConflict("研究调度器尚未启动")
    return worker


def _raise(exc: Exception) -> None:
    if isinstance(exc, ModelResearchNotFound):
        status = HTTPStatus.NOT_FOUND
    elif isinstance(exc, (ModelResearchConflict, DatasetCacheBusy)):
        status = HTTPStatus.CONFLICT
    elif isinstance(exc, (ArtifactError, FileNotFoundError)):
        status = HTTPStatus.NOT_FOUND
    elif isinstance(exc, (ModelResearchError, TypeError, ValueError)):
        status = HTTPStatus.BAD_REQUEST
    else:
        status = HTTPStatus.INTERNAL_SERVER_ERROR
    raise HTTPException(status_code=int(status), detail=str(exc)) from exc


def _dispatch(request: Request, job: dict[str, Any]) -> tuple[dict[str, Any], int]:
    return dispatch_model_job(repository, _worker(request), job)


def _model_validation_view(
    model: dict[str, Any], backtest: Any | None,
) -> dict[str, Any]:
    assessed = assess_model_validation(model.get("metrics_json"), backtest)
    stored = dict((model.get("manifest_json") or {}).get("validation") or {})
    if (
        stored.get("manual_override") is True
        and str(stored.get("backtest_job_id") or "")
        == str(assessed.get("backtest_job_id") or "")
    ):
        return stored
    return assessed


def _experiment_view(
    model: dict[str, Any], summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    experiment = dict(
        (model.get("job_config_json") or {}).get("experiment") or {}
    )
    if not experiment:
        return None
    strategy = str(experiment.get("strategy") or "")
    is_horizon_study = strategy == "horizon_grid"
    unit = {
        "horizon_grid": "标签周期",
        "factor_ablation": "因子消融方案",
        "model_ensemble": "模型",
    }.get(strategy, "参数组合")
    if summary is None:
        experiment.update({
            "backtest_allowed": False,
            "backtest_blocked_reason": f"{unit}选优状态尚不可用",
        })
        return experiment
    selection = dict(summary.get("selection") or {})
    trial = next((
        dict(item)
        for item in selection.get("trial_assessments") or []
        if str(item.get("job_id") or "") == str(model.get("job_id") or "")
    ), {})
    selected_job_id = str(selection.get("selected_job_id") or "")
    current_job_id = str(model.get("job_id") or "")
    allowed = selection.get("status") == "selected" and selected_job_id == current_job_id
    if allowed:
        blocked_reason = ""
    elif selection.get("status") == "evaluating":
        blocked_reason = f"{unit}研究尚未全部完成，暂不能进入Top20回测"
    elif selection.get("status") == "no_qualified_trials":
        blocked_reason = f"该实验没有{unit}通过验证集门槛"
    elif selected_job_id:
        blocked_reason = f"仅验证集排名最高且通过门槛的{unit}版本可进入Top20回测"
    else:
        blocked_reason = f"{unit}研究尚未选出可回测版本"
    experiment.update({
        "selection": {
            key: value for key, value in selection.items()
            if key != "trial_assessments"
        },
        "trial_assessment": trial or None,
        "is_selected": allowed,
        "backtest_allowed": allowed,
        "backtest_blocked_reason": blocked_reason,
    })
    return experiment


def _model_response(
    model: dict[str, Any], backtest: Any | None,
    experiment_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(model)
    validation = _model_validation_view(model, backtest)
    registry_state = str(model.get("state") or "candidate")
    dataset = dict(model.get("dataset_spec") or {})
    registry_scope = str(model.get("registry_scope") or "").strip() or (
        f"{dataset.get('research_target') or 'stock_selection'}:"
        f"{dataset.get('universe_id') or 'csi500'}"
    )
    result["registry_state"] = registry_state
    result["state"] = (
        "archived" if registry_state == "archived"
        else "validated" if validation.get("approved") is True
        else "candidate"
    )
    result["registry"] = {
        "stage": result["state"],
        "is_default": bool(model.get("is_default", False)),
        "scope": registry_scope,
        "note": str(model.get("registry_note") or ""),
        "archived_at": model.get("archived_at"),
        "can_set_default": (
            registry_state != "archived" and validation.get("approved") is True
        ),
        "can_archive": (
            registry_state != "archived" and not bool(model.get("is_default", False))
        ),
    }
    result["experiment"] = _experiment_view(model, experiment_summary)
    result["backtest_eligibility"] = _model_backtest_eligibility(model)
    result["latest_backtest"] = backtest.model_dump() if backtest is not None else None
    result["validation"] = validation
    return result


def _experiment_summary_for_model(model: dict[str, Any]) -> dict[str, Any] | None:
    experiment_id = str(
        ((model.get("job_config_json") or {}).get("experiment") or {}).get(
            "experiment_id"
        ) or ""
    )
    return repository.get_training_experiment(experiment_id) if experiment_id else None


def _assert_experiment_backtest_allowed(model: dict[str, Any]) -> None:
    summary = _experiment_summary_for_model(model)
    experiment = _experiment_view(model, summary)
    if experiment and experiment.get("backtest_allowed") is not True:
        raise ModelResearchConflict(str(
            experiment.get("backtest_blocked_reason")
            or "参数实验版本尚未获得Top20回测资格"
        ))


def _model_backtest_eligibility(model: dict[str, Any]) -> dict[str, Any]:
    validation_metrics = dict(
        (model.get("metrics_json") or {}).get("validation") or {}
    )
    assessment = assess_parameter_trial(validation_metrics)
    walk_forward = dict((model.get("manifest_json") or {}).get("walk_forward") or {})
    stability: dict[str, Any] = {}
    if walk_forward.get("enabled") is True:
        stability = dict(walk_forward.get("stability") or {})
        if not stability:
            stability = assess_walk_forward_stability(
                walk_forward.get("aggregate") or {},
                window_count=int(walk_forward.get("window_count") or 0),
            )
    walk_forward_passed = (
        walk_forward.get("enabled") is not True or stability.get("passed") is True
    )
    failed_labels = [
        str(item.get("label") or item.get("key") or "")
        for item in assessment.get("checks") or []
        if item.get("passed") is not True
    ]
    if not walk_forward_passed:
        failed_labels.append("Walk-Forward跨期稳定性")
    passed = assessment.get("passed") is True and walk_forward_passed
    return {
        **assessment,
        "policy": "alphablocks.top20-eligibility.v2",
        "passed": passed,
        "selection_split": (
            "validation+walk_forward_oos"
            if walk_forward.get("enabled") is True else "validation"
        ),
        "walk_forward_required": walk_forward.get("enabled") is True,
        "walk_forward_stability": stability or None,
        "reason": "" if passed else "正式回测准入未通过：" + "、".join(failed_labels),
    }


def _assert_validation_backtest_allowed(model: dict[str, Any]) -> None:
    eligibility = _model_backtest_eligibility(model)
    if eligibility.get("passed") is not True:
        raise ModelResearchConflict(
            str(eligibility.get("reason") or "验证集未达到Top20回测资格")
        )


def _assert_stock_prediction_scope(model: dict[str, Any], action: str) -> None:
    dataset = dict(model.get("dataset_spec") or {})
    scope = str(dataset.get("prediction_scope") or "stock").strip().lower()
    if scope != "stock":
        raise ModelResearchConflict(
            f"{dataset.get('research_target') or scope}模型不能直接{action}；"
            "请在模型架构中作为门控引擎使用"
        )


def _assert_model_active(model: dict[str, Any], action: str) -> None:
    if str(model.get("state") or "candidate") == "archived":
        raise ModelResearchConflict(f"已归档模型不能{action}；请先从模型池恢复")


def _validate_execution_node(payload: dict[str, Any]) -> None:
    execution = _execution_spec(payload.get("execution") or {
        "node_id": payload.get("execution_node_id") or "local",
    })
    from factor_service.research.remote import get_remote_node

    for node_id in execution.get("node_ids", [execution["node_id"]]):
        if node_id == "local":
            continue
        node = get_remote_node(node_id)
        if execution["max_runtime_minutes"] > int(node.max_runtime_minutes):
            raise ModelResearchConflict(
                f"训练节点“{node.name}”单次任务最长{node.max_runtime_minutes}分钟"
            )


def _freeze_training_data_bindings(payload: dict[str, Any]) -> None:
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        return
    required_ids = required_training_data_binding_ids(dataset)
    universe_source = dataset.get("universe_source")
    configured_source = (
        isinstance(universe_source, dict)
        and str(universe_source.get("source_kind") or "")
        == "configured_stock_pool"
    )
    settings = get_training_resource_settings() if configured_source else None
    if configured_source:
        current_sources = {
            str(item.get("source_id") or ""): {
                key: value
                for key, value in item.items()
                if key != "available"
            }
            for item in configured_stock_pool_sources(settings)
        }
        current = current_sources.get(
            str(universe_source.get("source_id") or "")
        )
        if current != universe_source:
            raise ModelResearchConflict(
                "模型训练股票池配置已变更，请重新校验Draft"
            )
    rotation_source = dataset.get("size_rotation_feature")
    try:
        rotation = normalize_size_rotation_feature(
            rotation_source, default_enabled=False,
        )
    except ValueError as exc:
        raise ModelResearchConflict(str(exc)) from exc
    if rotation_source is not None and rotation != rotation_source:
        raise ModelResearchConflict(
            "模型训练大小盘轮动特征不是规范化冻结规格"
        )
    dataset["size_rotation_feature"] = rotation
    if rotation["enabled"]:
        settings = settings or get_training_resource_settings()
        current_sources = {
            str(item.get("source_id") or ""): item
            for item in configured_stock_pool_sources(settings)
        }
        for pool_name in ("large_pool", "small_pool"):
            pool = rotation[pool_name]
            if current_sources.get(str(pool["source_id"])) != pool:
                raise ModelResearchConflict(
                    "大小盘轮动股票池配置已变更，请重新选择股票池"
                )
    if dataset.get("data_bindings"):
        try:
            frozen = normalize_frozen_training_data_bindings(
                dataset.get("data_bindings"),
            )
        except ValueError as exc:
            raise ModelResearchConflict(str(exc)) from exc
        missing = [
            binding_id for binding_id in required_ids
            if frozen_data_binding(frozen, binding_id) is None
        ]
        if missing:
            raise ModelResearchConflict(
                "dataset.data_bindings缺少冻结绑定: " + ", ".join(missing)
            )
        dataset["data_bindings"] = frozen
        feature = dataset.get("industry_feature") or {}
        if isinstance(feature, dict) and feature.get("enabled") is True:
            dataset["industry_feature"] = {
                **feature,
                "schema_version": "alphablocks.stock-industry-one-hot.v2",
                "data_binding": frozen["bindings"].get(
                    "stock_industry_one_hot"
                ),
            }
        return
    settings = settings or get_training_resource_settings()
    if str(settings.get("schema_version") or "").endswith(".v1"):
        feature = dataset.get("industry_feature") or {}
        if not isinstance(feature, dict) or feature.get("enabled") is not True:
            return
        try:
            binding = frozen_training_data_bindings(
                settings, ["stock_industry_one_hot"],
            )["bindings"]["stock_industry_one_hot"]
        except ValueError as exc:
            raise ModelResearchConflict(str(exc)) from exc
        dataset["industry_feature"] = {
            **feature,
            "schema_version": "alphablocks.stock-industry-one-hot.v2",
            "data_binding": binding,
        }
        return
    try:
        frozen = frozen_training_data_bindings(
            settings, required_ids,
        )
    except ValueError as exc:
        raise ModelResearchConflict(str(exc)) from exc
    dataset["data_bindings"] = frozen
    feature = dataset.get("industry_feature") or {}
    if isinstance(feature, dict) and feature.get("enabled") is True:
        dataset["industry_feature"] = {
            **feature,
            "schema_version": "alphablocks.stock-industry-one-hot.v2",
            "data_binding": frozen["bindings"].get(
                "stock_industry_one_hot"
            ),
        }


@router.get("/jobs")
def list_jobs(
    status: str = Query(default=""),
    experiment_id: str = Query(default=""),
    kind: str = Query(default=""),
    model_id: str = Query(default=""),
    model_version: int | None = Query(default=None, ge=1),
    trade_date: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "jobs": repository.list_jobs(
                status=status, experiment_id=experiment_id, kind=kind,
                model_id=model_id, model_version=model_version,
                trade_date=trade_date, limit=limit,
            ),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/training-targets")
def get_training_targets() -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "targets": DatasetBuilder(load_research_settings()).target_capabilities(),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/training-data-bindings")
def get_model_training_resource_settings() -> dict[str, Any]:
    try:
        settings = get_training_resource_settings()
        return {
            "ok": True,
            "storage": "postgresql",
            "capabilities": training_data_binding_catalog(),
            "settings": settings,
            "universe_sources": configured_stock_pool_sources(settings),
        }
    except Exception as exc:
        _raise(exc)


@router.put("/training-data-bindings")
def update_model_training_resource_settings(
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    try:
        source = payload.get("settings")
        if not isinstance(source, dict):
            raise ModelResearchError("settings必须是对象")
        expected_revision = payload.get("expected_revision")
        if not isinstance(expected_revision, int):
            raise ModelResearchError("expected_revision必须是整数")
        settings = TrainingResourceSettingsRepository().update(
            source, expected_revision=expected_revision,
        )
        return {
            "ok": True,
            "storage": "postgresql",
            "capabilities": training_data_binding_catalog(),
            "settings": settings,
            "universe_sources": configured_stock_pool_sources(settings),
        }
    except TrainingResourceRevisionConflict as exc:
        _raise(ModelResearchConflict(str(exc)))
    except Exception as exc:
        _raise(exc)


@router.get("/sample-filter-formulas/catalog")
def get_sample_filter_formula_catalog() -> dict[str, Any]:
    return {
        "ok": True,
        "max_formulas": MAX_CUSTOM_SAMPLE_FILTERS,
        "max_window": MAX_SAMPLE_FILTER_WINDOW,
        "examples": [
            {
                "name": "价格与流动性",
                "expression": "$close >= 5 && $amount >= 100000000",
            },
            {
                "name": "剔除停牌和北交所",
                "expression": "$is_suspended == 0 && $is_bjs == 0",
            },
            {
                "name": "站上20日均线",
                "expression": "$close > Mean($close, 20)",
            },
        ],
    }


@router.post("/sample-filter-formulas/validate")
def validate_sample_filter_formula(
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    try:
        formula = normalize_custom_sample_filters([payload])[0]
    except (TypeError, ValueError) as exc:
        return {
            "ok": True,
            "valid": False,
            "expression": str(payload.get("expression") or ""),
            "error_message": str(exc),
        }
    return {
        "ok": True,
        "valid": True,
        "formula": formula,
        "required_fields": formula["required_fields"],
        "max_window": formula["max_window"],
    }


@router.get("/market-history")
def get_market_history(
    entity_code: str = Query(..., min_length=2, max_length=32),
    through_date: str = Query(default=""),
    limit: int = Query(default=60, ge=10, le=240),
) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "market": model_repository.stock_market_history(
                entity_code, through_date=through_date, limit=limit,
            ),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/execution-nodes")
def get_execution_nodes() -> dict[str, Any]:
    try:
        from factor_service.research.remote import execution_nodes

        return {"ok": True, "storage": "postgresql", "nodes": execution_nodes()}
    except Exception as exc:
        _raise(exc)


@router.get("/execution-node-settings")
def get_execution_node_settings() -> dict[str, Any]:
    try:
        from factor_service.research.remote_settings import list_remote_node_settings

        return {
            "ok": True,
            "storage": "postgresql",
            "nodes": list_remote_node_settings(),
            "security": {
                "secret_storage": "postgresql_aes_gcm",
                "password_storage": "postgresql_encrypted",
                "password_values_returned": False,
                "ssh_private_key_storage": "postgresql_encrypted",
                "ssh_private_key_values_returned": False,
                "api_token_storage": "postgresql_encrypted",
                "api_token_values_returned": False,
                "master_key_storage": "process_environment",
            },
        }
    except Exception as exc:
        _raise(exc)


@router.post("/execution-node-settings", status_code=HTTPStatus.CREATED)
def create_execution_node_setting(
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    try:
        from factor_service.research.remote_settings import create_remote_node_setting

        return {"ok": True, "node": create_remote_node_setting(payload)}
    except Exception as exc:
        _raise(exc)


@router.put("/execution-node-settings/{node_id}")
def update_execution_node_setting(
    node_id: str, payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    try:
        from factor_service.research.remote_settings import update_remote_node_setting

        return {
            "ok": True,
            "node": update_remote_node_setting(node_id, payload),
        }
    except Exception as exc:
        _raise(exc)


@router.delete("/execution-node-settings/{node_id}")
def delete_execution_node_setting(node_id: str) -> dict[str, Any]:
    try:
        from factor_service.research.remote_settings import delete_remote_node_setting

        return {"ok": True, **delete_remote_node_setting(node_id)}
    except Exception as exc:
        _raise(exc)


@router.get("/execution-nodes/{node_id}/status")
def get_execution_node_status(node_id: str, request: Request) -> dict[str, Any]:
    try:
        if node_id == "local":
            status = _worker(request).status()
            return {
                "ok": True,
                "status": {
                    "id": "local", "name": "本机训练", "type": "local",
                    "online": bool(status.get("ready")),
                    "training_active": bool(status.get("busy")),
                    "active_job_id": status.get("active_job_id") or "",
                    "progress": status.get("progress") or {},
                },
            }
        from factor_service.research.autodl import sanitize_snapshot
        from factor_service.research.remote import (
            RemoteTransport,
            autodl_client,
            get_remote_node,
            node_with_autodl_endpoint,
        )

        node = get_remote_node(node_id)
        if node.lifecycle_provider != "autodl_pro":
            return {
                "ok": True,
                "status": RemoteTransport(node).collect_status(),
            }
        client = autodl_client(node)
        power_state = client.status()
        if power_state != "running":
            return {
                "ok": True,
                "status": {
                    **node.public(), "online": False,
                    "training_active": False, "power_state": power_state,
                },
            }
        snapshot = client.snapshot()
        active_node = node_with_autodl_endpoint(node, snapshot)
        status = RemoteTransport(active_node).collect_status()
        status.update({
            "power_state": power_state,
            "autodl": sanitize_snapshot(snapshot),
        })
        return {"ok": True, "status": status}
    except Exception as exc:
        _raise(exc)


@router.post("/execution-nodes/{node_id}/test")
def test_execution_node(node_id: str, request: Request) -> dict[str, Any]:
    try:
        if node_id == "local":
            status = _worker(request).status()
            return {
                "ok": True, "success": bool(status.get("ready")),
                "node_id": "local",
                "detail": "AlphaFactorService研究调度器可用",
            }
        from factor_service.research.remote import (
            RemoteTransport,
            autodl_client,
            get_remote_node,
            node_with_autodl_endpoint,
        )

        node = get_remote_node(node_id)
        if node.lifecycle_provider == "autodl_pro":
            client = autodl_client(node)
            power_state = client.status()
            if power_state != "running":
                return {
                    "ok": True, "success": False, "node_id": node_id,
                    "power_state": power_state,
                    "detail": f"AutoDL实例当前状态: {power_state}",
                }
            node = node_with_autodl_endpoint(node, client.snapshot())
        return {
            "ok": True,
            **RemoteTransport(node).test_connection(),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/execution-nodes/{node_id}/lifecycle")
def get_execution_node_lifecycle(node_id: str) -> dict[str, Any]:
    try:
        from factor_service.research.autodl import sanitize_snapshot
        from factor_service.research.remote import autodl_client, get_remote_node

        node = get_remote_node(node_id)
        client = autodl_client(node)
        power_state = client.status()
        snapshot = (
            sanitize_snapshot(client.snapshot())
            if power_state == "running" else None
        )
        return {
            "ok": True,
            "lifecycle": {
                "provider": "autodl_pro",
                "instance_uuid": node.instance_uuid,
                "configured": client.configured(),
                "power_state": power_state,
                "snapshot": snapshot,
                "capabilities": {
                    "power_on": True,
                    "power_off": True,
                    "save_image": True,
                    "list_images": True,
                    "create_instance_from_image": True,
                },
            },
        }
    except Exception as exc:
        _raise(exc)


@router.post("/execution-nodes/{node_id}/power-on")
def power_on_execution_node(node_id: str) -> dict[str, Any]:
    try:
        from factor_service.research.remote import autodl_client, get_remote_node

        node = get_remote_node(node_id)
        client = autodl_client(node)
        before = client.status()
        if before != "running":
            client.power_on()
        return {
            "ok": True,
            "lifecycle": {
                "provider": "autodl_pro", "instance_uuid": node.instance_uuid,
                "previous_state": before,
                "power_state": "running" if before == "running" else "starting",
            },
        }
    except Exception as exc:
        _raise(exc)


@router.post("/execution-nodes/{node_id}/power-off")
def power_off_execution_node(node_id: str, request: Request) -> dict[str, Any]:
    try:
        _assert_execution_node_idle(request, node_id)
        from factor_service.research.remote import autodl_client, get_remote_node

        node = get_remote_node(node_id)
        client = autodl_client(node)
        before = client.status()
        if before not in {"stopped", "shutdown", "closed"}:
            client.power_off()
        return {
            "ok": True,
            "lifecycle": {
                "provider": "autodl_pro", "instance_uuid": node.instance_uuid,
                "previous_state": before,
                "power_state": (
                    before if before in {"stopped", "shutdown", "closed"}
                    else "stopping"
                ),
            },
        }
    except Exception as exc:
        _raise(exc)


@router.get("/execution-nodes/{node_id}/images")
def list_execution_node_images(
    node_id: str,
    page_index: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=100),
) -> dict[str, Any]:
    try:
        from factor_service.research.remote import autodl_client, get_remote_node

        data = autodl_client(get_remote_node(node_id)).list_images(
            page_index=page_index, page_size=page_size,
        )
        images = [{
            key: item.get(key)
            for key in ("image_uuid", "name", "status", "image_size", "create_at")
            if item.get(key) is not None
        } for item in (data.get("list") or []) if isinstance(item, dict)]
        return {
            "ok": True, "images": images,
            "pagination": {
                key: data.get(key)
                for key in (
                    "page_index", "page_size", "max_page", "result_total",
                )
            },
        }
    except Exception as exc:
        _raise(exc)


@router.post("/execution-nodes/{node_id}/images", status_code=HTTPStatus.ACCEPTED)
def save_execution_node_image(
    node_id: str, request: Request, payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    try:
        _assert_execution_node_idle(request, node_id)
        from factor_service.research.remote import autodl_client, get_remote_node

        node = get_remote_node(node_id)
        client = autodl_client(node)
        power_state = client.status()
        if power_state not in {"stopped", "shutdown", "closed"}:
            raise ModelResearchConflict("保存镜像前必须先关闭AutoDL实例")
        image = client.save_image(str(payload.get("image_name") or ""))
        return {
            "ok": True,
            "image": {"image_uuid": str(image.get("image_uuid") or "")},
            "status": "saving",
        }
    except Exception as exc:
        _raise(exc)


def _assert_execution_node_idle(request: Request, node_id: str) -> None:
    worker_status = _worker(request).status()
    active_job_id = str(worker_status.get("active_job_id") or "")
    if not active_job_id:
        return
    try:
        active_job = repository.get_job(active_job_id)
    except Exception:
        raise ModelResearchConflict(
            f"研究调度器正在执行任务 {active_job_id}，不能关闭或保存节点",
        )
    active_node_id = str(
        ((active_job.get("config_json") or {}).get("execution") or {}).get(
            "node_id"
        ) or "local"
    )
    if active_node_id == node_id:
        raise ModelResearchConflict(
            f"节点正在执行训练任务 {active_job_id}，请等待完成或先取消任务",
        )


@router.post("/jobs", status_code=HTTPStatus.CREATED)
def create_job(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        _validate_execution_node(payload)
        _freeze_training_data_bindings(payload)
        return {"ok": True, "job": repository.create_training_job(payload)}
    except Exception as exc:
        _raise(exc)


@router.post("/experiments", status_code=HTTPStatus.CREATED)
def create_experiment(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        _validate_execution_node(payload)
        _freeze_training_data_bindings(payload)
        return {
            "ok": True,
            "experiment": repository.create_training_experiment(payload),
        }
    except Exception as exc:
        _raise(exc)


@router.post("/dataset-preflight")
def preflight_dataset(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Freeze bindings and validate calendar/split semantics without X/y."""
    try:
        _freeze_training_data_bindings(payload)
        return {
            "ok": True,
            "preflight": DatasetPreflightService().validate(payload),
        }
    except Exception as exc:
        _raise(exc)


@router.post("/dataset-previews", status_code=HTTPStatus.ACCEPTED)
def create_dataset_preview(
    background_tasks: BackgroundTasks,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    """Materialize the exact immutable X/y snapshot without training a model."""
    try:
        _freeze_training_data_bindings(payload)
        previews = DatasetPreviewService(repository)
        preview = previews.create(payload)
        if str(preview.get("status") or "") == "running":
            background_tasks.add_task(previews.run, str(preview["preview_id"]))
        return {"ok": True, "preview": preview}
    except Exception as exc:
        _raise(exc)


@router.get("/dataset-previews/{preview_id}")
def get_dataset_preview(preview_id: str) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "preview": DatasetPreviewService(repository).get(preview_id),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/dataset-previews/{preview_id}/sample")
def get_dataset_preview_sample(preview_id: str) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "sample": DatasetPreviewService(repository).sample(preview_id),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/dataset-previews/{preview_id}/events")
def list_dataset_preview_events(
    preview_id: str,
    after: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "events": DatasetPreviewService(repository).events(
                preview_id, after=after,
            ),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "experiment": repository.get_training_experiment(experiment_id),
        }
    except Exception as exc:
        _raise(exc)


@router.post(
    "/experiments/{experiment_id}/retry",
    status_code=HTTPStatus.CREATED,
)
def retry_experiment(experiment_id: str) -> dict[str, Any]:
    try:
        source = repository.get_training_experiment(experiment_id)
        for job in source.get("jobs") or []:
            _validate_execution_node(job.get("config_json") or {})
        return {
            "ok": True,
            "experiment": repository.restart_training_experiment(experiment_id),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return {"ok": True, "job": repository.get_job(job_id)}
    except Exception as exc:
        _raise(exc)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    try:
        return {"ok": True, "job": repository.cancel_job(job_id)}
    except Exception as exc:
        _raise(exc)


@router.post("/jobs/{job_id}/retry")
def retry_job(
    job_id: str,
    payload: dict[str, Any] | None = Body(default=None),
) -> dict[str, Any]:
    try:
        source = repository.get_job(job_id)
        _validate_execution_node(source.get("config_json") or {})
        return {
            "ok": True,
            "job": repository.retry_job(
                job_id,
                idempotency_key=str((payload or {}).get("idempotency_key") or ""),
            ),
        }
    except Exception as exc:
        _raise(exc)


@router.post("/jobs/{job_id}/register")
def register_training_result(
    job_id: str,
    payload: dict[str, Any] | None = Body(default=None),
) -> dict[str, Any]:
    try:
        requested_model_id = str((payload or {}).get("model_id") or "").strip()
        job = (
            repository.register_training_result(
                job_id,
                model_id=requested_model_id,
            )
            if requested_model_id
            else repository.register_training_result(job_id)
        )
        return {
            "ok": True,
            "job": job,
        }
    except Exception as exc:
        _raise(exc)


@router.post("/jobs/{job_id}/decline-registration")
def decline_training_result(job_id: str) -> dict[str, Any]:
    try:
        return {"ok": True, "job": repository.decline_training_result(job_id)}
    except Exception as exc:
        _raise(exc)


@router.post("/jobs/{job_id}/dispatch")
def dispatch_job(request: Request, job_id: str) -> JSONResponse:
    try:
        payload, status = _dispatch(request, repository.get_job(job_id))
        return JSONResponse(
            status_code=status,
            content=jsonable_encoder(payload),
        )
    except Exception as exc:
        _raise(exc)


@router.get("/jobs/{job_id}/events")
def list_events(
    job_id: str, after: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    try:
        return {"ok": True, "events": repository.list_events(job_id, after=after)}
    except Exception as exc:
        _raise(exc)


@router.get("/jobs/{job_id}/artifacts")
def list_artifacts(job_id: str) -> dict[str, Any]:
    try:
        return {"ok": True, "artifacts": repository.list_artifacts(job_id)}
    except Exception as exc:
        _raise(exc)


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(request: Request, artifact_id: str) -> FileResponse:
    usage = ExitStack()
    try:
        artifact = repository.get_artifact(artifact_id)
        if artifact.get("artifact_kind") in DATASET_FILES.values():
            usage.enter_context(dataset_files(
                str(artifact["dataset_hash"]), load_service_settings().model_artifacts_root,
            ))
        path = _worker(request).control.resolve_artifact(
            artifact_id,
            str(artifact.get("sha256") or ""),
        )
        return _PinnedDatasetFileResponse(
            path, media_type="application/octet-stream", filename=path.name, usage=usage,
        )
    except Exception as exc:
        usage.close()
        _raise(exc)


class _PinnedDatasetFileResponse(FileResponse):
    def __init__(self, *args, usage: ExitStack, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage = usage

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Includes client disconnect / streaming failure, not only success.
            import anyio
            from starlette.concurrency import run_in_threadpool
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(self.usage.close)


@router.get("/models")
def list_models(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    try:
        models = repository.list_models(limit=limit)
        backtests = model_repository.latest_model_backtests([
            (str(model["model_id"]), int(model["version"])) for model in models
        ])
        summaries: dict[str, dict[str, Any]] = {}
        responses = []
        for model in models:
            experiment_id = str(
                ((model.get("job_config_json") or {}).get("experiment") or {}).get(
                    "experiment_id"
                ) or ""
            )
            if experiment_id and experiment_id not in summaries:
                summaries[experiment_id] = repository.get_training_experiment(experiment_id)
            responses.append(_model_response(
                model,
                backtests.get((str(model["model_id"]), int(model["version"]))),
                summaries.get(experiment_id),
            ))
        return {"ok": True, "models": responses}
    except Exception as exc:
        _raise(exc)


@router.post("/model-scores/query")
def query_model_scores(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Resolve a model once and return its exact-date immutable score snapshot."""

    try:
        version_value = payload.get("model_version")
        resolved = repository.resolve_model_reference(
            str(payload.get("model") or payload.get("model_id") or ""),
            version=(int(version_value) if version_value not in (None, "") else None),
        )
        trade_date = datetime.fromisoformat(
            str(payload.get("date") or payload.get("trade_date") or "")[:10]
        ).date()
        topk_value = payload.get("topk")
        snapshot = model_repository.model_score_snapshot(
            model_id=str(resolved["model_id"]),
            model_version=int(resolved["model_version"]),
            trade_date=trade_date,
            topk=(int(topk_value) if topk_value not in (None, "") else None),
        )
        return {"ok": True, "resolved_model": resolved, "snapshot": snapshot}
    except Exception as exc:
        _raise(exc)


@router.post("/model-scores/query-range")
def query_model_scores_range(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Resolve one model revision and return bounded exact-date score snapshots."""

    try:
        version_value = payload.get("model_version")
        resolved = repository.resolve_model_reference(
            str(payload.get("model") or payload.get("model_id") or ""),
            version=(int(version_value) if version_value not in (None, "") else None),
        )
        date_start = datetime.fromisoformat(str(payload.get("date_start") or "")[:10]).date()
        date_end = datetime.fromisoformat(str(payload.get("date_end") or "")[:10]).date()
        topk_value = payload.get("topk")
        snapshots = model_repository.model_score_snapshots(
            model_id=str(resolved["model_id"]),
            model_version=int(resolved["model_version"]),
            date_start=date_start,
            date_end=date_end,
            topk=(int(topk_value) if topk_value not in (None, "") else None),
        )
        return {"ok": True, "resolved_model": resolved, "snapshots": snapshots}
    except Exception as exc:
        _raise(exc)


@router.get("/architectures")
def list_model_architectures(
    limit: int = Query(default=100, ge=1, le=200),
) -> dict[str, Any]:
    try:
        architectures = repository.list_model_architectures(limit=limit)
        return {
            "ok": True,
            "architectures": _architecture_views(architectures),
        }
    except Exception as exc:
        _raise(exc)


@router.post("/architectures", status_code=HTTPStatus.CREATED)
def create_model_architecture(
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "architecture": _architecture_views([
                repository.create_model_architecture(payload),
            ])[0],
        }
    except Exception as exc:
        _raise(exc)


@router.get("/architectures/{architecture_id}")
def get_model_architecture(architecture_id: str) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "architecture": _architecture_views([
                repository.get_model_architecture(architecture_id),
            ])[0],
        }
    except Exception as exc:
        _raise(exc)


@router.put("/architectures/{architecture_id}")
def update_model_architecture(
    architecture_id: str, payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "architecture": _architecture_views([
                repository.update_model_architecture(architecture_id, payload),
            ])[0],
        }
    except Exception as exc:
        _raise(exc)


@router.post("/architectures/{architecture_id}/activate")
def activate_model_architecture(architecture_id: str) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "architecture": _architecture_views([
                repository.activate_model_architecture(architecture_id),
            ])[0],
        }
    except Exception as exc:
        _raise(exc)


@router.post("/architectures/{architecture_id}/archive")
def archive_model_architecture(architecture_id: str) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "architecture": _architecture_views([
                repository.archive_model_architecture(architecture_id),
            ])[0],
        }
    except Exception as exc:
        _raise(exc)


@router.delete("/architectures/{architecture_id}")
def delete_model_architecture(architecture_id: str) -> dict[str, Any]:
    """Permanently delete an inactive architecture and its ClickHouse evidence."""

    try:
        architecture = repository.get_model_architecture(architecture_id)
        revision = int(architecture.get("revision") or 1)
        active_backtests = model_repository.active_model_backtest_jobs(
            model_id=architecture_id,
            model_version=revision,
        )
        if active_backtests:
            raise ModelResearchConflict(
                "模型架构仍有运行中的回测任务："
                + "、".join(
                    str(item["backtest_job_id"]) for item in active_backtests
                )
            )
        deleted = repository.delete_model_architecture(architecture_id)
        warnings: list[str] = []
        cleanup: dict[str, Any] = {}
        try:
            cleanup["clickhouse"] = model_repository.delete_model_data(
                model_id=architecture_id,
                model_version=revision,
            )
        except Exception as exc:
            warnings.append(f"ClickHouse架构回测清理失败：{exc}")
        return {
            "ok": True,
            "deleted": deleted,
            "cleanup": cleanup,
            "warnings": warnings,
        }
    except Exception as exc:
        _raise(exc)


def _architecture_views(
    architectures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = [
        (str(item["architecture_id"]), int(item["revision"]))
        for item in architectures
    ]
    backtests = model_repository.latest_model_backtest_jobs(keys)
    views = []
    for architecture in architectures:
        item = dict(architecture)
        latest = backtests.get((
            str(item["architecture_id"]), int(item["revision"]),
        ))
        item["latest_backtest"] = (
            latest.model_dump(mode="json") if latest is not None else None
        )
        item["runtime_capabilities"] = {
            "signal_composition": True,
            "research_backtest": True,
            "live_execution": False,
            "pipeline_modes": ["flat", "hierarchical"],
            "merge_methods": ["priority", "weighted_score", "union"],
        }
        views.append(item)
    return views


def _feature_labels(model: dict[str, Any]) -> dict[str, str]:
    manifest = dict(model.get("manifest_json") or {})
    feature_names = [str(item) for item in manifest.get("feature_names") or []]
    dataset = dict(model.get("dataset_spec") or {})
    if not dataset:
        dataset = dict((model.get("job_config_json") or {}).get("dataset") or {})
    labels = {}
    for factor in dataset.get("factors") or []:
        factor_id = str(factor.get("factor_id") or "")
        version = int(factor.get("factor_version") or 0)
        feature = next((
            name for name in feature_names
            if name.startswith(f"{factor_id}__v{version}__")
        ), "")
        if feature:
            labels[feature] = str(factor.get("label") or factor_id or feature)
    return labels


def _label_attribution_window(
    window: Any, model: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(window, dict):
        return None
    labels = _feature_labels(model)
    result = dict(window)
    result["features"] = [{
        **dict(item),
        "label": labels.get(str(item.get("factor") or ""), item.get("factor")),
    } for item in window.get("features") or []]
    return result


def _label_attribution_report(
    diagnostics: dict[str, Any], model: dict[str, Any],
) -> dict[str, Any]:
    report = dict(diagnostics)
    report["windows"] = [
        _label_attribution_window(item, model)
        for item in diagnostics.get("windows") or []
    ]
    weak_number = int((diagnostics.get("weak_window") or {}).get("window") or 0)
    report["weak_window"] = next((
        item for item in report["windows"]
        if int((item or {}).get("window") or 0) == weak_number
    ), _label_attribution_window(diagnostics.get("weak_window"), model))
    return report


def _architecture_attribution_findings(
    report: dict[str, Any], engines: list[dict[str, Any]],
) -> list[dict[str, str]]:
    weak = dict(report.get("weak_window") or {})
    findings = []
    if weak.get("all_profiles_negative"):
        regime = str(weak.get("market_regime") or "unknown")
        regime_text = {
            "strong_bull": "同期基准处于强势上涨状态",
            "bull": "同期基准上涨",
            "bear": "同期基准下跌",
            "sideways": "同期基准震荡",
        }.get(regime, "同期市场状态待确认")
        findings.append({
            "key": "common_window_failure",
            "severity": "danger",
            "title": "共同弱窗，不是单一门控故障",
            "detail": (
                f"W{weak.get('window')}三组消融全部为负超额，{regime_text}；"
                "门控只能缓解，无法单独修复底层选股信号。"
            ),
        })
    stock = next((item for item in engines if item.get("stage") == "stock_rank"), None)
    if stock:
        rank_ic = ((stock.get("weak_window") or {}).get("model_rank_ic"))
        if rank_ic is not None and float(rank_ic) < 0.02:
            findings.append({
                "key": "stock_signal_decay",
                "severity": "danger",
                "title": "个股排序信号接近失效",
                "detail": (
                    f"弱窗个股模型RankIC为{float(rank_ic):.4f}；应扩展因子组并以"
                    "全新冻结WFA窗口验证，而不是在当前测试窗内调参。"
                ),
            })
    industry_delta = weak.get("industry_gate_delta")
    if industry_delta is not None and float(industry_delta) > 0:
        findings.append({
            "key": "industry_gate_partial_help",
            "severity": "success",
            "title": "行业门控仍有缓冲作用",
            "detail": (
                f"相对仅个股增加{float(industry_delta):.2%}超额年化，"
                "但尚不足以让该窗口转正。"
            ),
        })
    for engine in engines:
        counts = dict((engine.get("weak_window") or {}).get("counts") or {})
        abnormal = int(counts.get("reversed") or 0) + int(counts.get("decayed") or 0)
        if abnormal:
            findings.append({
                "key": f"factor_decay:{engine.get('engine_key')}",
                "severity": "warning",
                "title": f"{engine.get('display_name')}因子关系改变",
                "detail": (
                    f"弱窗有{int(counts.get('reversed') or 0)}个方向翻转、"
                    f"{int(counts.get('decayed') or 0)}个明显衰减。"
                ),
            })
    return findings


def _attribution_window_conclusion(window: Any) -> str:
    if not isinstance(window, dict):
        return "该共同窗口没有可用因子归因。"
    counts = dict(window.get("counts") or {})
    number = int(window.get("window") or 0)
    if int(counts.get("reversed") or 0):
        return f"W{number}有{int(counts['reversed'])}个因子方向翻转。"
    if int(counts.get("decayed") or 0):
        return f"W{number}有{int(counts['decayed'])}个因子信号衰减。"
    if int(counts.get("shifted") or 0):
        return f"W{number}有{int(counts['shifted'])}个因子分布显著漂移。"
    return f"W{number}因子层未见一致性异常，需检查模型与市场状态。"


@router.post(
    "/architectures/{architecture_id}/backtests",
    status_code=HTTPStatus.CREATED,
)
def create_architecture_backtest(
    background_tasks: BackgroundTasks,
    architecture_id: str,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    try:
        architecture = repository.get_model_architecture(architecture_id)
        if (
            (architecture.get("readiness") or {}).get(
                "research_backtest_ready"
            ) is not True
        ):
            raise ModelResearchConflict("模型架构尚未通过研究回测的数据与预测检查")
        date_preset = str(payload.get("date_preset") or "3y")
        date_start = (
            datetime.fromisoformat(str(payload["date_start"])).date()
            if payload.get("date_start") else None
        )
        date_end = (
            datetime.fromisoformat(str(payload["date_end"])).date()
            if payload.get("date_end") else None
        )
        created = model_repository.create_architecture_backtest_job(
            architecture,
            date_preset=date_preset,
            date_start=date_start,
            date_end=date_end,
            ablation_profile=str(payload.get("ablation_profile") or "full"),
        )
        background_tasks.add_task(
            run_model_backtest_job, created.backtest_job_id,
        )
        return {"ok": True, "backtest": created}
    except Exception as exc:
        _raise(exc)


@router.get("/architectures/{architecture_id}/backtests")
def list_architecture_backtests(
    architecture_id: str, limit: int = 40,
) -> dict[str, Any]:
    try:
        architecture = repository.get_model_architecture(architecture_id)
        jobs = model_repository.list_architecture_backtest_jobs(
            architecture_id, int(architecture["revision"]), limit=limit,
        )
        return {
            "ok": True,
            "backtests": jobs,
            "profiles": model_repository.architecture_ablation_profiles(),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/architectures/{architecture_id}/walk-forward-attribution")
def get_architecture_walk_forward_attribution(
    architecture_id: str,
) -> dict[str, Any]:
    try:
        architecture = repository.get_model_architecture(architecture_id)
        jobs = model_repository.list_architecture_backtest_jobs(
            architecture_id, int(architecture["revision"]), limit=80,
        )
        report = architecture_walk_forward_attribution([
            job.model_dump(mode="json") for job in jobs
        ])
        if report.get("eligible") is not True:
            return {"ok": True, "diagnostics": report}
        artifact_root = load_service_settings().model_artifacts_root
        weak_window_number = int(
            (report.get("weak_window") or {}).get("window") or 0,
        )
        engines = []
        for engine in architecture.get("engines") or []:
            if engine.get("enabled") is not True:
                continue
            model = repository.get_model(
                str(engine.get("model_id") or ""),
                int(engine.get("model_version") or 0),
            )
            manifest = dict(model.get("manifest_json") or {})
            walk_forward = dict(manifest.get("walk_forward") or {})
            if walk_forward.get("enabled") is not True:
                continue
            attribution = dataset_walk_forward_attribution(
                str(model.get("dataset_hash") or ""),
                artifact_root,
                walk_forward,
            )
            weak = next((
                item for item in attribution.get("windows") or []
                if int(item.get("window") or 0) == weak_window_number
            ), attribution.get("weak_window"))
            engines.append({
                "engine_key": str(engine.get("engine_key") or ""),
                "display_name": str(engine.get("display_name") or model.get("name") or ""),
                "stage": str(engine.get("stage") or "stock_rank"),
                "role": str(engine.get("role") or "stock_selection"),
                "model_id": str(model.get("model_id") or ""),
                "model_version": int(model.get("version") or 0),
                "dataset_hash": str(model.get("dataset_hash") or ""),
                "primary_cause": attribution.get("primary_cause"),
                "conclusion": _attribution_window_conclusion(weak),
                "weak_window": _label_attribution_window(weak, model),
            })
        report["engines"] = engines
        report["findings"] = _architecture_attribution_findings(report, engines)
        return {"ok": True, "diagnostics": report}
    except Exception as exc:
        _raise(exc)


@router.post(
    "/architectures/{architecture_id}/ablation-backtests",
    status_code=HTTPStatus.CREATED,
)
def create_architecture_ablation_backtests(
    background_tasks: BackgroundTasks,
    architecture_id: str,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    try:
        architecture = repository.get_model_architecture(architecture_id)
        if str(architecture.get("pipeline_mode") or "flat") != "hierarchical":
            raise ModelResearchConflict("只有分层门控架构支持分层消融回测")
        if (
            (architecture.get("readiness") or {}).get(
                "research_backtest_ready"
            ) is not True
        ):
            raise ModelResearchConflict("模型架构尚未通过研究回测的数据与预测检查")
        date_preset = str(payload.get("date_preset") or "3y")
        date_start = (
            datetime.fromisoformat(str(payload["date_start"])).date()
            if payload.get("date_start") else None
        )
        date_end = (
            datetime.fromisoformat(str(payload["date_end"])).date()
            if payload.get("date_end") else None
        )
        jobs = []
        for profile in model_repository.architecture_ablation_profiles():
            created = model_repository.create_architecture_backtest_job(
                architecture,
                date_preset=date_preset,
                date_start=date_start,
                date_end=date_end,
                ablation_profile=str(profile["key"]),
            )
            jobs.append(created)
            background_tasks.add_task(
                run_model_backtest_job, created.backtest_job_id,
            )
        return {
            "ok": True,
            "backtests": jobs,
            "profiles": model_repository.architecture_ablation_profiles(),
        }
    except Exception as exc:
        _raise(exc)


@router.post("/model-comparisons")
def compare_models(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        requested = list(payload.get("models") or [])
        if not 2 <= len(requested) <= 8:
            raise ModelResearchError("模型对比必须选择2到8个模型版本")
        keys: list[tuple[str, int]] = []
        for item in requested:
            model_id = str(item.get("model_id") or "").strip()
            version = int(item.get("model_version") or item.get("version") or 0)
            if not model_id or version <= 0:
                raise ModelResearchError("模型对比版本无效")
            key = (model_id, version)
            if key in keys:
                raise ModelResearchError("模型对比不能包含重复版本")
            keys.append(key)
        raw_models = [repository.get_model(model_id, version) for model_id, version in keys]
        backtests = model_repository.latest_model_backtests(keys)
        models = [
            _model_response(model, backtests.get(key), _experiment_summary_for_model(model))
            for model, key in zip(raw_models, keys, strict=True)
        ]
        universes = {
            str((model.get("dataset_spec") or {}).get("universe_id") or "")
            for model in models
        }
        label_keys = {_comparison_label_key(model) for model in models}
        dataset_hashes = {str(model.get("dataset_hash") or "") for model in models}
        universe_equal = len(universes) == 1
        label_equal = len(label_keys) == 1
        horizon = next(iter(label_keys))[1] if label_equal else None
        relationship = model_repository.model_prediction_comparison(
            sources=[{
                "model_id": model["model_id"],
                "model_version": model["version"],
                "name": model.get("name"),
                "model_kind": model.get("model_kind"),
                "weight": 1.0,
            } for model in models],
            horizon=horizon,
        )
        aligned_metrics = {
            str(item.get("source_key")): item
            for item in relationship.get("metrics") or []
        }
        for model in models:
            model["comparison_metrics"] = aligned_metrics.get(
                f"{model['model_id']}::v{int(model['version'])}"
            )
        backtest_signatures = [
            _comparison_backtest_key(model.get("latest_backtest")) for model in models
        ]
        backtest_comparable = (
            all(signature is not None for signature in backtest_signatures)
            and len(set(backtest_signatures)) == 1
        )
        research_comparable = (
            universe_equal
            and label_equal
            and int(relationship.get("evaluation_days") or 0) > 0
        )
        issues: list[str] = []
        if not universe_equal:
            issues.append("股票池不一致，研究指标不能直接排名")
        if not label_equal:
            issues.append("标签或预测周期不一致，研究指标不能直接排名")
        if int(relationship.get("common_days") or 0) <= 0:
            issues.append("模型之间没有共同的PIT安全样本外预测")
        elif int(relationship.get("evaluation_days") or 0) <= 0:
            issues.append("共同预测无法与真实未来收益标签对齐")
        if len(dataset_hashes) > 1:
            issues.append("Dataset Hash不同：可以比较共同样本表现，但因子集或数据快照并不相同")
        if not backtest_comparable:
            issues.append("Top20回测区间或交易参数不一致，回测指标只展示不判优")
        return {
            "ok": True,
            "comparison": {
                "models": models,
                "compatibility": {
                    "research_comparable": research_comparable,
                    "backtest_comparable": backtest_comparable,
                    "universe_equal": universe_equal,
                    "label_equal": label_equal,
                    "dataset_hash_equal": len(dataset_hashes) == 1,
                    "issues": issues,
                },
                "prediction_relationship": relationship,
                "leaders": _comparison_leaders(
                    models,
                    research_comparable=research_comparable,
                    backtest_comparable=backtest_comparable,
                ),
            },
        }
    except Exception as exc:
        _raise(exc)


def _comparison_label_key(model: dict[str, Any]) -> tuple[str, int, tuple[Any, ...]]:
    label = dict((model.get("dataset_spec") or {}).get("label") or {})
    value_range = label.get("range") or []
    return (
        str(label.get("kind") or ""),
        int(label.get("horizon_trading_days") or 0),
        tuple(value_range) if isinstance(value_range, list) else (value_range,),
    )


def _comparison_backtest_key(backtest: Any) -> tuple[Any, ...] | None:
    if not isinstance(backtest, dict) or str(backtest.get("status") or "") != "success":
        return None
    return (
        str(backtest.get("universe_id") or ""),
        str(backtest.get("benchmark_code") or ""),
        str(backtest.get("date_start") or ""),
        str(backtest.get("date_end") or ""),
        int(backtest.get("top_n") or 0),
        int(backtest.get("rebalance_every") or 0),
        float(backtest.get("buy_cost_rate") or 0.0),
        float(backtest.get("sell_cost_rate") or 0.0),
    )


def _comparison_leaders(
    models: list[dict[str, Any]], *, research_comparable: bool,
    backtest_comparable: bool,
) -> list[dict[str, Any]]:
    definitions = [
        ("rank_ic", "共同样本RankIC", "comparison_metrics", "rank_ic", True, research_comparable),
        ("ic_ir", "共同样本ICIR", "comparison_metrics", "ic_ir", True, research_comparable),
        ("rmse", "共同样本RMSE", "comparison_metrics", "rmse", False, research_comparable),
        ("excess_annual_return", "超额年化", "latest_backtest", "excess_annual_return", True, backtest_comparable),
        ("sharpe_ratio", "夏普比率", "latest_backtest", "sharpe_ratio", True, backtest_comparable),
        ("max_drawdown", "最大回撤", "latest_backtest", "max_drawdown", True, backtest_comparable),
    ]
    leaders: list[dict[str, Any]] = []
    for metric, label, section, field, higher_is_better, enabled in definitions:
        if not enabled:
            continue
        candidates: list[tuple[float, dict[str, Any]]] = []
        for model in models:
            source = model.get(section) or {}
            try:
                value = float(source.get(field))
            except (TypeError, ValueError):
                continue
            if value == value and value not in {float("inf"), float("-inf")}:
                candidates.append((value, model))
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda item: item[0], reverse=higher_is_better)
        winner_value, winner = candidates[0]
        runner_value, runner = candidates[1]
        advantage = (
            winner_value - runner_value if higher_is_better
            else runner_value - winner_value
        )
        leaders.append({
            "metric": metric,
            "label": label,
            "scope": "common_oos" if section == "comparison_metrics" else "top20_backtest",
            "higher_is_better": higher_is_better,
            "winner": {
                "model_id": winner["model_id"],
                "model_version": int(winner["version"]),
                "name": winner.get("name"),
                "value": winner_value,
            },
            "runner_up": {
                "model_id": runner["model_id"],
                "model_version": int(runner["version"]),
                "name": runner.get("name"),
                "value": runner_value,
            },
            "advantage": advantage,
        })
    return leaders


@router.post("/ensembles", status_code=HTTPStatus.CREATED)
def create_ensemble(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    draft: dict[str, Any] | None = None
    try:
        draft = repository.reserve_ensemble_model(payload)
        ensemble = dict((draft.get("config_json") or {}).get("ensemble") or {})
        sources = list(ensemble.get("sources") or [])
        availability = model_repository.ensemble_prediction_availability(
            sources=sources,
        )
        if not availability.get("trade_date"):
            raise ModelResearchConflict("源模型没有可共同融合的样本外预测")
        prefix = f"ensemble_{str(ensemble.get('fingerprint') or '')[:16]}_"
        predictions = model_repository.materialize_ensemble_predictions(
            model_id=str(draft["model_id"]),
            model_version=int(draft["model_version"]),
            sources=sources,
            dataset_hash=str(draft["dataset_hash"]),
            inference_run_prefix=prefix,
        )
        horizon = int(
            ((draft.get("dataset_spec") or {}).get("label") or {}).get(
                "horizon_trading_days"
            ) or 5
        )
        evaluation = model_repository.evaluate_model_predictions(
            model_id=str(draft["model_id"]),
            model_version=int(draft["model_version"]),
            horizon=horizon,
        )
        model = repository.complete_ensemble_model(
            str(draft["job_id"]), predictions, evaluation,
        )
        return {
            "ok": True,
            "model": _model_response(model, None),
            "availability": availability,
        }
    except Exception as exc:
        if draft:
            try:
                repository.fail_ensemble_model(str(draft["job_id"]), str(exc))
            except Exception:
                pass
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}")
def get_model(model_id: str, version: int) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        backtest = model_repository.latest_model_backtests([(model_id, version)]).get(
            (model_id, version)
        )
        return {
            "ok": True,
            "model": _model_response(
                model, backtest, _experiment_summary_for_model(model),
            ),
        }
    except Exception as exc:
        _raise(exc)


@router.delete("/models/{model_id}/versions/{version}")
def delete_model(model_id: str, version: int) -> dict[str, Any]:
    """Permanently delete one model version after all durable references clear."""

    try:
        active_backtests = model_repository.active_model_backtest_jobs(
            model_id=model_id, model_version=version,
        )
        if active_backtests:
            raise ModelResearchConflict(
                "模型仍有运行中的回测任务："
                + "、".join(
                    str(item["backtest_job_id"]) for item in active_backtests
                )
            )
        deleted = repository.delete_model(model_id, version)
        warnings: list[str] = []
        cleanup: dict[str, Any] = {}
        try:
            cleanup["clickhouse"] = model_repository.delete_model_data(
                model_id=model_id, model_version=version,
            )
        except Exception as exc:
            warnings.append(f"ClickHouse历史数据清理失败：{exc}")
        try:
            artifact_store = ModelArtifactStore(
                load_research_settings().model_artifacts_root,
            )
            cleanup["artifacts"] = artifact_store.delete_job_artifacts(
                str(deleted["job_id"]),
            )
            if bool(deleted.get("dataset_deleted")):
                cleanup["dataset_artifacts_deleted"] = (
                    artifact_store.delete_dataset_artifacts(
                        str(deleted["dataset_hash"]),
                    )
                )
        except Exception as exc:
            warnings.append(f"模型文件清理失败：{exc}")
        return {
            "ok": True,
            "deleted": deleted,
            "cleanup": cleanup,
            "warnings": warnings,
        }
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/reproducibility-audit")
def get_model_reproducibility_audit(
    model_id: str, version: int,
) -> dict[str, Any]:
    """Verify that an exact-replay model reproduced its immutable source."""
    try:
        replay = repository.get_model(model_id, version)
        origin = dict((replay.get("job_config_json") or {}).get("research_origin") or {})
        if origin.get("mode") != "exact_replay":
            raise ModelResearchConflict("只有服务端确认的精确复现模型可以执行一致性审计")
        source_model_id = str(origin.get("source_model_id") or "")
        source_model_version = int(origin.get("source_model_version") or 0)
        if not source_model_id or source_model_version <= 0:
            raise ModelResearchConflict("精确复现任务缺少不可变来源模型版本")
        source = repository.get_model(source_model_id, source_model_version)
        predictions = model_repository.model_prediction_reproducibility_audit(
            source_model_id=source_model_id,
            source_model_version=source_model_version,
            replay_model_id=model_id,
            replay_model_version=version,
        )
        return {
            "ok": True,
            "audit": build_model_reproducibility_audit(
                source, replay, predictions,
            ),
        }
    except Exception as exc:
        _raise(exc)


@router.post("/models/{model_id}/versions/{version}/incremental-training-precheck")
def incremental_training_precheck(
    model_id: str,
    version: int,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "precheck": repository.incremental_training_precheck(
                model_id, version, payload,
            ),
        }
    except Exception as exc:
        _raise(exc)


@router.post("/models/{model_id}/versions/{version}/registry")
def update_model_registry(
    model_id: str,
    version: int,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    try:
        source = repository.get_model(model_id, version)
        backtest = model_repository.latest_model_backtests([(model_id, version)]).get(
            (model_id, version)
        )
        validation = _model_validation_view(source, backtest)
        action = str(payload.get("action") or "").strip().lower()
        if action == "set_default" and validation.get("approved") is not True:
            raise ModelResearchConflict("只有通过研究门槛的模型才能设为主模型")
        model = repository.update_model_registry(
            model_id,
            version,
            action=action,
            validation_approved=validation.get("approved") is True,
            note=str(payload.get("note") or ""),
        )
        return {
            "ok": True,
            "model": _model_response(
                model, backtest, _experiment_summary_for_model(model),
            ),
        }
    except Exception as exc:
        _raise(exc)


def _research_report_payload(model_id: str, version: int) -> tuple[dict[str, Any], str]:
    source = repository.get_model(model_id, version)
    backtest = model_repository.latest_model_backtests([(model_id, version)]).get(
        (model_id, version)
    )
    model = _model_response(
        source, backtest, _experiment_summary_for_model(source),
    )
    report = build_model_research_report(model)
    return report, render_model_research_report_markdown(report)


@router.get("/models/{model_id}/versions/{version}/research-report")
def get_model_research_report(model_id: str, version: int) -> dict[str, Any]:
    try:
        report, markdown = _research_report_payload(model_id, version)
        return {"ok": True, "report": report, "markdown": markdown}
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/research-report.md")
def download_model_research_report(model_id: str, version: int) -> PlainTextResponse:
    try:
        _, markdown = _research_report_payload(model_id, version)
        return PlainTextResponse(
            markdown,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{model_id}-v{int(version)}-research-report.md"'
                ),
            },
        )
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/ensemble-diagnostics")
def get_ensemble_diagnostics(model_id: str, version: int) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        if str(model.get("model_kind") or "") != "ensemble":
            raise ModelResearchConflict("当前模型不是融合模型")
        ensemble = dict((model.get("manifest_json") or {}).get("ensemble") or {})
        sources = list(ensemble.get("sources") or [])
        horizon = int(
            ((model.get("dataset_spec") or {}).get("label") or {}).get(
                "horizon_trading_days"
            ) or 5
        )
        return {
            "ok": True,
            "diagnostics": model_repository.ensemble_model_diagnostics(
                sources=sources, horizon=horizon,
            ),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/feature-drift")
def get_model_feature_drift(model_id: str, version: int) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        manifest = dict(model.get("manifest_json") or {})
        grouped: dict[str, list[dict[str, Any]]] = {}
        if str(model.get("model_kind") or "") == "ensemble":
            sources = list((manifest.get("ensemble") or {}).get("sources") or [])
            for source in sources:
                dataset_hash = str(source.get("dataset_hash") or "")
                if dataset_hash:
                    grouped.setdefault(dataset_hash, []).append({
                        "model_id": str(source.get("model_id") or ""),
                        "model_version": int(source.get("model_version") or 0),
                        "model_kind": str(source.get("model_kind") or ""),
                    })
        else:
            dataset_hash = str(model.get("dataset_hash") or "")
            if dataset_hash:
                grouped[dataset_hash] = [{
                    "model_id": model_id,
                    "model_version": int(version),
                    "model_kind": str(model.get("model_kind") or ""),
                }]
        if not grouped:
            raise ModelResearchConflict("模型没有可诊断的冻结数据集")
        artifact_root = load_service_settings().model_artifacts_root
        datasets = []
        for dataset_hash, sources in grouped.items():
            datasets.append({
                **dataset_feature_drift(dataset_hash, artifact_root),
                "sources": sources,
            })
        feature_importance = list(model.get("feature_importance_json") or [])
        importance_warning = ""
        if (
            str(model.get("model_kind") or "") != "ensemble"
            and not any(abs(float(item.get("importance") or 0.0)) > 0 for item in feature_importance)
        ):
            try:
                artifacts = repository.list_artifacts(str(model.get("job_id") or ""))
                bundle = next(
                    item for item in artifacts
                    if str(item.get("artifact_kind") or "") == "bundle"
                )
                bundle_path = ModelArtifactStore(artifact_root).resolve(
                    str(bundle.get("relative_path") or ""),
                )
                feature_names = [
                    str(item) for item in (manifest.get("feature_names") or [])
                ]
                feature_importance = artifact_model_feature_importance(
                    bundle_path, feature_names,
                )
            except Exception as exc:
                importance_warning = f"历史特征重要性无法重新解析：{exc}"
        return {
            "ok": True,
            "drift": datasets[0] if len(datasets) == 1 else None,
            "datasets": datasets,
            "multiple_datasets": len(datasets) > 1,
            "feature_importance": feature_importance,
            "importance_warning": importance_warning,
        }
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/walk-forward-attribution")
def get_model_walk_forward_attribution(
    model_id: str, version: int,
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        if str(model.get("model_kind") or "") == "ensemble":
            raise ModelResearchConflict("融合模型没有单一冻结WFA数据集")
        manifest = dict(model.get("manifest_json") or {})
        walk_forward = dict(manifest.get("walk_forward") or {})
        if walk_forward.get("enabled") is not True:
            raise ModelResearchConflict("当前模型没有启用WFA")
        diagnostics = dataset_walk_forward_attribution(
            str(model.get("dataset_hash") or ""),
            load_service_settings().model_artifacts_root,
            walk_forward,
        )
        return {
            "ok": True,
            "diagnostics": _label_attribution_report(diagnostics, model),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/permutation-importance")
def get_model_permutation_importance(
    model_id: str,
    version: int,
    limit: int = Query(default=20, ge=1, le=50),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        if str(model.get("model_kind") or "") == "ensemble":
            raise ModelResearchConflict("融合模型请使用源模型边际贡献诊断")
        manifest = dict(model.get("manifest_json") or {})
        feature_names = [str(item) for item in manifest.get("feature_names") or []]
        if not feature_names:
            raise ModelResearchConflict("模型没有可解释的冻结特征")
        artifact_root = load_service_settings().model_artifacts_root
        artifacts = repository.list_artifacts(str(model.get("job_id") or ""))
        bundle = next(
            item for item in artifacts
            if str(item.get("artifact_kind") or "") == "bundle"
        )
        bundle_path = ModelArtifactStore(artifact_root).resolve(
            str(bundle.get("relative_path") or ""),
        )
        stored_importance = list(model.get("feature_importance_json") or [])
        if not any(
            abs(float(item.get("importance") or 0.0)) > 0
            for item in stored_importance
        ):
            stored_importance = artifact_model_feature_importance(
                bundle_path, feature_names,
            )
        importance_order = {
            str(item.get("factor") or ""): index
            for index, item in enumerate(stored_importance)
        }
        model_kind = str(model.get("model_kind") or "")
        model_params = dict(manifest.get("model_params") or {})
        is_sequence_model = model_kind in SEQUENCE_MODEL_KINDS or (
            model_kind == "stacking"
            and any(
                str(item.get("kind") or "") in SEQUENCE_MODEL_KINDS
                for item in list(model_params.get("base_models") or [])
                if isinstance(item, dict)
            )
        )
        effective_limit = min(limit, 8) if is_sequence_model else limit
        selected = sorted(
            feature_names,
            key=lambda item: (importance_order.get(item, len(feature_names)), item),
        )[:effective_limit]
        dataset_hash = str(model.get("dataset_hash") or "")
        diagnostics_runner = (
            isolated_artifact_model_permutation_importance
            if is_sequence_model
            else artifact_model_permutation_importance
        )
        with dataset_files(dataset_hash, artifact_root):
            dataset_path = ModelArtifactStore(artifact_root).resolve(
                f"datasets/{dataset_hash}/dataset.parquet",
            )
            diagnostics = diagnostics_runner(
                bundle_path,
                dataset_path,
                model_kind=model_kind,
                segments=dict(manifest.get("segments") or {}),
                model_params=model_params,
                feature_names=selected,
            )
        return {
            "ok": True,
            "diagnostics": {
                **diagnostics,
                "dataset_hash": dataset_hash,
                "selected_feature_count": len(selected),
                "total_feature_count": len(feature_names),
                "truncated": len(selected) < len(feature_names),
                "requested_feature_limit": limit,
            },
        }
    except StopIteration as exc:
        _raise(ModelResearchConflict("模型缺少可解释的bundle产物"))
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/shap-summary")
def get_model_shap_summary(
    model_id: str,
    version: int,
    split: str = Query(default="valid", pattern="^(train|valid|test)$"),
    sample_rows: int = Query(default=30_000, ge=1_000, le=100_000),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        model_kind = str(model.get("model_kind") or "")
        if model_kind not in {"lightgbm", "xgboost", "catboost"}:
            raise ModelResearchConflict(
                "SHAP归因当前仅支持LightGBM、XGBoost和CatBoost",
            )
        manifest = dict(model.get("manifest_json") or {})
        feature_names = [str(item) for item in manifest.get("feature_names") or []]
        if not feature_names:
            raise ModelResearchConflict("模型没有可解释的冻结特征")
        artifact_root = load_service_settings().model_artifacts_root
        artifacts = repository.list_artifacts(str(model.get("job_id") or ""))
        bundle = next(
            item for item in artifacts
            if str(item.get("artifact_kind") or "") == "bundle"
        )
        store = ModelArtifactStore(artifact_root)
        bundle_path = store.resolve(str(bundle.get("relative_path") or ""))
        dataset_hash = str(model.get("dataset_hash") or "")
        with dataset_files(dataset_hash, artifact_root):
            dataset_path = store.resolve(f"datasets/{dataset_hash}/dataset.parquet")
            diagnostics = artifact_model_shap_summary(
                bundle_path,
                dataset_path,
                model_kind=model_kind,
                segments=dict(manifest.get("segments") or {}),
                feature_names=feature_names,
                split=split,
                sample_rows=sample_rows,
            )
        return {
            "ok": True,
            "diagnostics": {**diagnostics, "dataset_hash": dataset_hash},
        }
    except StopIteration as exc:
        _raise(ModelResearchConflict("模型缺少可解释的bundle产物"))
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/feature-redundancy")
def get_model_feature_redundancy(
    model_id: str,
    version: int,
    threshold: float = Query(default=0.85, ge=0.5, le=0.99),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        if str(model.get("model_kind") or "") == "ensemble":
            raise ModelResearchConflict("融合模型请使用源模型相关性矩阵")
        dataset_hash = str(model.get("dataset_hash") or "")
        diagnostics = dataset_feature_redundancy(
            dataset_hash,
            load_service_settings().model_artifacts_root,
            threshold=threshold,
        )
        return {"ok": True, "diagnostics": diagnostics}
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/factor-validation-audit")
def get_model_factor_validation_audit(
    model_id: str,
    version: int,
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        if str(model.get("model_kind") or "") == "ensemble":
            raise ModelResearchConflict("融合模型请分别审计源模型因子")
        diagnostics = dataset_factor_validation_audit(
            str(model.get("dataset_hash") or ""),
            load_service_settings().model_artifacts_root,
        )
        return {"ok": True, "diagnostics": diagnostics}
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/training-diagnostics")
def get_model_training_diagnostics(
    model_id: str,
    version: int,
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        if str(model.get("model_kind") or "") == "ensemble":
            raise ModelResearchConflict("融合模型请分别查看源模型训练过程")
        manifest = dict(model.get("manifest_json") or {})
        artifacts = repository.list_artifacts(str(model.get("job_id") or ""))
        bundle = next(
            item for item in artifacts
            if str(item.get("artifact_kind") or "") == "bundle"
        )
        artifact_root = load_service_settings().model_artifacts_root
        bundle_path = ModelArtifactStore(artifact_root).resolve(
            str(bundle.get("relative_path") or ""),
        )
        diagnostics = artifact_model_training_diagnostics(
            bundle_path,
            model_kind=str(model.get("model_kind") or manifest.get("model_kind") or ""),
            model_params=dict(manifest.get("model_params") or {}),
        )
        return {
            "ok": True,
            "diagnostics": {
                **diagnostics,
                "dataset_hash": str(model.get("dataset_hash") or ""),
            },
        }
    except StopIteration as exc:
        _raise(ModelResearchConflict("模型缺少训练过程bundle产物"))
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/quantile-diagnostics")
def get_model_quantile_diagnostics(
    model_id: str,
    version: int,
    quantiles: int = Query(default=10, ge=2, le=20),
    horizon: int | None = Query(default=None, ge=1, le=60),
    sample_interval: int = Query(default=1, ge=1, le=60),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        frozen_horizon = int(
            ((model.get("dataset_spec") or {}).get("label") or {}).get(
                "horizon_trading_days",
            ) or 5
        )
        evaluation_horizon = int(horizon or frozen_horizon)
        diagnostics = model_repository.model_prediction_quantile_diagnostics(
            model_id=model_id,
            model_version=version,
            horizon=evaluation_horizon,
            quantiles=quantiles,
            sample_interval=sample_interval,
        )
        return {
            "ok": True,
            "diagnostics": {
                **diagnostics,
                "dataset_hash": str(model.get("dataset_hash") or ""),
                "frozen_label_horizon_trading_days": frozen_horizon,
                "is_frozen_label_horizon": evaluation_horizon == frozen_horizon,
            },
        }
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/stability-diagnostics")
def get_model_stability_diagnostics(
    model_id: str,
    version: int,
    rolling_window: int = Query(default=20, ge=5, le=60),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        frozen_horizon = int(
            ((model.get("dataset_spec") or {}).get("label") or {}).get(
                "horizon_trading_days",
            ) or 5
        )
        diagnostics = model_repository.model_prediction_stability_diagnostics(
            model_id=model_id,
            model_version=version,
            horizon=frozen_horizon,
            rolling_window=rolling_window,
            quantiles=5,
        )
        return {
            "ok": True,
            "diagnostics": {
                **diagnostics,
                "dataset_hash": str(model.get("dataset_hash") or ""),
                "frozen_label_horizon_trading_days": frozen_horizon,
            },
        }
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/exposure-diagnostics")
def get_model_exposure_diagnostics(
    model_id: str,
    version: int,
    score_quantiles: int = Query(default=5, ge=3, le=10),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        frozen_horizon = int(
            ((model.get("dataset_spec") or {}).get("label") or {}).get(
                "horizon_trading_days",
            ) or 5
        )
        diagnostics = model_repository.model_prediction_exposure_diagnostics(
            model_id=model_id,
            model_version=version,
            horizon=frozen_horizon,
            score_quantiles=score_quantiles,
        )
        return {
            "ok": True,
            "diagnostics": {
                **diagnostics,
                "dataset_hash": str(model.get("dataset_hash") or ""),
                "frozen_label_horizon_trading_days": frozen_horizon,
            },
        }
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/prediction-distribution-diagnostics")
def get_model_prediction_distribution_diagnostics(
    model_id: str,
    version: int,
    bins: int = Query(default=10, ge=5, le=30),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        diagnostics = model_repository.model_prediction_distribution_diagnostics(
            model_id=model_id,
            model_version=version,
            bins=bins,
        )
        return {
            "ok": True,
            "diagnostics": {
                **diagnostics,
                "dataset_hash": str(model.get("dataset_hash") or ""),
            },
        }
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/strategy-deployments/{mode}")
def get_strategy_deployment(model_id: str, version: int, mode: str) -> dict[str, Any]:
    try:
        repository.get_model(model_id, version)
        return {
            "ok": True,
            "deployment": repository.get_strategy_deployment(
                model_id, version, mode=mode,
            ),
        }
    except Exception as exc:
        _raise(exc)


@router.post("/models/{model_id}/versions/{version}/strategy-deployments/{mode}")
def record_strategy_deployment(
    model_id: str,
    version: int,
    mode: str,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        _assert_model_active(model, "加入策略运行")
        _assert_stock_prediction_scope(model, "加入个股策略运行")
        backtest = model_repository.latest_model_backtests([(model_id, version)]).get(
            (model_id, version)
        )
        if _model_validation_view(model, backtest).get("approved") is not True:
            raise ModelResearchConflict("模型尚未通过研究门槛，不能加入策略运行")
        return {
            "ok": True,
            "deployment": repository.record_strategy_deployment(
                model_id,
                version,
                mode=mode,
                state=str(payload.get("state") or "active"),
                snapshot=dict(payload.get("snapshot") or {}),
            ),
        }
    except Exception as exc:
        _raise(exc)


@router.delete("/models/{model_id}/versions/{version}/strategy-deployments/{mode}")
def delete_strategy_deployment(
    model_id: str,
    version: int,
    mode: str,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    """Delete a verified orphan deployment reference."""

    try:
        repository.get_model(model_id, version)
        if mode != "paper" or payload.get("reason") != "orphaned_paper_strategy":
            raise ModelResearchConflict(
                "仅允许通过 orphaned_paper_strategy 清理 paper 残留部署"
            )
        deployment = repository.get_strategy_deployment(
            model_id, version, mode=mode,
        )
        snapshot = (
            deployment.get("strategy_snapshot_json")
            if isinstance(deployment, dict)
            else {}
        )
        if not deployment:
            return {"ok": True, "deleted": None}
        if str(payload.get("live_id") or "").strip() != str(
            (snapshot or {}).get("live_id") or ""
        ).strip():
            raise ModelResearchConflict("部署策略标识与模型部署快照不匹配")
        return {
            "ok": True,
            "deleted": repository.delete_strategy_deployment(
                model_id, version, mode=mode,
            ),
        }
    except Exception as exc:
        _raise(exc)


def _inference_availability(
    model: dict[str, Any], *, trade_date: str = "", data_cutoff: str = "",
) -> dict[str, Any]:
    requested_date = None
    if trade_date:
        requested_date = datetime.fromisoformat(trade_date).date()
    cutoff = None
    if data_cutoff:
        cutoff = datetime.fromisoformat(data_cutoff.replace("Z", "+00:00"))
    if str(model.get("model_kind") or "") == "ensemble":
        ensemble = dict((model.get("manifest_json") or {}).get("ensemble") or {})
        availability = model_repository.ensemble_prediction_availability(
            sources=list(ensemble.get("sources") or []),
            requested_trade_date=requested_date,
        )
        return {
            **availability,
            "factor_latest_date": availability.get("trade_date"),
            "market_latest_date": availability.get("trade_date"),
            "factor_count": int(availability.get("source_count") or 0),
            "data_cutoff": cutoff,
            "availability_mode": "source_model_predictions",
        }
    return model_repository.model_inference_availability(
        factors=list((model.get("dataset_spec") or {}).get("factors") or []),
        requested_trade_date=requested_date,
        data_cutoff=cutoff,
        universe_id=str((model.get("dataset_spec") or {}).get("universe_id") or "csi500"),
    )


def _inference_precheck_item(
    key: str, label: str, passed: bool, detail: str, *, blocking: bool = True,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "passed": bool(passed),
        "blocking": bool(blocking),
        "detail": str(detail),
    }


def _inference_schedule_response(schedule: dict[str, Any] | None) -> dict[str, Any] | None:
    if not schedule:
        return None
    fields = (
        "model_id", "model_version", "enabled", "run_after_local",
        "max_catchup_days", "last_checked_at", "last_submitted_trade_date",
        "last_error", "created_at", "updated_at", "name", "model_kind",
        "state", "is_default", "prediction_json",
    )
    return {key: schedule.get(key) for key in fields}


def _build_inference_precheck(
    model: dict[str, Any], *, mode: str, trade_date: str,
    data_cutoff: str, availability: dict[str, Any],
    availability_error: str, artifacts: list[dict[str, Any]],
    runs: list[dict[str, Any]], schedule: dict[str, Any] | None,
    validation: dict[str, Any],
) -> dict[str, Any]:
    model_kind = str(model.get("model_kind") or "")
    state = str(model.get("state") or "candidate")
    production = mode == "production"
    model_active = state != "archived"
    validation_ready = validation.get("approved") is True
    bundle_ready = model_kind == "ensemble" or any(
        str(item.get("artifact_kind") or "") == "bundle" for item in artifacts
    )
    requested_available = availability.get("requested_trade_date_available") is True
    inputs_ready = not availability_error and bool(availability.get("trade_date"))
    try:
        cutoff = datetime.fromisoformat(str(data_cutoff).replace("Z", "+00:00"))
        if cutoff.tzinfo is None:
            cutoff = cutoff.astimezone()
        signal_close = datetime.fromisoformat(f"{trade_date}T15:00:00+08:00")
        after_close = cutoff >= signal_close
    except (TypeError, ValueError):
        after_close = False
    active_run = next(
        (item for item in runs if str(item.get("status") or "") in {
            "queued", "leased", "running", "uploading",
        }),
        None,
    )
    prediction = dict(model.get("prediction_json") or {})
    latest_prediction_date = str(
        prediction.get("latest_trade_date")
        or prediction.get("last_inference_trade_date")
        or prediction.get("date_end") or ""
    )[:10]
    already_generated = bool(
        trade_date and latest_prediction_date and latest_prediction_date >= trade_date
    ) or any(str(item.get("status") or "") == "succeeded" for item in runs)
    items = [
        _inference_precheck_item(
            "model_active", "模型版本可用", model_active,
            "不可变模型版本未归档" if model_active else "模型已归档，请先从模型池恢复",
        ),
        _inference_precheck_item(
            "validation", "研究门槛", validation_ready,
            "已通过Top20研究门槛" if validation_ready else "尚未通过研究门槛，只能用于研究推理",
            blocking=production,
        ),
        _inference_precheck_item(
            "artifact", "模型产物", bundle_ready,
            "融合模型直接复用冻结源预测" if model_kind == "ensemble"
            else "训练Bundle存在且可调度" if bundle_ready else "缺少训练Bundle",
        ),
        _inference_precheck_item(
            "frozen_inputs", "冻结输入", inputs_ready,
            f"{int(availability.get('factor_count') or 0)}个冻结输入已校验"
            if inputs_ready else availability_error or "没有共同可用数据日",
        ),
        _inference_precheck_item(
            "trade_date", "目标交易日", requested_available,
            f"{trade_date}行情与全部输入完整" if requested_available
            else f"{trade_date or '所选日期'}不是完整可推理交易日",
        ),
        _inference_precheck_item(
            "signal_close", "收盘截止时间", after_close,
            "数据截止时间不早于T日15:00" if after_close
            else "每日推理只能在目标交易日收盘后运行",
        ),
        _inference_precheck_item(
            "schedule", "自动调度", bool(schedule and schedule.get("enabled")),
            "自动推理已启用" if schedule and schedule.get("enabled")
            else "自动推理未启用，不影响手动运行",
            blocking=False,
        ),
    ]
    dependencies_ready = all(
        item["passed"] or not item["blocking"] for item in items
    )
    if active_run:
        status = "running"
    elif dependencies_ready and already_generated:
        status = "up_to_date"
    elif dependencies_ready:
        status = "ready"
    else:
        status = "blocked"
    return {
        "checked_at": datetime.now().astimezone().isoformat(),
        "mode": mode,
        "status": status,
        "passed": dependencies_ready,
        "can_submit": dependencies_ready and not already_generated and active_run is None,
        "already_generated": already_generated,
        "active_job_id": str((active_run or {}).get("job_id") or ""),
        "trade_date": trade_date,
        "data_cutoff": data_cutoff,
        "latest_available_trade_date": str(availability.get("trade_date") or "")[:10],
        "latest_prediction_trade_date": latest_prediction_date,
        "availability": availability,
        "model": {
            "model_id": model.get("model_id"),
            "model_version": int(model.get("version") or 0),
            "name": model.get("name"),
            "model_kind": model_kind,
            "state": state,
            "is_default": bool(model.get("is_default", False)),
            "label_horizon_trading_days": int(
                (((model.get("dataset_spec") or {}).get("label") or {}).get(
                    "horizon_trading_days"
                ) or 0)
            ),
        },
        "schedule": _inference_schedule_response(schedule),
        "items": items,
    }


@router.get("/models/{model_id}/versions/{version}/inference-availability")
def inference_availability(model_id: str, version: int) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        return {"ok": True, "availability": _inference_availability(model)}
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/inference-precheck")
def inference_precheck(
    model_id: str,
    version: int,
    trade_date: str = Query(default=""),
    data_cutoff: str = Query(default=""),
    mode: str = Query(default="production", pattern="^(research|production)$"),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        cutoff = data_cutoff or datetime.now().astimezone().isoformat()
        availability_error = ""
        try:
            availability = _inference_availability(
                model, trade_date=trade_date, data_cutoff=cutoff,
            )
            target_date = trade_date or str(availability.get("trade_date") or "")[:10]
            if target_date and not trade_date:
                # The discovered target is by definition the latest fully
                # available date, so no second availability pass is needed.
                availability = dict(availability)
                availability["requested_trade_date"] = (
                    datetime.fromisoformat(target_date).date()
                )
                availability["requested_trade_date_available"] = True
        except Exception as exc:
            availability = {}
            target_date = trade_date
            availability_error = str(exc)
        if target_date:
            datetime.fromisoformat(target_date)
        artifacts = (
            [] if str(model.get("model_kind") or "") == "ensemble"
            else repository.list_artifacts(str(model.get("job_id") or ""))
        )
        runs = repository.list_inference_runs(
            model_id=model_id, model_version=version,
            trade_date=target_date, limit=20,
        ) if target_date else []
        schedule = next((
            item for item in repository.list_inference_schedules()
            if str(item.get("model_id")) == model_id
            and int(item.get("model_version") or 0) == int(version)
        ), None)
        backtest = model_repository.latest_model_backtests([(model_id, version)]).get(
            (model_id, version)
        )
        validation = _model_validation_view(model, backtest)
        return {
            "ok": True,
            "precheck": _build_inference_precheck(
                model,
                mode=mode,
                trade_date=target_date,
                data_cutoff=cutoff,
                availability=availability,
                availability_error=availability_error,
                artifacts=artifacts,
                runs=runs,
                schedule=schedule,
                validation=validation,
            ),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/inference-runs")
def list_inference_runs(
    status: str = Query(default=""),
    model_id: str = Query(default=""),
    model_version: int | None = Query(default=None, ge=1),
    trade_date: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    try:
        runs = repository.list_inference_runs(
            status=status, model_id=model_id, model_version=model_version,
            trade_date=trade_date, limit=limit,
        )
        counts: dict[str, int] = {}
        for item in runs:
            state = str(item.get("status") or "unknown")
            counts[state] = counts.get(state, 0) + 1
        return {"ok": True, "runs": runs, "status_counts": counts}
    except Exception as exc:
        _raise(exc)


@router.post("/models/{model_id}/versions/{version}/inferences")
def create_inference(
    request: Request,
    model_id: str,
    version: int,
    payload: dict[str, Any] = Body(default={}),
) -> JSONResponse:
    try:
        model = repository.get_model(model_id, version)
        _assert_model_active(model, "生成新预测")
        trade_date = str(payload.get("trade_date") or "")
        data_cutoff = str(payload.get("data_cutoff") or "")
        availability = _inference_availability(
            model, trade_date=trade_date, data_cutoff=data_cutoff,
        )
        trade_date = trade_date or str(availability.get("trade_date") or "")
        if not trade_date:
            raise ModelResearchConflict("模型因子与中证500行情没有共同可推理交易日")
        if trade_date > str(availability.get("trade_date") or ""):
            raise ModelResearchError(
                f"目标日{trade_date}尚不可推理，当前共同最新交易日为{availability.get('trade_date')}"
            )
        if not str(payload.get("trade_date") or ""):
            availability = _inference_availability(
                model, trade_date=trade_date, data_cutoff=data_cutoff,
            )
        if availability.get("requested_trade_date_available") is not True:
            raise ModelResearchError(f"目标日{trade_date}不是完整可推理交易日")
        if str(model.get("model_kind") or "") == "ensemble":
            ensemble = dict((model.get("manifest_json") or {}).get("ensemble") or {})
            prefix = f"ensemble_{str(ensemble.get('fingerprint') or '')[:16]}_"
            predictions = model_repository.materialize_ensemble_predictions(
                model_id=model_id,
                model_version=version,
                sources=list(ensemble.get("sources") or []),
                dataset_hash=str(model.get("dataset_hash") or ""),
                inference_run_prefix=prefix,
                trade_date=datetime.fromisoformat(trade_date).date(),
            )
            job = repository.record_ensemble_inference(
                model_id,
                version,
                trade_date=trade_date,
                data_cutoff=data_cutoff or datetime.now().astimezone().isoformat(),
                predictions=predictions,
                trigger="manual",
            )
            return JSONResponse(
                status_code=HTTPStatus.OK,
                content=jsonable_encoder({
                    "ok": True, "job": job, "availability": availability,
                }),
            )
        job = repository.create_inference_job(
            model_id,
            version,
            {**payload, "trade_date": trade_date, "trigger": "manual"},
        )
        if str(job.get("status")) == "queued":
            result, status = _dispatch(request, job)
            result["availability"] = availability
            return JSONResponse(status_code=status, content=jsonable_encoder(result))
        status = HTTPStatus.OK if str(job.get("status")) == "succeeded" else HTTPStatus.ACCEPTED
        return JSONResponse(
            status_code=int(status),
            content=jsonable_encoder({
                "ok": True,
                "job": job,
                "availability": availability,
            }),
        )
    except Exception as exc:
        _raise(exc)


@router.get("/predictions")
def list_predictions(
    model_id: str,
    model_version: int,
    trade_date: str = "",
    entity_code: str = "",
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict[str, Any]:
    try:
        parsed_date = datetime.fromisoformat(trade_date).date() if trade_date else None
        rows = model_repository.list_model_predictions(
            model_id=model_id,
            model_version=model_version,
            trade_date=parsed_date,
            entity_code=entity_code,
            limit=limit,
        )
        return {"ok": True, "predictions": rows}
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/prediction-overview")
def get_prediction_overview(
    model_id: str,
    version: int,
    trade_date: str = "",
    top_n: int = Query(default=20, ge=1, le=500),
    history_days: int = Query(default=120, ge=2, le=250),
) -> dict[str, Any]:
    try:
        repository.get_model(model_id, version)
        parsed_date = datetime.fromisoformat(trade_date).date() if trade_date else None
        return {
            "ok": True,
            "overview": model_repository.model_prediction_overview(
                model_id=model_id,
                model_version=version,
                trade_date=parsed_date,
                top_n=top_n,
                history_days=history_days,
            ),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/return-calibration")
def get_model_return_calibration(
    model_id: str,
    version: int,
    entity_code: str,
    trade_date: str,
    lookback_days: int = Query(default=252, ge=20, le=504),
    buckets: int = Query(default=10, ge=2, le=20),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        dataset = dict(model.get("dataset_spec") or {})
        if str(dataset.get("prediction_scope") or "stock") != "stock":
            raise ModelResearchConflict("只有个股选股模型支持单股收益率校准")
        horizon = int(
            (dataset.get("label") or {}).get("horizon_trading_days") or 5
        )
        parsed_date = datetime.fromisoformat(trade_date).date()
        return {
            "ok": True,
            "forecast": model_repository.model_stock_return_calibration(
                model_id=model_id,
                model_version=version,
                entity_code=entity_code,
                as_of_date=parsed_date,
                horizon=horizon,
                lookback_days=lookback_days,
                buckets=buckets,
            ),
        }
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/signals")
def list_signals(
    model_id: str,
    version: int,
    trade_date: str,
    top_n: int = Query(default=20, ge=1, le=500),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        _assert_model_active(model, "生成正式策略信号")
        _assert_stock_prediction_scope(model, "生成个股TopN信号")
        backtest = model_repository.latest_model_backtests([(model_id, version)]).get(
            (model_id, version)
        )
        if _model_validation_view(model, backtest).get("approved") is not True:
            raise ModelResearchConflict("模型尚未通过研究门槛，不能用于正式策略信号")
        rows = model_repository.list_model_signals(
            model_id=model_id,
            model_version=version,
            trade_date=datetime.fromisoformat(trade_date).date(),
            top_n=top_n,
        )
        return {"ok": True, "signals": rows}
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/selection/daily")
def get_daily_selection(
    model_id: str,
    version: int,
    strategy: str = Query(default="balanced"),
    trade_date: str = "",
    ignore_ma20: bool = Query(default=False),
) -> dict[str, Any]:
    """今日选股：市场状态 + 行业排行 + 候选股 + 被排除示例（对齐 QuantMind）。"""
    try:
        model = repository.get_model(model_id, version)
        _assert_stock_prediction_scope(model, "生成个股选股参考")
        result = run_daily_selection(
            model_id=model_id,
            model_version=version,
            strategy=strategy,
            trade_date=trade_date or None,
            ignore_ma20=ignore_ma20,
        )
        return {"ok": True, **result}
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/selection/negative")
def get_negative_selection(
    model_id: str,
    version: int,
    trade_date: str = "",
) -> dict[str, Any]:
    """负分多空参考：做空候选 + 错杀参考 + 分数×市值矩阵（对齐 QuantMind）。"""
    try:
        model = repository.get_model(model_id, version)
        _assert_stock_prediction_scope(model, "生成负分参考")
        result = run_negative_selection(
            model_id=model_id,
            model_version=version,
            trade_date=trade_date or None,
        )
        return {"ok": True, **result}
    except Exception as exc:
        _raise(exc)


@router.post("/models/{model_id}/versions/{version}/backtests", status_code=HTTPStatus.CREATED)
def create_backtest(
    background_tasks: BackgroundTasks,
    model_id: str,
    version: int,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        _assert_model_active(model, "运行Top20回测")
        _assert_stock_prediction_scope(model, "运行个股Top20回测")
        _assert_experiment_backtest_allowed(model)
        _assert_validation_backtest_allowed(model)
        trained_universe = str(
            (model.get("dataset_spec") or {}).get("universe_id") or "csi500"
        )
        created = model_repository.create_model_backtest_job(ModelBacktestJobCreate(
            model_id=model_id,
            model_version=version,
            universe_id=str(payload.get("universe_id") or trained_universe),
            top_n=20,
            rebalance_every=5,
            date_preset=str(payload.get("date_preset") or "3y"),
        ))
        background_tasks.add_task(run_model_backtest_job, created.backtest_job_id)
        return {"ok": True, "backtest": created}
    except Exception as exc:
        _raise(exc)


@router.post(
    "/models/{model_id}/versions/{version}/portfolio-sensitivity",
    status_code=HTTPStatus.CREATED,
)
def create_portfolio_sensitivity(
    background_tasks: BackgroundTasks,
    model_id: str,
    version: int,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        _assert_stock_prediction_scope(model, "运行TopN敏感性研究")
        raw_top_ns = payload.get("top_ns") or [20, 50, 100]
        if not isinstance(raw_top_ns, list):
            raise ModelResearchConflict("top_ns必须是数组")
        top_ns = sorted(set(int(item) for item in raw_top_ns))
        if not top_ns or len(top_ns) > 6 or any(item < 5 or item > 500 for item in top_ns):
            raise ModelResearchConflict("TopN敏感性只允许1至6组、每组5至500只")
        rebalance_every = int(payload.get("rebalance_every") or 5)
        if rebalance_every < 1 or rebalance_every > 60:
            raise ModelResearchConflict("调仓间隔必须在1至60个交易日之间")
        date_preset = str(payload.get("date_preset") or "3y")
        if date_preset not in {"3m", "1y", "3y", "10y"}:
            raise ModelResearchConflict("敏感性研究日期范围无效")
        trained_universe = str(
            (model.get("dataset_spec") or {}).get("universe_id") or "csi500"
        )
        jobs = []
        for top_n in top_ns:
            created = model_repository.create_model_backtest_job(
                ModelBacktestJobCreate(
                    model_id=model_id,
                    model_version=version,
                    universe_id=str(payload.get("universe_id") or trained_universe),
                    top_n=top_n,
                    rebalance_every=rebalance_every,
                    date_preset=date_preset,
                    research_only=True,
                ),
            )
            background_tasks.add_task(
                run_model_backtest_job, created.backtest_job_id,
            )
            jobs.append(created)
        return {
            "ok": True,
            "jobs": jobs,
            "guard": "敏感性结果仅用于诊断，不写入模型验证状态。",
        }
    except Exception as exc:
        _raise(exc)


@router.get("/models/{model_id}/versions/{version}/portfolio-sensitivity")
def list_portfolio_sensitivity(
    model_id: str,
    version: int,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    try:
        repository.get_model(model_id, version)
        jobs = model_repository.list_model_sensitivity_backtests(
            model_id, version, limit=limit,
        )
        return {
            "ok": True,
            "jobs": jobs,
            "guard": "敏感性结果不参与validated状态判定。",
        }
    except Exception as exc:
        _raise(exc)


@router.get("/model-backtests/{backtest_job_id}")
def get_backtest(backtest_job_id: str) -> dict[str, Any]:
    try:
        result = model_repository.get_model_backtest_job(backtest_job_id)
        if result is None:
            raise ModelResearchNotFound("模型回测任务不存在")
        is_architecture = (
            str((result.configuration or {}).get("signal_source") or "")
            == "model_architecture"
        )
        is_research_only = (
            (result.configuration or {}).get("research_only") is True
        )
        if result.status == "success" and not is_architecture and not is_research_only:
            model = repository.get_model(result.model_id, result.model_version)
            validation = _model_validation_view(model, result)
            stored = dict((model.get("manifest_json") or {}).get("validation") or {})
            already_recorded = (
                str(stored.get("backtest_job_id") or "") == backtest_job_id
                and str(stored.get("policy") or "") == str(validation.get("policy") or "")
                and bool(stored.get("manual_override")) == bool(validation.get("manual_override"))
                and bool(stored.get("passed")) == bool(validation.get("passed"))
            )
            if not already_recorded:
                repository.record_validation_result(
                    result.model_id,
                    result.model_version,
                    backtest_job_id,
                    approved=bool(validation.get("passed")),
                    validation=validation,
                )
        else:
            validation = None
        payload = result.model_dump()
        payload["validation"] = validation
        payload["backtest_kind"] = (
            "model_architecture" if is_architecture else
            "portfolio_sensitivity" if is_research_only else "model_version"
        )
        return {"ok": True, "backtest": payload}
    except Exception as exc:
        _raise(exc)


@router.get("/model-backtests/{backtest_job_id}/daily")
def get_backtest_daily(
    backtest_job_id: str, limit: int = Query(default=5000, ge=1, le=20000),
) -> dict[str, Any]:
    try:
        if model_repository.get_model_backtest_job(backtest_job_id) is None:
            raise ModelResearchNotFound("模型回测任务不存在")
        return {
            "ok": True,
            "daily": model_repository.list_model_backtest_daily(backtest_job_id, limit=limit),
        }
    except Exception as exc:
        _raise(exc)


@router.post("/models/{model_id}/versions/{version}/backtests/{backtest_job_id}/validate")
def validate_backtest(
    model_id: str,
    version: int,
    backtest_job_id: str,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    try:
        result = model_repository.get_model_backtest_job(backtest_job_id)
        if result is None or result.status != "success":
            raise ModelResearchConflict("模型回测尚未成功，不能标记为已验证")
        source = repository.get_model(model_id, version)
        _assert_model_active(source, "更新验证状态")
        validation = assess_model_validation(source.get("metrics_json"), result)
        override = bool(payload.get("override", False))
        reason = str(payload.get("reason") or "").strip()
        if validation["passed"] is not True and not override:
            labels = [
                str(item["label"])
                for item in validation["checks"]
                if item["passed"] is not True
            ]
            raise ModelResearchConflict("模型未通过研究门槛：" + "、".join(labels))
        if override and not reason:
            raise ModelResearchError("人工放行必须填写原因")
        validation.update({
            "approved": True,
            "manual_override": override,
            "override_reason": reason if override else "",
        })
        model = repository.mark_validated(
            model_id, version, backtest_job_id, validation=validation,
        )
        return {"ok": True, "model": model, "backtest": result, "validation": validation}
    except Exception as exc:
        _raise(exc)


@router.get("/inference-schedules")
def list_inference_schedules() -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "schedules": [
                _inference_schedule_response(item)
                for item in repository.list_inference_schedules()
            ],
        }
    except Exception as exc:
        _raise(exc)


@router.put("/models/{model_id}/versions/{version}/inference-schedule")
def update_inference_schedule(
    model_id: str, version: int, payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    try:
        model = repository.get_model(model_id, version)
        _assert_model_active(model, "启用自动推理")
        if bool(payload.get("enabled", True)):
            backtest = model_repository.latest_model_backtests([(model_id, version)]).get(
                (model_id, version)
            )
            if _model_validation_view(model, backtest).get("approved") is not True:
                raise ModelResearchConflict("只有通过研究门槛的模型才能启用自动推理")
        return {
            "ok": True,
            "schedule": repository.update_inference_schedule(model_id, version, payload),
        }
    except Exception as exc:
        _raise(exc)


__all__ = ["router"]
