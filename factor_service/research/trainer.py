from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import platform
import tarfile
from typing import Any

import clickhouse_connect
import numpy as np
import pandas as pd

from factor_service.model_validation import assess_walk_forward_stability
from factor_service.model_artifacts import ModelArtifactStore
from factor_service.research.config import Settings
from factor_service.research.dataset import (
    DatasetBuilder,
    PreparedDataset,
    split_trading_dates,
    walk_forward_segments,
)
from factor_service.research.job import CancellationToken, ProgressCallback
from factor_service.research.metric_logging import log_evaluation_history
from factor_service.research.industry_feature import normalize_industry_feature
from factor_service.research.preprocessing import (
    normalize_feature_preprocessing,
    preprocess_qlib_frame,
)
from factor_service.research.snapshot import DatasetSnapshotStore
from factor_service.research.dataset_archive import archive_for_settings
from factor_service.research.model_bundle import package_model_bundle
from factor_service.research.runtime_resources import release_training_memory
from factor_service.research.training_diagnostics import build_training_diagnostics


SEQUENCE_MODEL_KINDS = frozenset({
    "gru", "lstm", "alstm", "transformer", "tcn", "nativetft",
    "transformer_lstm",
})


@dataclass(frozen=True)
class TrainingResult:
    result: dict[str, Any]
    artifacts: list[tuple[str, Path]]
    predictions_path: Path


@dataclass(frozen=True)
class WalkForwardTrainingResult:
    prediction: pd.Series
    report: dict[str, Any]
    latest_model: Any
    latest_dataset: Any
    latest_evals_result: dict[str, Any]
    latest_segments: dict[str, tuple[str, str]]
    latest_model_params: dict[str, Any]


@dataclass
class QlibStackingModel:
    """Serializable Stacking bundle: fitted Qlib bases plus Ridge meta learner."""

    base_models: list[dict[str, Any]]
    meta_model: Any
    classification: bool = False

    def combine(self, predictions: list[np.ndarray]) -> np.ndarray:
        if len(predictions) != len(self.base_models):
            raise ValueError("Stacking基模型预测数量与模型产物不一致")
        matrix = np.column_stack([
            np.asarray(values, dtype=float).reshape(-1) for values in predictions
        ])
        raw = np.asarray(self.meta_model.predict(matrix), dtype=float).reshape(-1)
        if self.classification:
            return np.clip(raw, 1e-7, 1.0 - 1e-7)
        return raw

    def to_cpu(self) -> None:
        for item in self.base_models:
            model = item.get("model")
            if hasattr(model, "to_cpu"):
                model.to_cpu()
            _prepare_model_for_serialization(str(item.get("kind") or ""), model)


class QlibTrainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset_builder: DatasetBuilder | None = None
        self.snapshot_store = DatasetSnapshotStore(
            settings.model_artifacts_root, archive=archive_for_settings(settings),
        )
        self.distributed = None

    def train(
        self,
        job: dict[str, Any],
        work_dir: Path,
        *,
        cancellation: CancellationToken | None = None,
        progress: ProgressCallback | None = None,
    ) -> TrainingResult:
        # The isolated trainer must own its pin too: a parent worker exiting must
        # not allow another task to replace files still needed by this process.
        archive = self.snapshot_store.archive
        usage = (archive.cache_slot(str(job["dataset_hash"])) if archive is not None
                 else self.snapshot_store.artifacts.dataset_usage(str(job["dataset_hash"])))
        with usage:
            from factor_service.research.distributed_config import distributed_nodes
            if distributed_nodes(job):
                from factor_service.research.distributed import DistributedTraining
                with DistributedTraining(self.settings, job, work_dir, cancellation, progress) as execution:
                    self.distributed = execution
                    try:
                        return self._train(job, work_dir, cancellation=execution.cancellation, progress=progress)
                    finally:
                        self.distributed = None
            return self._train(job, work_dir, cancellation=cancellation, progress=progress)

    def _train(
        self,
        job: dict[str, Any],
        work_dir: Path,
        *,
        cancellation: CancellationToken | None = None,
        progress: ProgressCallback | None = None,
    ) -> TrainingResult:
        try:
            import qlib
            from qlib.data.dataset import DataHandlerLP, DatasetH
            from qlib.workflow import R
        except ImportError as exc:
            raise RuntimeError("Qlib模型环境尚未安装，请执行uv sync") from exc
        work_dir.mkdir(parents=True, exist_ok=True)
        _checkpoint(cancellation)
        def create_builder() -> DatasetBuilder:
            self.dataset_builder = self.dataset_builder or DatasetBuilder(self.settings)
            return self.dataset_builder

        snapshot = self.snapshot_store.get_or_create(
            job, work_dir, None, builder_factory=create_builder,
            cancellation=cancellation, progress=progress,
        )
        if self.distributed is not None:
            self.distributed.bind_snapshot(snapshot, self.snapshot_store)
        prepared = snapshot.prepared
        config = dict(job.get("config_json") or {})
        subtask = dict(config.get("_distributed_task") or {})
        dataset_spec = dict(job.get("dataset_spec") or config.get("dataset") or {})
        label_spec = dict(dataset_spec.get("label") or {})
        target_mode = str(
            dataset_spec.get("target_mode") or label_spec.get("mode") or "return"
        ).strip().lower()
        classification = target_mode == "classification"
        walk_forward_config = dict(config.get("walk_forward") or {})
        incremental_config = dict(config.get("incremental_training") or {})
        source_model: Any | None = None
        source_manifest: dict[str, Any] = {}
        training_prepared = prepared
        if incremental_config:
            source_model, source_manifest = _load_incremental_source(
                self.settings, incremental_config,
            )
            training_prepared = _incremental_prepared_dataset(
                prepared, incremental_config, source_manifest,
                horizon=int(
                    (job.get("dataset_spec") or {}).get("label", {}).get(
                        "horizon_trading_days", 5,
                    )
                ),
            )
            _progress(progress, "incremental_dataset_ready", 57, {
                "source_model_version": incremental_config["source_model_version"],
                "segments": training_prepared.segments,
            })
        dataset_path = snapshot.dataset_path
        raw_dataset_path = snapshot.raw_dataset_path
        dataset_manifest_path = snapshot.manifest_path
        model_spec = dict(config.get("model") or {})
        raw_params = dict(model_spec.get("params") or {})
        model_kind = str(model_spec.get("kind") or "lightgbm")
        validation_enabled = not (
            walk_forward_config.get("enabled") is True
            and int(walk_forward_config.get("valid_sessions", 60)) == 0
        )
        if not validation_enabled:
            raw_params["early_stopping_rounds"] = 0
            if model_kind == "catboost":
                raw_params["od_wait"] = 0
        dataset = None
        model: Any = None
        model_params: dict[str, Any] = dict(raw_params)
        if model_kind == "stacking" and incremental_config:
            raise ValueError("Stacking暂不支持增量续训")
        if model_kind == "stacking" and walk_forward_config.get("enabled") is True:
            raise ValueError("Stacking已使用时序OOF，不能同时开启Walk-Forward")
        recorder_root = work_dir / "mlruns"
        # DatasetH由已冻结的DataFrame驱动，不读取Qlib本地行情；0.9.7仍要求
        # provider_uri非空，因此给每个任务一个隔离的空Provider目录。
        provider_root = work_dir / "qlib_provider"
        provider_root.mkdir(parents=True, exist_ok=True)
        qlib.init(provider_uri=str(provider_root), expression_cache=None, dataset_cache=None)
        # MLflow 3.15开始默认拒绝新的文件目录型tracking store；SQLite既支持
        # Qlib Recorder，又能随模型产物一起打包，避免依赖外部MLflow服务。
        recorder_db = work_dir / "mlflow.db"
        recorder_uri = f"sqlite:///{recorder_db.as_posix()}"
        experiment_name = f"alphablocks_{job['model_id']}"
        _prepare_recorder_experiment(recorder_uri, experiment_name, recorder_root)
        optuna_result: dict[str, Any] | None = subtask.get("optuna_result")
        optuna_config = dict(config.get("optuna") or {})
        if optuna_config.get("enabled") is True:
            tuning_experiment_name = (
                f"{experiment_name}_optuna_v"
                f"{int(config.get('planned_model_version') or 1)}"
            )
            _prepare_recorder_experiment(
                recorder_uri, tuning_experiment_name, recorder_root,
            )
            with R.start(
                experiment_name=tuning_experiment_name,
                recorder_name=f"{job['job_id']}_optuna",
                uri=recorder_uri,
            ):
                optuna_result = _tune_tree_hyperparameters(
                    training_prepared,
                    model_kind=model_kind,
                    base_params=raw_params,
                    config=optuna_config,
                    walk_forward_config=walk_forward_config,
                    DataHandlerLP=DataHandlerLP,
                    DatasetH=DatasetH,
                    classification=classification,
                    cancellation=cancellation,
                    progress=progress,
                    distributed=self.distributed,
                    fixed_trial=subtask if subtask.get("kind") == "optuna_trial" else None,
                )
            if subtask.get("kind") == "optuna_trial":
                optuna_result.setdefault('attrs', {})['environment'] = {
                    'python': platform.python_version(), 'platform': platform.platform(),
                    'qlib': getattr(qlib, '__version__', 'unknown'), **_model_package_version(model_kind),
                }
                trial_path = work_dir / "trial_result.json"
                trial_path.write_text(json.dumps(optuna_result, ensure_ascii=False), encoding="utf-8")
                return TrainingResult({"trial": optuna_result}, [("trial_result", trial_path)], trial_path)
            raw_params = {**raw_params, **dict(optuna_result["best_params"])}
        if self.distributed is not None and walk_forward_config.get("enabled") is not True:
            return self.distributed.final_model(raw_params, optuna_result)
        if model_kind != "stacking" and walk_forward_config.get("enabled") is not True:
            handler = DataHandlerLP.from_df(training_prepared.frame)
            dataset = _dataset_for_model(
                handler, training_prepared.segments, model_kind, raw_params, DatasetH,
            )
            model, model_params = _create_model(
                model_kind, raw_params, len(training_prepared.feature_names),
            )
        evals_result: dict[str, Any] = {}
        walk_forward_report: dict[str, Any] | None = None
        walk_forward_prediction: pd.Series | None = None
        walk_forward_result: WalkForwardTrainingResult | None = None
        if walk_forward_config.get("enabled") is True:
            rolling_experiment_name = (
                f"{experiment_name}_rolling_v"
                f"{int(config.get('planned_model_version') or 1)}"
            )
            _prepare_recorder_experiment(
                recorder_uri, rolling_experiment_name, recorder_root,
            )
            walk_forward_result = _run_walk_forward(
                prepared,
                walk_forward_config,
                work_dir=work_dir,
                model_id=str(job["model_id"]),
                model_version=int(config.get("planned_model_version") or 1),
                model_kind=model_kind,
                raw_params=raw_params,
                DataHandlerLP=DataHandlerLP,
                DatasetH=DatasetH,
                recorder_uri=recorder_uri,
                experiment_name=rolling_experiment_name,
                cancellation=cancellation,
                progress=progress,
                distributed=self.distributed,
                only_window=int(subtask["window"]) if subtask.get("kind") == "window" else None,
            )
            if subtask.get("kind") == "window":
                return walk_forward_result
            walk_forward_prediction = walk_forward_result.prediction
            walk_forward_report = walk_forward_result.report
            model = walk_forward_result.latest_model
            dataset = walk_forward_result.latest_dataset
            evals_result = walk_forward_result.latest_evals_result
            model_params = walk_forward_result.latest_model_params
        with R.start(
            experiment_name=experiment_name,
            recorder_name=str(job["job_id"]),
            uri=recorder_uri,
        ):
            R.log_params(
                job_id=job["job_id"],
                model_id=job["model_id"],
                dataset_hash=job["dataset_hash"],
                schema_version="alphablocks.qlib-training.v1",
            )
            if model_kind == "stacking":
                stacking = _fit_stacking(
                    training_prepared,
                    model_spec=model_spec,
                    DataHandlerLP=DataHandlerLP,
                    DatasetH=DatasetH,
                    classification=classification,
                    cancellation=cancellation,
                    progress=progress,
                )
                model = stacking["model"]
                model_params = stacking["model_params"]
                evals_result = stacking["evals_result"]
                train_prediction = stacking["train_prediction"]
                valid_prediction = stacking["valid_prediction"]
                test_prediction = stacking["test_prediction"]
            elif walk_forward_result is not None:
                # RollingGen + TrainerR already fitted every immutable window.
                # The root model is only a latest-window compatibility alias;
                # date-aware inference must always load a window artifact.
                _progress(progress, "rolling_series_ready", 80, {
                    "window_count": walk_forward_report["window_count"],
                    "latest_segments": walk_forward_result.latest_segments,
                })
                train_prediction = _predict_training_dataset(
                    model, model_kind, dataset, "train",
                    classification=classification,
                )
                valid_prediction = (
                    _predict_dataset(
                        model, model_kind, dataset, "valid",
                        classification=classification,
                    )
                    if "valid" in walk_forward_result.latest_segments
                    else None
                )
                test_prediction = _predict_dataset(
                    model, model_kind, dataset, "test",
                    classification=classification,
                )
            else:
                training_start = 58
                _progress(progress, "training_final_model", training_start, {"iteration": 0})
                _fit_model(
                    model_kind, model, dataset, evals_result,
                    cancellation=cancellation, progress=progress,
                    stage="training_final_model",
                    progress_start=training_start,
                    progress_end=80,
                    metric_prefix="final.",
                    initial_model=source_model,
                )
                _checkpoint(cancellation)
                _progress(progress, "predicting", 82, {})
                train_prediction = _predict_training_dataset(
                    model, model_kind, dataset, "train", classification=classification,
                )
                valid_prediction = _predict_dataset(
                    model, model_kind, dataset, "valid", classification=classification,
                )
                test_prediction = _predict_dataset(
                    model, model_kind, dataset, "test", classification=classification,
                )
            if hasattr(model, "to_cpu"):
                model.to_cpu()
            if model_kind != "stacking":
                _prepare_model_for_serialization(model_kind, model)
            R.save_objects(trained_model=model)
            recorder_id = R.get_recorder().id
        # 研究回测只允许使用完全样本外的test段；train/valid预测不得进入模型信号库。
        # 开启Walk-Forward时发布各窗口拼接后的OOS预测；根模型只是最新窗口别名，
        # 不得被按日期推理直接使用。
        published_prediction = walk_forward_prediction if walk_forward_prediction is not None else test_prediction
        prediction_frame = _prediction_frame(published_prediction, training_prepared, job)
        predictions_path = work_dir / "predictions.parquet"
        prediction_frame.to_parquet(predictions_path, index=False)
        prediction_rows = len(prediction_frame)
        metric_frame = (
            prepared.raw_frame
            if walk_forward_result is not None and prepared.raw_frame is not None
            else training_prepared.frame
        )
        metric_segments = (
            walk_forward_result.latest_segments
            if walk_forward_result is not None
            else training_prepared.segments
        )
        standard_metrics = _metrics(
            test_prediction, metric_frame,
            metric_segments["test"],
            classification=classification,
        )
        train_metrics = _metrics(
            train_prediction, metric_frame,
            metric_segments["train"],
            classification=classification,
        )
        del train_prediction
        if valid_prediction is None or "valid" not in metric_segments:
            validation_metrics = {
                "enabled": False,
                "reason": "walk_forward_valid_sessions_zero",
                "rows": 0,
                "days": 0,
            }
        else:
            raw_validation_metrics = _metrics(
                valid_prediction, metric_frame,
                metric_segments["valid"],
                classification=classification,
            )
            validation_metrics = {
                "enabled": True,
                "rows": raw_validation_metrics["test_rows"],
                "days": raw_validation_metrics["test_days"],
                "rmse": raw_validation_metrics["rmse"],
                "ic": raw_validation_metrics["ic"],
                "rank_ic": raw_validation_metrics["rank_ic"],
                "ic_ir": raw_validation_metrics["ic_ir"],
                "rank_icir": raw_validation_metrics["rank_icir"],
                **{
                    key: raw_validation_metrics[key]
                    for key in (
                        "mse", "l2", "mae", "auc", "log_loss",
                        "accuracy", "precision", "recall", "f1",
                    )
                    if key in raw_validation_metrics
                },
            }
        normalized_train_metrics = {
            "rows": train_metrics["test_rows"],
            "days": train_metrics["test_days"],
            **{
                key: value
                for key, value in train_metrics.items()
                if key not in {"test_rows", "test_days"}
            },
        }
        if walk_forward_report is not None:
            metrics: dict[str, Any] = {
                **walk_forward_report["aggregate"],
                "evaluation_mode": "walk_forward",
                "walk_forward_windows": walk_forward_report["window_count"],
                "train": normalized_train_metrics,
                "standard_test": standard_metrics,
                "validation": validation_metrics,
            }
        else:
            metrics = {
                **standard_metrics,
                "evaluation_mode": "single_split",
                "train": normalized_train_metrics,
                "validation": validation_metrics,
            }
        if model_kind == "stacking":
            feature_importance = _stacking_feature_importance(
                model, training_prepared.feature_names,
            )
            training_diagnostics = _stacking_training_diagnostics(
                model, evals_result, model_params, train_prediction_rows=int(
                    metrics["train"]["rows"]
                ),
            )
        else:
            feature_importance = _feature_importance(
                model, training_prepared.feature_names,
            )
            training_diagnostics = build_training_diagnostics(
                model_kind, evals_result, model_params,
            )
        research_target = str(
            dataset_spec.get("research_target") or "stock_selection"
        )
        prediction_scope = str(
            dataset_spec.get("prediction_scope") or "stock"
        )
        manifest = {
            **training_prepared.manifest,
            "schema_version": "alphablocks.qlib-training.v1",
            "job_id": job["job_id"],
            "model_id": job["model_id"],
            "model_kind": model_kind,
            "algorithm_ref": {
                "id": model_kind,
                "version": int(model_spec.get("version") or 1),
            },
            "research_target": research_target,
            "prediction_scope": prediction_scope,
            "target_mode": target_mode,
            "execution": dict(subtask.get("parent_execution") or config.get("execution") or {"node_id": "local", "mode": "local"}),
            "model_version": int((job.get("config_json") or {}).get("planned_model_version") or 1),
            "qlib_recorder_id": recorder_id,
            "qlib_recorder_uri": recorder_uri,
            "root_model_role": (
                "latest_window_compatibility_alias"
                if walk_forward_report is not None
                else "primary_model"
            ),
            "model_params": model_params,
            "hyperparameter_optimization": optuna_result,
            "ensemble": (
                {
                    **dict(model_spec.get("ensemble") or {}),
                    "method": "stacking",
                    "n_folds": int(model_params.get("n_folds") or 3),
                    "meta_alpha": float(model_params.get("meta_alpha") or 1.0),
                    "meta_coefficients": [
                        float(value) for value in np.asarray(
                            model.meta_model.coef_, dtype=float,
                        ).reshape(-1)
                    ],
                }
                if model_kind == "stacking" else None
            ),
            "validation_metrics": validation_metrics,
            "training_diagnostics": training_diagnostics,
            "runtime_optimization": _runtime_optimization_profile(model),
            "walk_forward": walk_forward_report,
            "incremental_training": (
                {
                    **incremental_config,
                    "source_artifact": {
                        key: value
                        for key, value in dict(
                            incremental_config.get("source_artifact") or {}
                        ).items()
                        if key != "relative_path"
                    },
                    "segments": training_prepared.segments,
                    "source_tree_count": _tree_count(source_model),
                    "result_tree_count": _tree_count(model),
                }
                if incremental_config else None
            ),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "accelerator": str(os.environ.get("ALPHA_MODEL_ACCELERATOR") or "cpu"),
                "effective_num_threads": int(
                    os.environ.get("ALPHA_EFFECTIVE_NUM_THREADS") or 0
                ),
                "validation_sample_rows": int(
                    os.environ.get("ALPHA_VALIDATION_SAMPLE_ROWS") or 0
                ),
                "qlib": getattr(qlib, "__version__", "unknown"),
                **_model_package_version(model_kind),
            },
            "prediction_rows": prediction_rows,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = work_dir / "manifest.json"
        metrics_path = work_dir / "metrics.json"
        importance_path = work_dir / "feature_importance.json"
        training_diagnostics_path = work_dir / "training_diagnostics.json"
        optuna_path = work_dir / "optuna_trials.json"
        config_path = work_dir / "task_config.json"
        model_path = work_dir / "model.pkl"
        _progress(progress, "packaging", 85, {})
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        importance_path.write_text(json.dumps(feature_importance, ensure_ascii=False, indent=2), encoding="utf-8")
        training_diagnostics_path.write_text(
            json.dumps(training_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        if optuna_result is not None:
            optuna_path.write_text(
                json.dumps(optuna_result, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        config_path.write_text(json.dumps(job.get("config_json") or {}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        with model_path.open("wb") as target:
            pickle.dump(model, target)
        members = [(path.name, path) for path in [manifest_path, metrics_path, importance_path,
                   training_diagnostics_path, config_path, model_path, predictions_path, dataset_manifest_path]]
        if optuna_path.is_file():
            members.append((optuna_path.name, optuna_path))
        if recorder_db.exists():
            members.append((recorder_db.name, recorder_db))
        if recorder_root.exists():
            members.append(('mlruns', recorder_root))
        bundle_path, walk_forward_bundle_path = package_model_bundle(
            work_dir, manifest, members, has_windows=walk_forward_report is not None,
        )
        result = {
            "metrics": metrics,
            "feature_importance": feature_importance,
            "predictions": {
                "row_count": prediction_rows,
                "date_start": prediction_frame["trade_date"].min().isoformat(),
                "date_end": prediction_frame["trade_date"].max().isoformat(),
                "inference_run_id": str(job["job_id"]),
                "model_version": int((job.get("config_json") or {}).get("planned_model_version") or 1),
            },
            "manifest": manifest,
        }
        artifacts = [
            ("bundle", bundle_path),
            ("dataset", dataset_path),
            ("dataset_raw", raw_dataset_path),
            ("dataset_manifest", dataset_manifest_path),
            ("predictions", predictions_path),
            ("manifest", manifest_path),
            ("training_diagnostics", training_diagnostics_path),
        ]
        if walk_forward_report is not None:
            artifacts.append(("walk_forward_series", walk_forward_bundle_path))
        if optuna_path.is_file():
            artifacts.append(("optuna_trials", optuna_path))
        _checkpoint(cancellation)
        _progress(progress, "packaged", 88, {
            "artifact_count": len(artifacts), "prediction_rows": prediction_rows,
        })
        return TrainingResult(result=result, artifacts=artifacts, predictions_path=predictions_path)

    def publish_predictions(
        self,
        path: Path,
        job: dict[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> int:
        """Idempotently publish one inference run after artifacts are durable.

        AlphaBlocks only exposes predictions through a registered model version.  The
        reserved version is removed before insertion so retries or a newly reserved job
        cannot mix two inference runs under one model version.
        """
        _checkpoint(cancellation)
        frame = pd.read_parquet(path)
        required = {
            "trade_date", "entity_code", "raw_prediction", "rank_value", "percentile",
            "score", "feature_cutoff_at", "computed_at", "source_vintage",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError("预测文件缺少字段: " + ", ".join(missing))
        if frame.empty:
            raise ValueError("测试段预测结果为空")
        if frame.duplicated(["trade_date", "entity_code"]).any():
            raise ValueError("预测文件包含重复的日期与预测实体")
        dataset_spec = dict(
            job.get("dataset_spec")
            or (job.get("config_json") or {}).get("dataset")
            or {}
        )
        prediction_scope = str(
            dataset_spec.get("prediction_scope") or "stock"
        ).strip().lower()
        if prediction_scope not in {"stock", "industry"}:
            raise ValueError(f"不支持的预测实体类型: {prediction_scope}")
        client = clickhouse_connect.get_client(
            host=self.settings.clickhouse_host,
            port=self.settings.clickhouse_port,
            username=self.settings.clickhouse_user,
            password=self.settings.clickhouse_password,
            autogenerate_session_id=False,
        )
        model_version = int((job.get("config_json") or {}).get("planned_model_version") or 1)
        dates = sorted({pd.Timestamp(value).date() for value in frame["trade_date"]})
        delete_scope = "AND trade_date IN {trade_dates:Array(Date)}" if str(job.get("kind") or "train") == "infer" else ""
        parameters = {
            "model_id": str(job["model_id"]),
            "model_version": model_version,
        }
        if delete_scope:
            parameters["trade_dates"] = dates
        client.command(
            f"""
            ALTER TABLE {self.settings.model_database}.model_predictions_daily
            DELETE WHERE model_id = {{model_id:String}}
              AND model_version = {{model_version:UInt32}}
              {delete_scope}
            SETTINGS mutations_sync = 2
            """,
            parameters=parameters,
        )
        _checkpoint(cancellation)
        now = datetime.now()
        rows = [
            [
                row.trade_date, prediction_scope, row.entity_code, job["model_id"],
                model_version,
                row.raw_prediction, row.rank_value, row.percentile, row.score,
                row.feature_cutoff_at, row.computed_at, row.source_vintage,
                job["dataset_hash"], job["job_id"], now,
            ]
            for row in frame.itertuples(index=False)
        ]
        client.insert(
            f"{self.settings.model_database}.model_predictions_daily",
            rows,
            column_names=[
                "trade_date", "entity_type", "entity_code", "model_id",
                "model_version", "raw_prediction", "rank_value", "percentile",
                "score", "feature_cutoff_at", "computed_at", "source_vintage",
                "dataset_hash", "inference_run_id", "updated_at",
            ],
        )
        published = int(client.query(
            f"""
            SELECT count()
            FROM {self.settings.model_database}.model_predictions_daily FINAL
            WHERE model_id = {{model_id:String}}
              AND model_version = {{model_version:UInt32}}
              AND inference_run_id = {{inference_run_id:String}}
            """,
            parameters={
                "model_id": str(job["model_id"]),
                "model_version": model_version,
                "inference_run_id": str(job["job_id"]),
            },
        ).result_rows[0][0])
        if published != len(rows):
            raise ValueError(f"预测发布校验失败: 期望{len(rows)}，实际{published}")
        return published


def _prepare_recorder_experiment(
    recorder_uri: str,
    experiment_name: str,
    recorder_root: Path,
) -> None:
    """Keep Qlib Recorder artifacts inside the current job directory."""
    from mlflow.tracking import MlflowClient

    recorder_root.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=recorder_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name,
            artifact_location=recorder_root.resolve().as_uri(),
        )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_walk_forward(
    prepared: PreparedDataset,
    config: dict[str, Any],
    *,
    work_dir: Path,
    model_id: str,
    model_version: int,
    model_kind: str,
    raw_params: dict[str, Any],
    DataHandlerLP: Any,
    DatasetH: Any,
    recorder_uri: str,
    experiment_name: str,
    cancellation: CancellationToken | None,
    progress: ProgressCallback | None,
    distributed: Any = None,
    only_window: int | None = None,
) -> WalkForwardTrainingResult | TrainingResult:
    import qlib
    from qlib.model.trainer import TrainerR
    from qlib.workflow import R

    classification = str(raw_params.get("loss") or "mse") == "binary"
    raw_frame = prepared.raw_frame
    if raw_frame is None:
        raise ValueError("Walk-Forward缺少未填充的冻结数据集")
    dates = pd.Index(raw_frame.index.get_level_values("datetime").unique())
    windows = walk_forward_segments(
        dates,
        strategy=str(config.get("strategy") or "rolling"),
        train_sessions=int(config.get("train_sessions") or 756),
        valid_sessions=int(config.get("valid_sessions", 60)),
        test_sessions=int(config.get("test_sessions") or 20),
        step_sessions=int(config.get("step_sessions") or 20),
        embargo_sessions=int(config.get("embargo_sessions") or 5),
        oos_date_start=str(config.get("oos_date_start") or ""),
        oos_date_end=str(config.get("oos_date_end") or ""),
    )
    if not windows:
        raise ValueError("没有可执行的Walk-Forward窗口")

    series_root = work_dir / "walk_forward"
    series_root.mkdir(parents=True, exist_ok=False)
    total = len(windows)
    tasks = [
        {
            "model": {
                "kind": model_kind,
                "params": dict(raw_params),
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {"segments": segments},
            },
            "alphablocks": {
                "schema_version": "alphablocks.qlib-rolling-task.v1",
                "series_id": model_id,
                "series_revision": int(model_version),
                "window": index,
            },
        }
        for index, segments in enumerate(windows, start=1)
    ]
    # Keep only small manifests in memory. Retaining each DatasetH here used to
    # retain every overlapping window's full DataFrame until the final window.
    reports: list[dict[str, Any]] = []
    prediction_paths: list[Path] = []
    latest: dict[str, Any] = {}

    def train_task(
        task: dict[str, Any],
        task_experiment_name: str,
        recorder_name: str | None = None,
    ) -> Any:
        window_meta = dict(task["alphablocks"])
        index = int(window_meta["window"])
        segments = {
            name: (str(segment[0]), str(segment[1]))
            for name, segment in dict(
                task["dataset"]["kwargs"]["segments"]
            ).items()
        }
        validation_enabled = "valid" in segments
        training_segments = dict(segments)
        if not validation_enabled:
            # Qlib's model adapters expect a named valid segment.  Reusing the
            # train bounds is an internal API-compatibility alias only; every
            # trainer receives validation_enabled=False and must not early-stop,
            # select a checkpoint, or publish metrics from this alias.
            training_segments["valid"] = training_segments["train"]
        _checkpoint(cancellation)
        start_percent = 58 + int(14 * (index - 1) / total)
        end_percent = 58 + int(14 * index / total)
        details = {"window_index": index, "window_count": total, "segments": segments}
        _progress(progress, "walk_forward_training", start_percent, details)
        window_frame, medians = _walk_forward_frame(prepared, segments)
        window_dataset = _dataset_for_model(
            DataHandlerLP.from_df(window_frame), training_segments,
            model_kind, raw_params, DatasetH,
        )
        setattr(window_dataset, "_alphablocks_validation_enabled", validation_enabled)
        window_model, window_params = _create_model(
            model_kind, raw_params, len(prepared.feature_names),
        )
        window_evals: dict[str, Any] = {}
        effective_recorder_name = (
            recorder_name
            or f"{model_id}_v{model_version}_window_{index:04d}"
        )
        with R.start(
            experiment_name=task_experiment_name,
            recorder_name=effective_recorder_name,
        ):
            R.log_params(
                series_id=model_id,
                series_revision=int(model_version),
                window=index,
                segments=json.dumps(segments, sort_keys=True),
                task_generator="qlib.workflow.task.gen.RollingGen",
                trainer="qlib.model.trainer.TrainerR",
            )
            _fit_model(
                model_kind,
                window_model,
                window_dataset,
                window_evals,
                cancellation=cancellation,
                progress=progress,
                stage="walk_forward_training",
                progress_start=start_percent,
                progress_end=end_percent,
                progress_details=details,
                metric_prefix=f"walk_forward.window_{index}.",
            )
            prediction = _predict_dataset(
                window_model, model_kind, window_dataset, "test",
                classification=classification,
            )
            window_metrics = _metrics(
                prediction, raw_frame, segments["test"],
                classification=classification,
            )
            if hasattr(window_model, "to_cpu"):
                window_model.to_cpu()
            _prepare_model_for_serialization(model_kind, window_model)
            R.save_objects(**{
                "task": task,
                "params.pkl": window_model,
                "pred.pkl": prediction,
                "metrics.pkl": window_metrics,
            })
            recorder = R.get_recorder()
        _checkpoint(cancellation)
        window_root = series_root / f"window_{index:04d}"
        window_root.mkdir()
        window_model_path = window_root / "model.pkl"
        with window_model_path.open("wb") as target:
            pickle.dump(window_model, target)
        window_manifest = {
            "schema_version": "alphablocks.qlib-rolling-window.v2",
            "series_id": model_id,
            "series_revision": int(model_version),
            "window": index,
            "model_kind": model_kind,
            "segments": segments,
            "effective_date_start": segments["test"][0],
            "effective_date_end": segments["test"][1],
            "feature_names": list(prepared.feature_names),
            "train_medians": medians,
            "metrics": window_metrics,
            "qlib_recorder_id": recorder.id,
            "qlib_task": task,
            "model_sha256": _file_sha256(window_model_path),
            "model_params": window_params,
            "evals_result": window_evals,
            "dataset_hash": prepared.manifest.get("dataset_hash", ""),
            "environment": {
                "python": platform.python_version(), "platform": platform.platform(),
                "qlib": getattr(qlib, "__version__", "unknown"), **_model_package_version(model_kind),
            },
        }
        prediction_path = window_root / "prediction.parquet"
        prediction.to_frame("prediction").to_parquet(prediction_path)
        window_manifest["prediction_sha256"] = _file_sha256(prediction_path)
        window_manifest_path = window_root / "manifest.json"
        window_manifest_path.write_text(
            json.dumps(window_manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        prediction_paths.append(prediction_path)
        reports.append({
            "window": index,
            "segments": segments,
            "metrics": window_metrics,
            "train_medians": medians,
            "qlib_recorder_id": recorder.id,
            "effective_date_start": segments["test"][0],
            "effective_date_end": segments["test"][1],
            "artifact": {
                "path": f"walk_forward/window_{index:04d}",
                "model_sha256": window_manifest["model_sha256"],
                "manifest_sha256": _file_sha256(window_manifest_path),
            },
        })
        if index == total:
            latest.update({
                "model": window_model,
                "dataset": window_dataset,
                "evals_result": window_evals,
                "segments": segments,
                "model_params": window_params,
            })
        del window_frame, window_dataset, window_model, window_evals, prediction
        release_training_memory()
        _progress(progress, "walk_forward_window_saved", end_percent, {
            **details,
            "completed_windows": index,
            "retained_training_windows": int(index == total),
            "artifact_path": f"walk_forward/window_{index:04d}",
        })
        return recorder

    if only_window is not None:
        if not 1 <= only_window <= total:
            raise ValueError("分布式窗口序号超出冻结时间结构")
        with R.uri_context(recorder_uri):
            train_task(tasks[only_window - 1], experiment_name)
        window_root = series_root / f"window_{only_window:04d}"
        bundle = work_dir / f"window_{only_window:04d}.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            archive.add(window_root, arcname=window_root.name)
        return TrainingResult({"window": only_window, "segments": windows[only_window - 1]},
                              [("window", bundle)], window_root / "prediction.parquet")
    if distributed is not None:
        distributed.windows(tasks, raw_params, series_root)
        for index, segments in enumerate(windows, start=1):
            window_root = series_root / f"window_{index:04d}"
            manifest_path = window_root / "manifest.json"
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (saved["window"] != index or saved["series_id"] != model_id
                    or saved["series_revision"] != model_version
                    or saved.get("dataset_hash") != prepared.manifest["dataset_hash"]
                    or saved["segments"] != {k: list(v) for k, v in segments.items()}
                    or saved["model_sha256"] != _file_sha256(window_root / "model.pkl")
                    or saved["prediction_sha256"] != _file_sha256(window_root / "prediction.parquet")):
                raise ValueError("分布式窗口身份、日期或文件摘要不一致，拒绝发布")
            prediction_paths.append(window_root / "prediction.parquet")
            reports.append({
                key: saved[key] for key in ("window", "segments", "metrics", "train_medians",
                                            "qlib_recorder_id", "effective_date_start", "effective_date_end")
            })
            reports[-1]["artifact"] = {
                "path": f"walk_forward/window_{index:04d}", "model_sha256": saved["model_sha256"],
                "manifest_sha256": _file_sha256(manifest_path),
            }
            reports[-1]['environment'] = saved.get('environment', {})
            prediction_check = pd.read_parquet(window_root / 'prediction.parquet')['prediction']
            prediction_dates = pd.to_datetime(prediction_check.index.get_level_values('datetime'))
            if (prediction_check.empty or not np.isfinite(prediction_check.to_numpy()).all()
                    or (prediction_dates < pd.Timestamp(segments['test'][0])).any()
                    or (prediction_dates > pd.Timestamp(segments['test'][1])).any()):
                raise ValueError('分布式窗口预测为空、非有限值或超出冻结样本外日期')
            del prediction_check, prediction_dates
            if index == total:
                window_frame, _ = _walk_forward_frame(prepared, segments)
                training_segments = dict(segments)
                if "valid" not in training_segments:
                    training_segments["valid"] = training_segments["train"]
                window_dataset = _dataset_for_model(DataHandlerLP.from_df(window_frame), training_segments,
                                                    model_kind, raw_params, DatasetH)
                with (window_root / "model.pkl").open("rb") as stream:
                    window_model = pickle.load(stream)
                latest.update(model=window_model, dataset=window_dataset,
                              evals_result=saved["evals_result"], segments=segments,
                              model_params=saved["model_params"])
    else:
        qlib_trainer = TrainerR(experiment_name=experiment_name, train_func=train_task)
        with R.uri_context(recorder_uri):
            recorders = qlib_trainer.train(tasks)
            qlib_trainer.end_train(recorders)
        if len(recorders) != total:
            raise ValueError("Qlib滚动任务训练结果数量与窗口计划不一致")
    if len(reports) != total:
        raise ValueError("Qlib滚动任务训练结果数量与窗口计划不一致")

    predictions = [pd.read_parquet(path)["prediction"] for path in prediction_paths]
    stitched = pd.concat(predictions).sort_index()
    del predictions
    if stitched.index.duplicated().any():
        raise ValueError("Walk-Forward样本外预测日期重叠")
    aggregate = _metrics(
        stitched,
        raw_frame,
        (windows[0]["test"][0], windows[-1]["test"][1]),
        classification=classification,
    )
    window_ics = np.asarray([float(item["metrics"]["ic"]) for item in reports], dtype=float)
    aggregate.update({
        "window_ic_mean": float(window_ics.mean()),
        "window_ic_std": float(window_ics.std(ddof=1)) if len(window_ics) > 1 else 0.0,
        "positive_ic_window_ratio": float(np.mean(window_ics > 0)),
    })
    report = {
        "schema_version": "alphablocks.qlib-rolling-model-series.v2",
        "enabled": True,
        "series_id": model_id,
        "series_revision": int(model_version),
        "publish_state": "ready",
        "strategy": str(config.get("strategy") or "rolling"),
        "train_sessions": int(config.get("train_sessions") or 756),
        "valid_sessions": int(config.get("valid_sessions", 60)),
        "validation_enabled": int(config.get("valid_sessions", 60)) > 0,
        "test_sessions": int(config.get("test_sessions") or 20),
        "step_sessions": int(config.get("step_sessions") or 20),
        "embargo_sessions": int(config.get("embargo_sessions") or 5),
        "oos_date_start": str(config.get("oos_date_start") or ""),
        "oos_date_end": str(config.get("oos_date_end") or ""),
        "window_count": total,
        "windows": reports,
        "aggregate": aggregate,
        "prediction_date_start": windows[0]["test"][0],
        "prediction_date_end": windows[-1]["test"][1],
        "orchestration": {
            "execution_mode": "distributed" if distributed is not None else "sequential",
            "task_generator": "qlib.workflow.task.gen.RollingGen",
            "trainer": "qlib.model.trainer.TrainerR",
            "recorder_per_window": True,
            "prediction_collector": "alphablocks.strict_oos_stitch",
            "inference_router": "exact_effective_date_interval",
            "root_model_role": "latest_window_compatibility_alias",
        },
        "stability": assess_walk_forward_stability(
            aggregate, window_count=total,
        ),
    }
    return WalkForwardTrainingResult(
        prediction=stitched,
        report=report,
        latest_model=latest["model"],
        latest_dataset=latest["dataset"],
        latest_evals_result=latest["evals_result"],
        latest_segments=latest["segments"],
        latest_model_params=latest["model_params"],
    )


def _walk_forward_frame(
    prepared: PreparedDataset,
    segments: dict[str, tuple[str, str]],
) -> tuple[pd.DataFrame, dict[str, float]]:
    raw_frame = prepared.raw_frame
    if raw_frame is None:
        raise ValueError("Walk-Forward缺少原始特征")
    train_start, train_end = segments["train"]
    train_features = raw_frame.loc[
        pd.IndexSlice[train_start:train_end, :], pd.IndexSlice["feature", :]
    ]
    medians = {
        name: float(
            pd.to_numeric(train_features[("feature", name)], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .median()
        )
        for name in prepared.feature_names
    }
    missing = [name for name, value in medians.items() if not np.isfinite(value)]
    if missing:
        raise ValueError("Walk-Forward训练段无法计算因子中位数: " + ", ".join(missing))
    window_start = segments["train"][0]
    evaluation_segment = (
        segments.get("test") or segments.get("valid") or segments["train"]
    )
    window_end = evaluation_segment[1]
    window_raw = raw_frame.loc[
        pd.IndexSlice[window_start:window_end, :], :
    ]
    manifest = dict(getattr(prepared, "manifest", {}) or {})
    preprocessing = normalize_feature_preprocessing(
        manifest.get("preprocessing"), default_enabled=False,
    )
    frozen_frame = getattr(prepared, "frame", raw_frame)
    processed_frame = frozen_frame.loc[
        pd.IndexSlice[window_start:window_end, :], :
    ]
    frame = _training_frame_for_preprocessing_contract(
        window_raw,
        processed_frame,
        prepared.feature_names,
        preprocessing,
        fallback_values=medians,
        excluded_features=manifest.get("preprocessing_excluded_features") or [],
    )
    return frame, medians


def _load_incremental_source(
    settings: Settings,
    contract: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    artifact = dict(contract.get("source_artifact") or {})
    path = ModelArtifactStore(settings.model_artifacts_root).resolve(
        str(artifact.get("relative_path") or ""),
    )
    expected = str(artifact.get("sha256") or "").lower()
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("增量训练来源模型Bundle的SHA256不一致")
    payloads: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as archive:
        for required in ("model.pkl", "manifest.json"):
            try:
                member = archive.getmember(required)
            except KeyError as exc:
                raise ValueError(f"增量训练来源Bundle缺少{required}") from exc
            if not member.isfile() or member.size <= 0 or member.size > 256 * 1024 * 1024:
                raise ValueError(f"增量训练来源Bundle中的{required}无效")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"增量训练来源Bundle中的{required}不可读")
            payloads[required] = stream.read()
    try:
        model = pickle.loads(payloads["model.pkl"])
        manifest = json.loads(payloads["manifest.json"].decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"增量训练来源Bundle解析失败: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("增量训练来源manifest必须是对象")
    if str(manifest.get("model_kind") or "") != "lightgbm":
        raise ValueError("增量训练来源Bundle不是LightGBM模型")
    bundle_identity = dict(contract.get("source_bundle_identity") or {})
    expected_bundle_model_id = str(
        bundle_identity.get("model_id")
        or contract.get("source_model_id") or ""
    )
    expected_bundle_model_version = int(
        bundle_identity.get("model_version")
        or contract.get("source_model_version") or 0
    )
    if (
        str(manifest.get("model_id") or "") != expected_bundle_model_id
        or int(manifest.get("model_version") or 0)
        != expected_bundle_model_version
    ):
        raise ValueError("增量训练来源Bundle与锁定模型版本不一致")
    return model, manifest


def _incremental_prepared_dataset(
    prepared: PreparedDataset,
    contract: dict[str, Any],
    source_manifest: dict[str, Any],
    *,
    horizon: int,
) -> PreparedDataset:
    raw_frame = prepared.raw_frame
    if raw_frame is None:
        raise ValueError("增量训练缺少未填充的冻结数据集")
    expected_features = list(source_manifest.get("feature_names") or [])
    if expected_features != prepared.feature_names:
        raise ValueError("增量训练来源模型的特征顺序与新数据集不一致")
    source_medians = dict(source_manifest.get("medians") or {})
    missing_medians = [
        name for name in expected_features
        if name not in source_medians or not np.isfinite(float(source_medians[name]))
    ]
    if missing_medians:
        raise ValueError(
            "增量训练来源模型缺少训练段中位数: " + ", ".join(missing_medians)
        )
    source_preprocessing = normalize_feature_preprocessing(
        source_manifest.get("preprocessing"), default_enabled=False,
    )
    candidate_manifest = dict(prepared.manifest or {})
    candidate_preprocessing = normalize_feature_preprocessing(
        candidate_manifest.get("preprocessing"), default_enabled=False,
    )
    source_industry_feature = normalize_industry_feature(
        source_manifest.get("industry_feature"), default_enabled=False,
    )
    candidate_industry_feature = normalize_industry_feature(
        candidate_manifest.get("industry_feature"), default_enabled=False,
    )
    if source_industry_feature != candidate_industry_feature:
        raise ValueError("增量训练来源模型与新数据集的行业特征口径不一致")
    source_excluded = sorted(
        str(name)
        for name in source_manifest.get("preprocessing_excluded_features") or []
    )
    candidate_excluded = sorted(
        str(name)
        for name in candidate_manifest.get("preprocessing_excluded_features") or []
    )
    if (
        source_preprocessing != candidate_preprocessing
        or (
            source_preprocessing["enabled"] is True
            and source_excluded != candidate_excluded
        )
    ):
        raise ValueError("增量训练来源模型与新数据集的特征预处理口径不一致")
    source_end = pd.Timestamp(str(contract.get("source_date_end") or ""))
    dates = pd.to_datetime(raw_frame.index.get_level_values("datetime"))
    new_raw = raw_frame.loc[dates > source_end].copy()
    new_dates = pd.Index(sorted(
        pd.to_datetime(new_raw.index.get_level_values("datetime")).unique()
    ))
    minimum = int(contract.get("minimum_new_trading_sessions") or 60)
    if len(new_dates) < minimum:
        raise ValueError(
            f"增量训练新增有效交易日不足{minimum}天，当前只有{len(new_dates)}天"
        )
    segments = split_trading_dates(new_dates, embargo_days=max(1, int(horizon)))
    prepared_dates = pd.to_datetime(
        prepared.frame.index.get_level_values("datetime"),
    )
    new_processed = prepared.frame.loc[prepared_dates > source_end].copy()
    frame = _training_frame_for_preprocessing_contract(
        new_raw,
        new_processed,
        expected_features,
        source_preprocessing,
        fallback_values=source_medians,
        excluded_features=source_excluded,
    )
    manifest = {
        **prepared.manifest,
        "segments": segments,
        "medians": {name: float(source_medians[name]) for name in expected_features},
        "preprocessing": source_preprocessing,
        "preprocessing_excluded_features": source_excluded,
        "incremental_training": {
            "mode": contract.get("mode"),
            "source_model_id": contract.get("source_model_id"),
            "source_model_version": contract.get("source_model_version"),
            "source_date_end": contract.get("source_date_end"),
            "new_trading_sessions": len(new_dates),
            "segments": segments,
            "preprocessing": "reuse_source_preprocessing_contract",
        },
    }
    return PreparedDataset(
        frame=frame,
        segments=segments,
        feature_names=list(prepared.feature_names),
        coverage=dict(prepared.coverage),
        medians={name: float(source_medians[name]) for name in expected_features},
        manifest=manifest,
        raw_frame=new_raw,
    )


def _training_frame_for_preprocessing_contract(
    raw_frame: pd.DataFrame,
    processed_frame: pd.DataFrame,
    feature_names: list[str],
    preprocessing: dict[str, Any],
    *,
    fallback_values: dict[str, Any],
    excluded_features: list[str],
) -> pd.DataFrame:
    """Build a training slice without recomputing numeric daily sections.

    Enabled numeric features were transformed before labels were joined in the
    immutable dataset snapshot. Recomputing them from ``raw_frame`` would let
    future-label availability change the cross-section. Disabled preprocessing
    remains the legacy train-median transform. Non-scaled categorical/boolean
    features are refilled from the current training window/source contract.
    """
    if preprocessing["enabled"] is not True:
        return preprocess_qlib_frame(
            raw_frame,
            feature_names,
            preprocessing,
            fallback_values=fallback_values,
            excluded_features=excluded_features,
        )
    if not raw_frame.index.equals(processed_frame.index):
        raise ValueError("冻结数据集的原始帧与预处理帧索引不一致")
    frame = processed_frame.copy()
    for name in excluded_features:
        key = ("feature", name)
        if key not in raw_frame.columns:
            raise ValueError(f"冻结数据集缺少非缩放特征{name}")
        values = pd.to_numeric(raw_frame[key], errors="coerce").astype(float)
        values = values.where(np.isfinite(values), np.nan)
        fallback = float(fallback_values.get(name, 0.0))
        if not np.isfinite(fallback):
            fallback = 0.0
        frame[key] = values.fillna(fallback).to_numpy(dtype=np.float64)
    matrix = frame.loc[:, pd.IndexSlice["feature", feature_names]].to_numpy(
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("冻结数据集预处理切片仍包含非有限值")
    return frame


def _tree_count(model: Any | None) -> int:
    booster = getattr(model, "model", None) if model is not None else None
    if booster is None:
        return 0
    method = getattr(booster, "num_trees", None)
    if callable(method):
        try:
            return int(method())
        except Exception:
            return 0
    return 0


def _fit_stacking(
    prepared: PreparedDataset,
    *,
    model_spec: dict[str, Any],
    DataHandlerLP: Any,
    DatasetH: Any,
    classification: bool,
    cancellation: CancellationToken | None,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    """Fit expanding-window OOF bases and a Ridge meta learner."""
    from sklearn.linear_model import Ridge

    base_specs = list(model_spec.get("base_models") or [])
    if not 2 <= len(base_specs) <= 8:
        raise ValueError("Stacking必须配置2到8个基模型")
    params = dict(model_spec.get("params") or {})
    n_folds = int(params.get("n_folds") or 3)
    meta_alpha = float(params.get("meta_alpha") or 1.0)
    train_dates = _segment_trading_dates(
        prepared.frame, prepared.segments["train"],
    )
    if len(train_dates) < n_folds + 1:
        raise ValueError(
            f"训练段只有{len(train_dates)}个交易日，不足以生成{n_folds}折时序OOF"
        )
    fold_size = len(train_dates) // (n_folds + 1)
    if fold_size < 1:
        raise ValueError("训练段交易日不足，无法生成Stacking OOF")
    total_fits = len(base_specs) * (n_folds + 1)
    fit_index = 0
    base_items: list[dict[str, Any]] = []
    oof_by_kind: dict[str, pd.Series] = {}
    valid_by_kind: dict[str, pd.Series] = {}
    test_by_kind: dict[str, pd.Series] = {}
    evals_by_kind: dict[str, Any] = {}

    def fit_progress_bounds() -> tuple[int, int]:
        start = 58 + int(22 * fit_index / max(1, total_fits))
        end = 58 + int(22 * (fit_index + 1) / max(1, total_fits))
        return start, max(start + 1, end)

    for base_index, base_spec in enumerate(base_specs, start=1):
        kind = str(base_spec.get("kind") or "").strip().lower()
        raw_params = dict(base_spec.get("params") or {})
        oof_parts: list[pd.Series] = []
        for fold_index in range(n_folds):
            _checkpoint(cancellation)
            train_end_index = fold_size * (fold_index + 1)
            valid_start_index = train_end_index
            valid_end_index = (
                len(train_dates)
                if fold_index == n_folds - 1
                else min(len(train_dates), train_end_index + fold_size)
            )
            fold_segments = {
                "train": (
                    train_dates[0].strftime("%Y-%m-%d"),
                    train_dates[train_end_index - 1].strftime("%Y-%m-%d"),
                ),
                "valid": (
                    train_dates[valid_start_index].strftime("%Y-%m-%d"),
                    train_dates[valid_end_index - 1].strftime("%Y-%m-%d"),
                ),
                "test": (
                    train_dates[valid_start_index].strftime("%Y-%m-%d"),
                    train_dates[valid_end_index - 1].strftime("%Y-%m-%d"),
                ),
            }
            fold_dataset = _dataset_for_model(
                DataHandlerLP.from_df(prepared.frame),
                fold_segments, kind, raw_params, DatasetH,
            )
            fold_model, _ = _create_model(
                kind, raw_params, len(prepared.feature_names),
            )
            fold_evals: dict[str, Any] = {}
            start, end = fit_progress_bounds()
            _fit_model(
                kind, fold_model, fold_dataset, fold_evals,
                cancellation=cancellation,
                progress=progress,
                stage="stacking_oof_training",
                progress_start=start,
                progress_end=end,
                progress_details={
                    "base_model_kind": kind,
                    "base_model_index": base_index,
                    "base_model_count": len(base_specs),
                    "fold": fold_index + 1,
                    "fold_count": n_folds,
                },
                metric_prefix=f"stacking.{kind}.fold_{fold_index + 1}.",
            )
            fit_index += 1
            oof_parts.append(_predict_dataset(
                fold_model, kind, fold_dataset, "valid",
                classification=classification,
            ).rename(kind))
            del fold_model, fold_dataset
        oof_by_kind[kind] = pd.concat(oof_parts).sort_index()

        full_dataset = _dataset_for_model(
            DataHandlerLP.from_df(prepared.frame),
            prepared.segments, kind, raw_params, DatasetH,
        )
        full_model, normalized_params = _create_model(
            kind, raw_params, len(prepared.feature_names),
        )
        full_evals: dict[str, Any] = {}
        start, end = fit_progress_bounds()
        _fit_model(
            kind, full_model, full_dataset, full_evals,
            cancellation=cancellation,
            progress=progress,
            stage="stacking_base_training",
            progress_start=start,
            progress_end=end,
            progress_details={
                "base_model_kind": kind,
                "base_model_index": base_index,
                "base_model_count": len(base_specs),
            },
            metric_prefix=f"stacking.{kind}.final.",
        )
        fit_index += 1
        valid_by_kind[kind] = _predict_dataset(
            full_model, kind, full_dataset, "valid", classification=classification,
        ).rename(kind)
        test_by_kind[kind] = _predict_dataset(
            full_model, kind, full_dataset, "test", classification=classification,
        ).rename(kind)
        base_items.append({
            "kind": kind,
            "params": normalized_params,
            "model": full_model,
        })
        evals_by_kind[kind] = full_evals
        del full_dataset

    meta_train = pd.concat(oof_by_kind, axis=1, join="inner").dropna()
    if meta_train.empty:
        raise ValueError("Stacking OOF预测没有共同有效样本")
    label = prepared.frame[("label", "LABEL0")]
    meta_label = label.reindex(meta_train.index)
    valid_rows = meta_label.notna() & np.isfinite(meta_train).all(axis=1)
    meta_train = meta_train.loc[valid_rows]
    meta_label = meta_label.loc[valid_rows]
    if len(meta_train) < max(100, len(base_specs) * 20):
        raise ValueError(f"Stacking元学习器有效样本过少: {len(meta_train)}")
    meta_model = Ridge(alpha=meta_alpha, fit_intercept=True)
    meta_model.fit(meta_train.to_numpy(dtype=float), meta_label.to_numpy(dtype=float))
    ensemble_model = QlibStackingModel(
        base_models=base_items,
        meta_model=meta_model,
        classification=classification,
    )

    train_prediction = pd.Series(
        ensemble_model.combine([
            meta_train[item["kind"]].to_numpy(dtype=float) for item in base_items
        ]),
        index=meta_train.index,
        name="prediction",
    )
    valid_prediction = _combine_stacking_series(
        ensemble_model, valid_by_kind,
    )
    test_prediction = _combine_stacking_series(
        ensemble_model, test_by_kind,
    )
    _progress(progress, "stacking_meta_training", 81, {
        "base_model_kinds": [item["kind"] for item in base_items],
        "oof_rows": len(meta_train),
        "n_folds": n_folds,
        "meta_alpha": meta_alpha,
    })
    return {
        "model": ensemble_model,
        "model_params": {
            **params,
            "n_folds": n_folds,
            "meta_alpha": meta_alpha,
            "base_models": [
                {"kind": item["kind"], "params": item["params"]}
                for item in base_items
            ],
            "oof_rows": int(len(meta_train)),
        },
        "evals_result": evals_by_kind,
        "train_prediction": train_prediction,
        "valid_prediction": valid_prediction,
        "test_prediction": test_prediction,
    }


def _segment_trading_dates(
    frame: pd.DataFrame, segment: tuple[str, str],
) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(frame.index.get_level_values("datetime")).normalize()
    start, end = pd.Timestamp(segment[0]), pd.Timestamp(segment[1])
    return pd.DatetimeIndex(sorted(dates[(dates >= start) & (dates <= end)].unique()))


def _combine_stacking_series(
    model: QlibStackingModel, predictions: dict[str, pd.Series],
) -> pd.Series:
    frame = pd.concat(
        {item["kind"]: predictions[item["kind"]] for item in model.base_models},
        axis=1,
        join="inner",
    ).dropna()
    if frame.empty:
        raise ValueError("Stacking基模型预测没有共同有效样本")
    raw = model.combine([
        frame[item["kind"]].to_numpy(dtype=float) for item in model.base_models
    ])
    return pd.Series(raw, index=frame.index, name="prediction")


def _stacking_feature_importance(
    model: QlibStackingModel, feature_names: list[str],
) -> list[dict[str, float | str | int]]:
    coefficients = np.abs(
        np.asarray(model.meta_model.coef_, dtype=float).reshape(-1)
    )
    if coefficients.size != len(model.base_models) or not coefficients.any():
        coefficients = np.ones(len(model.base_models), dtype=float)
    coefficients = coefficients / coefficients.sum()
    combined = {name: 0.0 for name in feature_names}
    for weight, item in zip(coefficients, model.base_models):
        for row in _feature_importance(item["model"], feature_names):
            combined[str(row["factor"])] += float(weight) * abs(float(row["importance"]))
    rows: list[dict[str, float | str | int]] = [
        {"factor": name, "importance": float(combined[name])}
        for name in feature_names
    ]
    rows.sort(key=lambda item: (-float(item["importance"]), str(item["factor"])))
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank
    return rows


def _stacking_training_diagnostics(
    model: QlibStackingModel,
    evals_by_kind: dict[str, Any],
    model_params: dict[str, Any],
    *,
    train_prediction_rows: int,
) -> dict[str, Any]:
    coefficients = np.asarray(model.meta_model.coef_, dtype=float).reshape(-1)
    return {
        "schema_version": "alphablocks.stacking-diagnostics.v1",
        "status": "available",
        "model_kind": "stacking",
        "ensemble_method": "stacking",
        "n_folds": int(model_params.get("n_folds") or 3),
        "oof_rows": int(model_params.get("oof_rows") or train_prediction_rows),
        "meta_learner": {
            "kind": "ridge",
            "alpha": float(model_params.get("meta_alpha") or 1.0),
            "intercept": float(np.asarray(model.meta_model.intercept_).reshape(-1)[0]),
            "coefficients": {
                item["kind"]: float(coefficients[index])
                for index, item in enumerate(model.base_models)
            },
        },
        "base_models": [
            {
                "kind": item["kind"],
                "diagnostics": build_training_diagnostics(
                    item["kind"], evals_by_kind.get(item["kind"], {}), item["params"],
                ),
            }
            for item in model.base_models
        ],
    }


def _runtime_optimization_profile(model: Any) -> dict[str, Any]:
    if isinstance(model, QlibStackingModel):
        return {
            "kind": "stacking",
            "base_models": [
                {
                    "kind": str(item.get("kind") or ""),
                    **dict(getattr(item.get("model"), "runtime_profile", {}) or {}),
                }
                for item in model.base_models
            ],
        }
    return dict(getattr(model, "runtime_profile", {}) or {})


def _fit_model(
    model_kind: str,
    model: Any,
    dataset: Any,
    evals_result: dict[str, Any],
    *,
    cancellation: CancellationToken | None,
    progress: ProgressCallback | None,
    stage: str = "training",
    progress_start: int = 58,
    progress_end: int = 80,
    progress_details: dict[str, Any] | None = None,
    metric_prefix: str = "",
    initial_model: Any | None = None,
) -> None:
    """Fit one Qlib model; LightGBM retains per-iteration cancellation points."""
    validation_enabled = bool(
        getattr(dataset, "_alphablocks_validation_enabled", True)
    )
    if initial_model is not None and model_kind != "lightgbm":
        raise ValueError("首版增量续训只支持LightGBM")
    if model_kind != "lightgbm":
        _checkpoint(cancellation)
        if model_kind in {"xgboost", "catboost"}:
            early_stopping_rounds = int(
                getattr(model, "_alphablocks_early_stopping_rounds", 50)
            )
            num_boost_round = int(
                getattr(model, "_alphablocks_num_boost_round", 1000)
            )
            if validation_enabled:
                model.fit(
                    dataset,
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=20,
                    evals_result=evals_result,
                )
            else:
                from qlib.data.dataset import DataHandlerLP

                train = dataset.prepare(
                    "train", col_set=["feature", "label"],
                    data_key=DataHandlerLP.DK_L,
                )
                x_train = train["feature"]
                y_train = np.asarray(train["label"].values).reshape(-1)
                if model_kind == "xgboost":
                    import xgboost as xgb

                    native_evals: dict[str, Any] = {}
                    matrix = xgb.DMatrix(x_train.values, label=y_train)
                    model.model = xgb.train(
                        model._params,
                        dtrain=matrix,
                        num_boost_round=num_boost_round,
                        evals=[(matrix, "train")],
                        verbose_eval=20,
                        evals_result=native_evals,
                    )
                    evals_result["train"] = list(
                        native_evals["train"].values()
                    )[0]
                else:
                    from catboost import CatBoost, Pool

                    native_params = dict(model._params)
                    native_params.update({
                        "iterations": num_boost_round,
                        "verbose": 20,
                    })
                    native_params.pop("early_stopping_rounds", None)
                    model.model = CatBoost(native_params)
                    model.model.fit(
                        Pool(data=x_train, label=y_train),
                        use_best_model=False,
                    )
                    raw_evaluations = model.model.get_evals_result()
                    train_metrics = dict(raw_evaluations.get("learn") or {})
                    if train_metrics:
                        evals_result["train"] = list(
                            train_metrics.values()
                        )[0]
            if model_kind == "catboost" and validation_enabled and not evals_result:
                raw_evaluations = model.model.get_evals_result()
                train_metrics = dict(raw_evaluations.get("learn") or {})
                valid_metrics = dict(raw_evaluations.get("validation") or {})
                shared_metrics = [name for name in train_metrics if name in valid_metrics]
                if shared_metrics:
                    metric = shared_metrics[0]
                    evals_result.update({
                        "train": list(train_metrics[metric]),
                        "valid": list(valid_metrics[metric]),
                    })
        else:
            import inspect

            def mapped_progress(
                _stage: str, percent: int, details: dict[str, Any],
            ) -> None:
                ratio = min(1.0, max(0.0, (int(percent) - 58) / 22))
                mapped = progress_start + int((progress_end - progress_start) * ratio)
                _progress(progress, stage, mapped, {**(progress_details or {}), **details})

            try:
                fit_signature = inspect.signature(model.fit)
            except (TypeError, ValueError):
                fit_signature = None
            cooperative_fit = (
                fit_signature is not None
                and {"cancellation", "progress"} <= set(fit_signature.parameters)
            )
            if cooperative_fit:
                model.fit(
                    dataset, evals_result=evals_result,
                    cancellation=cancellation, progress=mapped_progress,
                )
            else:
                _checkpoint(cancellation)
                model.fit(dataset, evals_result=evals_result)
                if progress is not None:
                    _progress(
                        progress, stage, progress_end,
                        {**(progress_details or {}), "model_kind": model_kind},
                    )
        _checkpoint(cancellation)
        _progress(
            progress, stage, progress_end,
            {**(progress_details or {}), "model_kind": model_kind, "completed": True},
        )
        return

    import lightgbm as lgb
    prepared_sets = model._prepare_data(dataset)  # Qlib's canonical DatasetH adapter.
    datasets, names = list(zip(*prepared_sets))
    if not validation_enabled:
        datasets, names = datasets[:1], names[:1]
    total = max(1, int(model.num_boost_round))

    def cooperative_callback(environment: Any) -> None:
        _checkpoint(cancellation)
        iteration = int(environment.iteration) + 1
        if iteration == 1 or iteration % 20 == 0:
            percent = min(
                progress_end,
                progress_start + int((progress_end - progress_start) * iteration / total),
            )
            _progress(
                progress,
                stage,
                percent,
                {
                    **(progress_details or {}),
                    "iteration": iteration,
                    "total_iterations": total,
                },
            )

    cooperative_callback.order = 0  # type: ignore[attr-defined]
    cooperative_callback.before_iteration = True  # type: ignore[attr-defined]
    def training_callbacks() -> list[Any]:
        callbacks = [
            cooperative_callback,
            lgb.log_evaluation(period=20),
            lgb.record_evaluation(evals_result),
        ]
        if validation_enabled:
            callbacks.insert(1, lgb.early_stopping(model.early_stopping_rounds))
        return callbacks

    initial_booster = getattr(initial_model, "model", None) if initial_model is not None else None
    if initial_model is not None and initial_booster is None:
        raise ValueError("增量训练来源模型缺少LightGBM Booster")
    try:
        model.model = lgb.train(
            model.params,
            datasets[0],
            num_boost_round=model.num_boost_round,
            valid_sets=datasets,
            valid_names=names,
            callbacks=training_callbacks(),
            init_model=initial_booster,
            keep_training_booster=True,
        )
    except lgb.basic.LightGBMError as exc:
        gpu_requested = str(model.params.get("device_type") or "") == "gpu"
        gpu_error = any(token in str(exc).lower() for token in (
            "gpu tree learner", "opencl", "gpu device", "gpu not found",
        ))
        if not gpu_requested or not gpu_error:
            raise
        for key in ("device_type", "gpu_use_dp", "max_bin"):
            model.params.pop(key, None)
        evals_result.clear()
        _progress(progress, "lightgbm_gpu_fallback", progress_start, {
            **(progress_details or {}),
            "reason": str(exc)[:300],
            "fallback": "cpu",
        })
        model.model = lgb.train(
            model.params,
            datasets[0],
            num_boost_round=model.num_boost_round,
            valid_sets=datasets,
            valid_names=names,
            callbacks=training_callbacks(),
            init_model=initial_booster,
            keep_training_booster=True,
        )
    log_evaluation_history(
        {name: evals_result[name] for name in names},
        metric_prefix=metric_prefix,
        cancellation=cancellation,
        progress=progress,
        progress_percent=progress_end,
        progress_details={
            **(progress_details or {}),
            "model_kind": model_kind,
            "training_stage": stage,
        },
    )
    _progress(
        progress, stage, progress_end,
        {**(progress_details or {}), "model_kind": model_kind, "completed": True},
    )


def _suggest_tree_hyperparameters(trial: Any, model_kind: str) -> dict[str, Any]:
    """QuantMind-compatible tree search spaces with AlphaBlocks field names."""
    if model_kind == "lightgbm":
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.1, log=True,
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 20, 500,
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.4, 0.9,
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.4, 0.9,
            ),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        }
    if model_kind == "xgboost":
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.1, log=True,
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.4, 0.9,
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", 20, 300,
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        }
    if model_kind == "catboost":
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.1, log=True,
            ),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1.0, 10.0, log=True,
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 0.5, 5.0,
            ),
        }
    raise ValueError("Optuna只支持LightGBM、XGBoost或CatBoost")


def _validation_subsegments(
    frame: pd.DataFrame,
    validation_segment: tuple[str, str],
    requested_windows: int,
) -> list[tuple[str, str]]:
    """Split the frozen validation period into contiguous trading-date windows."""
    label = frame[("label", "LABEL0")]
    start, end = validation_segment
    validation = label.loc[pd.IndexSlice[start:end, :]]
    dates = pd.Index(
        validation.index.get_level_values("datetime").unique()
    ).sort_values()
    window_count = max(1, int(requested_windows))
    if window_count > 1 and len(dates) < window_count * 2:
        raise ValueError(
            f"Optuna稳健验证需要每个窗口至少2个交易日；"
            f"当前验证集{len(dates)}日，配置{window_count}个窗口"
        )
    chunks = np.array_split(dates.to_numpy(), window_count)
    return [
        (
            pd.Timestamp(chunk[0]).date().isoformat(),
            pd.Timestamp(chunk[-1]).date().isoformat(),
        )
        for chunk in chunks
        if len(chunk)
    ]


def _seeded_tree_params(
    base_params: dict[str, Any], model_kind: str, seed: int,
) -> dict[str, Any]:
    params = {**base_params, "seed": int(seed)}
    if model_kind == "catboost":
        params["random_seed"] = int(seed)
    return params


def _walk_forward_optuna_segments(
    prepared: PreparedDataset,
    walk_forward_config: dict[str, Any],
    fold_count: int,
) -> list[dict[str, tuple[str, str]]]:
    """Build chronological inner tuning folds before the sealed OOS start."""
    raw_frame = prepared.raw_frame
    if raw_frame is None:
        raise ValueError("Walk-Forward Optuna缺少未填充的冻结数据集")
    valid_sessions = int(walk_forward_config.get("valid_sessions", 60))
    if valid_sessions < 2:
        raise ValueError("Walk-Forward Optuna要求验证长度至少为2个交易日")
    dates = pd.Index(sorted(
        pd.to_datetime(
            raw_frame.index.get_level_values("datetime"),
        ).normalize().unique()
    ))
    outer_windows = walk_forward_segments(
        dates,
        strategy=str(walk_forward_config.get("strategy") or "rolling"),
        train_sessions=int(walk_forward_config.get("train_sessions") or 756),
        valid_sessions=valid_sessions,
        test_sessions=int(walk_forward_config.get("test_sessions") or 20),
        step_sessions=int(walk_forward_config.get("step_sessions") or 20),
        embargo_sessions=int(walk_forward_config.get("embargo_sessions") or 5),
        oos_date_start=str(walk_forward_config.get("oos_date_start") or ""),
        oos_date_end=str(walk_forward_config.get("oos_date_end") or ""),
    )
    if not outer_windows:
        raise ValueError("Walk-Forward Optuna无法生成正式样本外窗口")
    first = outer_windows[0]
    base_train_start = int(dates.get_loc(pd.Timestamp(first["train"][0])))
    base_train_end = int(dates.get_loc(pd.Timestamp(first["train"][1])))
    base_valid_start = int(dates.get_loc(pd.Timestamp(first["valid"][0])))
    base_valid_end = int(dates.get_loc(pd.Timestamp(first["valid"][1])))
    strategy = str(walk_forward_config.get("strategy") or "rolling")
    folds: list[dict[str, tuple[str, str]]] = []
    for reverse_offset in range(int(fold_count) - 1, -1, -1):
        shift = reverse_offset * valid_sessions
        train_start = 0 if strategy == "expanding" else base_train_start - shift
        train_end = base_train_end - shift
        valid_start = base_valid_start - shift
        valid_end = base_valid_end - shift
        if train_start < 0 or train_end - train_start + 1 < 252:
            raise ValueError(
                "Walk-Forward Optuna历史不足：请减少调参折数或推迟样本外开始日期"
            )
        folds.append({
            "train": (
                dates[train_start].date().isoformat(),
                dates[train_end].date().isoformat(),
            ),
            "valid": (
                dates[valid_start].date().isoformat(),
                dates[valid_end].date().isoformat(),
            ),
        })
    return folds


def _tune_tree_hyperparameters(
    prepared: PreparedDataset,
    *,
    model_kind: str,
    base_params: dict[str, Any],
    config: dict[str, Any],
    walk_forward_config: dict[str, Any] | None = None,
    DataHandlerLP: Any,
    DatasetH: Any,
    classification: bool,
    cancellation: CancellationToken | None,
    progress: ProgressCallback | None,
    distributed: Any = None,
    fixed_trial: dict | None = None,
) -> dict[str, Any]:
    """Search frozen validation data and return auditable trials.

    V2 scores fixed validation sub-windows. V3 uses those sub-windows for a
    single split, but switches to independently refitted inner folds when the
    job enables Walk-Forward. The sealed outer test segment is never read.
    """
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna尚未安装，请更新AlphaFactorService训练环境") from exc

    n_trials = int(config.get("n_trials") or 20)
    seed = int(config.get("seed") if config.get("seed") is not None else 42)
    search_space_version = str(
        config.get("search_space_version") or "alphablocks.tree-optuna.v1"
    )
    robust_selection = search_space_version in {
        "alphablocks.tree-optuna.v2", "alphablocks.tree-optuna.v3",
    }
    validation_mode = str(config.get("validation_mode") or (
        "walk_forward_folds"
        if search_space_version == "alphablocks.tree-optuna.v3"
        and dict(walk_forward_config or {}).get("enabled") is True
        else "fixed_subwindows"
    ))
    walk_forward_tuning = validation_mode == "walk_forward_folds"
    requested_windows = (
        int(config.get("validation_windows") or 3) if robust_selection else 1
    )
    seed_count = (
        int(config.get("seed_count") or 3) if robust_selection else 1
    )
    stability_penalty = (
        float(config.get("stability_penalty", 0.5))
        if robust_selection else 0.0
    )
    minimum_positive_window_ratio = (
        float(config.get("minimum_positive_window_ratio", 0.6))
        if robust_selection else 0.0
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    validation_segment = prepared.segments.get("valid")
    if not validation_segment:
        raise ValueError("Optuna需要非空验证集")
    tuning_folds = (
        _walk_forward_optuna_segments(
            prepared, dict(walk_forward_config or {}), requested_windows,
        )
        if walk_forward_tuning else []
    )
    validation_windows = (
        [fold["valid"] for fold in tuning_folds]
        if walk_forward_tuning
        else (
            _validation_subsegments(
                prepared.frame, validation_segment, requested_windows,
            )
            if robust_selection else [validation_segment]
        )
    )
    trial_seeds = [
        (seed + offset) % 2_147_483_648 for offset in range(seed_count)
    ]

    def objective(trial: Any) -> float:
        _checkpoint(cancellation)
        suggested = _suggest_tree_hyperparameters(trial, model_kind)
        seed_evaluations: list[dict[str, Any]] = []
        window_values: list[float] = []
        window_rank_ics: list[float] = []
        first_full_metrics: dict[str, Any] | None = None

        def fit_candidate(
            trial_params: dict[str, Any],
            frame: pd.DataFrame,
            segments: dict[str, tuple[str, str]],
            *,
            repeat_index: int,
            fold_number: int | None,
        ) -> pd.Series:
            trial_dataset = _dataset_for_model(
                DataHandlerLP.from_df(frame), segments,
                model_kind, trial_params, DatasetH,
            )
            trial_model, _ = _create_model(
                model_kind, trial_params, len(prepared.feature_names),
            )
            evals_result: dict[str, Any] = {}
            progress_details = {
                "trial": int(trial.number) + 1,
                "trial_count": n_trials,
                "seed_repeat": repeat_index,
                "seed_count": seed_count,
            }
            if fold_number is not None:
                progress_details.update({
                    "tuning_fold": fold_number,
                    "tuning_fold_count": len(tuning_folds),
                })
            _fit_model(
                model_kind, trial_model, trial_dataset, evals_result,
                cancellation=cancellation,
                progress=progress,
                stage="optuna_trial",
                progress_start=58,
                progress_end=58,
                progress_details=progress_details,
                metric_prefix=(
                    f"optuna.trial_{int(trial.number) + 1}."
                    f"seed_{repeat_index}."
                    + (f"fold_{fold_number}." if fold_number is not None else "")
                ),
            )
            prediction = _predict_dataset(
                trial_model, model_kind, trial_dataset, "valid",
                classification=classification,
            )
            if hasattr(trial_model, "to_cpu"):
                trial_model.to_cpu()
            _prepare_model_for_serialization(model_kind, trial_model)
            del trial_model, trial_dataset, evals_result
            release_training_memory()
            return prediction

        for repeat_index, trial_seed in enumerate(trial_seeds, start=1):
            _checkpoint(cancellation)
            trial_params = _seeded_tree_params(
                {**base_params, **suggested}, model_kind, trial_seed,
            )
            if walk_forward_tuning:
                windows: list[dict[str, Any]] = []
                metric_frame = (
                    prepared.raw_frame
                    if prepared.raw_frame is not None else prepared.frame
                )
                for fold_number, fold_segments in enumerate(tuning_folds, start=1):
                    fold_frame, _ = _walk_forward_frame(prepared, fold_segments)
                    prediction = fit_candidate(
                        trial_params, fold_frame, fold_segments,
                        repeat_index=repeat_index, fold_number=fold_number,
                    )
                    metrics = _metrics(
                        prediction, metric_frame, fold_segments["valid"],
                        classification=classification,
                    )
                    rank_icir = float(metrics["rank_icir"])
                    rank_ic = float(metrics["rank_ic"])
                    if not np.isfinite(rank_icir):
                        raise optuna.TrialPruned("验证折Rank ICIR不是有限值")
                    window_values.append(rank_icir)
                    window_rank_ics.append(rank_ic)
                    windows.append({
                        "window": fold_number,
                        "train": fold_segments["train"],
                        "date_start": fold_segments["valid"][0],
                        "date_end": fold_segments["valid"][1],
                        "test_rows": int(metrics["test_rows"]),
                        "test_days": int(metrics["test_days"]),
                        "rmse": float(metrics["rmse"]),
                        "rank_ic": rank_ic,
                        "rank_icir": rank_icir,
                    })
                    del fold_frame, prediction
                    release_training_memory()
                seed_evaluations.append({
                    "seed": trial_seed,
                    "windows": windows,
                })
                continue
            prediction = fit_candidate(
                trial_params, prepared.frame, prepared.segments,
                repeat_index=repeat_index, fold_number=None,
            )
            full_metrics = _metrics(
                prediction, prepared.frame, validation_segment,
                classification=classification,
            )
            if first_full_metrics is None:
                first_full_metrics = full_metrics
            windows: list[dict[str, Any]] = []
            for window_number, segment in enumerate(validation_windows, start=1):
                metrics = _metrics(
                    prediction, prepared.frame, segment,
                    classification=classification,
                )
                rank_icir = float(metrics["rank_icir"])
                rank_ic = float(metrics["rank_ic"])
                if not np.isfinite(rank_icir):
                    raise optuna.TrialPruned("验证窗口Rank ICIR不是有限值")
                window_values.append(rank_icir)
                window_rank_ics.append(rank_ic)
                windows.append({
                    "window": window_number,
                    "date_start": segment[0],
                    "date_end": segment[1],
                    "test_rows": int(metrics["test_rows"]),
                    "test_days": int(metrics["test_days"]),
                    "rmse": float(metrics["rmse"]),
                    "rank_ic": rank_ic,
                    "rank_icir": rank_icir,
                })
            seed_evaluations.append({
                "seed": trial_seed,
                "full_validation": {
                    key: full_metrics[key]
                    for key in ("test_rows", "test_days", "rmse", "rank_ic", "rank_icir")
                },
                "windows": windows,
            })
        mean_rank_icir = float(np.mean(window_values))
        std_rank_icir = float(np.std(window_values, ddof=0))
        positive_window_ratio = float(np.mean(np.asarray(window_rank_ics) > 0.0))
        value = mean_rank_icir - stability_penalty * std_rank_icir
        if first_full_metrics is None:
            first_windows = list(seed_evaluations[0]["windows"])
            first_full_metrics = {
                "test_rows": sum(item["test_rows"] for item in first_windows),
                "test_days": sum(item["test_days"] for item in first_windows),
                "rmse": float(np.mean([
                    item["rmse"] for item in first_windows
                ])),
                "rank_ic": float(np.mean([
                    item["rank_ic"] for item in first_windows
                ])),
                "rank_icir": float(np.mean([
                    item["rank_icir"] for item in first_windows
                ])),
            }
        full_metrics = first_full_metrics
        for key in ("test_rows", "test_days", "rmse", "rank_ic", "rank_icir"):
            trial.set_user_attr(key, full_metrics.get(key))
        trial.set_user_attr("validation_windows", seed_evaluations)
        trial.set_user_attr("mean_rank_icir", mean_rank_icir)
        trial.set_user_attr("std_rank_icir", std_rank_icir)
        trial.set_user_attr("positive_window_ratio", positive_window_ratio)
        trial.set_user_attr("stability_score", value)
        if not np.isfinite(value):
            raise optuna.TrialPruned("验证集稳健性得分不是有限值")
        return value

    if fixed_trial is not None:
        trial = optuna.trial.FixedTrial(fixed_trial["params"], number=int(fixed_trial["trial_number"]))
        try:
            value = objective(trial)
            return {"state": "complete", "value": value, "params": dict(trial.params),
                    "attrs": dict(trial.user_attrs)}
        except optuna.TrialPruned as exc:
            return {"state": "pruned", "value": None, "params": dict(trial.params),
                    "attrs": dict(trial.user_attrs), "reason": str(exc)}

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, constant_liar=distributed is not None),
        study_name=f"{model_kind}_validation_rank_icir",
    )

    def report_trial(_study: Any, frozen_trial: Any) -> None:
        _progress(progress, "optuna_trial_completed", 58, {
            "trial": int(frozen_trial.number) + 1,
            "trial_count": n_trials,
            "state": frozen_trial.state.name.lower(),
            "value": (
                float(frozen_trial.value)
                if frozen_trial.value is not None else None
            ),
        })

    if distributed is not None:
        study = distributed.optimize(study, n_trials, model_kind, report_trial)
    else:
        study.optimize(objective, n_trials=n_trials, callbacks=[report_trial],
                       show_progress_bar=False, gc_after_trial=True)
    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise ValueError("Optuna没有完成任何有效试验")
    eligible = [
        trial for trial in completed
        if float(trial.user_attrs.get("positive_window_ratio") or 0.0)
        >= minimum_positive_window_ratio
    ]
    selection_pool = eligible or completed
    selected_trial = max(
        selection_pool,
        key=lambda item: (
            float(item.value) if item.value is not None else float("-inf")
        ),
    )
    trials = []
    for trial in study.trials:
        duration = (
            (trial.datetime_complete - trial.datetime_start).total_seconds()
            if trial.datetime_start is not None and trial.datetime_complete is not None
            else None
        )
        if distributed is not None:
            duration = trial.user_attrs.get('distributed_elapsed_seconds')
        trials.append({
            "trial_number": int(trial.number) + 1,
            "state": trial.state.name.lower(),
            "value": float(trial.value) if trial.value is not None else None,
            "params": dict(trial.params),
            "validation": dict(trial.user_attrs),
            "duration_seconds": float(duration) if duration is not None else None,
        })
    result = {
        "schema_version": (
            "alphablocks.optuna-search-result.v3"
            if search_space_version == "alphablocks.tree-optuna.v3"
            else (
                "alphablocks.optuna-search-result.v2"
                if robust_selection else "alphablocks.optuna-search-result.v1"
            )
        ),
        "enabled": True,
        "backend": "optuna",
        "model_kind": model_kind,
        "objective": "validation_rank_icir",
        "direction": "maximize",
        "sampler": "tpe",
        "seed": seed,
        "search_space_version": search_space_version,
        "validation_mode": validation_mode,
        "tuning_fold_segments": tuning_folds,
        "selection_metric": (
            "validation_rank_icir_stability_score"
            if robust_selection else "validation_rank_icir"
        ),
        "validation_windows": len(validation_windows),
        "seed_count": seed_count,
        "stability_penalty": stability_penalty,
        "minimum_positive_window_ratio": minimum_positive_window_ratio,
        "positive_ratio_gate_satisfied": bool(eligible),
        "selection_fallback": (
            None if eligible else "no_trial_met_positive_window_ratio"
        ),
        "requested_trials": n_trials,
        "completed_trials": len(completed),
        "best_trial_number": int(selected_trial.number) + 1,
        "best_value": float(selected_trial.value),
        "best_params": dict(selected_trial.params),
        "trials": trials,
    }
    _progress(progress, "optuna_completed", 58, {
        "completed_trials": len(completed),
        "best_trial_number": result["best_trial_number"],
        "best_value": result["best_value"],
        "best_params": result["best_params"],
    })
    return result


def _checkpoint(cancellation: CancellationToken | None) -> None:
    if cancellation is not None:
        cancellation.checkpoint()


def _progress(
    callback: ProgressCallback | None, stage: str, percent: int, details: dict[str, Any],
) -> None:
    if callback is not None:
        callback(stage, percent, details)


def _effective_num_threads(source: dict[str, Any]) -> int:
    requested = max(1, int(source.get("num_threads", 4)))
    raw_override = str(os.environ.get("ALPHA_EFFECTIVE_NUM_THREADS") or "").strip()
    if not raw_override:
        return requested
    try:
        return max(1, min(32, int(raw_override)))
    except ValueError:
        return requested


def _qlib_lgb_params(source: dict[str, Any]) -> dict[str, Any]:
    params = {
        "loss": str(source.get("loss") or "mse"),
        "metric": str(source.get("metric") or (
            "auc" if source.get("loss") == "binary" else "rmse"
        )),
        "learning_rate": float(source.get("learning_rate", 0.02)),
        "num_leaves": int(source.get("num_leaves", 31)),
        "max_depth": int(source.get("max_depth", -1)),
        "num_boost_round": int(source.get("n_estimators", 2000)),
        "early_stopping_rounds": int(source.get("early_stopping_rounds", 50)),
        "bagging_fraction": float(source.get("bagging_fraction", source.get("subsample", 0.8))),
        "feature_fraction": float(source.get("feature_fraction", source.get("colsample_bytree", 0.7))),
        "lambda_l1": float(source.get("lambda_l1", source.get("reg_alpha", 0.5))),
        "lambda_l2": float(source.get("lambda_l2", source.get("reg_lambda", 1.0))),
        "min_data_in_leaf": int(source.get("min_data_in_leaf", 300)),
        "min_child_samples": int(source.get("min_child_samples", 150)),
        "path_smooth": float(source.get("path_smooth", 1.0)),
        "bagging_freq": int(source.get("bagging_freq", 5)),
        "num_threads": _effective_num_threads(source),
        "seed": int(source.get("seed", 42)),
        "feature_fraction_seed": int(source.get("seed", 42)),
        "bagging_seed": int(source.get("seed", 42)),
        "data_random_seed": int(source.get("seed", 42)),
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    if str(os.environ.get("ALPHA_MODEL_ACCELERATOR") or "cpu") == "cuda":
        params.update({
            "device_type": "gpu",
            "gpu_use_dp": False,
            "max_bin": 255,
        })
    return params


def _create_model(kind: str, source: dict[str, Any], feature_count: int) -> tuple[Any, dict[str, Any]]:
    loss = str(source.get("loss") or "mse").strip().lower()
    classification = loss == "binary"
    metric = str(source.get("metric") or ("auc" if classification else "rmse")).strip().lower()
    if kind == "lightgbm":
        from qlib.contrib.model.gbdt import LGBModel

        params = _qlib_lgb_params(source)
        return LGBModel(**params), params
    if kind == "xgboost":
        try:
            from qlib.contrib.model.xgboost import XGBModel
        except ImportError as exc:
            raise RuntimeError("XGBoost尚未安装，请执行uv sync") from exc
        params = {
            "objective": "binary:logistic" if classification else "reg:squarederror",
            "eval_metric": {
                "l2": "rmse", "rmse": "rmse", "mae": "mae",
                "auc": "auc", "binary_logloss": "logloss",
            }[metric],
            "eta": float(source.get("learning_rate", 0.02)),
            "max_depth": int(source.get("max_depth", 4)),
            "subsample": float(source.get("subsample", 0.7)),
            "colsample_bytree": float(source.get("colsample_bytree", 0.65)),
            "alpha": float(source.get("reg_alpha", 0.5)),
            "lambda": float(source.get("reg_lambda", 2.0)),
            "min_child_weight": float(source.get("min_child_weight", 100.0)),
            "nthread": _effective_num_threads(source),
            "seed": int(source.get("seed", 42)),
            "verbosity": 0,
        }
        if str(os.environ.get("ALPHA_MODEL_ACCELERATOR") or "cpu") == "cuda":
            params.update({"device": "cuda", "tree_method": "hist"})
        model = XGBModel(**params)
        model._alphablocks_num_boost_round = int(source.get("n_estimators", 2000))
        model._alphablocks_early_stopping_rounds = int(source.get("early_stopping_rounds", 50))
        return model, {
            **params,
            "loss": loss,
            "num_boost_round": model._alphablocks_num_boost_round,
            "early_stopping_rounds": model._alphablocks_early_stopping_rounds,
        }
    if kind == "catboost":
        try:
            from qlib.contrib.model.catboost_model import CatBoostModel
        except ImportError as exc:
            raise RuntimeError("CatBoost尚未安装，请执行uv sync") from exc
        params = {
            "eval_metric": {
                "l2": "RMSE", "rmse": "RMSE", "mae": "MAE",
                "auc": "AUC", "binary_logloss": "Logloss",
            }[metric],
            "learning_rate": float(source.get("learning_rate", 0.02)),
            "depth": int(source.get("depth", 6)),
            "l2_leaf_reg": float(source.get("l2_leaf_reg", 3.0)),
            "random_strength": float(source.get("random_strength", 1.5)),
            "bagging_temperature": float(source.get("bagging_temperature", 0.8)),
            "thread_count": _effective_num_threads(source),
            "random_seed": int(source.get("seed", 42)),
            "allow_writing_files": False,
        }
        if str(os.environ.get("ALPHA_MODEL_ACCELERATOR") or "cpu") == "cuda":
            params.update({"task_type": "GPU", "devices": "0"})
        model = CatBoostModel(loss="Logloss" if classification else "RMSE", **params)
        model._alphablocks_loss = loss
        model._alphablocks_num_boost_round = int(source.get("n_estimators", 2000))
        # Qlib's CatBoostModel.fit always forwards early_stopping_rounds to
        # CatBoost.  CatBoost treats that value and od_wait as aliases and
        # rejects a configuration containing both, so translate the UI's
        # od_wait into Qlib's single early-stopping setting instead of passing
        # both aliases to the native estimator.
        model._alphablocks_early_stopping_rounds = int(
            source.get("early_stopping_rounds", source.get("od_wait", 50))
        )
        return model, {
            **params,
            "loss": loss,
            "num_boost_round": model._alphablocks_num_boost_round,
            "early_stopping_rounds": model._alphablocks_early_stopping_rounds,
        }
    if kind == "mlp":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch尚未安装，请执行uv sync") from exc
        from factor_service.research.models import QlibTorchMLPModel

        raw_layers = source.get("hidden_layers")
        if isinstance(raw_layers, list) and raw_layers:
            hidden_layers = [int(width) for width in raw_layers]
        else:
            hidden_size = int(source.get("hidden_size", 64))
            hidden_layers = [
                max(4, hidden_size // (2 ** index))
                for index in range(int(source.get("layer_count", 2)))
            ]
        params = {
            "loss": loss,
            "learning_rate": float(source.get("learning_rate", 0.0001)),
            "max_steps": int(source.get("max_steps", 200)),
            "batch_size": int(source.get("batch_size", 4000)),
            "early_stopping_rounds": int(source.get("early_stopping_rounds", 20)),
            "eval_steps": int(source.get("eval_steps", 10)),
            "seed": int(source.get("seed", 42)),
            "weight_decay": float(source.get("weight_decay", 0.0001)),
            "input_dim": feature_count,
            "hidden_layers": hidden_layers,
            "num_threads": _effective_num_threads(source),
        }
        return QlibTorchMLPModel(**params), params
    if kind == "lstm":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch尚未安装，请执行uv sync") from exc
        from factor_service.research.models import QlibTorchLSTMModel

        params = {
            "loss": loss,
            "learning_rate": float(source.get("learning_rate", 0.001)),
            "lookback_window": int(source.get("lookback_window", 20)),
            "hidden_size": int(source.get("hidden_size", 64)),
            "num_layers": int(source.get("num_layers", 2)),
            "dropout": float(source.get("dropout", 0.2)),
            "max_steps": int(source.get("max_steps", 200)),
            "batch_size": int(source.get("batch_size", 4000)),
            "early_stopping_rounds": int(source.get("early_stopping_rounds", 20)),
            "eval_steps": int(source.get("eval_steps", 10)),
            "seed": int(source.get("seed", 42)),
            "weight_decay": float(source.get("weight_decay", 0.0001)),
            "input_dim": feature_count,
            "num_threads": _effective_num_threads(source),
        }
        return QlibTorchLSTMModel(**params), params
    if kind == "transformer_lstm":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch尚未安装，请执行uv sync") from exc
        from factor_service.research.models import QlibTorchTransformerLSTMModel

        params = {
            "loss": loss,
            "learning_rate": float(source.get("learning_rate", 0.001)),
            "lookback_window": int(source.get("lookback_window", 60)),
            "d_model": int(source.get("d_model", 64)),
            "nhead": int(source.get("nhead", 4)),
            "transformer_layers": int(source.get("transformer_layers", 2)),
            "dim_feedforward": int(source.get("dim_feedforward", 256)),
            "lstm_hidden_size": int(source.get("lstm_hidden_size", 128)),
            "lstm_layers": int(source.get("lstm_layers", 1)),
            "dropout": float(source.get("dropout", 0.2)),
            "max_steps": int(source.get("max_steps", 300)),
            "batch_size": int(source.get("batch_size", 256)),
            "early_stopping_rounds": int(source.get("early_stopping_rounds", 20)),
            "eval_steps": int(source.get("eval_steps", 10)),
            "seed": int(source.get("seed", 42)),
            "weight_decay": float(source.get("weight_decay", 0.0001)),
            "input_dim": feature_count,
            "num_threads": _effective_num_threads(source),
        }
        return QlibTorchTransformerLSTMModel(**params), params
    if kind in {"gru", "alstm"}:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch尚未安装，请执行uv sync") from exc
        if kind == "gru":
            from factor_service.research.models import QlibTorchGRUModel as ModelClass
        else:
            from factor_service.research.models import QlibTorchALSTMModel as ModelClass

        params = {
            "loss": loss,
            "learning_rate": float(source.get("learning_rate", 0.001)),
            "lookback_window": int(source.get("lookback_window", 20)),
            "hidden_size": int(source.get("hidden_size", 64)),
            "num_layers": int(source.get("num_layers", 2)),
            "dropout": float(source.get("dropout", 0.2)),
            "max_steps": int(source.get("max_steps", 200)),
            "batch_size": int(source.get("batch_size", 4000)),
            "early_stopping_rounds": int(source.get("early_stopping_rounds", 20)),
            "eval_steps": int(source.get("eval_steps", 10)),
            "seed": int(source.get("seed", 42)),
            "weight_decay": float(source.get("weight_decay", 0.0001)),
            "input_dim": feature_count,
            "num_threads": _effective_num_threads(source),
        }
        return ModelClass(**params), params
    if kind == "transformer":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch尚未安装，请执行uv sync") from exc
        from factor_service.research.models import QlibTorchTransformerModel

        params = {
            "loss": loss,
            "learning_rate": float(source.get("learning_rate", 0.0001)),
            "lookback_window": int(source.get("lookback_window", 20)),
            "d_model": int(source.get("d_model", 64)),
            "nhead": int(source.get("nhead", 4)),
            "transformer_layers": int(source.get("transformer_layers", 2)),
            "dim_feedforward": int(source.get("dim_feedforward", 256)),
            "dropout": float(source.get("dropout", 0.2)),
            "max_steps": int(source.get("max_steps", 200)),
            "batch_size": int(source.get("batch_size", 4000)),
            "early_stopping_rounds": int(source.get("early_stopping_rounds", 20)),
            "eval_steps": int(source.get("eval_steps", 10)),
            "seed": int(source.get("seed", 42)),
            "weight_decay": float(source.get("weight_decay", 0.0001)),
            "input_dim": feature_count,
            "num_threads": _effective_num_threads(source),
        }
        return QlibTorchTransformerModel(**params), params
    if kind == "tcn":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch尚未安装，请执行uv sync") from exc
        from factor_service.research.models import QlibTorchTCNModel

        params = {
            "loss": loss,
            "learning_rate": float(source.get("learning_rate", 0.0001)),
            "lookback_window": int(source.get("lookback_window", 20)),
            "hidden_size": int(source.get("hidden_size", 128)),
            "kernel_size": int(source.get("kernel_size", 5)),
            "num_layers": int(source.get("num_layers", 2)),
            "dropout": float(source.get("dropout", 0.2)),
            "max_steps": int(source.get("max_steps", 200)),
            "batch_size": int(source.get("batch_size", 4000)),
            "early_stopping_rounds": int(source.get("early_stopping_rounds", 20)),
            "eval_steps": int(source.get("eval_steps", 10)),
            "seed": int(source.get("seed", 42)),
            "weight_decay": float(source.get("weight_decay", 0.0001)),
            "input_dim": feature_count,
            "num_threads": _effective_num_threads(source),
        }
        return QlibTorchTCNModel(**params), params
    if kind == "nativetft":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch尚未安装，请执行uv sync") from exc
        from factor_service.research.models import QlibTorchNativeTFTModel

        params = {
            "loss": loss,
            "learning_rate": float(source.get("learning_rate", 0.0005)),
            "lookback_window": int(source.get("lookback_window", 20)),
            "d_model": int(source.get("d_model", 64)),
            "nhead": int(source.get("nhead", 4)),
            "gru_hidden_size": int(source.get("gru_hidden_size", 64)),
            "num_layers": int(source.get("num_layers", 2)),
            "dim_feedforward": int(source.get("dim_feedforward", 128)),
            "dropout": float(source.get("dropout", 0.2)),
            "max_steps": int(source.get("max_steps", 200)),
            "batch_size": int(source.get("batch_size", 4000)),
            "early_stopping_rounds": int(source.get("early_stopping_rounds", 20)),
            "eval_steps": int(source.get("eval_steps", 10)),
            "seed": int(source.get("seed", 42)),
            "weight_decay": float(source.get("weight_decay", 0.0001)),
            "input_dim": feature_count,
            "num_threads": _effective_num_threads(source),
        }
        return QlibTorchNativeTFTModel(**params), params
    if kind == "tabnet":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch尚未安装，请执行uv sync") from exc
        from factor_service.research.models import QlibNativeTabNetAdapter

        params = {
            "loss": loss,
            "lr": float(source.get("learning_rate", 0.005)),
            "n_d": int(source.get("n_d", 64)),
            "n_a": int(source.get("n_a", 64)),
            "n_steps": int(source.get("n_steps", 5)),
            "n_shared": int(source.get("n_shared", 2)),
            "n_ind": int(source.get("n_ind", 2)),
            "batch_size": int(source.get("batch_size", 4000)),
            "n_epochs": int(source.get("max_steps", 200)),
            "early_stop": int(source.get("early_stopping_rounds", 20)),
            "seed": int(source.get("seed", 42)),
            "pretrain": bool(source.get("pretrain", False)),
        }
        return QlibNativeTabNetAdapter(input_dim=feature_count, **params), params
    if kind == "linear":
        from factor_service.research.models import QlibSklearnRidgeModel

        params = {
            "loss": loss,
            "alpha": float(source.get("alpha", 3.0)),
            "fit_intercept": bool(source.get("fit_intercept", True)),
            "solver": str(source.get("solver", "auto")),
            "max_iter": int(source.get("max_iter", 1000)),
            "seed": int(source.get("seed", 42)),
            "num_threads": _effective_num_threads(source),
            "input_dim": feature_count,
        }
        return QlibSklearnRidgeModel(**params), params
    if kind == "random_forest":
        from factor_service.research.models import QlibSklearnRandomForestModel

        params = {
            "loss": loss,
            "n_estimators": int(source.get("n_estimators", 300)),
            "max_depth": int(source.get("max_depth", 0)),
            "min_samples_split": int(source.get("min_samples_split", 2)),
            "min_samples_leaf": int(source.get("min_samples_leaf", 1)),
            "max_features": source.get("max_features", "sqrt"),
            "seed": int(source.get("seed", 42)),
            "num_threads": _effective_num_threads(source),
            "input_dim": feature_count,
        }
        return QlibSklearnRandomForestModel(**params), params
    raise ValueError(f"不支持的模型: {kind}")


def _prepare_model_for_serialization(kind: str, model: Any) -> None:
    """Make a remotely trained model loadable by the CPU inference service."""
    if kind == "lightgbm":
        booster = getattr(model, "model", None)
        if booster is not None and hasattr(booster, "free_dataset"):
            # keep_training_booster=True retains native train/valid matrices.
            # Only tree state is needed for inference and immutable artifacts.
            booster.free_dataset()
        return
    if kind != "xgboost":
        return
    booster = getattr(model, "model", None)
    if booster is not None and hasattr(booster, "set_param"):
        booster.set_param({"device": "cpu"})


def _dataset_for_model(
    handler: Any,
    segments: dict[str, tuple[str, str]],
    model_kind: str,
    params: dict[str, Any],
    DatasetH: Any,
) -> Any:
    if model_kind not in SEQUENCE_MODEL_KINDS:
        return DatasetH(handler=handler, segments=segments)
    from qlib.data.dataset import TSDatasetH

    return TSDatasetH(
        handler=handler,
        segments=segments,
        step_len=int(params.get("lookback_window", 60)),
    )


def _predict_dataset(
    model: Any,
    model_kind: str,
    dataset: Any,
    segment: str,
    *,
    classification: bool,
) -> pd.Series:
    if model_kind == "catboost" and classification:
        from qlib.data.dataset import DataHandlerLP

        features = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I,
        )
        probabilities = _catboost_positive_probability(
            model.model, features.values,
        )
        return pd.Series(probabilities, index=features.index)
    return model.predict(dataset, segment=segment)


def _predict_training_dataset(
    model: Any,
    model_kind: str,
    dataset: Any,
    segment: str,
    *,
    classification: bool,
) -> pd.Series:
    """Use a deterministic sample only for expensive sequence train diagnostics."""
    try:
        sample_rows = max(
            0, int(os.environ.get("ALPHA_TRAIN_METRIC_SAMPLE_ROWS") or 0),
        )
    except ValueError:
        sample_rows = 0
    if (
        model_kind in SEQUENCE_MODEL_KINDS
        and sample_rows > 0
        and hasattr(model, "predict_sampled")
    ):
        return model.predict_sampled(dataset, segment, max_rows=sample_rows)
    return _predict_dataset(
        model, model_kind, dataset, segment, classification=classification,
    )


def _catboost_positive_probability(predictor: Any, values: Any) -> np.ndarray:
    """Return class-one probabilities for sklearn and Qlib CatBoost wrappers."""
    if hasattr(predictor, "predict_proba"):
        probabilities = np.asarray(predictor.predict_proba(values), dtype=float)
    else:
        # Qlib's CatBoostModel stores catboost.CatBoost rather than
        # CatBoostClassifier, so predict_proba is not available even when the
        # fitted loss is Logloss. CatBoost exposes the same probabilities via
        # predict(..., prediction_type="Probability").
        probabilities = np.asarray(
            predictor.predict(values, prediction_type="Probability"),
            dtype=float,
        )
    if probabilities.ndim == 1:
        return probabilities.reshape(-1)
    if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
        return probabilities[:, 1]
    if probabilities.ndim == 2 and probabilities.shape[1] == 1:
        return probabilities[:, 0]
    raise ValueError("CatBoost分类预测未返回有效概率")


def predict_feature_frame(model: Any, model_kind: str, features: pd.DataFrame) -> np.ndarray:
    """Predict an already preprocessed feature frame from a serialized Qlib model."""
    if model_kind == "xgboost":
        import xgboost as xgb

        return np.asarray(model.model.predict(xgb.DMatrix(features.values)), dtype=float)
    if model_kind == "mlp":
        return np.asarray(model.predict_frame(features), dtype=float).reshape(-1)
    if model_kind in SEQUENCE_MODEL_KINDS:
        raise ValueError("时序模型推理必须通过TSDatasetH提供按股票组织的历史窗口")
    predictor = getattr(model, "model", None)
    if predictor is None or not hasattr(predictor, "predict"):
        raise ValueError(f"{model_kind}模型产物不包含可用预测器")
    values = features.values if model_kind in {"lightgbm", "catboost"} else features
    if (
        model_kind == "catboost"
        and getattr(model, "_alphablocks_loss", "mse") == "binary"
    ):
        return _catboost_positive_probability(predictor, values)
    if getattr(model, "classification", False) and hasattr(predictor, "predict_proba"):
        probabilities = np.asarray(predictor.predict_proba(values), dtype=float)
        return probabilities[:, 1]
    return np.asarray(predictor.predict(values), dtype=float).reshape(-1)


def _model_package_version(kind: str) -> dict[str, str]:
    package = {
        "stacking": "sklearn",
        "lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost",
        "random_forest": "sklearn", "linear": "sklearn",
        "mlp": "torch", "gru": "torch", "lstm": "torch", "alstm": "torch",
        "transformer": "torch", "tabnet": "torch", "tcn": "torch",
        "nativetft": "torch", "transformer_lstm": "torch",
    }[kind]
    module = __import__(package)
    return {package: str(getattr(module, "__version__", "unknown"))}


def _prediction_frame(prediction: pd.Series, prepared: PreparedDataset, job: dict[str, Any]) -> pd.DataFrame:
    frame = prediction.rename("raw_prediction").reset_index()
    frame.rename(columns={"datetime": "trade_date", "instrument": "entity_code"}, inplace=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    grouped = frame.groupby("trade_date")["raw_prediction"]
    # rank_value=1始终代表当日预测最高的实体，便于页面和TopN语义一致。
    frame["rank_value"] = grouped.rank(method="first", ascending=False).astype(int)
    frame["percentile"] = grouped.rank(method="average", pct=True)
    dataset_spec = dict(
        job.get("dataset_spec")
        or (job.get("config_json") or {}).get("dataset")
        or {}
    )
    prediction_scope = str(
        dataset_spec.get("prediction_scope") or "stock"
    ).strip().lower()
    if prediction_scope == "industry":
        # 行业截面实体较少，使用完整区间映射，保证首尾为+1/-1，
        # 而不是普通pct rank在N个样本时只能覆盖(-1, 1]。
        counts = grouped.transform("size")
        frame["score"] = np.where(
            counts > 1,
            1.0 - 2.0 * (frame["rank_value"] - 1.0) / (counts - 1.0),
            0.0,
        )
    else:
        frame["score"] = (2.0 * frame["percentile"] - 1.0).clip(-1.0, 1.0)
    trade_dates = pd.to_datetime(frame["trade_date"])
    if trade_dates.dt.tz is None:
        signal_dates = trade_dates.dt.tz_localize("Asia/Shanghai")
    else:
        signal_dates = trade_dates.dt.tz_convert("Asia/Shanghai")
    # ClickHouse DateTime('Asia/Shanghai') must receive timezone-aware values.
    # Passing naive pandas timestamps makes clickhouse-connect interpret them as
    # UTC, shifting both PIT audit fields forward by eight hours.
    frame["feature_cutoff_at"] = signal_dates + pd.Timedelta(hours=15)
    computed = pd.Timestamp.now(tz="Asia/Shanghai")
    frame["computed_at"] = computed
    frame["source_vintage"] = f"qlib#{job['job_id']}@{computed.isoformat()}"
    return frame


def _metrics(
    prediction: pd.Series,
    qlib_frame: pd.DataFrame,
    test_segment: tuple[str, str],
    *,
    classification: bool = False,
) -> dict[str, float | int]:
    label = qlib_frame[("label", "LABEL0")]
    start, end = test_segment
    label = label.loc[pd.IndexSlice[start:end, :]]
    aligned = pd.concat([prediction.rename("prediction"), label.rename("label")], axis=1).dropna()
    daily_ic = aligned.groupby(level="datetime").apply(
        lambda group: group["prediction"].corr(group["label"], method="spearman")
        if group["prediction"].nunique() > 1 and group["label"].nunique() > 1 else np.nan,
        include_groups=False,
    ).dropna()
    errors = aligned["prediction"] - aligned["label"]
    mse = float(np.mean(np.square(errors)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    ic_mean = float(daily_ic.mean()) if not daily_ic.empty else 0.0
    ic_std = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else 0.0
    result: dict[str, float | int] = {
        "test_rows": int(len(aligned)),
        "test_days": int(aligned.index.get_level_values("datetime").nunique()),
        "rmse": rmse,
        "mse": mse,
        "l2": mse,
        "mae": mae,
        "ic": ic_mean,
        "rank_ic": ic_mean,
        "ic_ir": ic_mean / ic_std if ic_std else 0.0,
        "rank_icir": ic_mean / ic_std if ic_std else 0.0,
    }
    if classification and not aligned.empty:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            log_loss,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        labels = aligned["label"].astype(int).to_numpy()
        probabilities = np.clip(
            aligned["prediction"].astype(float).to_numpy(), 1e-7, 1.0 - 1e-7,
        )
        predicted = (probabilities >= 0.5).astype(int)
        result.update({
            "auc": (
                float(roc_auc_score(labels, probabilities))
                if np.unique(labels).size == 2 else 0.5
            ),
            "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
            "accuracy": float(accuracy_score(labels, predicted)),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "positive_rate": float(np.mean(labels)),
        })
    return result


def _feature_importance(model: Any, feature_names: list[str]) -> list[dict[str, float | str | int]]:
    if hasattr(model, "get_feature_importance"):
        raw_importance = model.get_feature_importance()
        if isinstance(raw_importance, pd.Series):
            mapped = {str(key): float(value) for key, value in raw_importance.items()}
            values = [
                mapped.get(
                    name,
                    mapped.get(
                        f"Column_{index}",
                        mapped.get(f"f{index}", 0.0),
                    ),
                )
                for index, name in enumerate(feature_names)
            ]
        else:
            values = list(raw_importance)
    elif hasattr(model, "dnn_model"):
        first_linear = next(
            (layer for layer in model.dnn_model.modules() if layer.__class__.__name__ == "Linear"), None,
        )
        if first_linear is None:
            values = [0.0] * len(feature_names)
        else:
            values = first_linear.weight.detach().abs().mean(dim=0).cpu().numpy().tolist()
    else:
        values = [0.0] * len(feature_names)
    rows = [
        {"factor": name, "importance": float(values[index]) if index < len(values) else 0.0}
        for index, name in enumerate(feature_names)
    ]
    rows.sort(key=lambda item: (-float(item["importance"]), str(item["factor"])))
    for index, item in enumerate(rows, start=1):
        item["rank"] = index
    return rows


__all__ = [
    "QlibStackingModel", "QlibTrainer", "TrainingResult",
    "_qlib_lgb_params", "predict_feature_frame",
]
