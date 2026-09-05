from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import pickle
import tarfile
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from factor_service.research.control import ResearchControl
from factor_service.research.config import Settings
from factor_service.research.dataset import DatasetBuilder, _feature_name
from factor_service.research.errors import PermanentJobError
from factor_service.research.industry_feature import (
    append_industry_one_hot_features,
    industry_feature_names,
    normalize_industry_feature,
)
from factor_service.research.job import CancellationToken, ProgressCallback
from factor_service.research.preprocessing import (
    DATASET_PIPELINE_VERSION,
    normalize_feature_preprocessing,
    preprocess_feature_panel,
)
from factor_service.research.rolling import effective_rolling_window
from factor_service.research.model_bundle import require_window_artifact
from factor_service.research.size_rotation_feature import (
    normalize_size_rotation_feature,
    size_rotation_feature_names,
)
from factor_service.research.training_resource_settings import (
    normalize_frozen_training_data_bindings,
)
from factor_service.research.universe_source import (
    normalize_universe_source,
)
from factor_service.research.trainer import (
    QlibStackingModel,
    SEQUENCE_MODEL_KINDS,
    TrainingResult,
    predict_feature_frame,
)


class DailyInferenceRunner:
    """Load an immutable training bundle and score one historical trading day."""

    def __init__(self, settings: Settings, control: ResearchControl) -> None:
        self.settings = settings
        self.control = control
        self.dataset_builder = DatasetBuilder(settings)

    def run(
        self,
        job: dict[str, Any],
        work_dir: Path,
        *,
        cancellation: CancellationToken | None = None,
        progress: ProgressCallback | None = None,
    ) -> TrainingResult:
        config = dict(job["config_json"])
        source = dict(config["source_model"])
        inference = dict(config["inference"])
        trade_date = str(inference["trade_date"])
        data_cutoff = datetime.fromisoformat(str(inference["data_cutoff"]).replace("Z", "+00:00"))
        cutoff_for_clickhouse = data_cutoff.astimezone(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
        work_dir.mkdir(parents=True, exist_ok=True)
        _checkpoint(cancellation)
        _progress(progress, "downloading_model", 10, {"artifact_id": source["artifact_id"]})
        bundle_path = self.control.download_artifact(
            str(source["artifact_id"]),
            work_dir / "source_model.tar.gz",
            str(source["artifact_sha256"]),
        )
        _checkpoint(cancellation)
        root_model, training_manifest = _load_bundle(bundle_path)
        window_bundle_path = None
        if (training_manifest.get('walk_forward') or {}).get('enabled') is True:
            window_bundle_path = _download_window_bundle(
                training_manifest, source, self.control, work_dir,
            )
        model, training_manifest, rolling_model = _load_model_for_trade_date(
            window_bundle_path,
            root_model,
            training_manifest,
            trade_date,
        )
        try:
            data_bindings = normalize_frozen_training_data_bindings(
                training_manifest.get("data_bindings"), allow_empty=True,
            )
        except ValueError as exc:
            raise PermanentJobError(str(exc)) from exc
        model_kind = str(training_manifest.get("model_kind") or "lightgbm")
        dataset_spec = dict(job.get("dataset_spec") or config.get("dataset") or {})
        try:
            training_universe_source = normalize_universe_source(
                training_manifest.get("universe_source"), allow_empty=True,
            )
            requested_universe_source = normalize_universe_source(
                dataset_spec.get("universe_source"), allow_empty=True,
            )
        except ValueError as exc:
            raise PermanentJobError(str(exc)) from exc
        if requested_universe_source != training_universe_source:
            raise PermanentJobError(
                "每日推理股票池与训练模型的冻结成员来源不一致"
            )
        requested_bindings = normalize_frozen_training_data_bindings(
            dataset_spec.get("data_bindings"), allow_empty=True,
        )
        if (
            requested_bindings.get("settings_revision", 0) > 0
            and requested_bindings != data_bindings
        ):
            raise PermanentJobError(
                "每日推理数据集与训练模型的数据能力绑定版本不一致"
            )
        research_target = str(
            dataset_spec.get("research_target") or "stock_selection"
        ).strip().lower()
        prediction_scope = str(
            dataset_spec.get("prediction_scope")
            or ("industry" if research_target == "industry_rotation"
                else "stock")
        ).strip().lower()
        if research_target not in {
            "stock_selection", "industry_rotation",
        }:
            raise PermanentJobError(f"训练目标{research_target}尚不支持每日推理")
        expected_names = list(training_manifest.get("feature_names") or [])
        medians = dict(training_manifest.get("medians") or {})
        preprocessing = normalize_feature_preprocessing(
            training_manifest.get("preprocessing"), default_enabled=False,
        )
        preprocessing_excluded_features = [
            str(name)
            for name in training_manifest.get(
                "preprocessing_excluded_features",
            ) or []
        ]
        unknown_exclusions = sorted(
            set(preprocessing_excluded_features) - set(expected_names)
        )
        if unknown_exclusions:
            raise PermanentJobError(
                "模型产物包含未知的非缩放特征: " + ", ".join(unknown_exclusions)
            )
        if dataset_spec.get("preprocessing") is not None:
            requested_preprocessing = normalize_feature_preprocessing(
                dataset_spec.get("preprocessing"), default_enabled=False,
            )
            if requested_preprocessing != preprocessing:
                raise PermanentJobError("每日推理数据集与训练模型的特征预处理口径不一致")
        industry_feature = normalize_industry_feature(
            training_manifest.get("industry_feature"), default_enabled=False,
        )
        if dataset_spec.get("industry_feature") is not None:
            requested_industry_feature = normalize_industry_feature(
                dataset_spec.get("industry_feature"), default_enabled=False,
            )
            if requested_industry_feature != industry_feature:
                raise PermanentJobError("每日推理数据集与训练模型的行业特征口径不一致")
        size_rotation_feature = normalize_size_rotation_feature(
            training_manifest.get("size_rotation_feature"),
            default_enabled=False,
        )
        if dataset_spec.get("size_rotation_feature") is not None:
            requested_size_rotation_feature = normalize_size_rotation_feature(
                dataset_spec.get("size_rotation_feature"),
                default_enabled=False,
            )
            if requested_size_rotation_feature != size_rotation_feature:
                raise PermanentJobError(
                    "每日推理数据集与训练模型的大小盘轮动特征口径不一致"
                )
        factors = list(job["dataset_spec"]["factors"])
        factor_feature_names = [_feature_name(item) for item in factors]
        expected_size_rotation_names = size_rotation_feature_names(
            size_rotation_feature,
        )
        expected_industry_names = industry_feature_names(industry_feature)
        actual_names = [
            *factor_feature_names,
            *expected_size_rotation_names,
            *expected_industry_names,
        ]
        if actual_names != expected_names or any(name not in medians for name in expected_names):
            raise PermanentJobError("模型产物中的特征顺序或训练中位数与冻结因子不一致")

        lookback_window = 1
        feature_date_start = trade_date
        universe_id = str(dataset_spec.get("universe_id") or "csi500")
        index_code = str(dataset_spec.get("index_code") or "000905.SH")
        sample_filters = dataset_spec.get("sample_filters")
        universe_field_filters = dataset_spec.get("universe_field_filters")
        universe_source = training_universe_source
        universe_label = {
            "csi300": "沪深300", "csi500": "中证500", "csi800": "中证800",
            "csi1000": "中证1000", "all_a": "全A",
        }.get(universe_id, universe_id)
        if model_kind in SEQUENCE_MODEL_KINDS or model_kind == "stacking":
            model_params = dict(training_manifest.get("model_params") or {})
            if model_kind == "stacking":
                lookback_window = max([
                    int(dict(item.get("params") or {}).get("lookback_window") or 1)
                    for item in model_params.get("base_models") or []
                    if str(item.get("kind") or "") in SEQUENCE_MODEL_KINDS
                ] or [1])
            else:
                lookback_window = int(model_params.get("lookback_window") or 60)
        if lookback_window > 1:
            try:
                sequence_dates = self.dataset_builder.trading_dates_ending_at(
                    trade_date, lookback_window,
                    index_code=index_code, universe_id=universe_id,
                    data_bindings=data_bindings,
                )
            except ValueError as exc:
                raise PermanentJobError(str(exc)) from exc
            feature_date_start = sequence_dates[0]

        _progress(progress, "building_inference_features", 35, {"trade_date": trade_date})
        target_membership = self.dataset_builder._membership(
            trade_date, trade_date, universe_id=universe_id, index_code=index_code,
            sample_filters=sample_filters,
            universe_field_filters=universe_field_filters,
            data_bindings=data_bindings,
            universe_source=universe_source,
            data_cutoff=data_cutoff.astimezone(timezone.utc).isoformat(),
        )
        if target_membership.empty:
            raise PermanentJobError(f"{trade_date}不是{universe_label}可推理交易日")
        membership = self.dataset_builder._membership(
            feature_date_start, trade_date,
            universe_id=universe_id, index_code=index_code,
            sample_filters=sample_filters,
            universe_field_filters=universe_field_filters,
            data_bindings=data_bindings,
            universe_source=universe_source,
            data_cutoff=data_cutoff.astimezone(timezone.utc).isoformat(),
        )
        features = membership[["trade_date", "instrument"]].drop_duplicates()
        coverages: dict[str, float] = {}
        expected_count = max(1, len(features))
        deterministic_factor_quantiles = (
            str(dataset_spec.get("pipeline_version") or "")
            == DATASET_PIPELINE_VERSION
        )
        for index, (factor, feature_name) in enumerate(
            zip(factors, factor_feature_names), start=1,
        ):
            _checkpoint(cancellation)
            _progress(progress, "loading_inference_factors", 35 + int(25 * (index - 1) / len(factors)), {
                "factor_id": factor["factor_id"],
                "factor_index": index,
                "factor_count": len(factors),
            })
            values = self.dataset_builder._factor_values(
                factor, cutoff_for_clickhouse, feature_date_start, trade_date,
                deterministic_quantiles=deterministic_factor_quantiles,
            ).rename(columns={"value": feature_name})
            eligible_values = values.merge(
                features[["trade_date", "instrument"]],
                on=["trade_date", "instrument"], how="inner",
            )
            coverages[str(factor["factor_id"])] = (
                eligible_values[["trade_date", "instrument"]].drop_duplicates().shape[0] / expected_count
            )
            features = features.merge(
                eligible_values[["trade_date", "instrument", feature_name]],
                on=["trade_date", "instrument"], how="left",
            )
        if size_rotation_feature["enabled"]:
            if research_target != "stock_selection":
                raise PermanentJobError("大小盘轮动特征仅支持个股选股每日推理")
            try:
                features, size_rotation_details = (
                    self.dataset_builder._size_rotation_features(
                        features,
                        date_start=feature_date_start,
                        date_end=trade_date,
                        index_code=index_code,
                        universe_id=universe_id,
                        size_rotation_feature=size_rotation_feature,
                        data_bindings=data_bindings,
                        data_cutoff=data_cutoff.astimezone(
                            timezone.utc,
                        ).isoformat(),
                    )
                )
            except ValueError as exc:
                raise PermanentJobError(str(exc)) from exc
            actual_size_names = list(
                size_rotation_details.get("feature_names") or []
            )
            if actual_size_names != expected_size_rotation_names:
                raise PermanentJobError(
                    "每日推理大小盘轮动特征顺序与训练模型不一致"
                )
            coverages.update({
                name: float(value)
                for name, value in dict(
                    size_rotation_details.get("coverage") or {}
                ).items()
            })
        minimum = float(job["dataset_spec"].get("minimum_factor_coverage") or 0.8)
        low = [name for name, coverage in coverages.items() if coverage < minimum]
        if low:
            raise PermanentJobError("每日推理因子覆盖率低于阈值: " + ", ".join(low))
        industry_feature_details: dict[str, Any] = {
            "feature_names": [], "mapped_coverage": None,
        }
        if industry_feature["enabled"]:
            if research_target != "stock_selection":
                raise PermanentJobError("行业编码特征仅支持个股选股每日推理")
            try:
                industry_membership = self.dataset_builder._industry_membership(
                    features[["trade_date", "instrument"]],
                    feature_date_start,
                    trade_date,
                    industry_feature=industry_feature,
                    data_bindings=data_bindings,
                )
                industry_source_details = dict(
                    industry_membership.attrs.get("training_data_binding") or {}
                )
                features, industry_feature_details = (
                    append_industry_one_hot_features(
                        features, industry_membership, industry_feature,
                    )
                )
                if industry_source_details:
                    industry_feature_details["data_binding"] = (
                        industry_source_details
                    )
            except ValueError as exc:
                raise PermanentJobError(str(exc)) from exc
            if (
                list(industry_feature_details.get("feature_names") or [])
                != expected_industry_names
            ):
                raise PermanentJobError("每日推理行业One-hot特征顺序与训练模型不一致")
            mapped_coverage = float(
                industry_feature_details.get("mapped_coverage") or 0.0
            )
            if mapped_coverage < minimum:
                raise PermanentJobError(
                    f"每日推理行业映射覆盖率{mapped_coverage:.2%}低于阈值"
                )
        if research_target == "industry_rotation":
            try:
                features = self.dataset_builder.industry_features(
                    features, expected_names, feature_date_start, trade_date,
                    data_bindings=data_bindings,
                )
            except ValueError as exc:
                raise PermanentJobError(str(exc)) from exc
        try:
            features = preprocess_feature_panel(
                features,
                expected_names,
                preprocessing,
                fallback_values={
                    name: float(medians[name]) for name in expected_names
                },
                excluded_features=preprocessing_excluded_features,
            )
        except ValueError as exc:
            raise PermanentJobError(str(exc)) from exc
        if features[expected_names].isna().any().any():
            raise PermanentJobError("每日推理特征填充后仍有缺失值")

        _checkpoint(cancellation)
        _progress(progress, "inferencing", 68, {"row_count": len(features)})
        sequence_coverage: float | None = None
        try:
            if model_kind == "stacking":
                if not isinstance(model, QlibStackingModel):
                    raise ValueError("Stacking模型产物类型无效")
                predictions = _predict_stacking(
                    model,
                    features=features,
                    feature_names=expected_names,
                    trade_date=trade_date,
                )
                raw = predictions["raw_prediction"].to_numpy(dtype=float)
                target_count = (
                    int(features.loc[
                        features["trade_date"] == pd.Timestamp(trade_date),
                        "instrument",
                    ].nunique()) if research_target == "industry_rotation"
                    else max(1, target_membership["instrument"].nunique())
                )
                sequence_coverage = len(predictions) / target_count
                if sequence_coverage < minimum:
                    raise ValueError(
                        f"Stacking共同预测覆盖率{sequence_coverage:.2%}低于阈值"
                    )
            elif model_kind in SEQUENCE_MODEL_KINDS:
                from qlib.data.dataset import DataHandlerLP, TSDatasetH

                sequence_frame = features.set_index(["trade_date", "instrument"])[expected_names]
                sequence_frame.index.names = ["datetime", "instrument"]
                sequence_frame.columns = pd.MultiIndex.from_tuples(
                    [("feature", name) for name in expected_names]
                )
                inference_dataset = TSDatasetH(
                    handler=DataHandlerLP.from_df(sequence_frame),
                    segments={"infer": (trade_date, trade_date)},
                    step_len=lookback_window,
                )
                sequence_prediction = model.predict(inference_dataset, segment="infer")
                predictions = sequence_prediction.rename("raw_prediction").reset_index()
                predictions.rename(
                    columns={"datetime": "trade_date", "instrument": "entity_code"},
                    inplace=True,
                )
                target_count = (
                    int(features.loc[
                        features["trade_date"] == pd.Timestamp(trade_date),
                        "instrument",
                    ].nunique()) if research_target == "industry_rotation"
                    else max(1, target_membership["instrument"].nunique())
                )
                sequence_coverage = len(predictions) / target_count
                if sequence_coverage < minimum:
                    raise ValueError(f"时序模型完整历史窗口覆盖率{sequence_coverage:.2%}低于阈值")
                raw = predictions["raw_prediction"].to_numpy(dtype=float)
            else:
                raw = predict_feature_frame(model, model_kind, features[expected_names])
                predictions = features[["trade_date", "instrument"]].rename(
                    columns={"instrument": "entity_code"},
                )
                predictions["raw_prediction"] = raw
        except (ImportError, RuntimeError, ValueError) as exc:
            raise PermanentJobError(f"{model_kind}模型推理失败: {exc}") from exc
        expected_prediction_rows = len(predictions)
        if raw.shape[0] != expected_prediction_rows or not np.isfinite(raw).all():
            raise PermanentJobError("模型推理结果数量不一致或包含非有限值")
        predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
        grouped = predictions.groupby("trade_date")["raw_prediction"]
        predictions["rank_value"] = grouped.rank(method="first", ascending=False).astype(int)
        predictions["percentile"] = grouped.rank(method="average", pct=True)
        if prediction_scope == "industry":
            counts = grouped.transform("size")
            predictions["score"] = np.where(
                counts > 1,
                1.0 - 2.0 * (predictions["rank_value"] - 1.0) / (counts - 1.0),
                0.0,
            )
        else:
            predictions["score"] = (
                2.0 * predictions["percentile"] - 1.0
            ).clip(-1.0, 1.0)
        predictions["feature_cutoff_at"] = pd.Timestamp(inference["feature_cutoff_at"])
        computed_at = pd.Timestamp.now(tz="Asia/Shanghai")
        predictions["computed_at"] = computed_at
        predictions["source_vintage"] = f"qlib-daily#{job['job_id']}@{computed_at.isoformat()}"
        predictions_path = work_dir / "predictions.parquet"
        predictions.to_parquet(predictions_path, index=False)
        future_function_guards = [
            "frozen factor definitions computed on demand from source data",
            "source rows limited to signal date and available by market close",
            "inference data_cutoff >= signal date close",
            "historical index membership",
            "causal per-instrument history ending at signal date",
        ]
        if rolling_model is not None:
            future_function_guards.append(
                "exact-date immutable rolling window model selected before inference"
            )
        if preprocessing["enabled"]:
            future_function_guards.append(
                "training-identical same-date cross-sectional median, 1/99 winsorization and z-score"
            )
        else:
            future_function_guards.append("training-fitted medians only")
        if research_target == "industry_rotation":
            future_function_guards.append(
                "exact-date SW2021 industry snapshots no earlier than 2021-12-13"
            )
        if industry_feature["enabled"]:
            future_function_guards.extend([
                "training-identical exact-date SW2021 L1 stock industry mapping",
                "training-identical frozen one-hot vocabulary and unknown bucket",
            ])
        manifest = {
            "schema_version": "alphablocks.qlib-inference.v1",
            "job_id": job["job_id"],
            "model_id": job["model_id"],
            "model_version": int(config["planned_model_version"]),
            "model_kind": model_kind,
            "research_target": research_target,
            "prediction_scope": prediction_scope,
            "dataset_hash": job["dataset_hash"],
            "training_job_id": source["training_job_id"],
            "rolling_model": rolling_model,
            "trade_date": trade_date,
            "feature_date_start": feature_date_start,
            "lookback_window": lookback_window,
            "data_cutoff": inference["data_cutoff"],
            "feature_cutoff_at": inference["feature_cutoff_at"],
            "feature_names": expected_names,
            "coverage": coverages,
            "sequence_coverage": sequence_coverage,
            "medians_source": "training_manifest",
            "preprocessing": preprocessing,
            "preprocessing_stage": str(
                training_manifest.get("preprocessing_stage")
                or "training_universe_after_factor_score"
            ),
            "preprocessing_excluded_features": preprocessing_excluded_features,
            "industry_feature": industry_feature,
            "industry_feature_details": industry_feature_details,
            "data_bindings": data_bindings,
            "row_count": len(predictions),
            "future_function_guards": future_function_guards,
            "created_at": computed_at.isoformat(),
        }
        manifest_path = work_dir / "inference_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8",
        )
        result = {
            "metrics": {},
            "feature_importance": [],
            "predictions": {
                "row_count": len(predictions),
                "date_start": trade_date,
                "date_end": trade_date,
                "inference_run_id": str(job["job_id"]),
                "model_version": int(config["planned_model_version"]),
            },
            "manifest": manifest,
        }
        _progress(progress, "packaged", 88, {"prediction_rows": len(predictions)})
        return TrainingResult(
            result=result,
            artifacts=[("inference_predictions", predictions_path), ("inference_manifest", manifest_path)],
            predictions_path=predictions_path,
        )


def _predict_stacking(
    model: QlibStackingModel,
    *,
    features: pd.DataFrame,
    feature_names: list[str],
    trade_date: str,
) -> pd.DataFrame:
    from qlib.data.dataset import DataHandlerLP, TSDatasetH

    target_date = pd.Timestamp(trade_date)
    date_values = pd.to_datetime(features["trade_date"])
    target = features.loc[date_values == target_date].copy()
    if target.empty:
        raise ValueError(f"{trade_date}没有可用于Stacking推理的目标截面")
    target_index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(target["trade_date"]),
            target["instrument"].astype(str),
        ],
        names=["datetime", "instrument"],
    )
    base_predictions: dict[str, pd.Series] = {}
    sequence_frame: pd.DataFrame | None = None
    for item in model.base_models:
        kind = str(item.get("kind") or "").strip().lower()
        base_model = item.get("model")
        params = dict(item.get("params") or {})
        if kind in SEQUENCE_MODEL_KINDS:
            if sequence_frame is None:
                sequence_frame = features.set_index(
                    ["trade_date", "instrument"],
                )[feature_names]
                sequence_frame.index.names = ["datetime", "instrument"]
                sequence_frame.columns = pd.MultiIndex.from_tuples(
                    [("feature", name) for name in feature_names]
                )
            dataset = TSDatasetH(
                handler=DataHandlerLP.from_df(sequence_frame),
                segments={"infer": (trade_date, trade_date)},
                step_len=int(params.get("lookback_window") or 60),
            )
            base_predictions[kind] = base_model.predict(
                dataset, segment="infer",
            ).rename(kind)
        else:
            values = predict_feature_frame(
                base_model, kind, target[feature_names],
            )
            base_predictions[kind] = pd.Series(
                np.asarray(values, dtype=float).reshape(-1),
                index=target_index,
                name=kind,
            )
    aligned = pd.concat(base_predictions, axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("Stacking基模型没有共同有效的当日预测")
    raw = model.combine([
        aligned[item["kind"]].to_numpy(dtype=float) for item in model.base_models
    ])
    result = pd.Series(raw, index=aligned.index, name="raw_prediction").reset_index()
    result.rename(columns={"instrument": "entity_code"}, inplace=True)
    return result


def _load_bundle(path: Path) -> tuple[Any, dict[str, Any]]:
    payloads: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as archive:
        if any(member.name == 'walk_forward' or member.name.startswith('walk_forward/')
               for member in archive.getmembers()):
            raise PermanentJobError('主模型包不允许内嵌滚动窗口，必须使用独立序列包')
        for required in ("model.pkl", "manifest.json"):
            member = archive.getmember(required)
            if not member.isfile() or member.size <= 0 or member.size > 256 * 1024 * 1024:
                raise PermanentJobError(f"模型产物中的{required}无效")
            source = archive.extractfile(member)
            if source is None:
                raise PermanentJobError(f"无法读取模型产物中的{required}")
            payloads[required] = source.read()
    try:
        manifest = json.loads(payloads["manifest.json"].decode("utf-8"))
    except Exception as exc:
        raise PermanentJobError(f"模型产物解析失败: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PermanentJobError("训练manifest必须是JSON对象")
    if (manifest.get('walk_forward') or {}).get('enabled') is True:
        try:
            require_window_artifact(manifest)
        except ValueError as exc:
            raise PermanentJobError(str(exc)) from exc
    try:
        model = pickle.loads(payloads['model.pkl'])
    except Exception as exc:
        raise PermanentJobError(f'模型产物解析失败: {exc}') from exc
    return model, manifest


def _download_window_bundle(manifest, source, control, work_dir):
    try:
        reference = require_window_artifact(manifest)
    except ValueError as exc:
        raise PermanentJobError(str(exc)) from exc
    frozen = source.get('walk_forward_artifact') or {}
    if (reference.get('artifact_kind') != 'walk_forward_series'
            or not frozen.get('artifact_id')
            or frozen.get('sha256') != reference.get('sha256')
            or int(frozen.get('size_bytes') or 0) != int(reference.get('size_bytes') or 0)):
        raise PermanentJobError('滚动序列产物与冻结模型清单不一致')
    path = control.download_artifact(frozen['artifact_id'], work_dir / 'source_windows.tar.gz', frozen['sha256'])
    if path.stat().st_size != int(reference['size_bytes']):
        raise PermanentJobError('滚动序列产物大小与清单不一致')
    return path


def _load_model_for_trade_date(
    series_path: Path | None,
    root_model: Any,
    root_manifest: dict[str, Any],
    trade_date: str,
) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    """Load the immutable window model whose OOS interval contains trade_date."""

    try:
        selected = effective_rolling_window(
            root_manifest.get("walk_forward"), trade_date,
        )
    except ValueError as exc:
        raise PermanentJobError(str(exc)) from exc
    if selected is None:
        return root_model, root_manifest, None
    try:
        require_window_artifact(root_manifest)
    except ValueError as exc:
        raise PermanentJobError(str(exc)) from exc
    if series_path is None:
        raise PermanentJobError('滚动模型缺少独立序列包')

    artifact = dict(selected.get("artifact") or {})
    relative_root = str(artifact.get("path") or "").strip("/")
    if not relative_root.startswith("walk_forward/window_"):
        raise PermanentJobError("滚动模型窗口产物路径无效")
    model_member = f"{relative_root}/model.pkl"
    manifest_member = f"{relative_root}/manifest.json"
    payloads: dict[str, bytes] = {}
    with tarfile.open(series_path, "r:gz") as archive:
        for member_name in (model_member, manifest_member):
            try:
                member = archive.getmember(member_name)
            except KeyError as exc:
                raise PermanentJobError(
                    f"独立序列包缺少滚动窗口产物{member_name}"
                ) from exc
            if (
                not member.isfile()
                or member.size <= 0
                or member.size > 256 * 1024 * 1024
            ):
                raise PermanentJobError(f"滚动窗口产物{member_name}无效")
            source = archive.extractfile(member)
            if source is None:
                raise PermanentJobError(f"滚动窗口产物{member_name}不可读")
            payloads[member_name] = source.read()
    try:
        window_manifest = json.loads(payloads[manifest_member].decode("utf-8"))
        model = pickle.loads(payloads[model_member])
    except Exception as exc:
        raise PermanentJobError(f"滚动窗口模型解析失败: {exc}") from exc
    if not isinstance(window_manifest, dict):
        raise PermanentJobError("滚动窗口manifest必须是JSON对象")

    expected_model_sha256 = str(
        window_manifest.get("model_sha256")
        or artifact.get("model_sha256")
        or ""
    ).lower()
    actual_model_sha256 = sha256(payloads[model_member]).hexdigest()
    if expected_model_sha256 != actual_model_sha256:
        raise PermanentJobError("滚动窗口模型SHA256与序列合同不一致")
    if (
        str(window_manifest.get("series_id") or "")
        != str(root_manifest.get("model_id") or "")
        or int(window_manifest.get("series_revision") or 0)
        != int(root_manifest.get("model_version") or 0)
    ):
        raise PermanentJobError("滚动窗口模型身份与根manifest不一致")
    if (
        str(window_manifest.get("effective_date_start") or "")[:10]
        != str(selected.get("effective_date_start") or "")[:10]
        or str(window_manifest.get("effective_date_end") or "")[:10]
        != str(selected.get("effective_date_end") or "")[:10]
    ):
        raise PermanentJobError("滚动窗口生效区间与序列合同不一致")

    merged_manifest = dict(root_manifest)
    merged_manifest["medians"] = dict(
        window_manifest.get("train_medians") or {}
    )
    merged_manifest["active_rolling_window"] = window_manifest
    routing = {
        "series_id": str(window_manifest["series_id"]),
        "series_revision": int(window_manifest["series_revision"]),
        "window": int(window_manifest["window"]),
        "effective_date_start": str(window_manifest["effective_date_start"]),
        "effective_date_end": str(window_manifest["effective_date_end"]),
        "qlib_recorder_id": str(window_manifest.get("qlib_recorder_id") or ""),
        "model_sha256": actual_model_sha256,
        "routing_policy": "exact_effective_date_interval",
    }
    return model, merged_manifest, routing


def _checkpoint(cancellation: CancellationToken | None) -> None:
    if cancellation is not None:
        cancellation.checkpoint()


def _progress(
    callback: ProgressCallback | None, stage: str, percent: int, details: dict[str, Any],
) -> None:
    if callback is not None:
        callback(stage, percent, details)


__all__ = ["DailyInferenceRunner"]
