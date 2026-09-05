from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd
from psycopg.types.json import Jsonb

from factor_service.model_research_repository import (
    ModelResearchConflict,
    ModelResearchNotFound,
    ModelResearchRepository,
    _bounded_identity,
    _canonical_json,
    _dataset_spec,
    _training_dataset_source,
)
from factor_service.research.config import load_settings
from factor_service.research.dataset import DatasetBuilder, PreparedDataset
from factor_service.research.errors import JobCanceled
from factor_service.research.job import CancellationToken
from factor_service.research.snapshot import DatasetSnapshotStore
from factor_service.research.dataset_archive import archive_for_settings


MAX_PREVIEW_ROWS = 500
PREVIEW_KIND = "dataset_preview"


class DatasetPreviewService:
    """Build the exact immutable training snapshot without starting a model run."""

    def __init__(self, repository: ModelResearchRepository) -> None:
        self.repository = repository
        self.database = repository.database

    def create(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        spec = _dataset_spec(_training_dataset_source(payload))
        preview_request = _preview_request(payload)
        idempotency_key = _bounded_identity(
            payload.get("idempotency_key"), "idempotency_key",
        )
        client_preview_id = _bounded_identity(
            payload.get("client_preview_id"), "client_preview_id",
        )
        spec_json = _canonical_json(spec)
        dataset_hash = sha256(spec_json.encode("utf-8")).hexdigest()
        request_hash = sha256(_canonical_json({
            "dataset": spec,
            "preview": preview_request,
            "client_preview_id": client_preview_id,
            "title": str(payload.get("title") or "训练数据预览")[:160],
        }).encode("utf-8")).hexdigest()
        dataset_id = f"dataset_{dataset_hash[:24]}"
        now = _utcnow()
        with self.database.connection() as conn:
            with conn.transaction():
                if idempotency_key or client_preview_id:
                    for token in sorted({
                        value for value in (
                            f"key:{idempotency_key}" if idempotency_key else "",
                            f"client:{client_preview_id}" if client_preview_id else "",
                        ) if value
                    }):
                        conn.execute(
                            "SELECT pg_advisory_xact_lock(hashtext(%s))",
                            (f"dataset-preview:{token}",),
                        )
                    existing = conn.execute(
                        """
                        SELECT job_id, config_json FROM model_jobs
                        WHERE kind = %s
                          AND (
                                (%s <> '' AND config_json -> 'preview' ->> 'idempotency_key' = %s)
                             OR (%s <> '' AND config_json -> 'preview' ->> 'client_preview_id' = %s)
                          )
                        ORDER BY requested_at DESC
                        LIMIT 1
                        """,
                        (
                            PREVIEW_KIND,
                            idempotency_key, idempotency_key,
                            client_preview_id, client_preview_id,
                        ),
                    ).fetchone()
                    if existing:
                        existing_config = dict(existing.get("config_json") or {})
                        stored_preview = dict(existing_config.get("preview") or {})
                        stored_hash = str(stored_preview.get("request_hash") or "")
                        legacy_matches = (
                            not stored_hash
                            and _canonical_json(existing_config.get("dataset") or {}) == spec_json
                            and {
                                "split": stored_preview.get("split"),
                                "rows": stored_preview.get("rows"),
                                "view": stored_preview.get("view"),
                            } == preview_request
                            and str(stored_preview.get("client_preview_id") or "")
                            in {"", client_preview_id}
                        )
                        if stored_hash != request_hash and not legacy_matches:
                            raise ModelResearchConflict(
                                "幂等键或client_preview_id已用于不同Dataset Preview请求"
                            )
                        return self.get(str(existing["job_id"]))
                conn.execute(
                    """
                    INSERT INTO model_dataset_specs(
                        dataset_id, spec_hash, name, universe_id, factor_count,
                        data_cutoff, spec_json, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (spec_hash) DO NOTHING
                    """,
                    (
                        dataset_id,
                        dataset_hash,
                        str(spec.get("name") or "训练数据预览"),
                        str(spec.get("universe_id") or ""),
                        len(spec.get("factors") or []),
                        spec["data_cutoff"],
                        Jsonb(spec),
                        now,
                    ),
                )
                dataset_row = conn.execute(
                    "SELECT dataset_id FROM model_dataset_specs WHERE spec_hash = %s",
                    (dataset_hash,),
                ).fetchone()
                dataset_id = str(dataset_row["dataset_id"])
                preview_seed = client_preview_id or idempotency_key
                preview_id = (
                    "dataset_preview_"
                    + sha256(("preview:" + preview_seed).encode("utf-8")).hexdigest()[:32]
                    if preview_seed else f"dataset_preview_{uuid4().hex}"
                )
                config = {
                    "schema_version": "alphablocks.dataset-preview.v1",
                    "dataset": spec,
                    "preview": {
                        **preview_request,
                        "idempotency_key": idempotency_key,
                        "client_preview_id": client_preview_id,
                        "request_hash": request_hash,
                    },
                }
                conn.execute(
                    """
                    INSERT INTO model_jobs(
                        job_id, dataset_id, model_id, kind, model_kind, title,
                        status, config_json, progress_json, attempt_count,
                        max_attempts, requested_at, started_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, 'none', %s,
                        'running', %s, %s, 1, 1, %s, %s, %s
                    )
                    """,
                    (
                        preview_id,
                        dataset_id,
                        preview_id,
                        PREVIEW_KIND,
                        str(payload.get("title") or "训练数据预览")[:160],
                        Jsonb(config),
                        Jsonb({"stage": "queued", "progress": 0}),
                        now,
                        now,
                        now,
                    ),
                )
                self._event(
                    conn,
                    preview_id,
                    "preview.queued",
                    stage="queued",
                    payload={
                        "dataset_hash": dataset_hash,
                        **preview_request,
                    },
                )
        return self.get(preview_id)

    def run(self, preview_id: str) -> None:
        archive = None
        usage = None
        try:
            job = self._preview_job(preview_id)
            cancellation = CancellationToken()

            def progress(stage: str, percent: int, details: dict[str, Any]) -> None:
                current = self._preview_job(preview_id)
                if current.get("cancel_requested") is True:
                    cancellation.cancel("Dataset Preview 已请求取消")
                cancellation.checkpoint()
                self._progress(preview_id, stage, percent, details)

            settings = load_settings()
            work_dir = Path(settings.work_root) / preview_id
            archive = archive_for_settings(settings)
            snapshot_store = DatasetSnapshotStore(settings.model_artifacts_root, archive=archive)
            usage = (archive.cache_slot(str(job["dataset_hash"])) if archive is not None
                     else snapshot_store.artifacts.dataset_usage(str(job["dataset_hash"])))
            usage.__enter__()
            snapshot = snapshot_store.get_or_create(
                job,
                work_dir,
                None,
                builder_factory=lambda: DatasetBuilder(settings),
                cancellation=cancellation,
                progress=progress,
            )
            request_spec = dict(
                (job.get("config_json") or {}).get("preview") or {}
            )
            sample = _dataset_sample(snapshot.prepared, request_spec)
            result = {
                "schema_version": "alphablocks.dataset-preview-result.v1",
                "dataset_hash": str(job["dataset_hash"]),
                "snapshot_reused": bool(snapshot.reused),
                "manifest": {
                    key: value
                    for key, value in dict(snapshot.prepared.manifest).items()
                    if key not in {"files"}
                },
                "sample": sample,
            }
            now = _utcnow()
            with self.database.connection() as conn:
                with conn.transaction():
                    conn.execute(
                        """
                        UPDATE model_jobs
                        SET status = 'succeeded', result_json = %s,
                            progress_json = %s, finished_at = %s, updated_at = %s
                        WHERE job_id = %s AND kind = %s
                        """,
                        (
                            Jsonb(result),
                            Jsonb({"stage": "succeeded", "progress": 100}),
                            now,
                            now,
                            preview_id,
                            PREVIEW_KIND,
                        ),
                    )
                    self._event(
                        conn,
                        preview_id,
                        "preview.succeeded",
                        stage="succeeded",
                        payload={
                            "dataset_hash": str(job["dataset_hash"]),
                            "snapshot_reused": bool(snapshot.reused),
                            "sample_rows": int(sample["row_count"]),
                        },
                    )
        except JobCanceled as exc:
            self._fail(preview_id, str(exc), canceled=True)
        except Exception as exc:
            self._fail(preview_id, str(exc))
        finally:
            if usage is not None:
                usage.__exit__(None, None, None)

    def get(self, preview_id: str) -> dict[str, Any]:
        return _preview_view(self._preview_job(preview_id))

    def sample(self, preview_id: str) -> dict[str, Any]:
        job = self._preview_job(preview_id)
        if str(job.get("status") or "") != "succeeded":
            raise ModelResearchConflict("Dataset Preview 尚未完成")
        sample = dict((job.get("result_json") or {}).get("sample") or {})
        if not sample:
            raise ModelResearchConflict("Dataset Preview 未生成抽样结果")
        return sample

    def events(self, preview_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        self._preview_job(preview_id)
        return self.repository.list_events(preview_id, after=after)

    def _preview_job(self, preview_id: str) -> dict[str, Any]:
        job = self.repository.get_job(preview_id)
        if str(job.get("kind") or "") != PREVIEW_KIND:
            raise ModelResearchNotFound("Dataset Preview 不存在")
        return job

    def _progress(
        self,
        preview_id: str,
        stage: str,
        percent: int,
        details: Mapping[str, Any],
    ) -> None:
        now = _utcnow()
        progress = {
            "stage": str(stage),
            "progress": max(0, min(int(percent), 100)),
            "details": dict(details),
        }
        with self.database.connection() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE model_jobs SET progress_json = %s, updated_at = %s
                    WHERE job_id = %s AND kind = %s AND status = 'running'
                    """,
                    (Jsonb(progress), now, preview_id, PREVIEW_KIND),
                )
                self._event(
                    conn,
                    preview_id,
                    "preview.progress",
                    stage=str(stage),
                    payload=progress,
                )

    def _fail(
        self, preview_id: str, message: str, *, canceled: bool = False,
    ) -> None:
        now = _utcnow()
        with self.database.connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    """
                    SELECT job_id, cancel_requested
                    FROM model_jobs
                    WHERE job_id = %s AND kind = %s
                    """,
                    (preview_id, PREVIEW_KIND),
                ).fetchone()
                if not row:
                    return
                was_canceled = canceled or bool(row.get("cancel_requested"))
                status = "canceled" if was_canceled else "failed"
                conn.execute(
                    """
                    UPDATE model_jobs
                    SET status = %s, error_message = %s,
                        progress_json = %s, finished_at = %s, updated_at = %s
                    WHERE job_id = %s
                    """,
                    (
                        status,
                        "" if was_canceled else str(message)[:4000],
                        Jsonb({"stage": status, "progress": 100}),
                        now,
                        now,
                        preview_id,
                    ),
                )
                self._event(
                    conn,
                    preview_id,
                    f"preview.{status}",
                    stage=status,
                    message=str(message)[:1000],
                )

    @staticmethod
    def _event(
        conn: Any,
        job_id: str,
        event_type: str,
        *,
        stage: str,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO model_job_events(
                job_id, event_type, stage, message, payload_json
            ) VALUES (%s, %s, %s, %s, %s)
            """,
            (
                job_id,
                event_type,
                stage,
                message,
                Jsonb(dict(payload or {})),
            ),
        )


def _preview_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    split = str(payload.get("split") or "train").strip().lower()
    if split == "validation":
        split = "valid"
    if split not in {"train", "valid"}:
        raise ModelResearchConflict("Dataset Preview 只开放 train 和 validation")
    view = str(payload.get("view") or "processed").strip().lower()
    if view not in {"processed", "raw"}:
        raise ModelResearchConflict("Dataset Preview view 只支持 processed 或 raw")
    try:
        rows = int(payload.get("rows") or 100)
    except (TypeError, ValueError) as exc:
        raise ModelResearchConflict("Dataset Preview rows 必须是正整数") from exc
    if not 1 <= rows <= MAX_PREVIEW_ROWS:
        raise ModelResearchConflict(
            f"Dataset Preview rows 必须在 1 至 {MAX_PREVIEW_ROWS} 之间"
        )
    return {"split": split, "rows": rows, "view": view}


def _dataset_sample(
    prepared: PreparedDataset,
    request_spec: Mapping[str, Any],
) -> dict[str, Any]:
    split = str(request_spec.get("split") or "train")
    rows = int(request_spec.get("rows") or 100)
    view = str(request_spec.get("view") or "processed")
    frame = (
        prepared.raw_frame
        if view == "raw" and prepared.raw_frame is not None
        else prepared.frame
    )
    segment = prepared.segments.get(split)
    if not segment:
        raise ModelResearchConflict(f"Dataset Preview 切分不存在: {split}")
    dates = pd.to_datetime(frame.index.get_level_values("datetime"))
    mask = (dates >= pd.Timestamp(segment[0])) & (dates <= pd.Timestamp(segment[1]))
    selected = frame.loc[mask].sort_index().head(rows)
    if selected.empty:
        raise ModelResearchConflict("Dataset Preview 指定切分没有可用样本")
    if not isinstance(selected.columns, pd.MultiIndex):
        raise ModelResearchConflict("Dataset Preview 快照列结构无效")
    features = selected["feature"].copy()
    labels = selected["label"].copy()
    return {
        "schema_version": "alphablocks.dataset-sample.v1",
        "split": "validation" if split == "valid" else split,
        "view": view,
        "row_count": len(selected),
        "X": _table(features),
        "y": _table(labels),
        "segments": dict(prepared.segments),
        "feature_names": list(prepared.feature_names),
        "coverage": dict(prepared.coverage),
        "filter_steps": list(
            dict(prepared.manifest).get("universe_filter_steps") or []
        ),
    }


def _table(frame: pd.DataFrame) -> dict[str, Any]:
    table = frame.reset_index()
    table.columns = [str(column) for column in table.columns]
    table = table.astype(object).where(pd.notna(table), None)
    rows = [
        {str(key): _json_scalar(value) for key, value in record.items()}
        for record in table.to_dict(orient="records")
    ]
    return {
        "schema_version": "alphablocks.table.v1",
        "columns": [str(column) for column in table.columns],
        "rows": rows,
        "row_count": len(rows),
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not pd.notna(value):
        return None
    return value


def _preview_view(job: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(job.get("config_json") or {})
    request_spec = dict(config.get("preview") or {})
    result = dict(job.get("result_json") or {})
    progress = dict(job.get("progress_json") or {})
    status = str(job.get("status") or "running")
    return {
        "schema_version": "alphablocks.dataset-preview.v1",
        "preview_id": str(job.get("job_id") or ""),
        "status": status,
        "stage": str(progress.get("stage") or status),
        "progress": int(progress.get("progress") or 0),
        "dataset_hash": str(job.get("dataset_hash") or ""),
        "split": "validation" if request_spec.get("split") == "valid" else request_spec.get("split"),
        "rows": int(request_spec.get("rows") or 0),
        "view": str(request_spec.get("view") or "processed"),
        "sample_available": bool(result.get("sample")),
        "snapshot_reused": result.get("snapshot_reused"),
        "error": (
            {
                "code": "dataset_preview_failed",
                "message": str(job.get("error_message") or ""),
            }
            if status == "failed"
            else None
        ),
        "created_at": job.get("requested_at"),
        "updated_at": job.get("updated_at"),
        "finished_at": job.get("finished_at"),
    }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["DatasetPreviewService", "MAX_PREVIEW_ROWS", "PREVIEW_KIND"]
