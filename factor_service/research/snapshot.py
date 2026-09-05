from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable

import pandas as pd

from factor_service.model_artifacts import ModelArtifactStore
from factor_service.research.dataset import DatasetBuilder, PreparedDataset, _frame_fingerprint
from factor_service.research.job import CancellationToken, ProgressCallback


@dataclass(frozen=True)
class DatasetSnapshot:
    prepared: PreparedDataset
    dataset_path: Path
    raw_dataset_path: Path
    manifest_path: Path
    reused: bool


class DatasetSnapshotStore:
    """Materialize once in work_root, then train only from artifact-root Parquet."""

    def __init__(self, artifact_root: str | Path, *, archive: Any = None) -> None:
        self.artifacts = ModelArtifactStore(artifact_root)
        self.archive = archive

    def get_or_create(
        self,
        job: dict[str, Any],
        work_dir: Path,
        builder: DatasetBuilder | None,
        *,
        cancellation: CancellationToken | None = None,
        progress: ProgressCallback | None = None,
        builder_factory: Callable[[], DatasetBuilder] | None = None,
    ) -> DatasetSnapshot:
        dataset_hash = str(job.get("dataset_hash") or "").strip().lower()
        canonical_dir = self.artifacts.root / "datasets" / dataset_hash
        dataset_path = canonical_dir / "dataset.parquet"
        raw_dataset_path = canonical_dir / "dataset_raw.parquet"
        manifest_path = canonical_dir / "dataset_manifest.json"
        _checkpoint(cancellation)
        _progress(progress, "checking_dataset_snapshot", 4, {"dataset_hash": dataset_hash})
        slot = self.archive.cache_slot(dataset_hash) if self.archive is not None else nullcontext()
        with slot, self.artifacts.dataset_lock(dataset_hash):
            if self.archive is not None:
                self.archive.restore_locked(
                    dataset_hash, checkpoint=lambda: _checkpoint(cancellation),
                )
            if manifest_path.is_file():
                prepared = self._load(
                    dataset_hash, dataset_path, raw_dataset_path, manifest_path,
                )
                self.artifacts.touch_dataset(dataset_hash)
                self._archive(job, cancellation, progress)
                _progress(progress, "dataset_ready", 56, {
                    "dataset_hash": dataset_hash,
                    "row_count": len(prepared.frame),
                    "feature_count": len(prepared.feature_names),
                    "snapshot_reused": True,
                })
                return DatasetSnapshot(
                    prepared, dataset_path, raw_dataset_path, manifest_path, True,
                )

            if builder is None and builder_factory is not None:
                builder = builder_factory()
            if builder is None:
                raise ValueError("冻结数据集快照不存在，且当前执行器不能访问源数据")

            _progress(progress, "materializing_dataset", 5, {"dataset_hash": dataset_hash})
            prepared = builder.build(job, cancellation=cancellation, progress=progress)
            _checkpoint(cancellation)
            staging_dir = work_dir / "dataset_staging"
            staging_dir.mkdir(parents=True, exist_ok=True)
            staged_dataset = staging_dir / "dataset.parquet"
            staged_raw = staging_dir / "dataset_raw.parquet"
            staged_manifest = staging_dir / "dataset_manifest.json"
            prepared.frame.to_parquet(staged_dataset)
            raw_frame = prepared.raw_frame if prepared.raw_frame is not None else prepared.frame
            raw_frame.to_parquet(staged_raw)
            # Parquet canonicalizes semantically equivalent NaN payloads. Fingerprint
            # the persisted representation so a valid round trip remains verifiable.
            persisted_frame = pd.read_parquet(staged_dataset)
            persisted_raw_frame = pd.read_parquet(staged_raw)
            snapshot_manifest = {
                **prepared.manifest,
                "schema_version": "alphablocks.qlib-dataset-snapshot.v1",
                "dataset_hash": dataset_hash,
                "dataset_spec_hash": dataset_hash,
                "content_fingerprint": _frame_fingerprint(persisted_frame),
                "raw_content_fingerprint": _frame_fingerprint(persisted_raw_frame),
                "files": {
                    "dataset.parquet": {"sha256": _file_sha256(staged_dataset)},
                    "dataset_raw.parquet": {"sha256": _file_sha256(staged_raw)},
                },
            }
            staged_manifest.write_text(
                json.dumps(snapshot_manifest, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _checkpoint(cancellation)
            _progress(progress, "publishing_dataset_snapshot", 55, {"dataset_hash": dataset_hash})
            for kind, path in (
                ("dataset", staged_dataset),
                ("dataset_raw", staged_raw),
                ("dataset_manifest", staged_manifest),
            ):
                self.artifacts.publish_file(
                    job_id=str(job["job_id"]),
                    artifact_kind=kind,
                    source_path=path,
                    dataset_hash=dataset_hash,
                )
            # Reloading from canonical files is intentional: DatasetH must consume the
            # immutable artifact, never the in-memory frame used to create it.
            loaded = self._load(dataset_hash, dataset_path, raw_dataset_path, manifest_path)
            self.artifacts.touch_dataset(dataset_hash)
            try:
                shutil.rmtree(staging_dir)
            except OSError:
                pass
            # The canonical copy is verified above. Keep it (not a second staging
            # copy) if MinIO upload or PostgreSQL registration subsequently fails.
            self._archive(job, cancellation, progress)
            return DatasetSnapshot(loaded, dataset_path, raw_dataset_path, manifest_path, False)

    def _archive(self, job: dict, cancellation: Any, progress: Any) -> None:
        if self.archive is None:
            return
        dataset_hash = str(job["dataset_hash"])
        _progress(progress, "archiving_dataset", 55, {"dataset_hash": dataset_hash})
        self.archive.archive_locked(
            dataset_hash,
            spec=dict(job.get("dataset_spec") or (job.get("config_json") or {}).get("dataset") or {}),
            checkpoint=lambda: _checkpoint(cancellation),
        )
        _progress(progress, "dataset_archived", 56, {"dataset_hash": dataset_hash})

    @staticmethod
    def _load(
        dataset_hash: str,
        dataset_path: Path,
        raw_dataset_path: Path,
        manifest_path: Path,
    ) -> PreparedDataset:
        if not dataset_path.is_file() or not raw_dataset_path.is_file():
            raise ValueError("冻结数据集快照不完整")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("冻结数据集manifest无法读取") from exc
        if not isinstance(manifest, dict) or manifest.get("dataset_spec_hash") != dataset_hash:
            raise ValueError("冻结数据集manifest与dataset_hash不一致")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("冻结数据集manifest缺少文件摘要")
        for path in (dataset_path, raw_dataset_path):
            expected = (files.get(path.name) or {}).get("sha256")
            if expected != _file_sha256(path):
                raise ValueError(f"冻结数据集文件校验失败: {path.name}")
        frame = pd.read_parquet(dataset_path)
        raw_frame = pd.read_parquet(raw_dataset_path)
        if manifest.get("content_fingerprint") != _frame_fingerprint(frame):
            raise ValueError("冻结数据集内容指纹校验失败")
        if manifest.get("raw_content_fingerprint") != _frame_fingerprint(raw_frame):
            raise ValueError("冻结原始数据集内容指纹校验失败")
        segments = {
            str(name): (str(value[0]), str(value[1]))
            for name, value in dict(manifest.get("segments") or {}).items()
        }
        return PreparedDataset(
            frame=frame,
            raw_frame=raw_frame,
            segments=segments,
            feature_names=[str(item) for item in manifest.get("feature_names") or []],
            coverage={str(key): float(value) for key, value in dict(manifest.get("coverage") or {}).items()},
            medians={str(key): float(value) for key, value in dict(manifest.get("medians") or {}).items()},
            manifest=manifest,
        )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(cancellation: CancellationToken | None) -> None:
    if cancellation is not None:
        cancellation.checkpoint()


def _progress(
    callback: ProgressCallback | None, stage: str, percent: int, details: dict[str, Any],
) -> None:
    if callback is not None:
        callback(stage, percent, details)


def prune_stale_dataset_staging(
    work_root: str | Path,
    *,
    retention_seconds: float = 24 * 60 * 60,
    now: float | None = None,
) -> dict[str, object]:
    """Remove interrupted dataset staging copies after their recovery window."""

    retention = float(retention_seconds)
    if retention <= 0:
        raise ValueError("训练数据暂存保留时间必须大于0")
    root = Path(work_root).resolve()
    result: dict[str, object] = {
        "scanned": 0,
        "deleted": 0,
        "reclaimed_bytes": 0,
    }
    if not root.is_dir():
        return result
    cutoff = (time.time() if now is None else float(now)) - retention
    for directory in sorted(root.rglob("dataset_staging")):
        resolved = directory.resolve()
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or directory.name != "dataset_staging"
            or root not in resolved.parents
        ):
            continue
        result["scanned"] = int(result["scanned"]) + 1
        timestamps = [directory.stat().st_mtime]
        timestamps.extend(
            path.stat().st_mtime for path in directory.rglob("*") if path.is_file()
        )
        if max(timestamps) > cutoff:
            continue
        size_bytes = sum(
            path.stat().st_size for path in directory.rglob("*") if path.is_file()
        )
        shutil.rmtree(directory)
        result["deleted"] = int(result["deleted"]) + 1
        result["reclaimed_bytes"] = int(result["reclaimed_bytes"]) + size_bytes
    return result


__all__ = [
    "DatasetSnapshot",
    "DatasetSnapshotStore",
    "prune_stale_dataset_staging",
]
