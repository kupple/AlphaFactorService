from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
import fcntl
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

from factor_service.dataset_archive_repository import DatasetArchiveRepository
from factor_service.model_artifacts import ArtifactError, ModelArtifactStore
from factor_service.model_object_store import ModelObjectStore
from factor_service.research.errors import RetryableJobError


DATASET_FILES = {
    "dataset.parquet": "dataset",
    "dataset_raw.parquet": "dataset_raw",
    "dataset_manifest.json": "dataset_manifest",
}
logger = logging.getLogger(__name__)


class DatasetCacheBusy(RetryableJobError):
    """A different snapshot still has live users; retry after they release it."""

    code = "dataset_cache_busy"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DatasetArchive:
    """MinIO owns bytes; disk retains one pinned, replaceable dataset snapshot."""

    def __init__(self, artifacts: ModelArtifactStore, objects: ModelObjectStore,
                 repository: Any = None, active_hashes: Any = None) -> None:
        self.artifacts = artifacts
        self.objects = objects
        self.repository = repository or DatasetArchiveRepository()
        self._active_hashes = active_hashes

    def active_hashes(self) -> set[str]:
        if self._active_hashes is not None:
            return self._active_hashes()
        from factor_service.model_research_repository import ModelResearchRepository
        return ModelResearchRepository().active_dataset_hashes()

    def directory(self, dataset_hash: str) -> Path:
        clean = self.artifacts._dataset_hash(dataset_hash)
        target = self.artifacts.root / "datasets" / clean
        if target.is_symlink() or target.resolve() != target:
            raise ArtifactError("数据集缓存路径不能是符号链接")
        return target

    def record(self, dataset_hash: str) -> dict | None:
        row = self.repository.get(dataset_hash)
        if row is not None:
            files = row.get("files_json") or {}
            if set(files) != set(DATASET_FILES):
                raise ArtifactError("数据库中的数据集归档不完整")
            if row.get("dataset_hash") != dataset_hash:
                raise ArtifactError("数据库中的数据集归档身份不一致")
        return row

    @contextmanager
    def _cache_lock(self) -> Iterator[None]:
        path = self.artifacts.root / ".dataset-cache-slot.lock"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield

    def _cache_directories(self) -> list[Path]:
        root = self.artifacts.root / "datasets"
        if root.is_symlink() or root.resolve() != root:
            raise ArtifactError("数据集缓存路径不能是符号链接")
        entries = sorted(root.iterdir()) if root.exists() else []
        for path in entries:
            self.artifacts._dataset_hash(path.name)
            if path.is_symlink() or not path.is_dir():
                raise ArtifactError("数据集缓存含未知路径，已保留")
        return entries

    def _eviction_files(self, dataset_hash: str) -> list[Path] | None:
        """Caller holds both usage-exclusive and publication locks."""
        directory = self.directory(dataset_hash)
        paths = list(directory.iterdir())
        if not paths:
            return []  # An abandoned reservation contains no data to recover.
        allowed = {*DATASET_FILES, ".last_used"}
        if any(p.is_symlink() or not p.is_file() or p.name not in allowed for p in paths):
            raise ArtifactError("数据集缓存包含未知文件，保留本地副本")
        record = self.record(dataset_hash)
        if record is None:
            return None
        for name, identity in record["files_json"].items():
            self.objects.verify_file(identity)
            path = directory / name
            if path.exists():
                self._verify_local(path, identity)
        return paths

    def _replace_others(self, target: str, *, protected_hashes: set[str] | frozenset[str] = frozenset()) -> dict:
        """Under cache lock, validate/lock ALL old copies before deleting ANY."""
        obsolete = [p for p in self._cache_directories() if p.name != target]
        result = {"deleted": [], "reclaimed_bytes": 0}
        with ExitStack() as locks:
            prepared = []
            for directory in obsolete:
                key = directory.name
                if key in protected_hashes:
                    raise DatasetCacheBusy("旧数据集仍被活动任务保护，暂不清理缓存")
                try:
                    locks.enter_context(self.artifacts.dataset_usage(key, exclusive=True, blocking=False))
                    locks.enter_context(self.artifacts.dataset_lock(key, blocking=False))
                except BlockingIOError as exc:
                    raise DatasetCacheBusy("另一份数据集正在生成、训练或读取，请稍后重试；本地缓存最多保留一份") from exc
                paths = self._eviction_files(key)
                if paths is None:
                    raise ArtifactError(f"旧数据集{key}尚未归档MinIO，已保留；请先完成归档再切换数据集")
                prepared.append((directory, paths))
            for directory, paths in prepared:
                result["reclaimed_bytes"] += sum(p.stat().st_size for p in paths)
                for path in paths:
                    path.unlink()
                directory.rmdir()
                result["deleted"].append(directory.name)
        return result

    @contextmanager
    def cache_slot(self, dataset_hash: str) -> Iterator[Path]:
        """Reserve the only cache slot before building/hydrating; retain on exit.

        Shared dataset pins allow concurrent users of the SAME snapshot. Another
        hash cannot enter until all current readers/builders have released theirs.
        Queued tasks alone do not pin disk; they can hydrate later from MinIO.
        """
        clean = self.artifacts._dataset_hash(dataset_hash)
        with ExitStack() as usage:
            with self._cache_lock():
                self._replace_others(clean)
                directory = self.directory(clean)
                directory.mkdir(parents=True, exist_ok=True)
                usage.enter_context(self.artifacts.dataset_usage(clean))
            yield directory

    def restore_locked(self, dataset_hash: str, *, checkpoint: Any = None) -> bool:
        """Caller holds publication lock. A broken archive never triggers rebuild."""
        record = self.record(dataset_hash)
        if record is None:
            return False
        directory = self.directory(dataset_hash)
        directory.mkdir(parents=True, exist_ok=True)
        # Manifest is installed last, so interruption cannot publish a partial snapshot.
        for name in DATASET_FILES:
            if checkpoint:
                checkpoint()
            item = record["files_json"][name]
            destination = directory / name
            if destination.is_symlink():
                raise ArtifactError("数据集文件不能是符号链接")
            if destination.is_file():
                self._verify_local(destination, item)
                continue
            self.objects.download_file(
                object_uri=item["object_uri"], version_id=item.get("version_id", ""),
                destination=destination, digest=item["sha256"], size_bytes=item["size_bytes"],
                checkpoint=checkpoint,
            )
        self.artifacts.touch_dataset(dataset_hash)
        return True

    def archive_locked(self, dataset_hash: str, *, spec: dict | None = None,
                       checkpoint: Any = None) -> dict:
        """Upload + full readback + atomic registry commit, never delete here."""
        directory = self.directory(dataset_hash)
        manifest_path = directory / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_spec_hash") != dataset_hash:
            raise ArtifactError("数据集清单与Dataset Hash不一致")
        existing = self.record(dataset_hash)
        files = {}
        for name, kind in DATASET_FILES.items():
            if checkpoint:
                checkpoint()
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ArtifactError(f"待归档数据集文件无效: {name}")
            digest, size = file_sha256(path), path.stat().st_size
            if name != "dataset_manifest.json" and (
                manifest.get("files", {}).get(name, {}).get("sha256") != digest
            ):
                raise ArtifactError(f"待归档数据集SHA256与清单不一致: {name}")
            if existing:
                item = existing["files_json"][name]
                self._verify_local(path, item)
            else:
                item = self.objects.publish_file(
                    job_id=f"dataset_{dataset_hash[:24]}", model_id="datasets",
                    model_version=0, artifact_kind=kind, source_path=path,
                    digest=digest, size_bytes=size, dataset_hash=dataset_hash,
                    checkpoint=checkpoint,
                )
                if item is None:
                    raise ArtifactError("MinIO数据集归档未启用")
                # Metadata alone is not proof of stored bytes. Stream readback before
                # committing the only durable identity or allowing any local eviction.
                self.objects.verify_file(item, content=True, checkpoint=checkpoint)
            files[name] = {k: item[k] for k in (
                "object_uri", "version_id", "sha256", "size_bytes",
            )}
        if checkpoint:
            checkpoint()
        if existing:
            return existing
        return self.repository.register(
            dataset_hash, spec=spec or {}, manifest=manifest, files=files,
        )

    @contextmanager
    def use(self, dataset_hash: str) -> Iterator[Path]:
        """Hydrate/pin for the reader's lifetime; retain this most recent snapshot."""
        with self.cache_slot(dataset_hash) as directory:
            with self.artifacts.dataset_lock(dataset_hash):
                self.restore_locked(dataset_hash)
            yield directory

    def evict(self, dataset_hash: str, *, protected_hashes: set[str] | None = None) -> int:
        protected = self.active_hashes() if protected_hashes is None else protected_hashes
        if dataset_hash in protected:
            return 0
        try:
            with self._cache_lock(), self.artifacts.dataset_usage(dataset_hash, exclusive=True, blocking=False):
                with self.artifacts.dataset_lock(dataset_hash, blocking=False):
                    directory = self.directory(dataset_hash)
                    if not directory.is_dir():
                        return 0
                    paths = self._eviction_files(dataset_hash)
                    if paths is None:
                        return 0  # Never evict an unarchived / failed-upload dataset.
                    size = sum(p.stat().st_size for p in paths)
                    for path in paths:
                        path.unlink()
                    directory.rmdir()
                    return size
        except BlockingIOError:
            return 0

    def try_evict(self, dataset_hash: str) -> int:
        try:
            return self.evict(dataset_hash)
        except Exception:
            logger.warning("数据集临时副本清理失败，已保留文件: %s", dataset_hash, exc_info=True)
            return 0

    def cleanup(self, *, protected_hashes: set[str] | None = None) -> dict:
        protected = self.active_hashes() if protected_hashes is None else protected_hashes
        result = {"scanned": 0, "deleted": [], "reclaimed_bytes": 0, "errors": [], "capacity": 1, "retained": ""}
        try:
            with self._cache_lock():
                directories = self._cache_directories()
                result["scanned"] = len(directories)
                if directories:
                    latest = max(directories, key=lambda p: (p / ".last_used").stat().st_mtime
                                 if (p / ".last_used").is_file() else p.stat().st_mtime)
                    result["retained"] = latest.name
                    result.update(self._replace_others(latest.name, protected_hashes=protected))
        except Exception as exc:
            result["errors"].append({"error": str(exc)})
        return result

    @staticmethod
    def _verify_local(path: Path, item: dict) -> None:
        if path.stat().st_size != int(item["size_bytes"]) or file_sha256(path) != item["sha256"]:
            raise ArtifactError(f"本地数据集与已登记的MinIO归档不同: {path.name}")


def archive_for_settings(settings: Any, *, repository: Any = None,
                         active_hashes: Any = None) -> DatasetArchive | None:
    config = getattr(settings, "model_object_store", None)
    if config is None or not config.enabled:
        return None  # Offline snapshot-only remote runners never receive credentials.
    return DatasetArchive(
        ModelArtifactStore(settings.model_artifacts_root), ModelObjectStore(config),
        repository=repository, active_hashes=active_hashes,
    )


@contextmanager
def dataset_files(dataset_hash: str, artifact_root: str | Path) -> Iterator[Path]:
    """Application read boundary; unrelated/offline roots keep existing behavior."""
    from factor_service.research.config import load_settings
    settings = load_settings()
    root = Path(artifact_root).resolve()
    archive = archive_for_settings(settings) if root == settings.model_artifacts_root.resolve() else None
    context = archive.use(dataset_hash) if archive else nullcontext(root / "datasets" / dataset_hash)
    with context as directory:
        yield directory


def archive_existing_job_dataset(job_id: str, *, evict: bool = False,
                                 verify_restore: bool = False) -> dict:
    """Explicit maintenance operation: archive existing bytes, never rebuild/train."""
    from factor_service.model_research_repository import ModelResearchRepository
    from factor_service.research.config import load_settings
    settings = load_settings()
    archive = archive_for_settings(settings)
    if archive is None:
        raise ArtifactError("请先启用MinIO存储")
    job = ModelResearchRepository().get_job(job_id)
    dataset_hash = str(job['dataset_hash'])
    # Maintenance must be able to archive legacy unarchived copies one by one.
    # A shared pin protects existing files without admitting an additional copy;
    # only hydration of missing files requires the single-slot reservation.
    with archive.artifacts.dataset_usage(dataset_hash):
        existing = (archive.directory(dataset_hash) / 'dataset_manifest.json').is_file()
        with nullcontext() if existing else archive.cache_slot(dataset_hash):
            with archive.artifacts.dataset_lock(dataset_hash):
                if not existing and not archive.restore_locked(dataset_hash):
                    raise ArtifactError("本地和MinIO都没有该数据集；不会自动重新计算")
                record = archive.archive_locked(dataset_hash, spec=dict(job.get('dataset_spec') or {}))
    result = {'job_id': job_id, 'dataset_hash': dataset_hash, 'files': record['files_json']}
    if evict:
        result['reclaimed_bytes'] = archive.evict(dataset_hash)
    if verify_restore:
        if archive.directory(dataset_hash).exists():
            raise ArtifactError("下载验证要求先成功清理本地副本；数据集可能仍在使用")
        with archive.use(dataset_hash):
            result['restored_and_verified'] = True
        result['temporary_files_removed'] = not archive.directory(dataset_hash).exists()
        result['cache_retained'] = archive.directory(dataset_hash).exists()
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='归档既有训练数据集，不重新生成因子或训练模型')
    parser.add_argument('job_id')
    parser.add_argument('--evict', action='store_true')
    parser.add_argument('--verify-restore', action='store_true')
    args = parser.parse_args()
    print(json.dumps(archive_existing_job_dataset(
        args.job_id, evict=args.evict, verify_restore=args.verify_restore,
    ), ensure_ascii=False), flush=True)
