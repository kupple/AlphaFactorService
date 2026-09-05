from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from minio.error import S3Error

from factor_service.model_object_store import (
    ModelObjectStore,
    ModelObjectStoreConfig,
    ModelObjectStoreConfigurationError,
    ModelObjectStoreIntegrityError,
)


def test_upload_progress_reports_bytes_and_reuses_identical_object(tmp_path):
    source = tmp_path / 'model.tar.gz'
    source.write_bytes(b'archive-body')
    client = _MemoryMinio()
    store = ModelObjectStore(_config(), client=client)
    events = []
    params = dict(job_id='job', model_id='model', model_version=1, artifact_kind='bundle',
                  source_path=source, digest=sha256(source.read_bytes()).hexdigest(), size_bytes=source.stat().st_size,
                  progress=lambda done, total: events.append((done, total)))
    store.publish_file(**params)
    assert events[0] == (0, source.stat().st_size)
    assert events[-1] == (source.stat().st_size, source.stat().st_size)
    assert [e[0] for e in events] == sorted(e[0] for e in events)
    store.publish_file(**params)
    assert len(client.uploads) == 1


def test_upload_checkpoint_cancellation_is_not_swallowed(tmp_path):
    from factor_service.research.errors import JobCanceled
    source = tmp_path / 'model.tar.gz'
    source.write_bytes(b'archive-body')
    client = _MemoryMinio()
    store = ModelObjectStore(_config(), client=client)
    def checkpoint():
        raise JobCanceled('cancel upload')
    with pytest.raises(JobCanceled):
        store.publish_file(job_id='job', model_id='model', model_version=1, artifact_kind='bundle',
                           source_path=source, digest=sha256(source.read_bytes()).hexdigest(),
                           size_bytes=source.stat().st_size, checkpoint=checkpoint)
    assert not client.uploads


class _MemoryMinio:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], SimpleNamespace] = {}
        self.bodies: dict[tuple[str, str], bytes] = {}
        self.uploads: list[dict[str, object]] = []

    def stat_object(self, bucket: str, object_key: str, *, version_id=None):
        stored = self.objects.get((bucket, object_key))
        if stored is None:
            raise S3Error(
                None, "NoSuchKey", "missing", object_key, "request", "host",
                bucket, object_key,
            )
        return stored

    def fput_object(
        self, bucket: str, object_key: str, source: str,
        *, content_type: str, metadata: dict[str, str], progress=None,
    ):
        size = Path(source).stat().st_size
        if progress is not None:
            progress.set_meta(object_name=object_key, total_length=size)
            progress.update(size)
        self.bodies[(bucket, object_key)] = Path(source).read_bytes()
        stored = SimpleNamespace(
            size=size,
            version_id=f"version-{len(self.uploads) + 1}",
            metadata={f"X-Amz-Meta-{key}": value for key, value in metadata.items()},
        )
        self.objects[(bucket, object_key)] = stored
        self.uploads.append({
            "bucket": bucket,
            "object_key": object_key,
            "source": source,
            "content_type": content_type,
            "metadata": metadata,
        })
        return stored

    def get_object(self, bucket: str, object_key: str, *, version_id=None):
        body = self.bodies[(bucket, object_key)]

        class _Response:
            def __init__(self, value: bytes) -> None:
                from io import BytesIO

                self.source = BytesIO(value)

            def read(self, size: int) -> bytes:
                return self.source.read(size)

            def close(self) -> None:
                self.source.close()

            def release_conn(self) -> None:
                pass

        return _Response(body)


def _config() -> ModelObjectStoreConfig:
    return ModelObjectStoreConfig(
        enabled=True,
        endpoint_url="http://10.126.126.5:9000",
        bucket="alphablocks-models",
        access_key="app-user",
        secret_key="app-secret",
    )


def test_remote_training_config_import_does_not_require_minio() -> None:
    source = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'minio' or name.startswith('minio.'):
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from factor_service.research.config import Settings
assert Settings
"""

    result = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_final_model_upload_is_verified_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "qlib_experiment.tar.gz"
    body = b"final model bundle"
    source.write_bytes(body)
    digest = sha256(body).hexdigest()
    client = _MemoryMinio()
    store = ModelObjectStore(_config(), client=client)

    first = store.publish_file(
        job_id="job-1", model_id="size-model", model_version=3,
        artifact_kind="bundle", source_path=source,
        digest=digest, size_bytes=len(body),
    )
    second = store.publish_file(
        job_id="job-1", model_id="size-model", model_version=3,
        artifact_kind="bundle", source_path=source,
        digest=digest, size_bytes=len(body),
    )

    assert len(client.uploads) == 1
    assert first == {
        "provider": "s3",
        "bucket": "alphablocks-models",
        "object_key": "models/size-model/versions/3/bundle/qlib_experiment.tar.gz",
        "object_uri": (
            "s3://alphablocks-models/models/size-model/versions/3/"
            "bundle/qlib_experiment.tar.gz"
        ),
        "http_url": (
            "http://10.126.126.5:9000/alphablocks-models/models/size-model/"
            "versions/3/bundle/qlib_experiment.tar.gz"
        ),
        "version_id": "version-1",
        "sha256": digest,
        "size_bytes": len(body),
        "uploaded": True,
    }
    assert second is not None and second["uploaded"] is False
    assert client.uploads[0]["metadata"]["sha256"] == digest


def test_non_final_artifact_is_not_uploaded(tmp_path: Path) -> None:
    store = ModelObjectStore(_config(), client=_MemoryMinio())

    assert store.publish_file(
        job_id="job-1", model_id="size-model", model_version=3,
        artifact_kind="predictions", source_path=tmp_path / "missing.parquet",
        digest="0" * 64, size_bytes=0,
    ) is None


def test_upload_rejects_failed_remote_sha_verification(tmp_path: Path) -> None:
    source = tmp_path / "walk_forward_series.tar.gz"
    source.write_bytes(b"walk forward models")
    digest = sha256(source.read_bytes()).hexdigest()

    class _CorruptingMinio(_MemoryMinio):
        def fput_object(self, *args, **kwargs):
            result = super().fput_object(*args, **kwargs)
            result.metadata["X-Amz-Meta-sha256"] = "0" * 64
            return result

    store = ModelObjectStore(_config(), client=_CorruptingMinio())

    with pytest.raises(ModelObjectStoreIntegrityError, match="校验失败"):
        store.publish_file(
            job_id="job-1", model_id="size-model", model_version=3,
            artifact_kind="walk_forward_series", source_path=source,
            digest=digest, size_bytes=source.stat().st_size,
        )


def test_download_uses_registered_version_and_verifies_content(tmp_path: Path) -> None:
    source = tmp_path / "qlib_experiment.tar.gz"
    body = b"portable qlib model"
    source.write_bytes(body)
    digest = sha256(body).hexdigest()
    client = _MemoryMinio()
    store = ModelObjectStore(_config(), client=client)
    identity = store.publish_file(
        job_id="job-1", model_id="size-model", model_version=3,
        artifact_kind="bundle", source_path=source,
        digest=digest, size_bytes=len(body),
    )

    destination = tmp_path / "cache" / source.name
    downloaded = store.download_file(
        object_uri=str(identity["object_uri"]),
        version_id=str(identity["version_id"]),
        destination=destination,
        digest=digest,
        size_bytes=len(body),
    )

    assert downloaded == destination
    assert destination.read_bytes() == body


def test_download_rejects_uri_from_another_bucket(tmp_path: Path) -> None:
    store = ModelObjectStore(_config(), client=_MemoryMinio())

    with pytest.raises(ModelObjectStoreConfigurationError, match="Bucket"):
        store.download_file(
            object_uri="s3://another-bucket/models/model.bin",
            version_id="version-1",
            destination=tmp_path / "model.bin",
            digest="a" * 64,
            size_bytes=1,
        )


def test_upload_cancellation_preserves_error_and_local_source(tmp_path: Path) -> None:
    from factor_service.research.errors import JobCanceled

    source = tmp_path / "dataset.parquet"
    source.write_bytes(b"frozen dataset")
    client = _MemoryMinio()
    store = ModelObjectStore(_config(), client=client)

    def cancel():
        raise JobCanceled("canceled")

    with pytest.raises(JobCanceled):
        store.publish_file(
            job_id="job-1", model_id="datasets", model_version=0,
            artifact_kind="dataset", source_path=source,
            digest=sha256(source.read_bytes()).hexdigest(), size_bytes=source.stat().st_size,
            dataset_hash="a" * 64, checkpoint=cancel,
        )
    assert source.read_bytes() == b"frozen dataset"
    assert client.uploads == []


def test_download_cancellation_does_not_publish_partial_file(tmp_path: Path) -> None:
    from factor_service.research.errors import JobCanceled

    source = tmp_path / "dataset.parquet"
    source.write_bytes(b"frozen dataset")
    client = _MemoryMinio()
    store = ModelObjectStore(_config(), client=client)
    identity = store.publish_file(
        job_id="job-1", model_id="datasets", model_version=0,
        artifact_kind="dataset", source_path=source,
        digest=sha256(source.read_bytes()).hexdigest(), size_bytes=source.stat().st_size,
        dataset_hash="a" * 64,
    )
    destination = tmp_path / "restore" / source.name

    def cancel():
        raise JobCanceled("canceled")

    with pytest.raises(JobCanceled):
        store.download_file(
            object_uri=identity["object_uri"], version_id=identity["version_id"],
            destination=destination, digest=identity["sha256"],
            size_bytes=identity["size_bytes"], checkpoint=cancel,
        )
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []
    assert source.read_bytes() == b"frozen dataset"
