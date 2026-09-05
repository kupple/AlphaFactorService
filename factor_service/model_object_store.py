from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any
from urllib.parse import quote, urlsplit

from factor_service.research.errors import JobError, PermanentJobError, RetryableJobError


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_MISSING_OBJECT_CODES = {"NoSuchKey", "NoSuchObject", "NotFound"}
_PERMANENT_S3_CODES = {
    "AccessDenied",
    "AuthorizationHeaderMalformed",
    "InvalidAccessKeyId",
    "InvalidBucketName",
    "InvalidRequest",
    "NoSuchBucket",
    "SignatureDoesNotMatch",
}


class ModelObjectStoreConfigurationError(PermanentJobError):
    code = "model_object_store_configuration"


class ModelObjectStoreIntegrityError(PermanentJobError):
    code = "model_object_store_integrity"


class ModelObjectStoreUploadError(RetryableJobError):
    code = "model_object_store_upload"


class ModelObjectStoreDownloadError(RetryableJobError):
    code = "model_object_store_download"


@dataclass(frozen=True)
class ModelObjectStoreConfig:
    enabled: bool = False
    endpoint_url: str = ""
    bucket: str = ""
    region: str = "us-east-1"
    access_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)
    prefix: str = "models"
    artifact_kinds: tuple[str, ...] = ("bundle", "walk_forward_series")


class ModelObjectStore:
    """Mirror final model bundles to an S3-compatible versioned bucket."""

    def __init__(
        self,
        config: ModelObjectStoreConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config or ModelObjectStoreConfig()
        self._client = client
        if self.config.enabled:
            self._validate_config()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def enabled_for(self, artifact_kind: str) -> bool:
        return self.enabled and str(artifact_kind) in self.config.artifact_kinds

    def publish_file(
        self,
        *,
        job_id: str,
        model_id: str,
        model_version: int,
        artifact_kind: str,
        source_path: str | Path,
        digest: str,
        size_bytes: int,
        dataset_hash: str = "",
        checkpoint: Any = None,
        progress: Any = None,
    ) -> dict[str, object] | None:
        if not (self.enabled if dataset_hash else self.enabled_for(artifact_kind)):
            return None
        source = Path(source_path)
        if not source.is_file():
            raise ModelObjectStoreIntegrityError(f"待归档模型不存在: {source}")
        expected_size = int(size_bytes)
        if source.stat().st_size != expected_size:
            raise ModelObjectStoreIntegrityError("本地模型大小在归档前发生变化")
        expected_digest = str(digest or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise ModelObjectStoreIntegrityError("待归档模型缺少有效SHA256")

        clean_job = self._component(job_id, "job_id")
        clean_model = self._component(model_id, "model_id")
        clean_kind = self._component(artifact_kind, "artifact_kind")
        clean_file = self._component(source.name, "file_name")
        version = int(model_version)
        if version <= 0 and not dataset_hash:
            raise ModelObjectStoreConfigurationError("模型归档缺少有效model_version")
        object_key = "/".join(filter(None, (
            self.config.prefix.strip("/"), clean_model,
            "versions", str(version), clean_kind, clean_file,
        )))
        metadata = {
            "sha256": expected_digest,
            "job-id": clean_job,
            "model-id": clean_model,
            "model-version": str(version),
            "artifact-kind": clean_kind,
        }
        if dataset_hash:
            if not re.fullmatch(r"[0-9a-f]{64}", dataset_hash):
                raise ModelObjectStoreConfigurationError("数据集归档缺少有效Dataset Hash")
            if clean_kind not in {"dataset", "dataset_raw", "dataset_manifest"}:
                raise ModelObjectStoreConfigurationError("数据集归档类型无效")
            # Dataset Spec hash is not a file hash. Include both, preventing a
            # failed/concurrent rebuild from overwriting previously archived bytes.
            object_key = "/".join((
                self.config.prefix.strip("/"), "datasets", dataset_hash,
                expected_digest, clean_file,
            ))
            metadata["dataset-hash"] = dataset_hash

        try:
            current = self._stat_or_none(object_key)
            if current is not None and self._matches(
                current, digest=expected_digest, size_bytes=expected_size,
            ):
                if progress:
                    progress(expected_size, expected_size)
                return self._identity(
                    object_key, current, expected_digest, expected_size,
                    uploaded=False,
                )
            result = self.client.fput_object(
                self.config.bucket,
                object_key,
                str(source),
                content_type=_content_type(clean_file),
                metadata=metadata,
                **({"progress": _UploadCheckpoint(checkpoint, progress, expected_size)} if checkpoint or progress else {}),
            )
            stored = self.client.stat_object(self.config.bucket, object_key)
            if not self._matches(
                stored, digest=expected_digest, size_bytes=expected_size,
            ):
                raise ModelObjectStoreIntegrityError(
                    "MinIO模型上传后的大小或SHA256元数据校验失败",
                )
            version_id = str(
                getattr(result, "version_id", "")
                or getattr(stored, "version_id", "")
                or ""
            )
            if progress:
                progress(expected_size, expected_size)
            return self._identity(
                object_key, stored, expected_digest, expected_size,
                uploaded=True, version_id=version_id,
            )
        except JobError:
            raise
        except Exception as exc:
            if _is_s3_error(exc) and str(exc.code or "") in _PERMANENT_S3_CODES:
                raise ModelObjectStoreConfigurationError(
                    f"MinIO模型归档配置或权限错误: {exc.code}",
                ) from exc
            if _is_s3_error(exc):
                raise ModelObjectStoreUploadError(
                    f"MinIO模型归档失败: {exc.code or type(exc).__name__}",
                ) from exc
            raise ModelObjectStoreUploadError(
                f"MinIO模型归档连接失败: {type(exc).__name__}",
            ) from exc

    def verify_file(
        self, identity: dict[str, Any], *, content: bool = False,
        checkpoint: Any = None,
    ) -> None:
        """Verify the exact registered version; optionally read back all bytes."""
        if not self.enabled:
            raise ModelObjectStoreConfigurationError("MinIO存储未启用")
        key = self._object_key(str(identity["object_uri"]))
        version = str(identity.get("version_id") or "") or None
        digest = str(identity["sha256"])
        size = int(identity["size_bytes"])
        response = None
        try:
            stored = self.client.stat_object(self.config.bucket, key, version_id=version)
            if not self._matches(stored, digest=digest, size_bytes=size):
                raise ModelObjectStoreIntegrityError("MinIO数据集大小或SHA256元数据不一致")
            if content:
                response = self.client.get_object(self.config.bucket, key, version_id=version)
                actual = sha256()
                count = 0
                while True:
                    if checkpoint:
                        checkpoint()
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > size:
                        raise ModelObjectStoreIntegrityError("MinIO数据集大小超过登记值")
                    actual.update(chunk)
                if count != size or actual.hexdigest() != digest:
                    raise ModelObjectStoreIntegrityError("MinIO数据集内容SHA256校验失败")
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def download_file(
        self,
        *,
        object_uri: str,
        version_id: str,
        destination: str | Path,
        digest: str,
        size_bytes: int,
        checkpoint: Any = None,
    ) -> Path:
        """Download one immutable model object and verify its durable identity."""
        if not self.enabled:
            raise ModelObjectStoreConfigurationError("MinIO模型存储未启用")
        object_key = self._object_key(object_uri)
        expected_digest = str(digest or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise ModelObjectStoreIntegrityError("待下载模型缺少有效SHA256")
        expected_size = int(size_bytes)
        if expected_size < 0:
            raise ModelObjectStoreIntegrityError("待下载模型大小无效")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".download", dir=target.parent,
        )
        response: Any | None = None
        try:
            requested_version = str(version_id or "").strip() or None
            stored = self.client.stat_object(
                self.config.bucket, object_key, version_id=requested_version,
            )
            if not self._matches(
                stored, digest=expected_digest, size_bytes=expected_size,
            ):
                raise ModelObjectStoreIntegrityError(
                    "MinIO模型下载前的大小或SHA256元数据校验失败",
                )
            response = self.client.get_object(
                self.config.bucket, object_key, version_id=requested_version,
            )
            actual_digest = sha256()
            actual_size = 0
            with os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                while True:
                    if checkpoint:
                        checkpoint()
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    actual_size += len(chunk)
                    if actual_size > expected_size:
                        raise ModelObjectStoreIntegrityError(
                            "MinIO模型下载大小超过登记值",
                        )
                    actual_digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if actual_size != expected_size or actual_digest.hexdigest() != expected_digest:
                raise ModelObjectStoreIntegrityError(
                    "MinIO模型下载后的大小或SHA256校验失败",
                )
            os.replace(temporary_name, target)
            return target
        except JobError:
            raise
        except Exception as exc:
            if _is_s3_error(exc) and str(exc.code or "") in _PERMANENT_S3_CODES:
                raise ModelObjectStoreConfigurationError(
                    f"MinIO模型下载配置或权限错误: {exc.code}",
                ) from exc
            if _is_s3_error(exc) and str(exc.code or "") in _MISSING_OBJECT_CODES:
                raise ModelObjectStoreIntegrityError("MinIO模型对象或指定版本不存在") from exc
            if _is_s3_error(exc):
                raise ModelObjectStoreDownloadError(
                    f"MinIO模型下载失败: {exc.code or type(exc).__name__}",
                ) from exc
            raise ModelObjectStoreDownloadError(
                f"MinIO模型下载连接失败: {type(exc).__name__}",
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            Path(temporary_name).unlink(missing_ok=True)
            if response is not None:
                response.close()
                response.release_conn()

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from minio import Minio
            except ModuleNotFoundError as exc:
                raise ModelObjectStoreConfigurationError(
                    "MinIO模型归档已启用，但运行环境未安装minio依赖",
                ) from exc
            endpoint = urlsplit(self.config.endpoint_url)
            self._client = Minio(
                endpoint.netloc,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=endpoint.scheme == "https",
                region=self.config.region,
            )
        return self._client

    def public_config(self) -> dict[str, object]:
        return {
            "provider": "s3",
            "endpoint_url": self.config.endpoint_url,
            "bucket": self.config.bucket,
            "prefix": self.config.prefix,
            "artifact_kinds": list(self.config.artifact_kinds),
        }

    def _stat_or_none(self, object_key: str) -> Any | None:
        try:
            return self.client.stat_object(self.config.bucket, object_key)
        except Exception as exc:
            if _is_s3_error(exc) and str(exc.code or "") in _MISSING_OBJECT_CODES:
                return None
            raise

    @staticmethod
    def _matches(stored: Any, *, digest: str, size_bytes: int) -> bool:
        metadata = {
            str(key).lower(): str(value)
            for key, value in dict(getattr(stored, "metadata", {}) or {}).items()
        }
        remote_digest = (
            metadata.get("x-amz-meta-sha256")
            or metadata.get("sha256")
            or ""
        ).lower()
        return int(getattr(stored, "size", -1)) == int(size_bytes) and remote_digest == digest

    def _identity(
        self,
        object_key: str,
        stored: Any,
        digest: str,
        size_bytes: int,
        *,
        uploaded: bool,
        version_id: str = "",
    ) -> dict[str, object]:
        endpoint = self.config.endpoint_url.rstrip("/")
        return {
            "provider": "s3",
            "bucket": self.config.bucket,
            "object_key": object_key,
            "object_uri": f"s3://{self.config.bucket}/{object_key}",
            "http_url": f"{endpoint}/{self.config.bucket}/{quote(object_key, safe='/')}",
            "version_id": version_id or str(getattr(stored, "version_id", "") or ""),
            "sha256": digest,
            "size_bytes": int(size_bytes),
            "uploaded": bool(uploaded),
        }

    def _validate_config(self) -> None:
        endpoint = urlsplit(str(self.config.endpoint_url or "").strip())
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
            raise ModelObjectStoreConfigurationError(
                "research.storage.object_store.endpoint_url必须是HTTP(S)地址",
            )
        if endpoint.path not in {"", "/"} or endpoint.query or endpoint.fragment:
            raise ModelObjectStoreConfigurationError("MinIO endpoint_url不能包含路径或查询参数")
        self._component(self.config.bucket, "bucket")
        if not self.config.access_key or not self.config.secret_key:
            raise ModelObjectStoreConfigurationError("MinIO凭据不完整")
        prefix = str(self.config.prefix or "").strip("/")
        if not prefix or any(
            not _SAFE_COMPONENT.fullmatch(part) for part in prefix.split("/")
        ):
            raise ModelObjectStoreConfigurationError("MinIO对象前缀无效")
        if not self.config.artifact_kinds:
            raise ModelObjectStoreConfigurationError("MinIO模型制品类型不能为空")
        for artifact_kind in self.config.artifact_kinds:
            self._component(artifact_kind, "artifact_kind")

    def _object_key(self, object_uri: str) -> str:
        parsed = urlsplit(str(object_uri or "").strip())
        if parsed.scheme != "s3" or parsed.netloc != self.config.bucket:
            raise ModelObjectStoreConfigurationError("模型对象URI与当前MinIO Bucket不匹配")
        if parsed.query or parsed.fragment:
            raise ModelObjectStoreConfigurationError("模型对象URI不能包含查询参数")
        object_key = parsed.path.lstrip("/")
        parts = object_key.split("/")
        if not object_key or any(
            not _SAFE_COMPONENT.fullmatch(part) or part in {".", ".."}
            for part in parts
        ):
            raise ModelObjectStoreConfigurationError("模型对象URI包含非法路径")
        prefix = self.config.prefix.strip("/")
        if object_key != prefix and not object_key.startswith(prefix + "/"):
            raise ModelObjectStoreConfigurationError("模型对象URI不属于配置的对象前缀")
        return object_key

    @staticmethod
    def _component(value: object, field_name: str) -> str:
        clean = str(value or "").strip()
        if not _SAFE_COMPONENT.fullmatch(clean) or clean in {".", ".."}:
            raise ModelObjectStoreConfigurationError(f"MinIO {field_name}包含非法字符")
        return clean


class _UploadCheckpoint:
    def __init__(self, checkpoint: Any, progress: Any = None, total: int = 0) -> None:
        self.checkpoint = checkpoint
        self.progress, self.total = progress, total
        self.done, self.last = 0, 0.0
        self.lock = threading.Lock()

    def set_meta(self, **kwargs: Any) -> None:
        if self.checkpoint:
            self.checkpoint()
        if self.progress:
            self.progress(0, self.total)

    def update(self, length: int) -> None:
        if self.checkpoint:
            self.checkpoint()
        # MinIO multipart workers may call this concurrently. Bound DB writes.
        with self.lock:
            self.done = min(self.total, self.done + max(0, length))
            now = time.monotonic()
            if self.progress and now - self.last >= 2:
                self.last = now
                self.progress(self.done, self.total)


def _is_s3_error(exc: BaseException) -> bool:
    try:
        from minio.error import S3Error
    except ModuleNotFoundError:
        return False
    return isinstance(exc, S3Error)


def load_secret_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ModelObjectStoreConfigurationError(f"MinIO凭据文件不存在: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ModelObjectStoreConfigurationError("MinIO凭据文件格式无效")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _content_type(file_name: str) -> str:
    if file_name.endswith((".tar.gz", ".tgz")):
        return "application/gzip"
    return "application/octet-stream"


__all__ = [
    "ModelObjectStore",
    "ModelObjectStoreConfig",
    "ModelObjectStoreConfigurationError",
    "ModelObjectStoreDownloadError",
    "ModelObjectStoreIntegrityError",
    "ModelObjectStoreUploadError",
    "load_secret_env",
]
