"""Standalone, standard-library-only operations for one remote dataset cache.

Executed over SSH (or in the configured Docker image). The owning controller
holds a token lock until upload, training and result retrieval have finished.
Locks never expire by time: a disconnected controller may still have a runner.
"""
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import re


FILES = {"dataset.parquet", "dataset_raw.parquet", "dataset_manifest.json"}
MARKER = ".alphablocks-cache-complete"
HASH = re.compile(r"[0-9a-f]{64}")


def _directory(path: Path) -> Path:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"缓存路径不是普通目录，已保留: {path}")
    if path.resolve() != path:
        raise ValueError(f"缓存路径包含符号链接，已保留: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _entries(root: Path) -> dict[str, Path]:
    result = {}
    for path in root.iterdir():
        if not HASH.fullmatch(path.name) or path.is_symlink() or not path.is_dir():
            raise ValueError(f"数据集缓存含未知路径，已保留: {path.name}")
        for child in path.iterdir():
            # rsync can leave one temporary file when a transfer is interrupted.
            temporary = any(re.fullmatch(r"\." + re.escape(name) + r"\.[A-Za-z0-9]+", child.name) for name in FILES)
            if child.is_symlink() or not child.is_file() or (
                child.name not in FILES | {MARKER, ".last_used"} and not temporary
            ):
                raise ValueError(f"数据集缓存含未知文件，已保留: {child}")
        result[path.name] = path
    return result


def _matches(path: Path, expected: dict) -> bool:
    if set(expected) != FILES:
        raise ValueError("数据集校验清单不完整")
    for name, identity in expected.items():
        if not HASH.fullmatch(str(identity.get("sha256", ""))):
            raise ValueError("数据集校验SHA256无效")
        item = path / name
        if not item.is_file() or item.is_symlink() or item.stat().st_size != identity["size_bytes"]:
            return False
        digest = sha256()
        with item.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != identity["sha256"]:
            return False
    return True


def operate(request: dict) -> dict:
    root = Path(request["work_dir"])
    if not root.is_absolute() or ".." in root.parts or len(root.parts) < 3 or root == Path.home():
        raise ValueError("节点工作目录范围不安全")
    root = _directory(root)
    cache = _directory(root / "cache")
    # Serialize individual remote operations as well: an SSH timeout during a
    # large hash check must not race a subsequent release/replacement command.
    descriptor = os.open(cache / ".dataset-operation.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as operation_lock:
        fcntl.flock(operation_lock.fileno(), fcntl.LOCK_EX)
        return _operate_locked(request, root, cache)


def _operate_locked(request: dict, root: Path, cache: Path) -> dict:
    datasets = _directory(cache / "datasets")
    lock = cache / ".dataset-use.lock"
    token = str(request["token"])
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise ValueError("节点缓存占用令牌无效")
    action = request["action"]
    if action == "owner" and not lock.exists() and not lock.is_symlink():
        return {"owner": None}
    if action == "acquire":
        try:
            lock.mkdir(mode=0o700)
        except FileExistsError:
            raise ValueError("节点数据集缓存被占用；若此前连接中断，请先确认旧任务结束再处理占用锁") from None
        (lock / "owner.json").write_text(json.dumps({"token": token, "job_id": request.get("job_id", "")}), encoding="utf-8")
        return {"acquired": True}
    if lock.is_symlink() or not lock.is_dir() or (lock / "owner.json").is_symlink():
        raise ValueError("节点数据集缓存占用锁无效")
    owner = json.loads((lock / "owner.json").read_text(encoding="utf-8"))
    if action == "owner":
        return {"owner": owner}
    if owner.get("token") != token:
        raise ValueError("节点数据集缓存由其他任务占用")
    if action == "release":
        (lock / "owner.json").unlink()
        lock.rmdir()
        return {"released": True}
    entries = _entries(datasets)
    if action == "inventory":
        return {"datasets": sorted(entries)}
    target = str(request["dataset_hash"])
    if not HASH.fullmatch(target):
        raise ValueError("节点数据集Hash无效")
    if action == "prepare":
        obsolete = set(entries) - {target}
        # The controller checks all MinIO identities before authorizing these exact
        # directories. Validate the whole inventory before deleting any one file.
        if obsolete != set(request["archived_hashes"]):
            raise ValueError("节点缓存清单已变化，未清理")
        for name in sorted(obsolete):
            for child in entries[name].iterdir():
                child.unlink()
            entries[name].rmdir()
        _directory(datasets / target)
        return {"removed": sorted(obsolete)}
    if action in {"check", "commit"}:
        path = datasets / target
        if set(entries) != {target}:
            raise ValueError("节点必须只有当前一套数据集缓存")
        valid = _matches(path, request["files"])
        marker = path / MARKER
        if not valid:
            marker.unlink(missing_ok=True)
            if action == "commit":
                raise ValueError("节点数据集传输校验失败，未标记为可训练")
        else:
            manifest = json.loads((path / "dataset_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("dataset_spec_hash") != target:
                marker.unlink(missing_ok=True)
                raise ValueError("节点数据集清单Hash与任务不一致")
            marker.write_text(json.dumps({"dataset_hash": target, "files": request["files"]}), encoding="utf-8")
        return {"valid": valid}
    raise ValueError("未知节点缓存操作")
