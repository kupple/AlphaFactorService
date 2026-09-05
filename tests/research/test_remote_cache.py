from hashlib import sha256
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

import factor_service.research.remote as remote_module
from factor_service.research.errors import PermanentJobError, RetryableJobError
from factor_service.research.job import CancellationToken
from factor_service.research.node_dataset_cache import MARKER, operate
from factor_service.research.remote_cache import RemoteDatasetCache


A, B, C = (letter * 64 for letter in "abc")


def snapshot(root, identity):
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset.parquet").write_bytes(b"synthetic-data-" + identity.encode())
    (root / "dataset_raw.parquet").write_bytes(b"synthetic-raw-" + identity.encode())
    files = {name: {"sha256": sha256((root / name).read_bytes()).hexdigest()}
             for name in ("dataset.parquet", "dataset_raw.parquet")}
    (root / "dataset_manifest.json").write_text(json.dumps({"dataset_spec_hash": identity, "files": files}))
    (root / ".last_used").touch()
    return SimpleNamespace(dataset_path=root / "dataset.parquet", manifest_path=root / "dataset_manifest.json")


class LocalTransport:
    """Run the actual SSH command locally, but only under pytest's private root."""
    def __init__(self):
        self.status = {"online": True, "training_active": False}
        self.uploads = []
        self.fail_upload = False

    def collect_status(self):
        return self.status

    def ssh(self, command, *, timeout, cancellation=None):
        if cancellation:
            cancellation.checkpoint()
        return subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=timeout)

    def push(self, source, destination, **kwargs):
        self.uploads.append(kwargs)
        # At no point should another dataset directory coexist with this target.
        target = Path(destination)
        assert [p.name for p in target.parent.iterdir()] == [target.name]
        if self.fail_upload:
            (target / "dataset.parquet").write_bytes(b"partial")
            raise RetryableJobError("synthetic upload interrupted")
        shutil.copytree(source, destination, dirs_exist_ok=True)


@pytest.fixture
def env(tmp_path):
    root = tmp_path.resolve() / "node"
    node = SimpleNamespace(work_dir=str(root), node_id="test-node", runner="direct_python", python_executable=sys.executable)
    transport = LocalTransport()
    records = {h: {"files_json": {name: {"object_uri": h + "/" + name} for name in
               ("dataset.parquet", "dataset_raw.parquet", "dataset_manifest.json")}} for h in (A, B, C)}
    verified = []
    archive = SimpleNamespace(record=records.get, objects=SimpleNamespace(verify_file=lambda item, **kw: verified.append(item)))
    cache = RemoteDatasetCache(node, transport, archive, "test-job")
    cache.acquire(CancellationToken())
    yield SimpleNamespace(root=root, cache=cache, transport=transport, records=records,
                          verified=verified, tmp=tmp_path, archive=archive, node=node)
    cache.release_when_idle()


def prepare(env, identity):
    data = snapshot(env.tmp / ("source-" + identity), identity)
    return env.cache.prepare(data, identity, cancellation=CancellationToken(), progress=lambda *a: None)


def test_same_dataset_reused_and_different_dataset_replaced(env):
    assert prepare(env, A) is False
    assert prepare(env, A) is True
    assert len(env.transport.uploads) == 1
    assert prepare(env, B) is False
    assert len(env.verified) == 3
    assert list((env.root / "cache/datasets").iterdir()) == [env.root / "cache/datasets" / B]
    assert (env.root / "cache/datasets" / B / MARKER).is_file()
    assert env.transport.uploads[-1]["checksum"] is True


def test_ssh_failure_does_not_remove_cache_or_release_uncertain_lock(env, monkeypatch):
    from factor_service.research.errors import NodeSSHAuthenticationError
    prepare(env, A)
    def fail(*a, **k):
        raise NodeSSHAuthenticationError('node authentication failed')
    monkeypatch.setattr(env.transport, 'ssh', fail)
    with pytest.raises(NodeSSHAuthenticationError):
        env.cache._call('prepare', dataset_hash=A, archived_hashes=[])
    env.cache.release_when_idle()
    assert env.cache.owned
    assert (env.root / 'cache/.dataset-use.lock').exists()
    assert (env.root / 'cache/datasets' / A / 'dataset.parquet').is_file()


def test_offline_status_preserves_ssh_error_type(env):
    from factor_service.research.errors import NodeSSHAuthenticationError
    env.transport.status = dict(online=False, error_code='node_ssh_authentication_failed', error='authentication failed')
    with pytest.raises(NodeSSHAuthenticationError):
        env.cache._idle()
    env.transport.status = dict(online=True, training_active=False)


def test_migrates_all_old_hash_directories_but_preserves_code_and_runs(env):
    cache_root = env.root / "cache/datasets"
    snapshot(cache_root / A, A)
    snapshot(cache_root / B, B)
    code = env.root / "cache/source/version/keep.py"
    code.parent.mkdir(parents=True)
    code.write_text("keep")
    result = env.root / "runs/old/work/bundle.tar.gz"
    result.parent.mkdir(parents=True)
    result.write_bytes(b"keep")
    prepare(env, C)
    assert [p.name for p in cache_root.iterdir()] == [C]
    assert len(env.verified) == 6
    assert code.read_text() == "keep" and result.read_bytes() == b"keep"


@pytest.mark.parametrize("missing_archive", [True, False])
def test_missing_or_unavailable_minio_preserves_every_old_cache(env, missing_archive):
    snapshot(env.root / "cache/datasets" / A, A)
    snapshot(env.root / "cache/datasets" / B, B)
    if missing_archive:
        del env.records[B]
    else:
        def fail(*args, **kwargs):
            raise ConnectionError("MinIO unavailable")
        env.archive.objects.verify_file = fail
    with pytest.raises((PermanentJobError, ConnectionError)):
        prepare(env, C)
    assert sorted(p.name for p in (env.root / "cache/datasets").iterdir()) == [A, B]
    assert not env.transport.uploads


def test_interrupted_transfer_not_ready_and_same_hash_can_retry(env):
    prepare(env, A)
    env.transport.fail_upload = True
    with pytest.raises(RetryableJobError):
        prepare(env, B)
    target = env.root / "cache/datasets" / B
    assert not (target / MARKER).exists()
    assert [p.name for p in target.parent.iterdir()] == [B]
    env.transport.fail_upload = False
    assert prepare(env, B) is False
    assert (target / MARKER).is_file()


def test_corrupt_cache_is_checked_and_retransferred(env):
    prepare(env, A)
    path = env.root / "cache/datasets" / A / "dataset.parquet"
    path.write_bytes(b"x" * path.stat().st_size)
    assert prepare(env, A) is False
    assert len(env.transport.uploads) == 2


def test_corrupt_upload_never_commits_ready_marker(env):
    def corrupt(source, destination, **kwargs):
        shutil.copytree(source, destination, dirs_exist_ok=True)
        (Path(destination) / "dataset.parquet").write_bytes(b"broken")
    env.transport.push = corrupt
    with pytest.raises(PermanentJobError, match="传输校验失败"):
        prepare(env, A)
    assert not (env.root / "cache/datasets" / A / MARKER).exists()


def test_atomic_lock_blocks_second_controller_and_wrong_token(env):
    other = RemoteDatasetCache(env.node, env.transport, env.archive, "other-job")
    with pytest.raises(RetryableJobError, match="缓存正在被其他任务占用"):
        other.acquire(CancellationToken())
    with pytest.raises(PermanentJobError, match="其他任务占用"):
        other._call("release")
    assert env.cache._call("inventory") == {"datasets": []}


def test_recovery_only_releases_this_parents_idle_old_jobs(env):
    other = RemoteDatasetCache(env.node, env.transport, env.archive, "new-child")
    assert other._call('owner')['owner']['job_id'] == 'test-job'
    other.recover_previous_jobs({'foreign-child'}, CancellationToken())
    assert other._call('owner')['owner']['job_id'] == 'test-job'
    env.transport.status['training_active'] = True
    with pytest.raises(RetryableJobError, match='运行中的训练'):
        other.recover_previous_jobs({'test-job'}, CancellationToken())
    env.transport.status['training_active'] = False
    other.recover_previous_jobs({'test-job'}, CancellationToken())
    env.cache.owned = False
    assert other._call('owner')['owner'] is None
    other.acquire(CancellationToken())
    other.release_when_idle()
    env.cache.release_when_idle()
    other.acquire(CancellationToken())
    other.release_when_idle()


@pytest.mark.parametrize("status", [{"online": False}, {"online": True, "training_active": True}])
def test_unknown_or_active_node_never_replaces_data(env, status):
    prepare(env, A)
    env.transport.status = status
    try:
        with pytest.raises(RetryableJobError):
            prepare(env, B)
        assert (env.root / "cache/datasets" / A / "dataset.parquet").is_file()
    finally:
        env.transport.status = {"online": True, "training_active": False}


@pytest.mark.parametrize("kind", ["unknown_file", "symlink_file", "symlink_dataset", "unknown_directory"])
def test_unknown_or_symlink_files_fail_closed_before_deletion(env, kind):
    root = env.root / "cache/datasets"
    snapshot(root / A, A)
    outside = env.tmp / "user-evidence"
    outside.mkdir()
    (outside / "important").write_text("keep")
    if kind == "unknown_file":
        (root / A / "important").write_text("keep")
    elif kind == "symlink_file":
        (root / A / "dataset.parquet").unlink()
        (root / A / "dataset.parquet").symlink_to(outside / "important")
    elif kind == "symlink_dataset":
        (root / B).symlink_to(outside, target_is_directory=True)
    else:
        (root / "user-data").mkdir()
    with pytest.raises(PermanentJobError, match="未知"):
        prepare(env, C)
    assert (outside / "important").read_text() == "keep"
    assert (root / A).exists()


def test_symlink_cache_parent_does_not_touch_external_directory(tmp_path):
    root = tmp_path.resolve() / "node"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "cache").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="普通目录"):
        operate({"work_dir": str(root), "action": "acquire", "token": "a" * 32})
    assert list(outside.iterdir()) == []


def test_docker_helper_uses_existing_image_and_no_network():
    seen = []
    node = SimpleNamespace(work_dir="/root/alphablocks-research", runner="docker", docker_image="test:latest")
    def ssh(command, **kwargs):
        seen.extend(shlex.split(command))
        return subprocess.CompletedProcess([], 0, '{"acquired": true}', "")
    cache = RemoteDatasetCache(node, SimpleNamespace(ssh=ssh), None, "test")
    cache._call("acquire")
    assert seen[:5] == ["mkdir", "-p", "--", "/root/alphablocks-research", "&&"]
    assert seen[5:10] == ["docker", "run", "--rm", "--network", "none"]
    assert seen[10:12] == ["--user", "$(id -u):$(id -g)"]
    assert seen[12:16] == ["-v", "/root/alphablocks-research:/root/alphablocks-research", "test:latest", "python"]


def test_busy_rsync_preserves_lock_and_dataset(env):
    original = env.transport.ssh
    def ssh(command, **kwargs):
        if command == "ps -ax -o command=":
            return subprocess.CompletedProcess([], 0, f"rsync --server -a . {env.root}/cache/datasets/{A}/", "")
        return original(command, **kwargs)
    env.transport.ssh = ssh
    try:
        with pytest.raises(RetryableJobError, match="仍在传输"):
            prepare(env, A)
    finally:
        env.transport.ssh = original


def test_release_does_not_expire_lock_for_running_task(env, monkeypatch):
    import factor_service.research.remote_cache as module
    monkeypatch.setattr(module.time, "monotonic", iter([0, 16]).__next__)
    env.transport.status = {"online": True, "training_active": True}
    env.cache.release_when_idle()
    assert env.cache.owned
    assert (env.root / "cache/.dataset-use.lock/owner.json").is_file()
    env.transport.status = {"online": True, "training_active": False}
    monkeypatch.undo()


@pytest.mark.parametrize("failure", ["busy", "upload", "launch", "ssh", "none"])
def test_executor_uses_cache_guard_and_cleans_up_only_owned_runner(tmp_path, monkeypatch, failure):
    calls = []
    data = snapshot(tmp_path / "dataset", A)
    data.reused = False
    class Guard:
        def __init__(self, *args):
            self.owned = False
        def acquire(self, cancellation):
            calls.append("acquire")
            if failure == "busy":
                raise RetryableJobError("busy")
            self.owned = True
        def prepare(self, *args, **kwargs):
            calls.append("prepare")
            if failure == "upload":
                raise RetryableJobError("upload failed")
            return False
        def release_when_idle(self):
            calls.append("release")
            self.owned = False
    monkeypatch.setattr(remote_module, "RemoteDatasetCache", Guard)
    executor = object.__new__(remote_module.RemoteResearchExecutor)
    executor.lifecycle = object()
    executor.node = SimpleNamespace(node_id="test", work_dir="/safe/remote", lifecycle_provider="autodl_pro",
                                    runner="direct_python", gpus="0", cleanup_success=False)
    executor.snapshot_store = SimpleNamespace(archive=None, get_or_create=lambda *a, **k: data)
    executor.settings = SimpleNamespace(model_artifacts_root=tmp_path / "artifacts")
    executor._prepare_lifecycle = lambda **kw: calls.append("lifecycle")
    executor._remote_cache_ready = lambda *a: True
    executor._remote_cpu_cores = lambda *a: 4
    executor._direct_python_command = lambda *a: "launch"
    executor._wait_for_process = lambda *a, **kw: calls.append("wait")
    executor._stop_remote_runner = lambda *a: calls.append("stop")
    executor._power_off_after_job = lambda *a: calls.append("power_off")
    def ssh(command, **kwargs):
        if command == "launch":
            calls.append("launch")
            if failure == "launch":
                raise TimeoutError("launch uncertain")
            if failure == "ssh":
                from factor_service.research.errors import NodeSSHConnectionError
                raise NodeSSHConnectionError("launch connection uncertain")
        return subprocess.CompletedProcess([], 0, "", "")
    executor.transport = SimpleNamespace(ssh=ssh, push=lambda *a, **k: None, pull=lambda *a, **k: None)
    monkeypatch.setattr(remote_module, "_load_remote_result", lambda *a: "result")
    job = {"job_id": "model_job_test", "dataset_hash": A, "config_json": {}}
    def run():
        return executor.train(job, tmp_path, cancellation=CancellationToken(), progress=lambda *a: None)
    if failure != "none":
        from factor_service.research.errors import NodeSSHError
        with pytest.raises((RetryableJobError, TimeoutError, NodeSSHError)):
            run()
    else:
        assert run() == "result"
        assert calls.index("prepare") < calls.index("launch") < calls.index("wait") < calls.index("release")
    assert ("stop" in calls) == (failure in {"launch", "none"})
    assert ("power_off" in calls) == (failure not in {"busy", "ssh"})
    assert ("release" in calls) == (failure != "ssh")
