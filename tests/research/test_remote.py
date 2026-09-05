from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from factor_service.research import remote as remote_module
from factor_service.research.config import Settings
from factor_service.research.job import CancellationToken
from factor_service.research.errors import NodeMemoryBudgetExceeded, classify_exception
from factor_service.research.remote import (
    RemoteResearchExecutor,
    RemoteTransport,
    _effective_memory_usage_mb,
    _effective_remote_thread_count,
    _load_remote_result,
    _requested_job_threads,
    _remote_result_is_complete,
    _raise_remote_resource_failure,
    _source_fingerprint,
    load_remote_nodes,
    remote_node_storage_payload,
)
from factor_service.research.remote_runner import _scoped_path
from factor_service.research.trainer import QlibTrainer


def _runtime(tmp_path: Path) -> dict:
    return {
        "research": {
            "execution": {
                "remote_nodes": [{
                    "id": "autodl-gpu-01",
                    "name": "AutoDL GPU",
                    "host": "gpu.example.test",
                    "port": 22022,
                    "user": "root",
                    "authentication_type": "password",
                    "ssh_password": "private-value",
                    "work_dir": "/root/alphablocks-research",
                    "docker_image": "alphafactor-research:latest",
                    "gpus": "all",
                    "known_hosts": str(tmp_path / "known_hosts"),
                }],
            },
        },
    }


def test_remote_node_public_contract_does_not_expose_secret(
    tmp_path: Path,
) -> None:
    node = load_remote_nodes(_runtime(tmp_path))[0]

    public = node.public()

    assert public["id"] == "autodl-gpu-01"
    assert public["compute_type"] == "gpu"
    assert public["available"] is True
    assert public["credential_type"] == "password"
    assert "ssh_password" not in public
    assert "private-value" not in json.dumps(public)


def test_remote_node_supports_autodl_direct_python_runner(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    runtime["research"]["execution"]["remote_nodes"][0].update({
        "runner": "direct_python",
        "python_executable": "/root/miniconda3/bin/python",
        "work_dir": "/root/autodl-tmp/alphablocks-research",
    })

    node = load_remote_nodes(runtime)[0]

    assert node.runner == "direct_python"
    assert node.python_executable == "/root/miniconda3/bin/python"
    assert node.public()["runner"] == "direct_python"


def test_remote_node_storage_excludes_secret_values(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    config = runtime["research"]["execution"]["remote_nodes"][0]

    node = load_remote_nodes(runtime)[0]
    stored = remote_node_storage_payload(node)

    assert stored["authentication_type"] == "password"
    assert stored["compute_type"] == "gpu"
    assert "ssh_password" not in stored
    assert "private-value" not in repr(stored)


def test_legacy_cpu_node_infers_compute_type_from_disabled_gpu_mount(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    runtime["research"]["execution"]["remote_nodes"][0]["gpus"] = "0"

    node = load_remote_nodes(runtime)[0]

    assert node.compute_type == "cpu"
    assert node.public()["compute_type"] == "cpu"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("host", "gpu;touch /tmp/pwned"),
        ("work_dir", "/root/../tmp"),
        ("docker_image", "image;shutdown"),
        ("authentication_type", "password;shutdown"),
        ("runner", "direct;shutdown"),
        ("compute_type", "tpu"),
    ],
)
def test_remote_node_rejects_command_injection_fields(
    tmp_path: Path, field: str, value: str,
) -> None:
    runtime = _runtime(tmp_path)
    runtime["research"]["execution"]["remote_nodes"][0][field] = value
    with pytest.raises(ValueError):
        load_remote_nodes(runtime)


def test_remote_node_rejects_compute_type_and_gpu_mount_mismatch(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    runtime["research"]["execution"]["remote_nodes"][0].update({
        "compute_type": "cpu",
        "gpus": "all",
    })

    with pytest.raises(ValueError, match="CPU节点必须禁用GPU挂载"):
        load_remote_nodes(runtime)


def test_direct_python_runner_rejects_relative_python_path(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime["research"]["execution"]["remote_nodes"][0].update({
        "runner": "direct_python",
        "python_executable": "python",
    })

    with pytest.raises(ValueError, match="安全绝对路径"):
        load_remote_nodes(runtime)


def test_password_transport_does_not_offer_unrelated_agent_keys(tmp_path, monkeypatch):
    transport = RemoteTransport(load_remote_nodes(_runtime(tmp_path))[0])
    captured = []

    def fake_run(args, *, timeout, cancellation):
        captured.append(args)
        return subprocess.CompletedProcess(args, 0, "ok", "")

    monkeypatch.setattr(transport, "_run", fake_run)
    transport.ssh("true", timeout=30)
    assert "PubkeyAuthentication=no" in captured[0]
    assert "PreferredAuthentications=password,keyboard-interactive" in captured[0]
    assert "NumberOfPasswordPrompts=1" in captured[0]
    rsync_ssh = transport._ssh_base(include_target=False)
    assert "PubkeyAuthentication=no" in rsync_ssh
    assert "StrictHostKeyChecking=yes" in rsync_ssh
    assert "private-value" not in " ".join(captured[0] + rsync_ssh)


def test_control_sessions_are_private_scoped_and_explicitly_closed(tmp_path):
    first = RemoteTransport(load_remote_nodes(_runtime(tmp_path))[0])
    second = RemoteTransport(load_remote_nodes(_runtime(tmp_path))[0])
    assert first._control_path != second._control_path
    assert len(str(first._control_path).encode()) < 104
    assert first._control_path.parent.stat().st_mode & 0o777 == 0o700
    assert 'ControlMaster=auto' in first._ssh_base()
    assert 'ControlPersist=60' in first._ssh_base()
    assert f'ControlPath={first._control_path}' in first._ssh_base(include_target=False)
    directory = first._control_path.parent
    first.close()
    first.close()  # Idempotent; must not open a connection on cleanup.
    assert not directory.exists()
    assert second._control_path.parent.exists()
    second.close()


def test_control_cleanup_uses_only_its_owned_local_socket(tmp_path, monkeypatch):
    transport = RemoteTransport(load_remote_nodes(_runtime(tmp_path))[0])
    transport._control_path.touch()
    calls = []
    monkeypatch.setattr('factor_service.research.remote.subprocess.run', lambda args, **kw: calls.append((args, kw)))
    monkeypatch.setenv('ALPHA_REMOTE_NODE_SECRET_KEY', 'do-not-pass')
    monkeypatch.setenv('SSHPASS', 'do-not-pass')
    transport.close()
    assert len(calls) == 1
    assert calls[0][0][:5] == ['ssh', '-S', str(transport._control_path), '-O', 'exit']
    assert 'ALPHA_REMOTE_NODE_SECRET_KEY' not in calls[0][1]['env']
    assert 'SSHPASS' not in calls[0][1]['env']


@pytest.mark.parametrize('operation', ['ssh', 'push', 'pull'])
def test_authentication_failure_is_typed_in_all_transports(tmp_path, monkeypatch, operation):
    from factor_service.research.errors import NodeSSHAuthenticationError
    transport = RemoteTransport(load_remote_nodes(_runtime(tmp_path))[0])
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 255, '', 'Too many authentication failures')
    monkeypatch.setattr(transport, '_run', run)
    with pytest.raises(NodeSSHAuthenticationError):
        if operation == 'ssh':
            transport.ssh('true', timeout=5)
        elif operation == 'push':
            transport.push(tmp_path / 'source', '/remote/destination')
        else:
            transport.pull('/remote/source', tmp_path / 'destination')
    assert len(calls) == 1  # No repeated password attempts inside the transport.


def test_status_keeps_ssh_error_code_for_scheduler(tmp_path, monkeypatch):
    from factor_service.research.errors import NodeSSHAuthenticationError
    transport = RemoteTransport(load_remote_nodes(_runtime(tmp_path))[0])
    def fail(*a, **k):
        raise NodeSSHAuthenticationError('node authentication failed')
    monkeypatch.setattr(transport, 'ssh', fail)
    status = transport.collect_status()
    assert status['online'] is False
    assert status['error_code'] == 'node_ssh_authentication_failed'
    assert 'ssh_password' not in status


def test_remote_transport_ssh_includes_target_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path)
    node_config = runtime["research"]["execution"]["remote_nodes"][0]
    node_config["authentication_type"] = "ssh_private_key"
    node_config["ssh_password"] = ""
    node_config["ssh_private_key"] = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "dGVzdA==\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )
    transport = RemoteTransport(load_remote_nodes(runtime)[0])
    captured: list[str] = []

    def fake_run(args, *, timeout, cancellation):
        captured.extend(args)
        return subprocess.CompletedProcess(args, 0, "ok", "")

    monkeypatch.setattr(transport, "_run", fake_run)

    transport.ssh("printf ok", timeout=30)

    assert captured.count("root@gpu.example.test") == 1
    assert captured[-1] == "printf ok"
    assert "IdentitiesOnly=yes" in captured
    assert transport._ssh_key_path is not None
    assert transport._ssh_key_path.read_text(encoding="utf-8").startswith(
        "-----BEGIN OPENSSH PRIVATE KEY-----",
    )
    assert oct(transport._ssh_key_path.stat().st_mode & 0o777) == "0o600"


def test_remote_transport_drains_large_stdout_and_stderr() -> None:
    transport = object.__new__(RemoteTransport)
    transport.node = SimpleNamespace(ssh_password="")
    size = 1_000_000

    completed = transport._run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.stdout.write('o' * {size}); "
                f"sys.stderr.write('e' * {size})"
            ),
        ],
        timeout=5,
        cancellation=None,
    )

    assert completed.returncode == 0
    assert len(completed.stdout) == size
    assert len(completed.stderr) == size


def test_remote_transport_uses_portable_utf8_locale() -> None:
    transport = object.__new__(RemoteTransport)
    transport.node = SimpleNamespace(ssh_password="")

    completed = transport._run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "print(os.environ['LANG']); "
                "print(os.environ['LC_ALL'])"
            ),
        ],
        timeout=5,
        cancellation=None,
    )

    assert completed.returncode == 0
    assert completed.stdout.splitlines() == ["C.UTF-8", "C.UTF-8"]


def test_effective_memory_usage_prefers_smaller_cgroup_limit() -> None:
    total, used = _effective_memory_usage_mb(
        [514_631, 34_669], [17_179_869_184, 1_073_741_824],
    )

    assert total == 16_384
    assert used == 1_024


def test_effective_memory_usage_uses_host_when_cgroup_is_unlimited() -> None:
    assert _effective_memory_usage_mb([32_768, 2_048], [0, 0]) == (32_768, 2_048)


def test_remote_transport_rsync_is_compatible_with_macos_openrsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "dataset.parquet"
    source.write_text("test-data", encoding="utf-8")
    runtime = _runtime(tmp_path)
    node_config = runtime["research"]["execution"]["remote_nodes"][0]
    node_config["authentication_type"] = "ssh_private_key"
    node_config["ssh_password"] = ""
    node_config["ssh_private_key"] = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "dGVzdA==\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )
    transport = RemoteTransport(load_remote_nodes(runtime)[0])
    captured: list[str] = []

    def fake_run(args, *, timeout, cancellation):
        captured.extend(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(transport, "_run", fake_run)

    transport.push(source, "/root/alphablocks-research/dataset.parquet")

    assert captured[0] == "rsync"
    assert "-z" not in captured
    assert "-az" not in captured
    assert "--protect-args" not in captured
    assert captured[-2] == str(source)
    assert captured[-1] == (
        "root@gpu.example.test:/root/alphablocks-research/dataset.parquet"
    )


def test_remote_progress_reads_only_unseen_lines() -> None:
    payloads = [
        "\n".join([
            json.dumps({
                "stage": "training", "percent": 64, "details": {"step": 1},
            }),
            json.dumps({
                "stage": "training", "percent": 65, "details": {"step": 2},
            }),
        ]),
        json.dumps({
            "stage": "training", "percent": 66, "details": {"step": 3},
        }),
    ]

    class Transport:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def ssh(self, command, *, timeout, cancellation):
            self.commands.append(command)
            return subprocess.CompletedProcess([], 0, payloads.pop(0), "")

    executor = object.__new__(RemoteResearchExecutor)
    executor.transport = Transport()
    executor.node = SimpleNamespace(node_id="autodl-test")
    events: list[tuple[str, int, dict]] = []

    def progress(stage, percent, details) -> None:
        events.append((stage, percent, details))

    seen = executor._forward_progress(
        "/root/run", 0,
        cancellation=CancellationToken(), progress=progress,
    )
    seen = executor._forward_progress(
        "/root/run", seen,
        cancellation=CancellationToken(), progress=progress,
    )

    assert seen == 3
    assert "tail -n +1 --" in executor.transport.commands[0]
    assert "tail -n +3 --" in executor.transport.commands[1]
    assert [event[2]["step"] for event in events] == [1, 2, 3]


def test_remote_resource_heartbeat_preserves_window_and_fresh_failure_details():
    events = [
        {"stage": "walk_forward_training", "percent": 66, "details": {"window_index": 19}},
        {"stage": "resource_heartbeat", "percent": 0, "details": {"resources": {"available": 3}}},
        {"stage": "node_memory_budget_exceeded", "percent": 66, "details": {"resources": {"available": 1}}},
    ]
    executor = object.__new__(RemoteResearchExecutor)
    executor.node = SimpleNamespace(node_id="test")
    executor.transport = SimpleNamespace(ssh=lambda *a, **k: subprocess.CompletedProcess(
        [], 0, "\n".join(json.dumps(event) for event in events), "",
    ))
    forwarded = []
    executor._forward_progress("/root/run", 0, cancellation=CancellationToken(), progress=lambda *args: forwarded.append(args))
    assert forwarded[1][0] == "remote.walk_forward_training"
    assert forwarded[1][1] == 66
    assert forwarded[1][2]["window_index"] == 19
    assert forwarded[1][2]["resources"]["available"] == 3
    assert forwarded[2][2]["resources"]["available"] == 1


def test_remote_memory_failure_has_nonretryable_actionable_code():
    transport = SimpleNamespace(ssh=lambda *a, **k: subprocess.CompletedProcess(
        [], 0, json.dumps({"error_code": "node_memory_budget_exceeded", "message": "内存预算不足"}), "",
    ))
    with pytest.raises(NodeMemoryBudgetExceeded, match="内存预算不足") as caught:
        _raise_remote_resource_failure(transport, "/root/run", None)
    assert classify_exception(caught.value) == (False, "node_memory_budget_exceeded")


def test_remote_cpu_probe_prefers_affinity_aware_nproc():
    commands = []
    executor = object.__new__(RemoteResearchExecutor)

    def ssh(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess([], 0, "8\n", "")

    executor.transport = SimpleNamespace(ssh=ssh)
    assert executor._remote_cpu_cores(CancellationToken()) == 8
    assert commands[0].startswith("nproc ")
    assert "sysctl -n hw.logicalcpu" in commands[0]


def test_direct_node_status_uses_supervisor_resource_reader(tmp_path):
    raw = _runtime(tmp_path)
    raw["research"]["execution"]["remote_nodes"][0].update({
        "runner": "direct_python", "python_executable": "/usr/bin/python3",
    })
    transport = object.__new__(RemoteTransport)
    transport.node = load_remote_nodes(raw)[0]
    commands = []
    def ssh(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess([], 0, "cpu_cores=10\nload=1.5\nmem=16384,4096\nmem_cgroup=0,0\ndisk=1000,200\ngpu=unavailable\ncontainers=", "")
    transport.ssh = ssh
    result = transport.collect_status()
    assert result["cpu_cores"] == 10
    assert result["mem_total_mb"] == 16384
    assert result["mem_used_mb"] == 4096
    assert not result["training_active"]
    assert "/usr/bin/vm_stat" in commands[0]
    assert "ps -ax -o pid=,command=" in commands[0]


def test_direct_python_launch_creates_owned_session_without_linux_setsid(tmp_path):
    executor = object.__new__(RemoteResearchExecutor)
    executor.node = SimpleNamespace(gpus="0", python_executable=sys.executable)
    package = tmp_path / "source/factor_service/research"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package.parent / "__init__.py").write_text("")
    (package / "remote_runner.py").write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(sys.argv[3]).write_text(json.dumps({'session': os.getsid(0), 'threads': os.environ['ALPHA_EFFECTIVE_NUM_THREADS']}))\n"
    )
    command = executor._direct_python_command(str(tmp_path), 2)
    assert "nohup setsid" not in command
    subprocess.run(command, shell=True, check=True, timeout=5, cwd=tmp_path)
    deadline = time.monotonic() + 5
    while not (tmp_path / "runner.exit").exists() and time.monotonic() < deadline:
        time.sleep(.05)
    assert (tmp_path / "runner.exit").read_text().strip() == "0"
    result = json.loads((tmp_path / "remote_result.json").read_text())
    assert result["session"] == int((tmp_path / "runner.pid").read_text())
    assert result["threads"] == "2"


def test_remote_poll_recovers_from_one_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(remote_module, "_REMOTE_POLL_RETRY_SECONDS", 0)
    executor = object.__new__(RemoteResearchExecutor)
    executor.node = SimpleNamespace(node_id="autodl-test")
    calls = 0
    events: list[tuple[str, int, dict]] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary SSH timeout")
        return "recovered"

    result = executor._poll_with_recovery(
        operation,
        cancellation=CancellationToken(),
        progress=lambda stage, percent, details: events.append(
            (stage, percent, details),
        ),
    )

    assert result == "recovered"
    assert calls == 2
    assert [event[0] for event in events] == [
        "remote_poll_degraded", "remote_poll_recovered",
    ]


def test_remote_resource_planner_uses_large_node_without_oversubscription() -> None:
    assert _effective_remote_thread_count(96, 4) == 32
    assert _effective_remote_thread_count(8, 4) == 4
    assert _effective_remote_thread_count(6, 16) == 6
    assert _requested_job_threads({
        "config_json": {"model": {
            "kind": "stacking",
            "params": {},
            "base_models": [
                {"kind": "lstm", "params": {"num_threads": 4}},
                {"kind": "gru", "params": {"num_threads": 12}},
            ],
        }},
    }) == 12


def test_source_fingerprint_changes_with_training_code(tmp_path: Path) -> None:
    source = tmp_path / "factor_service"
    source.mkdir()
    module = source / "trainer.py"
    module.write_text("VERSION = 1\n", encoding="utf-8")
    first = _source_fingerprint(source)

    module.write_text("VERSION = 2\n", encoding="utf-8")

    assert len(first) == 24
    assert _source_fingerprint(source) != first


def test_nonzero_remote_cleanup_exit_accepts_atomic_packaged_result() -> None:
    class _Transport:
        @staticmethod
        def ssh(command, *, timeout, cancellation):
            assert "test -s" in command
            assert "remote_packaged" in command
            return subprocess.CompletedProcess([], 0, "", "")

    assert _remote_result_is_complete(
        _Transport(), "/root/run", cancellation=None,
    ) is True


def test_remote_result_rejects_path_traversal(tmp_path: Path) -> None:
    work = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    work.mkdir()
    artifacts.mkdir()
    result = work / "remote_result.json"
    result.write_text(json.dumps({
        "schema_version": "alphablocks.remote-training-result.v1",
        "result": {},
        "artifacts": [{"kind": "bundle", "scope": "work", "path": "../escape"}],
        "predictions": {"scope": "work", "path": "predictions.parquet"},
    }), encoding="utf-8")

    with pytest.raises(RuntimeError, match="路径无效"):
        _load_remote_result(result, work, artifacts)


def test_remote_result_scope_preserves_shared_dataset_symlink(tmp_path: Path) -> None:
    work = tmp_path / "run" / "work"
    artifacts = tmp_path / "run" / "artifacts"
    cache = tmp_path / "cache" / "dataset-hash"
    work.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    cache.mkdir(parents=True)
    (cache / "dataset.parquet").write_bytes(b"parquet")
    (artifacts / "datasets").mkdir()
    (artifacts / "datasets" / "dataset-hash").symlink_to(
        cache, target_is_directory=True,
    )

    scoped = _scoped_path(
        artifacts / "datasets" / "dataset-hash" / "dataset.parquet",
        work,
        artifacts,
    )

    assert scoped == {
        "scope": "artifact_root",
        "path": "datasets/dataset-hash/dataset.parquet",
    }


def test_qlib_trainer_does_not_connect_to_clickhouse_during_init(tmp_path: Path) -> None:
    settings = Settings(
        clickhouse_host="unreachable.invalid",
        clickhouse_port=1,
        clickhouse_user="none",
        clickhouse_password="",
        factor_database="ab_factor",
        model_database="ab_model",
        source_database="starlight",
        work_root=tmp_path / "work",
        model_artifacts_root=tmp_path / "artifacts",
        scheduler_enabled=False,
        scheduler_refresh_seconds=60,
    )

    trainer = QlibTrainer(settings)

    assert trainer.dataset_builder is None
