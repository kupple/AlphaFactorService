from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable
import weakref

from factor_service.research.config import Settings
from factor_service.research.autodl import (
    AutoDLProClient,
    validate_api_token,
    validate_instance_uuid,
)
from factor_service.research.dataset import DatasetBuilder
from factor_service.research.errors import (
    NodeMemoryBudgetExceeded, NodeOutOfMemory, NodeResourceUnavailable,
    RetryableJobError, NodeSSHError,
)
from factor_service.research.job import CancellationToken
from factor_service.research.remote_node_repository import (
    get_remote_node_repository,
)
from factor_service.research.remote_node_secrets import REMOTE_NODE_SECRET_KEY_ENV
from factor_service.research.snapshot import DatasetSnapshotStore
from factor_service.research.dataset_archive import archive_for_settings
from factor_service.research.remote_cache import RemoteDatasetCache
from factor_service.research.trainer import TrainingResult
from factor_service.runtime_config import section


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,252}$")
_USER = re.compile(r"^[A-Za-z_][A-Za-z0-9._-]{0,63}$")
_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:-]{0,255}$")
_GPU_SPEC = re.compile(r"^[A-Za-z0-9=,._-]{0,128}$")
_COMPUTE_TYPES = {"cpu", "gpu"}
_REMOTE_RUNNERS = {"docker", "direct_python"}
_LIFECYCLE_PROVIDERS = {"", "autodl_pro"}
_AUTHENTICATION_TYPES = {"ssh_private_key", "password"}
_REMOTE_MAX_THREADS = 32
_REMOTE_VALIDATION_SAMPLE_ROWS = 200_000
_REMOTE_TRAIN_METRIC_SAMPLE_ROWS = 200_000
_REMOTE_POLL_RECOVERY_SECONDS = 90
_REMOTE_POLL_RETRY_SECONDS = 2


def _resource_probe_command(python: str) -> str:
    # Probe with the exact lightweight reader used by the supervisor, including
    # before a node has received its first source bundle. Never import ML here.
    source = Path(__file__).with_name("runtime_resources.py").read_text(encoding="utf-8")
    source += "\nr = read_runtime_resources()\n"
    source += "if r.memory_limit_bytes <= 0: raise RuntimeError('无法读取节点真实内存限制')\n"
    source += "print('cpu_cores=' + str(r.cpu_cores))\n"
    source += "print('load=' + str(os.getloadavg()[0]))\n"
    source += "print('mem=' + str(r.memory_limit_bytes // MIB) + ',' + str((r.memory_limit_bytes - r.memory_available_bytes) // MIB))\n"
    source += "print('mem_cgroup=0,0')\n"
    return f"{shlex.quote(python)} -c {shlex.quote(source)}"


@dataclass(frozen=True)
class RemoteNode:
    node_id: str
    name: str
    host: str
    port: int
    user: str
    work_dir: str
    runner: str
    python_executable: str
    docker_image: str
    compute_type: str
    gpus: str
    authentication_type: str
    ssh_private_key: str = field(repr=False)
    ssh_password: str = field(repr=False)
    known_hosts: Path | None
    enabled: bool
    cleanup_success: bool
    max_runtime_minutes: int
    lifecycle_provider: str
    instance_uuid: str
    api_token: str = field(repr=False)
    auto_start: bool
    auto_stop: bool
    boot_timeout_minutes: int
    known_hosts_config: str = ""

    def public(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": "remote",
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "work_dir": self.work_dir,
            "runner": self.runner,
            "python_executable": self.python_executable,
            "docker_image": self.docker_image,
            "compute_type": self.compute_type,
            "gpus": self.gpus,
            "enabled": self.enabled,
            "available": (
                self.enabled
                and self.credentials_available()
                and (not self.lifecycle_provider or self.lifecycle_available())
            ),
            "credential_type": self.authentication_type,
            "credential_configured": self.credentials_available(),
            "known_hosts_configured": self.known_hosts is not None,
            "max_runtime_minutes": self.max_runtime_minutes,
            "lifecycle_provider": self.lifecycle_provider or "manual",
            "instance_uuid": self.instance_uuid,
            "api_token_configured": bool(self.api_token),
            "auto_start": self.auto_start,
            "auto_stop": self.auto_stop,
            "boot_timeout_minutes": self.boot_timeout_minutes,
        }

    def credentials_available(self) -> bool:
        if self.authentication_type == "ssh_private_key":
            return bool(self.ssh_private_key)
        return bool(self.authentication_type == "password" and self.ssh_password)

    def lifecycle_available(self) -> bool:
        return bool(
            self.lifecycle_provider == "autodl_pro"
            and self.instance_uuid
            and self.api_token
        )


def load_remote_nodes(runtime: dict[str, Any] | None = None) -> list[RemoteNode]:
    if runtime is not None:
        raw_nodes = _runtime_remote_node_payloads(runtime)
    else:
        raw_nodes = get_remote_node_repository().list_nodes()
    if not isinstance(raw_nodes, list):
        raise ValueError("PostgreSQL远程训练节点配置必须是数组")
    nodes: list[RemoteNode] = []
    seen: set[str] = set()
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            raise ValueError("远程训练节点配置必须是对象")
        node = _normalize_node(raw)
        if node.node_id in seen:
            raise ValueError(f"远程训练节点ID重复: {node.node_id}")
        seen.add(node.node_id)
        nodes.append(node)
    return nodes


def _runtime_remote_node_payloads(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    execution = section(runtime, "research", "execution")
    raw_nodes = execution.get("remote_nodes") or []
    if not isinstance(raw_nodes, list):
        raise ValueError("research.execution.remote_nodes必须是数组")
    if any(not isinstance(raw, dict) for raw in raw_nodes):
        raise ValueError("远程训练节点配置必须是对象")
    return [dict(raw) for raw in raw_nodes]


def remote_node_storage_payload(node: RemoteNode) -> dict[str, Any]:
    return {
        "id": node.node_id,
        "name": node.name,
        "enabled": node.enabled,
        "host": node.host,
        "port": node.port,
        "user": node.user,
        "authentication_type": node.authentication_type,
        "known_hosts": node.known_hosts_config or str(node.known_hosts or ""),
        "work_dir": node.work_dir,
        "runner": node.runner,
        "python_executable": node.python_executable,
        "docker_image": node.docker_image,
        "compute_type": node.compute_type,
        "gpus": node.gpus,
        "max_runtime_minutes": node.max_runtime_minutes,
        "cleanup_success": node.cleanup_success,
        "lifecycle_provider": node.lifecycle_provider,
        "instance_uuid": node.instance_uuid,
        "auto_start": node.auto_start,
        "auto_stop": node.auto_stop,
        "boot_timeout_minutes": node.boot_timeout_minutes,
    }


def execution_nodes() -> list[dict[str, Any]]:
    local = {
        "id": "local",
        "type": "local",
        "name": "本机训练",
        "description": "AlphaFactorService短生命周期隔离进程",
        "enabled": True,
        "available": True,
    }
    return [local, *(node.public() for node in load_remote_nodes())]


def get_remote_node(node_id: str) -> RemoteNode:
    clean = str(node_id or "").strip()
    node = next((item for item in load_remote_nodes() if item.node_id == clean), None)
    if node is None:
        raise ValueError(f"远程训练节点未配置: {clean}")
    if not node.enabled:
        raise ValueError(f"远程训练节点已停用: {clean}")
    return node


class RemoteTransport:
    def __init__(self, node: RemoteNode) -> None:
        self.node = node
        self._ssh_key_path: Path | None = None
        self._ssh_key_cleanup: weakref.finalize | None = None
        if node.authentication_type == "ssh_private_key" and node.ssh_private_key:
            self._materialize_private_key(node.ssh_private_key)
        self._ensure_client_tools()
        # Short, private path stays below macOS's Unix socket path limit. Each
        # transport owns its session: never share auth with another node/user.
        self._control_directory = tempfile.TemporaryDirectory(prefix="abssh-", dir="/tmp")
        self._control_path = Path(self._control_directory.name) / "control"
        self._control_cleanup = weakref.finalize(
            self, _close_control_session, self._control_path, self._target(),
            self.node.port, self._control_directory.cleanup,
        )

    def close(self):
        self._control_cleanup()

    def test_connection(self) -> dict[str, Any]:
        if self.node.runner == "direct_python":
            python = shlex.quote(self.node.python_executable)
            dependency_check = shlex.quote(
                "import catboost, lightgbm, optuna, pandas, pyarrow, qlib, sklearn, torch, xgboost; "
                "print('python=ok'); print('torch=' + torch.__version__); "
                "print('cuda=' + str(torch.cuda.is_available()).lower())"
            )
            command = (
                "set -e; printf 'ssh=ok\\nrunner=direct_python\\n'; "
                f"test -x {python}; {_resource_probe_command(self.node.python_executable)}; "
                f"{python} -c {dependency_check}; "
                "if command -v nvidia-smi >/dev/null 2>&1; then "
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits; "
                "else printf 'gpu=unavailable\\n'; fi"
            )
        else:
            command = (
                "set -e; printf 'ssh=ok\\nrunner=docker\\n'; command -v docker >/dev/null; "
                "docker version --format 'docker={{.Server.Version}}'; "
                f"docker image inspect {shlex.quote(self.node.docker_image)} "
                "--format 'image=ok' >/dev/null; "
                "if command -v nvidia-smi >/dev/null 2>&1; then "
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits; "
                "else printf 'gpu=unavailable\\n'; fi"
            )
        completed = self.ssh(command, timeout=30)
        return {
            "success": completed.returncode == 0,
            "node": self.node.public(),
            "detail": completed.stdout.strip(),
            "error": completed.stderr.strip() if completed.returncode else "",
        }

    def collect_status(self) -> dict[str, Any]:
        runner_status = (
            "printf '\\ncontainers='; ps -ax -o pid=,command= | grep '[f]actor_service.research.remote_runner' "
            "2>/dev/null | tr '\\n' ';'"
            if self.node.runner == "direct_python"
            else "printf '\\ncontainers='; docker ps --filter label=alphablocks.research=1 "
            "--format '{{.Names}}|{{.Status}}' 2>/dev/null | tr '\\n' ';'"
        )
        resource_command = (
            "printf 'cpu_cores='; nproc 2>/dev/null || printf 0; "
            "printf '\\nload='; awk '{print $1}' /proc/loadavg 2>/dev/null || printf 0; "
            "printf '\\nmem='; free -m 2>/dev/null | awk '/^Mem:/{print $2\",\"$3}'; "
            "printf '\\nmem_cgroup='; "
            "if test -r /sys/fs/cgroup/memory.max; then "
            "limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); "
            "used=$(cat /sys/fs/cgroup/memory.current 2>/dev/null); "
            "test \"$limit\" = max && limit=0; printf '%s,%s' \"${limit:-0}\" \"${used:-0}\"; "
            "elif test -r /sys/fs/cgroup/memory/memory.limit_in_bytes; then "
            "limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null); "
            "used=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null); "
            "printf '%s,%s' \"${limit:-0}\" \"${used:-0}\"; "
            "else printf '0,0'; fi; "
        )
        if self.node.runner == "direct_python":
            resource_command = f"{_resource_probe_command(self.node.python_executable)}; "
        command = "set -e; " + resource_command + (
            "printf '\\ndisk='; df -Pk / 2>/dev/null | awk 'NR==2{print $2\",\"$3}'; "
            "printf '\\ngpu='; if command -v nvidia-smi >/dev/null 2>&1; then "
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu "
            "--format=csv,noheader,nounits | tr '\\n' ';'; else printf unavailable; fi; "
            + runner_status
        )
        try:
            completed = self.ssh(command, timeout=30)
        except Exception as exc:
            from factor_service.research.errors import error_payload
            return {**self.node.public(), "online": False, "error": str(exc), **error_payload(exc)}
        if completed.returncode != 0:
            return {
                **self.node.public(), "online": False,
                "error": completed.stderr.strip() or "SSH连接失败",
            }
        parsed = _parse_key_values(completed.stdout)
        gpus = []
        if parsed.get("gpu") and parsed["gpu"] != "unavailable":
            for line in parsed["gpu"].split(";"):
                parts = [item.strip() for item in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "name": parts[0], "util": _int(parts[1]),
                        "mem_used_mb": _int(parts[2]), "mem_total_mb": _int(parts[3]),
                        "temp_c": _int(parts[4]),
                    })
        containers = []
        for line in str(parsed.get("containers") or "").split(";"):
            if self.node.runner == "direct_python" and line.strip():
                containers.append({"name": line.strip(), "status": "running"})
            elif "|" in line:
                name, status = line.split("|", 1)
                containers.append({"name": name, "status": status})
        mem = [_int(item) for item in str(parsed.get("mem") or "").split(",")]
        cgroup_mem = [
            _int(item)
            for item in str(parsed.get("mem_cgroup") or "").split(",")
        ]
        mem_total_mb, mem_used_mb = _effective_memory_usage_mb(mem, cgroup_mem)
        disk = [_int(item) for item in str(parsed.get("disk") or "").split(",")]
        return {
            **self.node.public(), "online": True,
            "cpu_cores": _int(parsed.get("cpu_cores")),
            "cpu_load": _float(parsed.get("load")),
            "mem_total_mb": mem_total_mb,
            "mem_used_mb": mem_used_mb,
            "disk_total_kb": disk[0] if disk else 0,
            "disk_used_kb": disk[1] if len(disk) > 1 else 0,
            "gpus": gpus, "containers": containers, "runner": self.node.runner,
            "training_active": bool(containers),
        }

    def ssh(
        self,
        command: str,
        *,
        timeout: int,
        cancellation: CancellationToken | None = None,
    ) -> subprocess.CompletedProcess[str]:
        completed = self._run(
            [*self._auth_prefix(), *self._ssh_base(), command],
            timeout=timeout, cancellation=cancellation,
        )
        self._check_ssh_failure(completed)
        return completed

    def _check_ssh_failure(self, completed):
        from factor_service.research.errors import ssh_error_from_result
        error = ssh_error_from_result(self.node.node_id, completed.returncode, completed.stderr)
        if error is not None:
            raise error

    def push(
        self,
        source: Path,
        remote_path: str,
        *,
        directory: bool = False,
        delete: bool = False,
        checksum: bool = False,
        cancellation: CancellationToken | None = None,
    ) -> None:
        # macOS ships openrsync 2.6.9, which does not support the newer
        # --protect-args flag. Node configuration is validated before it reaches
        # this transport and subprocess receives an argv list, so no shell
        # interpolation is involved here.
        # Parquet and model bundles are already compressed. Recompressing them
        # with rsync's -z consumes a CPU core and slows large AutoDL transfers.
        args = [*self._auth_prefix(), "rsync", "-a", "--partial"]
        if delete:
            args.append("--delete")
        if checksum:
            args.append("--checksum")
        if directory:
            args.extend(["--exclude", "__pycache__", "--exclude", "*.pyc"])
        args.extend(["-e", shlex.join(self._ssh_base(include_target=False))])
        local = str(source) + ("/" if directory else "")
        remote = f"{self._target()}:{remote_path}" + ("/" if directory else "")
        completed = self._run(args + [local, remote], timeout=3600, cancellation=cancellation)
        self._check_ssh_failure(completed)
        if completed.returncode != 0:
            raise RetryableJobError(f"推送远程文件失败: {completed.stderr.strip()[-1000:]}")

    def pull(
        self,
        remote_path: str,
        destination: Path,
        *,
        directory: bool = False,
        cancellation: CancellationToken | None = None,
    ) -> None:
        destination.mkdir(parents=True, exist_ok=True) if directory else destination.parent.mkdir(parents=True, exist_ok=True)
        args = [
            *self._auth_prefix(), "rsync", "-a", "--partial",
            "-e", shlex.join(self._ssh_base(include_target=False)),
        ]
        remote = f"{self._target()}:{remote_path}" + ("/" if directory else "")
        local = str(destination) + ("/" if directory else "")
        completed = self._run(args + [remote, local], timeout=3600, cancellation=cancellation)
        self._check_ssh_failure(completed)
        if completed.returncode != 0:
            raise RetryableJobError(f"拉取远程产物失败: {completed.stderr.strip()[-1000:]}")

    def _run(
        self,
        args: list[str],
        *,
        timeout: int,
        cancellation: CancellationToken | None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop(REMOTE_NODE_SECRET_KEY_ENV, None)
        # Do not forward a workstation-specific locale through SSH.  Minimal
        # remote images (including MatPool) often do not install en_US.UTF-8;
        # OpenSSH then emits a misleading shell warning before the actual
        # runner diagnostics.  C.UTF-8 is available on the supported Linux
        # images and keeps Python/rsync output UTF-8 safe.
        environment["LANG"] = "C.UTF-8"
        environment["LC_ALL"] = "C.UTF-8"
        if self.node.ssh_password:
            environment["SSHPASS"] = self.node.ssh_password
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment,
        )
        deadline = time.monotonic() + max(1, int(timeout))
        try:
            while True:
                if cancellation is not None:
                    cancellation.checkpoint()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.terminate()
                    from factor_service.research.errors import NodeSSHConnectionError
                    raise NodeSSHConnectionError(f"节点 {self.node.node_id} SSH命令或传输超时")
                try:
                    # Short communicate calls drain both pipes while preserving
                    # cancellation checks for long-running SSH and rsync commands.
                    stdout, stderr = process.communicate(
                        timeout=min(0.2, remaining),
                    )
                except subprocess.TimeoutExpired:
                    continue
                return subprocess.CompletedProcess(
                    args, process.returncode, stdout, stderr,
                )
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise

    def _auth_prefix(self) -> list[str]:
        return ["sshpass", "-e"] if self.node.ssh_password else []

    def _ssh_base(self, *, include_target: bool = True) -> list[str]:
        args = [
            "ssh", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3", "-p", str(self.node.port),
            "-o", "ControlMaster=auto", "-o", "ControlPersist=60",
            "-o", f"ControlPath={self._control_path}",
        ]
        if self.node.known_hosts is not None:
            args.extend([
                "-o", "StrictHostKeyChecking=yes",
                "-o", f"UserKnownHostsFile={self.node.known_hosts}",
            ])
        else:
            args.extend(["-o", "StrictHostKeyChecking=accept-new"])
        if self._ssh_key_path is not None:
            args.extend(["-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-i", str(self._ssh_key_path)])
        else:
            args.extend(["-o", "BatchMode=no"])
            if self.node.ssh_password:
                # Agent/default keys can exhaust sshd MaxAuthTries before
                # sshpass receives a password prompt, including rsync's SSH.
                args.extend([
                    "-o", "PubkeyAuthentication=no",
                    "-o", "PreferredAuthentications=password,keyboard-interactive",
                    "-o", "NumberOfPasswordPrompts=1",
                ])
        if include_target:
            args.append(self._target())
        return args

    def _target(self) -> str:
        return f"{self.node.user}@{self.node.host}"

    def _ensure_client_tools(self) -> None:
        required = ["ssh", "rsync", *( ["sshpass"] if self.node.ssh_password else [])]
        missing = [name for name in required if shutil.which(name) is None]
        if missing:
            raise RuntimeError("本机缺少远程训练命令: " + ", ".join(missing))
        if not self.node.credentials_available():
            raise ValueError(f"远程节点{self.node.node_id}认证信息不可用")

    def _materialize_private_key(self, value: str) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="alphablocks-ssh-", delete=False,
        ) as handle:
            handle.write(value.rstrip("\n") + "\n")
            path = Path(handle.name)
        path.chmod(0o600)
        self._ssh_key_path = path
        self._ssh_key_cleanup = weakref.finalize(self, path.unlink, missing_ok=True)


def _close_control_session(path, target, port, cleanup):
    try:
        if path.exists():
            environment = os.environ.copy()
            environment.pop(REMOTE_NODE_SECRET_KEY_ENV, None)
            environment.pop("SSHPASS", None)
            # -O exit is a local control-socket operation; it never logs in again.
            subprocess.run(["ssh", "-S", str(path), "-O", "exit", "-p", str(port), target],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=3, check=False, env=environment)
    except (OSError, subprocess.TimeoutExpired):
        pass  # An idle orphan also expires via ControlPersist.
    finally:
        cleanup()


class RemoteResearchExecutor:
    def __init__(self, settings: Settings, node: RemoteNode) -> None:
        self.settings = settings
        self.node = node
        self.transport = RemoteTransport(node)
        self.lifecycle = autodl_client(node) if node.lifecycle_provider else None
        self.snapshot_store = DatasetSnapshotStore(
            settings.model_artifacts_root, archive=archive_for_settings(settings),
        )

    def train(
        self,
        job: dict[str, Any],
        work_dir: Path,
        *,
        cancellation: CancellationToken,
        progress: Callable[[str, int, dict[str, Any]], None],
    ) -> TrainingResult:
        self._remote_resource_details = {}
        self._last_remote_training_progress = {'stage': 'training', 'percent': 63, 'details': {}}
        job_id = str(job["job_id"])
        attempt = max(1, int(job.get("attempt_count") or 1))
        remote_root = f"{self.node.work_dir}/runs/{job_id}/attempt-{attempt:03d}"
        cache_root = f"{self.node.work_dir}/cache"
        container = f"ab-research-{job_id[-32:]}-{attempt:03d}"
        progress("remote_materializing_dataset", 4, {"node_id": self.node.node_id})
        snapshot = self.snapshot_store.get_or_create(
            job, work_dir, None, builder_factory=lambda: DatasetBuilder(self.settings),
            cancellation=cancellation, progress=progress,
        )
        progress("remote_dataset_staged", 56, {
            "node_id": self.node.node_id,
            "dataset_hash": job["dataset_hash"],
            "snapshot_reused": snapshot.reused,
        })
        cancellation.checkpoint()
        remote_ready = False
        runner_started = False
        lifecycle_attempted = False
        cache_acquisition_attempted = False
        dataset_cache_guard = RemoteDatasetCache(
            self.node, self.transport, getattr(self.snapshot_store, "archive", None), job_id,
        )
        try:
            lifecycle_attempted = self.lifecycle is not None
            self._prepare_lifecycle(cancellation=cancellation, progress=progress)
            # AutoDL can update the resolved SSH endpoint during startup.
            dataset_cache_guard.node = self.node
            dataset_cache_guard.transport = self.transport
            if getattr(self, "recoverable_cache_jobs", None):
                dataset_cache_guard.recover_previous_jobs(self.recoverable_cache_jobs, cancellation)
            cache_acquisition_attempted = True
            dataset_cache_guard.acquire(cancellation)
            remote_ready = True
            progress("remote_preparing", 60, {"node_id": self.node.node_id})
            source_root = Path(__file__).resolve().parents[1]
            source_fingerprint = _source_fingerprint(source_root)
            source_cache = f"{cache_root}/source/{source_fingerprint}"
            dataset_cache = f"{cache_root}/datasets/{job['dataset_hash']}"
            mkdir = " ".join(shlex.quote(path) for path in (
                remote_root,
                f"{remote_root}/source",
                f"{remote_root}/artifacts/datasets",
                f"{remote_root}/work",
                f"{cache_root}/source",
                f"{cache_root}/datasets",
            ))
            completed = self.transport.ssh(
                f"mkdir -p {mkdir}", timeout=60, cancellation=cancellation,
            )
            if completed.returncode != 0:
                raise RetryableJobError(completed.stderr.strip() or "无法创建远程任务目录")

            descriptor = work_dir / "remote_job.json"
            descriptor.write_text(
                json.dumps(job, ensure_ascii=False, sort_keys=True, default=str),
                encoding="utf-8",
            )
            source_cache_hit = self._remote_cache_ready(source_cache, cancellation)
            if not source_cache_hit:
                self.transport.push(
                    source_root, source_cache,
                    directory=True, delete=True, cancellation=cancellation,
                )
                self._mark_remote_cache_ready(source_cache, cancellation)
            dataset_cache_hit = dataset_cache_guard.prepare(
                snapshot, str(job["dataset_hash"]),
                cancellation=cancellation, progress=progress,
            )
            links = (
                f"rm -rf -- {shlex.quote(remote_root + '/source/factor_service')} "
                f"{shlex.quote(remote_root + '/artifacts/datasets/' + str(job['dataset_hash']))}; "
                f"ln -s {shlex.quote(source_cache)} "
                f"{shlex.quote(remote_root + '/source/factor_service')}; "
                f"ln -s {shlex.quote(dataset_cache)} "
                f"{shlex.quote(remote_root + '/artifacts/datasets/' + str(job['dataset_hash']))}"
            )
            linked = self.transport.ssh(
                links, timeout=60, cancellation=cancellation,
            )
            if linked.returncode != 0:
                raise RetryableJobError(
                    linked.stderr.strip() or "无法挂载远程训练缓存"
                )
            incremental = dict((job.get("config_json") or {}).get("incremental_training") or {})
            source_artifact = dict(incremental.get("source_artifact") or {})
            source_relative = str(source_artifact.get("relative_path") or "").strip()
            if source_relative:
                source_path = self.snapshot_store.artifacts.resolve(source_relative)
                remote_source = f"{remote_root}/artifacts/{source_relative}"
                remote_parent = str(PurePosixPath(remote_source).parent)
                created = self.transport.ssh(
                    f"mkdir -p {shlex.quote(remote_parent)}",
                    timeout=60, cancellation=cancellation,
                )
                if created.returncode != 0:
                    raise RetryableJobError("无法创建远程增量模型目录")
                self.transport.push(
                    source_path, remote_source, cancellation=cancellation,
                )
            self.transport.push(
                descriptor, f"{remote_root}/job.json", cancellation=cancellation,
            )
            progress("remote_snapshot_uploaded", 61, {
                "node_id": self.node.node_id,
                "dataset_hash": job["dataset_hash"],
                "dataset_cache_hit": dataset_cache_hit,
                "source_cache_hit": source_cache_hit,
            })

            cpu_cores = self._remote_cpu_cores(cancellation)
            thread_count = _effective_remote_thread_count(
                cpu_cores, _requested_job_threads(job),
            )
            progress("remote_resources_ready", 62, {
                "node_id": self.node.node_id,
                "cpu_cores": cpu_cores,
                "training_threads": thread_count,
                "gpu_enabled": bool(self.node.gpus and self.node.gpus != "0"),
                    "amp_enabled": bool(self.node.gpus and self.node.gpus != "0"),
                    "validation_sample_rows": _REMOTE_VALIDATION_SAMPLE_ROWS,
                    "train_metric_sample_rows": _REMOTE_TRAIN_METRIC_SAMPLE_ROWS,
            })
            if self.node.runner == "direct_python":
                command = self._direct_python_command(remote_root, thread_count)
            else:
                docker = [
                    "docker", "run", "-d", "--name", container,
                    "--label", "alphablocks.research=1",
                    "--label", f"alphablocks.job_id={job_id}",
                ]
                if self.node.gpus and self.node.gpus != "0":
                    docker.extend(["--gpus", self.node.gpus])
                docker.extend([
                    "-e", "PYTHONPATH=/opt/alphafactor",
                    "-e", f"OMP_NUM_THREADS={thread_count}",
                    "-e", f"MKL_NUM_THREADS={thread_count}",
                    "-e", f"ALPHA_TORCH_DEVICE={'cuda' if self.node.gpus and self.node.gpus != '0' else 'cpu'}",
                    "-e", f"ALPHA_MODEL_ACCELERATOR={'cuda' if self.node.gpus and self.node.gpus != '0' else 'cpu'}",
                    "-e", f"ALPHA_EFFECTIVE_NUM_THREADS={thread_count}",
                    "-e", "ALPHA_TORCH_AMP=1",
                    "-e", "ALPHA_TORCH_TF32=1",
                    "-e", "ALPHA_TORCH_AUTO_BATCH=1",
                    "-e", f"ALPHA_VALIDATION_SAMPLE_ROWS={_REMOTE_VALIDATION_SAMPLE_ROWS}",
                    "-e", f"ALPHA_TRAIN_METRIC_SAMPLE_ROWS={_REMOTE_TRAIN_METRIC_SAMPLE_ROWS}",
                    "-v", f"{remote_root}:/workspace",
                    "-v", f"{remote_root}/source:/opt/alphafactor:ro",
                    self.node.docker_image,
                    "python", "-m", "factor_service.research.remote_runner",
                    "/workspace/job.json", "/workspace/work",
                    "/workspace/remote_result.json", "/workspace/artifacts",
                    "/workspace/progress.jsonl",
                ])
                command = "docker rm -f {name} >/dev/null 2>&1 || true; {run}".format(
                    name=shlex.quote(container), run=shlex.join(docker),
                )
            runner_started = True
            launched = self.transport.ssh(
                command, timeout=120, cancellation=cancellation,
            )
            if launched.returncode != 0:
                raise RetryableJobError(
                    f"远程{self.node.runner}启动失败: {launched.stderr.strip()[-1000:]}"
                )
            progress("remote_training", 63, {
                "node_id": self.node.node_id,
                "runner": self.node.runner,
                **({"container": container} if self.node.runner == "docker" else {}),
            })
            if self.node.runner == "direct_python":
                self._wait_for_process(
                    remote_root, cancellation=cancellation, progress=progress,
                )
            else:
                self._wait_for_container(
                    container, remote_root, cancellation=cancellation, progress=progress,
                )
            progress("remote_downloading", 88, {"node_id": self.node.node_id})
            self.transport.pull(
                f"{remote_root}/work", work_dir,
                directory=True, cancellation=cancellation,
            )
            result_path = work_dir / "remote_result.json"
            self.transport.pull(
                f"{remote_root}/remote_result.json", result_path,
                cancellation=cancellation,
            )
            result = _load_remote_result(
                result_path, work_dir, self.settings.model_artifacts_root,
            )
            if self.node.cleanup_success:
                self.transport.ssh(
                    f"rm -rf -- {shlex.quote(remote_root)}",
                    timeout=120, cancellation=cancellation,
                )
            return result
        finally:
            ssh_failed = isinstance(sys.exc_info()[1], NodeSSHError)
            if ssh_failed:
                # The connection cannot prove remote ownership is idle. Do not
                # hammer failed credentials, release the lock, or power it off.
                try:
                    progress("remote_cleanup_pending", 63, {"node_id": self.node.node_id,
                             "job_id": job_id, "remote_root": remote_root,
                             "message": "SSH不可用，保留缓存占用锁；节点恢复后需确认旧任务已结束"})
                except Exception:
                    pass  # Preserve the original typed SSH error.
            else:
                if runner_started:
                    self._stop_remote_runner(container, remote_root)
                dataset_cache_guard.release_when_idle()
            # A failed lock acquisition must never shut down someone else's job.
            if not ssh_failed and lifecycle_attempted and (
                not cache_acquisition_attempted
                or (remote_ready and not dataset_cache_guard.owned)
            ):
                self._power_off_after_job(job, progress)

    def _remote_cache_ready(
        self, remote_path: str, cancellation: CancellationToken,
    ) -> bool:
        marker = shlex.quote(f"{remote_path}/.alphablocks-cache-complete")
        completed = self.transport.ssh(
            f"test -f {marker}", timeout=30, cancellation=cancellation,
        )
        return completed.returncode == 0

    def _mark_remote_cache_ready(
        self, remote_path: str, cancellation: CancellationToken,
    ) -> None:
        marker = shlex.quote(f"{remote_path}/.alphablocks-cache-complete")
        completed = self.transport.ssh(
            f"touch {marker}", timeout=30, cancellation=cancellation,
        )
        if completed.returncode != 0:
            raise RetryableJobError(
                completed.stderr.strip() or "无法写入远程缓存完成标记"
            )

    def _remote_cpu_cores(self, cancellation: CancellationToken) -> int:
        completed = self.transport.ssh(
            "nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || /usr/sbin/sysctl -n hw.logicalcpu 2>/dev/null || printf '4\\n'",
            timeout=30, cancellation=cancellation,
        )
        if completed.returncode != 0:
            return 4
        match = re.search(r"\d+", completed.stdout)
        return max(1, int(match.group(0))) if match else 4

    def _prepare_lifecycle(
        self,
        *,
        cancellation: CancellationToken,
        progress: Callable[[str, int, dict[str, Any]], None],
    ) -> None:
        if self.lifecycle is None:
            return
        progress("remote_checking_power", 57, {
            "node_id": self.node.node_id,
            "provider": self.node.lifecycle_provider,
        })
        cancellation.checkpoint()
        state = self.lifecycle.status()
        if state != "running":
            if not self.node.auto_start:
                raise RetryableJobError(
                    f"AutoDL实例未开机（当前状态: {state}），请先开机或启用自动开机",
                )
            progress("remote_powering_on", 58, {
                "node_id": self.node.node_id, "power_state": state,
            })
            self.lifecycle.power_on()
            state = self.lifecycle.wait_for_running(
                timeout_seconds=self.node.boot_timeout_minutes * 60,
                checkpoint=cancellation.checkpoint,
                on_status=lambda current: progress("remote_powering_on", 58, {
                    "node_id": self.node.node_id, "power_state": current,
                }),
            )
        snapshot = self.lifecycle.snapshot()
        self.node = node_with_autodl_endpoint(self.node, snapshot)
        self.transport = RemoteTransport(self.node)
        progress("remote_waiting_for_ssh", 59, {
            "node_id": self.node.node_id,
            "power_state": state,
            "host": self.node.host,
            "port": self.node.port,
        })
        deadline = time.monotonic() + self.node.boot_timeout_minutes * 60
        last_error = ""
        while time.monotonic() < deadline:
            cancellation.checkpoint()
            try:
                completed = self.transport.ssh(
                    "true", timeout=20, cancellation=cancellation,
                )
                if completed.returncode == 0:
                    return
                last_error = completed.stderr.strip()[-500:]
            except Exception as exc:
                last_error = str(exc)[-500:]
            time.sleep(3)
        raise TimeoutError(
            f"AutoDL实例已开机但SSH未就绪: {last_error or '连接超时'}",
        )

    def _power_off_after_job(
        self,
        job: dict[str, Any],
        progress: Callable[[str, int, dict[str, Any]], None],
    ) -> None:
        if self.lifecycle is None or not self.node.auto_stop:
            return
        if self._experiment_has_pending_remote_jobs(job):
            try:
                progress("remote_kept_alive", 89, {
                    "node_id": self.node.node_id,
                    "reason": "experiment_jobs_pending",
                })
            except Exception:
                pass
            return
        try:
            progress("remote_powering_off", 89, {
                "node_id": self.node.node_id,
            })
        except Exception:
            pass
        try:
            self.lifecycle.power_off()
        except Exception as exc:
            # A shutdown failure must be visible without retrying an already
            # completed and potentially expensive training run.
            try:
                progress("remote_power_off_failed", 89, {
                    "node_id": self.node.node_id,
                    "error": str(exc)[:500],
                })
            except Exception:
                pass

    def _experiment_has_pending_remote_jobs(self, job: dict[str, Any]) -> bool:
        config = dict(job.get("config_json") or {})
        experiment_id = str(
            dict(config.get("experiment") or {}).get("experiment_id") or ""
        ).strip()
        if not experiment_id:
            return False
        try:
            from factor_service.model_research_repository import (
                ModelResearchRepository,
            )

            jobs = ModelResearchRepository().list_jobs(
                experiment_id=experiment_id,
                limit=100,
            )
        except Exception:
            return False
        current_job_id = str(job.get("job_id") or "")
        for candidate in jobs:
            if str(candidate.get("job_id") or "") == current_job_id:
                continue
            if str(candidate.get("status") or "") not in {
                "queued", "leased", "running", "uploading",
            }:
                continue
            execution = dict(
                dict(candidate.get("config_json") or {}).get("execution") or {}
            )
            dataset_hash = str(candidate.get("dataset_hash") or "").strip().lower()
            if (
                str(execution.get("node_id") or "local") == self.node.node_id
                and self._local_dataset_snapshot_ready(dataset_hash)
            ):
                return True
        return False

    def _local_dataset_snapshot_ready(self, dataset_hash: str) -> bool:
        if not dataset_hash:
            return False
        root = self.snapshot_store.artifacts.root / "datasets" / dataset_hash
        return all((root / name).is_file() for name in (
            "dataset.parquet", "dataset_raw.parquet", "dataset_manifest.json",
        ))

    def _direct_python_command(self, remote_root: str, thread_count: int) -> str:
        gpu_enabled = bool(self.node.gpus and self.node.gpus != "0")
        paths = {
            "job": f"{remote_root}/job.json",
            "work": f"{remote_root}/work",
            "result": f"{remote_root}/remote_result.json",
            "artifacts": f"{remote_root}/artifacts",
            "progress": f"{remote_root}/progress.jsonl",
            "log": f"{remote_root}/runner.log",
            "pid": f"{remote_root}/runner.pid",
            "exit": f"{remote_root}/runner.exit",
        }
        runner = [
            "env",
            f"PYTHONPATH={remote_root}/source",
            f"OMP_NUM_THREADS={thread_count}",
            f"MKL_NUM_THREADS={thread_count}",
            f"ALPHA_TORCH_DEVICE={'cuda' if gpu_enabled else 'cpu'}",
            f"ALPHA_MODEL_ACCELERATOR={'cuda' if gpu_enabled else 'cpu'}",
            f"ALPHA_EFFECTIVE_NUM_THREADS={thread_count}",
            "ALPHA_TORCH_AMP=1",
            "ALPHA_TORCH_TF32=1",
            "ALPHA_TORCH_AUTO_BATCH=1",
            f"ALPHA_VALIDATION_SAMPLE_ROWS={_REMOTE_VALIDATION_SAMPLE_ROWS}",
            f"ALPHA_TRAIN_METRIC_SAMPLE_ROWS={_REMOTE_TRAIN_METRIC_SAMPLE_ROWS}",
            self.node.python_executable,
            "-m", "factor_service.research.remote_runner",
            paths["job"], paths["work"], paths["result"], paths["artifacts"],
            paths["progress"],
        ]
        inner = (
            f"{shlex.join(runner)}; code=$?; "
            f"printf '%s\\n' \"$code\" > {shlex.quote(paths['exit'])}; exit \"$code\""
        )
        # macOS does not ship the Linux setsid executable. Python's POSIX setsid
        # gives the shell wrapper the same dedicated, cancelable process group.
        detach = shlex.join([
            self.node.python_executable, "-c",
            "import os, sys; os.setsid(); os.execv('/bin/sh', ['sh', '-c', sys.argv[1]])",
            inner,
        ])
        return (
            f"rm -f {shlex.quote(paths['exit'])} {shlex.quote(paths['pid'])}; "
            f"nohup {detach} > {shlex.quote(paths['log'])} 2>&1 < /dev/null & "
            f"printf '%s\\n' \"$!\" > {shlex.quote(paths['pid'])}"
        )

    def _stop_remote_runner(self, container: str, remote_root: str) -> None:
        try:
            if self.node.runner == "direct_python":
                pid_path = shlex.quote(f"{remote_root}/runner.pid")
                command = (
                    f"if test -f {pid_path}; then pid=$(cat {pid_path}); "
                    "kill -- \"-$pid\" >/dev/null 2>&1 || "
                    "kill \"$pid\" >/dev/null 2>&1 || true; fi"
                )
            else:
                command = (
                    f"docker rm -f {shlex.quote(container)} >/dev/null 2>&1 || true"
                )
            self.transport.ssh(command, timeout=30)
        except Exception:
            pass

    def _wait_for_process(
        self,
        remote_root: str,
        *,
        cancellation: CancellationToken,
        progress: Callable[[str, int, dict[str, Any]], None],
    ) -> None:
        deadline = time.monotonic() + self.node.max_runtime_minutes * 60
        seen = 0
        exit_path = shlex.quote(f"{remote_root}/runner.exit")
        pid_path = shlex.quote(f"{remote_root}/runner.pid")
        while True:
            cancellation.checkpoint()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"远程训练超过{self.node.max_runtime_minutes}分钟上限",
                )
            def poll_process() -> tuple[int, subprocess.CompletedProcess[str]]:
                next_seen = self._forward_progress(
                    remote_root, seen,
                    cancellation=cancellation, progress=progress,
                )
                state = self.transport.ssh(
                    f"if test -f {exit_path}; then printf 'exited '; cat {exit_path}; "
                    f"elif test -f {pid_path} && kill -0 $(cat {pid_path}) 2>/dev/null; "
                    "then printf 'running -1'; else printf 'missing -1'; fi",
                    timeout=30, cancellation=cancellation,
                )
                if state.returncode != 0:
                    raise RetryableJobError(
                        "远程训练状态检查失败: " + state.stderr.strip()[-1000:],
                    )
                return next_seen, state

            seen, state = self._poll_with_recovery(
                poll_process,
                cancellation=cancellation,
                progress=progress,
            )
            values = state.stdout.strip().split()
            process_state = values[0] if values else "missing"
            exit_code = values[1] if len(values) > 1 else "-1"
            if process_state == "exited" and exit_code == "0":
                return
            if process_state in {"exited", "missing"}:
                _raise_remote_resource_failure(self.transport, remote_root, cancellation)
            if (
                process_state == "exited"
                and _remote_result_is_complete(
                    self.transport, remote_root, cancellation=cancellation,
                )
            ):
                # Qlib/MLflow may fail during interpreter teardown while trying
                # to inspect a non-git remote work directory. The result file is
                # written atomically before the packaged marker, so both are a
                # stronger completion signal than that late cleanup exit code.
                progress("remote_process_cleanup_warning", 88, {
                    "node_id": self.node.node_id,
                    "exit_code": exit_code,
                    "result_complete": True,
                })
                return
            if process_state in {"exited", "missing"}:
                logs = self.transport.ssh(
                    f"tail -n 200 {shlex.quote(remote_root + '/runner.log')} 2>&1 || true",
                    timeout=60,
                )
                raise RuntimeError(
                    f"远程训练进程异常结束(status={process_state}, exit={exit_code}):\n"
                    + logs.stdout[-20_000:]
                )
            time.sleep(2)

    def _wait_for_container(
        self,
        container: str,
        remote_root: str,
        *,
        cancellation: CancellationToken,
        progress: Callable[[str, int, dict[str, Any]], None],
    ) -> None:
        deadline = time.monotonic() + self.node.max_runtime_minutes * 60
        seen = 0
        while True:
            cancellation.checkpoint()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"远程训练超过{self.node.max_runtime_minutes}分钟上限",
                )
            def poll_container() -> tuple[int, subprocess.CompletedProcess[str]]:
                next_seen = self._forward_progress(
                    remote_root, seen,
                    cancellation=cancellation, progress=progress,
                )
                inspect = self.transport.ssh(
                    "docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' "
                    f"{shlex.quote(container)} 2>/dev/null || printf 'missing -1'",
                    timeout=30, cancellation=cancellation,
                )
                if inspect.returncode != 0:
                    raise RetryableJobError(
                        "远程训练状态检查失败: " + inspect.stderr.strip()[-1000:],
                    )
                return next_seen, inspect

            seen, inspect = self._poll_with_recovery(
                poll_container,
                cancellation=cancellation,
                progress=progress,
            )
            state = inspect.stdout.strip().split()
            status = state[0] if state else "missing"
            exit_code = state[1] if len(state) > 1 else "-1"
            if status in {"exited", "dead", "missing"}:
                if status == "exited" and exit_code == "0":
                    return
                _raise_remote_resource_failure(self.transport, remote_root, cancellation)
                if (
                    status == "exited"
                    and _remote_result_is_complete(
                        self.transport, remote_root, cancellation=cancellation,
                    )
                ):
                    progress("remote_process_cleanup_warning", 88, {
                        "node_id": self.node.node_id,
                        "exit_code": exit_code,
                        "result_complete": True,
                    })
                    return
                logs = self.transport.ssh(
                    f"docker logs --tail 200 {shlex.quote(container)} 2>&1 || true",
                    timeout=60,
                )
                raise RuntimeError(
                    f"远程训练容器异常结束(status={status}, exit={exit_code}):\n"
                    + logs.stdout[-20_000:]
                )
            time.sleep(2)

    def _poll_with_recovery(
        self,
        operation: Callable[[], Any],
        *,
        cancellation: CancellationToken,
        progress: Callable[[str, int, dict[str, Any]], None],
    ) -> Any:
        failure_started_at: float | None = None
        failure_count = 0
        while True:
            cancellation.checkpoint()
            try:
                result = operation()
            except (TimeoutError, RetryableJobError) as exc:
                now = time.monotonic()
                failure_started_at = failure_started_at or now
                failure_count += 1
                elapsed = now - failure_started_at
                if elapsed >= _REMOTE_POLL_RECOVERY_SECONDS:
                    raise RetryableJobError(
                        "远程训练监控连续失败"
                        f"{int(elapsed)}秒({failure_count}次): {exc}"
                    ) from exc
                progress("remote_poll_degraded", 63, {
                    "node_id": self.node.node_id,
                    "failure_count": failure_count,
                    "recovery_window_seconds": _REMOTE_POLL_RECOVERY_SECONDS,
                    "error": str(exc)[:500],
                })
                time.sleep(_REMOTE_POLL_RETRY_SECONDS)
                continue
            if failure_count:
                progress("remote_poll_recovered", 63, {
                    "node_id": self.node.node_id,
                    "failure_count": failure_count,
                    "degraded_seconds": round(
                        time.monotonic() - (failure_started_at or time.monotonic()),
                        3,
                    ),
                })
            return result

    def _forward_progress(
        self,
        remote_root: str,
        seen: int,
        *,
        cancellation: CancellationToken,
        progress: Callable[[str, int, dict[str, Any]], None],
    ) -> int:
        start_line = max(1, int(seen) + 1)
        progress_result = self.transport.ssh(
            f"tail -n +{start_line} -- "
            f"{shlex.quote(remote_root + '/progress.jsonl')} 2>/dev/null || true",
            timeout=30, cancellation=cancellation,
        )
        lines = [line for line in progress_result.stdout.splitlines() if line.strip()]
        consumed = 0
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                break
            if item.get("stage") == "resource_heartbeat":
                # A supervisor heartbeat must not move training backwards if it
                # raced with a newer child progress event.
                self._remote_resource_details = dict(item.get("details") or {})
                item = getattr(self, "_last_remote_training_progress", {
                    "stage": "training", "percent": 63, "details": {},
                })
                details = {**dict(item.get("details") or {}), **self._remote_resource_details}
            else:
                self._last_remote_training_progress = item
                details = {**getattr(self, "_remote_resource_details", {}), **dict(item.get("details") or {})}
            progress(
                f"remote.{item.get('stage') or 'training'}",
                min(88, max(63, int(item.get("percent") or 0))),
                {
                    "node_id": self.node.node_id,
                    **details,
                },
            )
            consumed += 1
        return seen + consumed


def _raise_remote_resource_failure(
    transport: RemoteTransport, remote_root: str,
    cancellation: CancellationToken | None,
) -> None:
    path = shlex.quote(f"{remote_root}/remote_failure.json")
    result = transport.ssh(
        f"if test -s {path}; then cat {path}; fi",
        timeout=30, cancellation=cancellation,
    )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    error_type = {
        "node_memory_budget_exceeded": NodeMemoryBudgetExceeded,
        "node_out_of_memory": NodeOutOfMemory,
        "node_resource_unavailable": NodeResourceUnavailable,
    }.get(payload.get("error_code"))
    if error_type is not None:
        raise error_type(str(payload.get("message") or payload["error_code"]))


def _remote_result_is_complete(
    transport: RemoteTransport,
    remote_root: str,
    *,
    cancellation: CancellationToken | None,
) -> bool:
    result_path = shlex.quote(f"{remote_root}/remote_result.json")
    progress_path = shlex.quote(f"{remote_root}/progress.jsonl")
    completed = transport.ssh(
        f"test -s {result_path} && grep -q '\"stage\": \"remote_packaged\"' "
        f"{progress_path}",
        timeout=30,
        cancellation=cancellation,
    )
    return completed.returncode == 0


def _source_fingerprint(source_root: Path) -> str:
    digest = sha256()
    files = sorted(
        path for path in source_root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    for path in files:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def _requested_job_threads(job: dict[str, Any]) -> int:
    model = dict((job.get("config_json") or {}).get("model") or {})
    candidates = [dict(model.get("params") or {}).get("num_threads")]
    candidates.extend(
        dict(item.get("params") or {}).get("num_threads")
        for item in list(model.get("base_models") or [])
        if isinstance(item, dict)
    )
    values = []
    for value in candidates:
        try:
            values.append(max(1, int(value)))
        except (TypeError, ValueError):
            continue
    return max(values, default=4)


def _effective_remote_thread_count(cpu_cores: int, requested: int) -> int:
    available = max(1, int(cpu_cores or 1))
    configured = max(1, int(requested or 1))
    automatic = max(4, available // 2)
    return min(available, _REMOTE_MAX_THREADS, max(configured, automatic))


def _effective_memory_usage_mb(
    host_memory_mb: list[int], cgroup_memory_bytes: list[int],
) -> tuple[int, int]:
    host_total = host_memory_mb[0] if host_memory_mb else 0
    host_used = host_memory_mb[1] if len(host_memory_mb) > 1 else 0
    cgroup_limit = cgroup_memory_bytes[0] if cgroup_memory_bytes else 0
    cgroup_used = cgroup_memory_bytes[1] if len(cgroup_memory_bytes) > 1 else 0
    limit_mb = cgroup_limit // (1024 * 1024) if cgroup_limit > 0 else 0
    used_mb = cgroup_used // (1024 * 1024) if cgroup_used > 0 else 0
    if limit_mb and (not host_total or limit_mb < host_total):
        return limit_mb, min(used_mb, limit_mb)
    return host_total, host_used


def _normalize_node(source: dict[str, Any]) -> RemoteNode:
    node_id = str(source.get("id") or "").strip()
    host = str(source.get("host") or "").strip()
    user = str(source.get("user") or "root").strip()
    work_dir = str(source.get("work_dir") or "/root/alphablocks-research").strip().rstrip("/")
    runner = str(source.get("runner") or "docker").strip().lower()
    python_executable = str(source.get("python_executable") or "python").strip()
    image = str(source.get("docker_image") or "alphafactor-research:latest").strip()
    configured_gpus = str(source.get("gpus") or "").strip()
    configured_compute_type = str(
        source.get("compute_type") or ""
    ).strip().lower()
    compute_type = configured_compute_type or (
        "cpu" if configured_gpus == "0" else "gpu"
    )
    gpus = configured_gpus or ("0" if compute_type == "cpu" else "all")
    authentication_type = str(
        source.get("authentication_type") or ""
    ).strip().lower()
    ssh_private_key = str(source.get("ssh_private_key") or "")
    ssh_password = str(source.get("ssh_password") or "")
    if not authentication_type:
        authentication_type = "password" if ssh_password else "ssh_private_key"
    lifecycle_provider = str(
        source.get("lifecycle_provider") or ""
    ).strip().lower()
    if lifecycle_provider in {"manual", "none"}:
        lifecycle_provider = ""
    instance_uuid = str(source.get("instance_uuid") or "").strip()
    api_token = str(source.get("api_token") or "").strip()
    if not _IDENTIFIER.fullmatch(node_id) or node_id == "local":
        raise ValueError(f"远程训练节点ID无效: {node_id}")
    if not _HOST.fullmatch(host):
        raise ValueError(f"远程训练节点host无效: {host}")
    if not _USER.fullmatch(user):
        raise ValueError(f"远程训练节点user无效: {user}")
    if not work_dir.startswith("/") or any(part in {"", ".", ".."} for part in PurePosixPath(work_dir).parts[1:]):
        raise ValueError(f"远程训练work_dir必须是安全绝对路径: {work_dir}")
    if runner not in _REMOTE_RUNNERS:
        raise ValueError("远程训练runner只允许docker或direct_python")
    if runner == "direct_python" and (
        not python_executable.startswith("/")
        or any(
            part in {"", ".", ".."}
            for part in PurePosixPath(python_executable).parts[1:]
        )
    ):
        raise ValueError("direct_python的python_executable必须是安全绝对路径")
    if not _IMAGE.fullmatch(image):
        raise ValueError(f"远程训练docker_image无效: {image}")
    if not _GPU_SPEC.fullmatch(gpus):
        raise ValueError(f"远程训练gpus配置无效: {gpus}")
    if compute_type not in _COMPUTE_TYPES:
        raise ValueError("远程训练compute_type只允许gpu或cpu")
    if (compute_type == "cpu") != (gpus == "0"):
        raise ValueError("CPU节点必须禁用GPU挂载，GPU节点必须配置GPU挂载")
    if authentication_type not in _AUTHENTICATION_TYPES:
        raise ValueError("远程训练认证只允许ssh_private_key或password")
    if "\x00" in ssh_password:
        raise ValueError("远程训练SSH密码包含无效NUL字符")
    if ssh_private_key and (
        "-----BEGIN " not in ssh_private_key
        or "PRIVATE KEY-----" not in ssh_private_key
        or "\x00" in ssh_private_key
    ):
        raise ValueError("远程训练SSH私钥格式无效")
    if lifecycle_provider not in _LIFECYCLE_PROVIDERS:
        raise ValueError("远程节点生命周期只支持manual或autodl_pro")
    if lifecycle_provider == "autodl_pro":
        instance_uuid = validate_instance_uuid(instance_uuid)
        if api_token:
            api_token = validate_api_token(api_token)
    else:
        instance_uuid = ""
        api_token = ""
    port = int(source.get("port") or 22)
    if not 1 <= port <= 65535:
        raise ValueError("远程训练SSH端口必须在1到65535之间")
    known_hosts_text = str(source.get("known_hosts") or "").strip()
    known_hosts = Path(known_hosts_text).expanduser().resolve() if known_hosts_text else None
    return RemoteNode(
        node_id=node_id,
        name=str(source.get("name") or node_id).strip()[:80],
        host=host,
        port=port,
        user=user,
        work_dir=work_dir,
        runner=runner,
        python_executable=python_executable,
        docker_image=image,
        compute_type=compute_type,
        gpus=gpus,
        authentication_type=authentication_type,
        ssh_private_key=(
            ssh_private_key if authentication_type == "ssh_private_key" else ""
        ),
        ssh_password=(ssh_password if authentication_type == "password" else ""),
        known_hosts=known_hosts,
        enabled=bool(source.get("enabled", True)),
        cleanup_success=bool(source.get("cleanup_success", True)),
        max_runtime_minutes=max(10, min(int(source.get("max_runtime_minutes") or 240), 1440)),
        lifecycle_provider=lifecycle_provider,
        instance_uuid=instance_uuid,
        api_token=api_token,
        auto_start=(
            bool(source.get("auto_start", False))
            if lifecycle_provider else False
        ),
        auto_stop=(
            bool(source.get("auto_stop", False))
            if lifecycle_provider else False
        ),
        boot_timeout_minutes=max(
            2, min(int(source.get("boot_timeout_minutes") or 15), 60),
        ),
        known_hosts_config=known_hosts_text,
    )


def autodl_client(node: RemoteNode) -> AutoDLProClient:
    if node.lifecycle_provider != "autodl_pro":
        raise ValueError(f"远程节点{node.node_id}未启用AutoDL Pro API")
    return AutoDLProClient(node.instance_uuid, node.api_token)


def node_with_autodl_endpoint(
    node: RemoteNode, snapshot: dict[str, Any],
) -> RemoteNode:
    host = str(snapshot.get("proxy_host") or "").strip()
    try:
        port = int(snapshot.get("ssh_port") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("AutoDL实例详情缺少有效SSH端口") from exc
    if not _HOST.fullmatch(host):
        raise ValueError("AutoDL实例详情缺少有效SSH地址")
    if not 1 <= port <= 65535:
        raise ValueError("AutoDL实例详情缺少有效SSH端口")
    return replace(node, host=host, port=port)


def _load_remote_result(
    path: Path, work_dir: Path, artifact_root: Path,
) -> TrainingResult:
    if not path.is_file():
        raise RuntimeError("远程训练没有返回结果描述文件")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "alphablocks.remote-training-result.v1":
        raise RuntimeError("远程训练结果协议版本不受支持")
    artifacts = []
    for item in payload.get("artifacts") or []:
        kind = str((item or {}).get("kind") or "")
        resolved = _resolve_scoped_path(item, work_dir, artifact_root)
        if not kind or not resolved.is_file():
            raise RuntimeError(f"远程训练产物不完整: {kind or 'unknown'}")
        artifacts.append((kind, resolved))
    predictions = _resolve_scoped_path(
        payload.get("predictions") or {}, work_dir, artifact_root,
    )
    if not predictions.is_file():
        raise RuntimeError("远程训练预测文件不存在")
    return TrainingResult(
        result=dict(payload.get("result") or {}),
        artifacts=artifacts,
        predictions_path=predictions,
    )


def _resolve_scoped_path(source: dict[str, Any], work_dir: Path, artifact_root: Path) -> Path:
    scope = str(source.get("scope") or "")
    root = work_dir.resolve() if scope == "work" else artifact_root.resolve() if scope == "artifact_root" else None
    if root is None:
        raise RuntimeError("远程训练产物scope无效")
    relative = PurePosixPath(str(source.get("path") or ""))
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError("远程训练产物路径无效")
    target = (root / Path(*relative.parts)).resolve()
    if root not in target.parents:
        raise RuntimeError("远程训练产物路径越界")
    return target


def _parse_key_values(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in str(value or "").splitlines():
        if "=" in line:
            key, item = line.split("=", 1)
            result[key.strip()] = item.strip()
    return result


def _int(value: Any) -> int:
    try:
        return int(float(str(value or 0).strip()))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(str(value or 0).strip())
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "RemoteNode", "RemoteResearchExecutor", "RemoteTransport",
    "autodl_client", "execution_nodes", "get_remote_node", "load_remote_nodes",
    "node_with_autodl_endpoint", "remote_node_storage_payload",
]
