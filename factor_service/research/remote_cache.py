"""Controller-side single-dataset cache policy; MinIO remains authoritative."""
import json
import logging
from pathlib import Path
import re
import shlex
import time
from uuid import uuid4

from factor_service.research.dataset_archive import DATASET_FILES, file_sha256
from factor_service.research.errors import PermanentJobError, RetryableJobError, NodeSSHError, restore_job_error


logger = logging.getLogger(__name__)


class RemoteDatasetCache:
    def __init__(self, node, transport, archive, job_id: str) -> None:
        self.node, self.transport, self.archive = node, transport, archive
        self.token, self.job_id = uuid4().hex, job_id
        self.owned = False

    def _call(self, action, *, cancellation=None, **payload):
        request = dict(payload, action=action, token=self.token,
                       job_id=self.job_id, work_dir=self.node.work_dir)
        source = Path(__file__).with_name("node_dataset_cache.py").read_text(encoding="utf-8")
        source += "\nimport sys\nprint(json.dumps(operate(json.loads(sys.argv[1]))))\n"
        if self.node.runner == "direct_python":
            command = [self.node.python_executable, "-c", source, json.dumps(request)]
            shell_command = shlex.join(command)
        else:
            command = ["-v", f"{self.node.work_dir}:{self.node.work_dir}", self.node.docker_image,
                       "python", "-c", source, json.dumps(request)]
            shell_command = (f"mkdir -p -- {shlex.quote(self.node.work_dir)} && "
                             'docker run --rm --network none --user "$(id -u):$(id -g)" '
                             + shlex.join(command))
        completed = self.transport.ssh(shell_command, timeout=3600 if action in {"check", "commit"} else 60,
                                       cancellation=cancellation)
        if completed.returncode != 0:
            if action == "acquire" and "缓存被占用" in completed.stderr:
                raise RetryableJobError("节点数据集缓存正在被其他任务占用")
            raise PermanentJobError(completed.stderr.strip()[-1500:] or "节点缓存操作失败，数据已保留")
        return json.loads(completed.stdout)

    def _idle(self):
        status = self.transport.collect_status()
        if not status.get("online"):
            error = restore_job_error(status)
            if isinstance(error, NodeSSHError):
                raise error
            raise RetryableJobError("无法确认节点任务状态，禁止替换数据集缓存")
        if status.get("training_active"):
            raise RetryableJobError("节点仍有运行中的训练任务，禁止替换数据集缓存")
        processes = self.transport.ssh("ps -ax -o command=", timeout=30)
        if processes.returncode != 0:
            raise RetryableJobError("无法确认节点数据传输状态，禁止替换缓存")
        datasets = f"{self.node.work_dir}/cache/datasets"
        if any(datasets in line and re.search(r"(?:^|/|\s)rsync\s.*--server\b", line)
               for line in processes.stdout.splitlines()):
            raise RetryableJobError("节点仍在传输数据集，禁止替换缓存")

    def acquire(self, cancellation):
        self._idle()
        self._call("acquire", cancellation=cancellation)
        self.owned = True
        # Recheck after atomic acquisition, including runners from older versions.
        try:
            self._idle()
        except BaseException:
            self._call("release")
            self.owned = False
            raise

    def recover_previous_jobs(self, job_ids, cancellation):
        """Only the fenced parent coordinator may recover its own idle attempts.

        Never expire a lock by age or release a different parent's cache lock.
        Token comparison in the standalone helper fences a concurrent change.
        """
        self._idle()
        owner = self._call("owner", cancellation=cancellation)["owner"]
        if owner is None or owner.get("job_id") not in set(job_ids):
            return
        self._idle()
        saved_token = self.token
        try:
            self.token = owner["token"]
            self._call("release", cancellation=cancellation)
        finally:
            self.token = saved_token

    def prepare(self, snapshot, dataset_hash, *, cancellation, progress):
        inventory = self._call("inventory", cancellation=cancellation)["datasets"]
        obsolete = sorted(set(inventory) - {dataset_hash})
        # Never drop a potentially unique copy, even when a legacy cache predates
        # MinIO archival. Do all checks before allowing any remote deletion.
        for old_hash in obsolete:
            cancellation.checkpoint()
            record = self.archive.record(old_hash) if self.archive is not None else None
            if record is None:
                raise PermanentJobError(f"节点旧数据集{old_hash}尚未归档MinIO，已保留；请先归档再切换数据集")
            for identity in record["files_json"].values():
                self.archive.objects.verify_file(identity, checkpoint=cancellation.checkpoint)
        self._idle()
        result = self._call("prepare", dataset_hash=dataset_hash, archived_hashes=obsolete,
                            cancellation=cancellation)
        manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
        files = {}
        for name in DATASET_FILES:
            path = snapshot.dataset_path.parent / name
            digest = file_sha256(path) if name == "dataset_manifest.json" else manifest["files"][name]["sha256"]
            files[name] = {"sha256": digest, "size_bytes": path.stat().st_size}
        progress("remote_dataset_cache_prepared", 60, {
            "node_id": self.node.node_id, "removed_dataset_hashes": result["removed"],
            "dataset_hash": dataset_hash, "cache_capacity": 1,
        })
        hit = self._call("check", dataset_hash=dataset_hash, files=files,
                         cancellation=cancellation)["valid"]
        if not hit:
            self.transport.push(snapshot.dataset_path.parent,
                                f"{self.node.work_dir}/cache/datasets/{dataset_hash}",
                                directory=True, delete=True, checksum=True, cancellation=cancellation)
            self._call("commit", dataset_hash=dataset_hash, files=files, cancellation=cancellation)
        return hit

    def release_when_idle(self):
        if not self.owned:
            return
        # Cancellation may take the supervisor a few seconds to stop its children.
        # An unreachable/running node keeps its lock; never evict by elapsed time.
        deadline = time.monotonic() + 15
        while True:
            try:
                self._idle()
                self._call("release")
                self.owned = False
                return
            except Exception as exc:
                if isinstance(exc, NodeSSHError) or time.monotonic() >= deadline:
                    logger.warning("节点缓存占用锁已保留，需确认旧任务结束: %s: %s", self.node.node_id, exc)
                    return
                time.sleep(1)
