from __future__ import annotations

import json
import re
from typing import Any

import requests


class JobError(RuntimeError):
    """Base class for errors whose retry semantics are known."""

    retryable = False
    code = "job_error"


class PermanentJobError(JobError):
    retryable = False
    code = "permanent_error"


class RetryableJobError(JobError):
    retryable = True
    code = "retryable_error"


class JobCanceled(PermanentJobError):
    code = "canceled"


class TrainingTimeout(PermanentJobError):
    code = "training_timeout"


class NodeMemoryBudgetExceeded(PermanentJobError):
    code = "node_memory_budget_exceeded"


class NodeOutOfMemory(PermanentJobError):
    code = "node_out_of_memory"


class NodeResourceUnavailable(PermanentJobError):
    code = "node_resource_unavailable"


class NodeSSHError(PermanentJobError):
    """Retire this node for the attempt; only the coordinator may reassign work."""
    code = "node_ssh_error"


class NodeSSHAuthenticationError(NodeSSHError):
    code = "node_ssh_authentication_failed"


class NodeSSHConnectionError(NodeSSHError):
    code = "node_ssh_connection_failed"


class NodeExecutionUnavailable(PermanentJobError):
    code = "node_execution_unavailable"


class WorkerShutdown(RetryableJobError):
    code = "worker_shutdown"


def classify_exception(exc: BaseException) -> tuple[bool, str]:
    if isinstance(exc, JobError):
        return bool(exc.retryable), str(exc.code)
    if isinstance(exc, (requests.Timeout, requests.ConnectionError, TimeoutError, ConnectionError)):
        return True, "network_error"
    module = type(exc).__module__
    name = type(exc).__name__
    if module.startswith(('psycopg', 'psycopg_pool')):
        retryable = name in {'OperationalError', 'InterfaceError', 'ConnectionTimeout',
                             'PoolTimeout', 'SerializationFailure', 'DeadlockDetected'}
        return retryable, 'postgresql_transient' if retryable else 'postgresql_query'
    if module.startswith("clickhouse_connect"):
        retryable = name in {"OperationalError", "InterfaceError"}
        return retryable, "clickhouse_transient" if retryable else "clickhouse_query"
    if isinstance(exc, (KeyError, TypeError, ValueError, json.JSONDecodeError)):
        return False, "invalid_job_or_data"
    return False, "unexpected_error"


def error_payload(exc: BaseException) -> dict[str, Any]:
    retryable, code = classify_exception(exc)
    return {"retryable": retryable, "error_code": code, "error_type": type(exc).__name__}


def restore_job_error(payload: dict[str, Any]) -> JobError | None:
    classes = (JobError, PermanentJobError, RetryableJobError, JobCanceled,
               TrainingTimeout, NodeMemoryBudgetExceeded, NodeOutOfMemory,
               NodeResourceUnavailable, WorkerShutdown, NodeSSHError,
               NodeSSHAuthenticationError, NodeSSHConnectionError, NodeExecutionUnavailable)
    error_type = next((cls for cls in classes if cls.code == payload.get("error_code")), None)
    if error_type is None and payload.get("retryable") is True:
        error_type = RetryableJobError
    return error_type(str(payload.get("error") or "任务执行失败")) if error_type else None


def ssh_error_from_result(node_id: str, returncode: int, stderr: str) -> JobError | None:
    """Recognize transport failures, not remote Python/data/file-permission errors."""
    if returncode == 0:
        return None
    text = str(stderr).lower()
    if "host key verification failed" in text or "remote host identification has changed" in text:
        return PermanentJobError(f"节点 {node_id} SSH主机身份校验失败；未放宽校验，请核对节点身份")
    if any(term in text for term in ("too many authentication failures",
                                    "permission denied, please try again",
                                    "permission denied (publickey", "permission denied (password")):
        return NodeSSHAuthenticationError(
            f"节点 {node_id} SSH认证失败；本轮不再反复尝试认证，请检查账号或凭据")
    if returncode in (255, 10, 12, 30, 35) and any(term in text for term in (
        "connection reset", "connection refused", "connection timed out", "connection closed",
        "no route to host", "network is unreachable", "could not resolve hostname",
        "broken pipe", "kex_exchange_identification", "ssh_exchange_identification",
    )):
        return NodeSSHConnectionError(f"节点 {node_id} SSH连接中断或不可达；请检查网络及SSH服务")
    return None


def legacy_ssh_failure(message: str) -> NodeSSHError | None:
    """Recover only the old wrapper's explicit final SSH exception, not log mentions."""
    if not str(message).startswith("[unexpected_error] RuntimeError: 隔离模型进程失败(returncode="):
        return None
    match = re.search(r"An exception has been raised\[PermanentJobError: (.*?)\]\.\s", message, re.S)
    if match is None:
        return None
    detail = match.group(1)
    host = re.search(r"(?:from|to) ([\w.:-]+) port \d+", detail)
    error = ssh_error_from_result(host.group(1) if host else "远程节点", 255, detail)
    return error if isinstance(error, NodeSSHError) else None


__all__ = [
    "JobCanceled", "JobError", "PermanentJobError", "RetryableJobError",
    "TrainingTimeout", "WorkerShutdown", "classify_exception", "error_payload",
    "NodeMemoryBudgetExceeded", "NodeOutOfMemory", "NodeResourceUnavailable",
    "NodeSSHError", "NodeSSHAuthenticationError", "NodeSSHConnectionError",
    "NodeExecutionUnavailable", "restore_job_error", "ssh_error_from_result", "legacy_ssh_failure",
]
