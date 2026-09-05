from __future__ import annotations

import requests
import pytest

from factor_service.research.control import ResearchControlError
from factor_service.research.errors import PermanentJobError, classify_exception
from factor_service.research.errors import (
    NodeSSHAuthenticationError, NodeSSHConnectionError, legacy_ssh_failure,
    ssh_error_from_result, restore_job_error, error_payload,
)


@pytest.mark.parametrize('code,message,expected', [
    (255, 'Received disconnect from 10.0.0.5 port 22:2: Too many authentication failures', NodeSSHAuthenticationError),
    (5, 'Permission denied, please try again.', NodeSSHAuthenticationError),
    (255, 'user@node: Permission denied (publickey,password).', NodeSSHAuthenticationError),
    (255, 'ssh: connect to host node port 22: Connection refused', NodeSSHConnectionError),
    (12, 'Connection reset by peer', NodeSSHConnectionError),
    (255, 'Host key verification failed.', PermanentJobError),
])
def test_ssh_error_classification_and_round_trip(code, message, expected):
    error = ssh_error_from_result('selected-node', code, message)
    assert type(error) is expected
    assert 'selected-node' in str(error)
    assert error.retryable is False  # Bounded coordinator reassignment, no automatic parent loop.
    restored = restore_job_error(dict(error=str(error), **error_payload(error)))
    assert type(restored) is expected and str(restored) == str(error)


@pytest.mark.parametrize('code,message', [
    (0, 'Too many authentication failures'),
    (1, "PermissionError: [Errno 13] Permission denied: '/cache/dataset.parquet'"),
    (1, 'ValueError: invalid date range'),
    (23, 'rsync: open file: Permission denied (13)'),
    (1, 'unknown remote failure'),
])
def test_non_transport_failures_are_not_reassigned(code, message):
    assert ssh_error_from_result('node', code, message) is None


def test_legacy_wrapper_requires_explicit_final_ssh_exception():
    prefix = '[unexpected_error] RuntimeError: 隔离模型进程失败(returncode=1):\n'
    body = ('INFO qlib initialized\nAn exception has been raised[PermanentJobError: '
            'Received disconnect from 10.0.0.5 port 22:2: Too many authentication failures].\nTraceback')
    assert isinstance(legacy_ssh_failure(prefix + body), NodeSSHAuthenticationError)
    assert legacy_ssh_failure(body) is None
    assert legacy_ssh_failure(prefix + 'old log: Too many authentication failures\nValueError: bad date') is None
    assert legacy_ssh_failure(prefix + body.replace('PermanentJobError:', 'ValueError:')) is None
    assert legacy_ssh_failure(prefix + body.replace('Too many authentication failures', 'Permission denied: /data')) is None


def test_error_classification_is_explicit() -> None:
    assert classify_exception(requests.ConnectionError("offline"))[0] is True
    assert classify_exception(ResearchControlError(
        "offline", retryable=True, code="control_database_transient",
    ))[0] is True
    assert classify_exception(ResearchControlError(
        "bad job", retryable=False, code="model_research_rejected",
    ))[0] is False
    assert classify_exception(PermanentJobError("bad data"))[0] is False
    assert classify_exception(RuntimeError("unknown"))[0] is False


def test_postgresql_connection_failures_are_retryable_but_query_errors_are_not():
    from psycopg import OperationalError
    from psycopg.errors import ConnectionTimeout, UndefinedTable
    from psycopg_pool import PoolTimeout
    for error in (OperationalError('offline'), ConnectionTimeout('timeout'), PoolTimeout('busy')):
        assert classify_exception(error) == (True, 'postgresql_transient')
    assert classify_exception(UndefinedTable('schema missing')) == (False, 'postgresql_query')
