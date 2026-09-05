from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import tarfile
import threading
import time
from types import SimpleNamespace

import optuna
import pytest

from factor_service.research.distributed import DistributedTraining, extract_verified_archive
from factor_service.research.distributed_config import execution_spec
from factor_service.research.errors import (
    JobCanceled, NodeMemoryBudgetExceeded, NodeOutOfMemory, RetryableJobError,
    NodeSSHAuthenticationError, NodeSSHConnectionError, NodeExecutionUnavailable,
)
from factor_service.research.job import CancellationToken
from factor_service.research.trainer import TrainingResult


class Ledger:
    def __init__(self):
        self.rows = {}
        self.lock = threading.RLock()

    @contextmanager
    def coordinator(self, parent):
        yield

    def list(self, parent, kind=None):
        with self.lock:
            return [deepcopy(v) for k, v in sorted(self.rows.items())
                    if k[0] == parent and (kind is None or v['kind'] == kind)]

    def plan(self, parent, key, plan_hash, kind, payload):
        with self.lock:
            row = self.rows.setdefault((parent, key), dict(parent_job_id=parent, task_key=key,
                plan_hash=plan_hash, kind=kind, payload_json=deepcopy(payload), state='pending',
                attempt_count=0, result_json={}))
            if (row['plan_hash'], row['payload_json'], row['kind']) != (plan_hash, payload, kind):
                raise ValueError('frozen plan mismatch')
            return deepcopy(row)

    def start(self, parent, key, owner, node):
        with self.lock:
            row = self.rows[parent, key]
            row.update(state='running', owner_token=owner, node_id=node,
                       attempt_count=row['attempt_count'] + 1)
            return deepcopy(row)

    def heartbeat(self, parent, key, owner):
        assert self.rows[parent, key]['owner_token'] == owner

    def finish(self, parent, key, owner, state, result=None, error=''):
        with self.lock:
            row = self.rows[parent, key]
            assert row['owner_token'] == owner
            row.update(state=state, result_json=result or {}, error=error)


class Objects:
    def enabled_for(self, kind):
        return True

    def __init__(self, root):
        self.root = root
        root.mkdir()
        self.fail_verify = False
        self.downloads = 0

    def publish_file(self, **kwargs):
        path = self.root / kwargs['source_path'].name
        shutil.copyfile(kwargs['source_path'], path)
        return dict(object_uri=str(path), version_id='', sha256=kwargs['digest'], size_bytes=kwargs['size_bytes'])

    def verify_file(self, identity, **kwargs):
        if self.fail_verify:
            raise ValueError('archive verification failed')
        assert sha256(Path(identity['object_uri']).read_bytes()).hexdigest() == identity['sha256']

    def download_file(self, **kwargs):
        self.downloads += 1
        self.verify_file(dict(object_uri=kwargs['object_uri'], sha256=kwargs['digest']))
        shutil.copyfile(kwargs['object_uri'], kwargs['destination'])


@dataclass
class Node:
    node_id: str
    host: str
    port: int = 22
    user: str = 'test'
    auto_stop: bool = False


@pytest.fixture
def env(tmp_path):
    repo = Ledger()
    objects = Objects(tmp_path / 'objects')
    job = dict(job_id='parent', model_id='model', dataset_hash='a' * 64, config_json={
        'execution': execution_spec(dict(node_ids=['one', 'two'])),
        'model': {'kind': 'lightgbm', 'params': {}}, 'optuna': {'enabled': True}})
    shared = dict(active={}, maximum=0, calls=[], fail_once=False, cancel=False, executor_nodes=[], closed_nodes=[])
    lock = threading.Lock()

    class Executor:
        def __init__(self, settings, node):
            self.node = node
            shared['executor_nodes'].append(node.node_id)
            self.transport = SimpleNamespace(close=lambda: shared['closed_nodes'].append(node.node_id))

        def train(self, child, work, *, cancellation, progress):
            task = child['config_json']['_distributed_task']
            with lock:
                assert not shared['active'].get(self.node.node_id)
                shared['active'][self.node.node_id] = True
                shared['maximum'] = max(shared['maximum'], sum(shared['active'].values()))
                shared['calls'].append((self.node.node_id, deepcopy(task), child['attempt_count']))
            try:
                time.sleep(0.02)
                if shared['cancel']:
                    raise JobCanceled('requested')
                if self.node.node_id in shared.get('node_errors', {}):
                    raise shared['node_errors'][self.node.node_id]('node failure')
                if self.node.node_id in shared.get('memory_fail_nodes', set()):
                    raise shared.get('memory_error', NodeMemoryBudgetExceeded)('observed memory pressure')
                if shared.get('invalid_data'):
                    raise ValueError('invalid dataset')
                if shared['fail_once']:
                    shared['fail_once'] = False
                    raise RetryableJobError('connection interrupted')
                cancellation.checkpoint()
                path = work / 'trial.json'
                trial = dict(state='complete', params=task['params'],
                             value=float(task.get('trial_number', 0)), attrs={'fold_count': 3})
                path.write_text(json.dumps(trial))
                return TrainingResult({'trial': trial}, [('trial_result', path)], path)
            finally:
                with lock:
                    shared['active'][self.node.node_id] = False

    def controller(folder, token=None, nodes=None):
        run = DistributedTraining(SimpleNamespace(model_object_store=SimpleNamespace(enabled=True)),
            job, tmp_path / folder, cancellation=token, repository=repo, objects=objects,
            nodes=nodes or [Node('one', 'one.test'), Node('two', 'two.test')], executor_factory=Executor)
        run.bind_snapshot(SimpleNamespace(dataset_path=tmp_path / 'dataset.parquet'),
                          SimpleNamespace(artifacts=None, archive=None))
        return run
    return SimpleNamespace(repo=repo, objects=objects, shared=shared, controller=controller, job=job)


def study():
    return optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42, constant_liar=True))


def test_trial_parallelism_global_budget_and_durable_replay(env):
    with env.controller('first') as run:
        result = run.optimize(study(), 5, 'lightgbm', lambda *a: None)
    assert len(result.trials) == 5
    assert env.shared['maximum'] == 2
    assert len(env.shared['calls']) == 5
    assert sorted(env.shared['executor_nodes']) == ['one', 'two']
    assert sorted(env.shared['closed_nodes']) == ['one', 'two']
    assert len({json.dumps(t.params, sort_keys=True) for t in result.trials}) == 5
    assert all(t.user_attrs['fold_count'] == 3 for t in result.trials)
    with env.controller('resumed') as run:
        restored = run.optimize(study(), 5, 'lightgbm', lambda *a: None)
    assert [t.value for t in result.trials] == [t.value for t in restored.trials]
    assert len(env.shared['calls']) == 5  # No double-budget on restart.


def test_distributed_observation_persists_durations_and_reports_reuse(env):
    events = []
    with env.controller('observed') as run:
        run.progress = lambda s, p, d: events.append((s, p, d))
        run.optimize(study(), 3, 'lightgbm', lambda *a: None)
    rows = env.repo.list('parent')
    assert all(r['result_json']['execution_observation']['elapsed_seconds'] > 0 for r in rows)
    assert events[-1][2]['distributed']['counts']['optuna_trial']['completed'] == 3
    calls = len(env.shared['calls'])
    with env.controller('resumed_observed') as run:
        run.progress = lambda s, p, d: events.append((s, p, d))
        run.optimize(study(), 3, 'lightgbm', lambda *a: None)
    assert len(env.shared['calls']) == calls
    assert events[-1][2]['distributed']['counts']['optuna_trial']['reused'] == 3


def test_connection_retry_keeps_frozen_params_and_archives_before_complete(env):
    env.shared['fail_once'] = True
    with env.controller('retry') as run:
        run.optimize(study(), 1, 'lightgbm', lambda *a: None)
        row = env.repo.list('parent')[0]
        restored = run._run(row)
    assert len(env.shared['calls']) == 2
    assert env.shared['calls'][0][1]['params'] == env.shared['calls'][1][1]['params']
    assert row['attempt_count'] == 2 and row['state'] == 'complete'
    assert restored.result['trial']['params'] == row['payload_json']['params']
    assert env.objects.downloads == 1


@pytest.mark.parametrize('error', [NodeMemoryBudgetExceeded, NodeOutOfMemory])
def test_memory_failure_reassigns_to_busy_healthy_node_and_reuses_results(env, error):
    env.shared.update(memory_fail_nodes={'one'}, memory_error=error)
    events = []
    with env.controller('memory-failover') as run:
        run.progress = lambda *event: events.append(event)
        result = run.optimize(study(), 6, 'lightgbm', lambda *a: None)
    assert len(result.trials) == 6
    assert len(env.shared['calls']) == 7
    assert env.shared['maximum'] <= 2
    failed = [call for call in env.shared['calls'] if call[0] == 'one']
    assert len(failed) == 1  # The unhealthy node is retired for this parent attempt.
    original = failed[0][1]
    retry = [call for call in env.shared['calls'] if call[0] == 'two'
             and call[1]['trial_number'] == original['trial_number']]
    assert len(retry) == 1 and retry[0][1]['params'] == original['params']
    assert all(row['state'] == 'complete' for row in env.repo.list('parent'))
    reassigned = [e for e in events if e[0] == 'distributed_task_reassigned']
    assert reassigned[0][2]['from_node_id'] == 'one'
    assert reassigned[0][2]['node_id'] == 'two'
    with env.controller('memory-resumed') as run:
        run.optimize(study(), 6, 'lightgbm', lambda *a: None)
    assert len(env.shared['calls']) == 7


def test_memory_failover_can_try_more_than_two_selected_nodes(env):
    env.shared['memory_fail_nodes'] = {'one', 'two'}
    nodes = [Node(name, name + '.test') for name in ('one', 'two', 'three')]
    with env.controller('three-nodes', nodes=nodes) as run:
        run.optimize(study(), 1, 'lightgbm', lambda *a: None)
    assert [call[0] for call in env.shared['calls']] == ['one', 'two', 'three']
    assert env.repo.list('parent')[0]['state'] == 'complete'


@pytest.mark.parametrize('error_type', [NodeSSHAuthenticationError, NodeSSHConnectionError])
def test_ssh_failure_reassigns_without_repeating_bad_node_or_completed_trials(env, error_type):
    env.shared['node_errors'] = {'one': error_type}
    events = []
    with env.controller('ssh-failover') as run:
        run.progress = lambda *event: events.append(event)
        run.optimize(study(), 5, 'lightgbm', lambda *a: None)
    assert len(env.shared['calls']) == 6
    assert sum(call[0] == 'one' for call in env.shared['calls']) == 1
    original = next(call[1] for call in env.shared['calls'] if call[0] == 'one')
    assert any(call[0] == 'two' and call[1] == original for call in env.shared['calls'])
    assert any(event[0] == 'distributed_node_ssh_failed' for event in events)
    assert any(event[0] == 'distributed_task_reassigned' and event[2]['reason'] == 'ssh' for event in events)
    with env.controller('ssh-replay') as run:
        run.optimize(study(), 5, 'lightgbm', lambda *a: None)
    assert len(env.shared['calls']) == 6


def test_all_selected_nodes_unavailable_reports_mixed_causes(env):
    env.shared['node_errors'] = {'one': NodeSSHAuthenticationError, 'two': NodeMemoryBudgetExceeded}
    with env.controller('mixed-failure') as run:
        with pytest.raises(NodeExecutionUnavailable, match='暂无节点可承接'):
            run.optimize(study(), 4, 'lightgbm', lambda *a: None)
    assert len(env.shared['calls']) == 2


def test_final_model_is_reassigned_with_same_frozen_parameters(env):
    env.shared['node_errors'] = {'one': NodeSSHConnectionError}
    params = {'num_leaves': 31}
    with env.controller('final') as run:
        result = run.final_model(params, {'best_trial': 2})
    assert len(env.shared['calls']) == 2
    assert env.shared['calls'][0][1] == env.shared['calls'][1][1]
    assert result.result['trial']['params'] == params


def test_all_nodes_memory_failed_exits_without_waiting_forever(env):
    env.shared['memory_fail_nodes'] = {'one', 'two'}
    with env.controller('all-memory-failed') as run:
        with pytest.raises(NodeMemoryBudgetExceeded, match='暂无节点可承接'):
            run.optimize(study(), 5, 'lightgbm', lambda *a: None)
    assert len(env.shared['calls']) == 2
    assert all(row['state'] == 'failed' for row in env.repo.list('parent'))


def test_node_wait_remains_cancelable(env):
    token = CancellationToken()
    errors = []
    with env.controller('cancel-wait', token=token) as run:
        run.unavailable_nodes['one'] = 'memory pressure'
        held = run._take_node()

        def wait_for_node():
            try:
                run._take_node()
            except Exception as exc:
                errors.append(exc)

        waiter = threading.Thread(target=wait_for_node)
        waiter.start()
        token.cancel('user cancellation')
        waiter.join(timeout=2)
        assert not waiter.is_alive()
        run._return_node(held)
    assert len(errors) == 1 and isinstance(errors[0], JobCanceled)


def test_invalid_data_is_not_misclassified_as_memory_or_reassigned(env):
    env.shared['invalid_data'] = True
    with env.controller('invalid') as run:
        with pytest.raises(ValueError, match='invalid dataset'):
            run.optimize(study(), 1, 'lightgbm', lambda *a: None)
        assert not run.unavailable_nodes
    assert len(env.shared['calls']) == 1


@pytest.mark.parametrize('error_type', [NodeMemoryBudgetExceeded, NodeSSHAuthenticationError, NodeSSHConnectionError])
def test_window_memory_failover_preserves_window_identity(env, tmp_path, error_type):
    env.shared['node_errors'] = {'one': error_type}
    with env.controller('windows') as run:
        base = run.factory

        class WindowExecutor(base):
            def train(self, child, work, **kwargs):
                original = super().train(child, work, **kwargs)
                task = child['config_json']['_distributed_task']
                bundle = work / 'window.tar.gz'
                with tarfile.open(bundle, 'w:gz') as archive:
                    info = tarfile.TarInfo(f"window-{task['window']}.json")
                    content = json.dumps(task['segments']).encode()
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
                return TrainingResult({'window': task['window']}, [('window', bundle)], original.predictions_path)

        run.factory = WindowExecutor
        tasks = [{'alphablocks': {'window': n}, 'dataset': {'kwargs': {'segments': {'test': [str(n), str(n + 1)]}}}}
                 for n in range(3)]
        run.windows(tasks, {'num_leaves': 31}, tmp_path / 'series')
    assert all((tmp_path / 'series' / f'window-{n}.json').exists() for n in range(3))
    assert all(row['state'] == 'complete' for row in env.repo.list('parent'))
    assert len(env.shared['calls']) == 4


def test_archive_verification_failure_is_not_complete_and_can_resume(env):
    env.objects.fail_verify = True
    with env.controller('failed') as run, pytest.raises(ValueError, match='archive verification'):
        run.optimize(study(), 1, 'lightgbm', lambda *a: None)
    row = env.repo.list('parent')[0]
    assert row['state'] == 'failed'
    params = row['payload_json']['params']
    env.objects.fail_verify = False
    with env.controller('retry') as run:
        result = run.optimize(study(), 1, 'lightgbm', lambda *a: None)
    assert result.trials[0].params == params
    assert len(env.shared['calls']) == 2


def test_cancellation_is_not_retried_and_node_slot_is_released(env):
    env.shared['cancel'] = True
    with env.controller('cancel') as run:
        with pytest.raises(JobCanceled):
            run.optimize(study(), 1, 'lightgbm', lambda *a: None)
        assert run.available.qsize() == 2
    assert len(env.shared['calls']) == 1


def test_changed_config_does_not_reuse_old_trials(env):
    with env.controller('first') as run:
        run.optimize(study(), 1, 'lightgbm', lambda *a: None)
    env.job['config_json']['model']['params']['n_estimators'] = 5
    with env.controller('changed') as run, pytest.raises(ValueError, match='恢复计划'):
        run.optimize(study(), 1, 'lightgbm', lambda *a: None)


def test_restart_finishes_frozen_proposal_then_fills_remaining_budget(env):
    with env.controller('first') as run:
        initial = study()
        trial = initial.ask()
        from factor_service.research.trainer import _suggest_tree_hyperparameters
        params = _suggest_tree_hyperparameters(trial, 'lightgbm')
        run.plan('optuna_trial', 'trial_00000', dict(trial_number=0, params=params,
            distributions={k: optuna.distributions.distribution_to_json(v) for k, v in trial.distributions.items()}))
    with env.controller('resumed') as run:
        completed = run.optimize(study(), 3, 'lightgbm', lambda *a: None)
    assert len(completed.trials) == 3 and completed.trials[0].params == params
    assert len({json.dumps(t.params, sort_keys=True) for t in completed.trials}) == 3


@pytest.mark.parametrize('name,kind', [('../escape', 'file'), ('/absolute', 'file'),
    ('link', 'link'), ('duplicate', 'duplicate')])
def test_unsafe_archive_rejected_before_extraction(tmp_path, name, kind):
    bundle = tmp_path / 'bundle.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        info = tarfile.TarInfo(name)
        if kind == 'link':
            info.type, info.linkname = tarfile.SYMTYPE, '../escape'
            archive.addfile(info)
        else:
            info.size = 1
            archive.addfile(info, io.BytesIO(b'a'))
            if kind == 'duplicate':
                archive.addfile(info, io.BytesIO(b'a'))
    with pytest.raises(ValueError, match='不安全'):
        extract_verified_archive(bundle, tmp_path / 'output')
    assert not (tmp_path / 'escape').exists()


@pytest.mark.parametrize('nodes', [[], ['one', 'one'], ['local', 'one'], ['distributed'], ['a'] * 17])
def test_invalid_node_selection(nodes):
    with pytest.raises(ValueError):
        execution_spec({'node_ids': nodes})


def test_single_and_distributed_execution_contracts():
    assert execution_spec({})['mode'] == 'local'
    assert execution_spec({'node_id': 'one'})['mode'] == 'remote_ssh_docker'
    result = execution_spec({'node_ids': ['one', 'two']})
    assert result['mode'] == 'distributed' and result['tasks_per_node'] == 1
