"""One coordinator, trial-level Optuna parallelism, independent rolling windows.

Nodes never receive PostgreSQL or object-store credentials. The coordinator owns
the durable ledger; node output is archived and verified before completion.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import queue
import shutil
import tarfile
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

from factor_service.model_object_store import ModelObjectStore
from factor_service.research.dataset_archive import file_sha256
from factor_service.research.distributed_config import distributed_nodes
from factor_service.research.errors import (
    NodeMemoryBudgetExceeded, NodeOutOfMemory, NodeSSHError,
    NodeExecutionUnavailable, RetryableJobError,
)
from factor_service.research.job import CancellationToken
from factor_service.training_subtask_repository import TrainingSubtaskRepository
from factor_service.research.execution_progress import DistributedProgress


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def extract_verified_archive(path, destination):
    """No links, absolute paths, traversal, duplicate entries or special files."""
    destination = Path(destination).resolve()
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        seen = set()
        for member in members:
            name = PurePosixPath(member.name)
            target = destination.joinpath(*name.parts)
            if (name.is_absolute() or '..' in name.parts or not name.parts
                    or member.name in seen or not (member.isdir() or member.isfile())
                    or target.resolve() != target or not target.is_relative_to(destination)):
                raise ValueError('分布式制品包含不安全或重复路径')
            seen.add(member.name)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)


class DistributedTraining:
    def __init__(self, settings, job, work_dir, cancellation=None, progress=None,
                 *, repository=None, objects=None, nodes=None, executor_factory=None):
        from factor_service.research.remote import get_remote_node, RemoteResearchExecutor
        self.settings, self.job = settings, deepcopy(job)
        self.root = Path(work_dir) / 'distributed'
        self.root.mkdir(parents=True, exist_ok=True)
        self.cancellation = cancellation or CancellationToken(timeout_seconds=(
            job['config_json']['execution'].get('max_runtime_minutes', 720) * 60))
        self.progress = progress or (lambda *args: None)
        self.repository = repository or TrainingSubtaskRepository()
        self.objects = objects or ModelObjectStore(settings.model_object_store)
        if not settings.model_object_store.enabled or not self.objects.enabled_for('bundle'):
            raise ValueError('分布式训练需要启用MinIO以保存可恢复的子任务制品')
        self.nodes = nodes or [get_remote_node(key) for key in distributed_nodes(job)]
        if len(self.nodes) < 2:
            raise ValueError('分布式训练至少需要两个执行节点')
        identities = {(node.host, node.port, node.user) for node in self.nodes}
        if len(identities) != len(self.nodes):
            raise ValueError('所选节点包含重复SSH目标，不能在同一机器叠加训练')
        self.factory = executor_factory or RemoteResearchExecutor
        self.parent = str(job['job_id'])
        self.plan_hash = fingerprint({
            'dataset_hash': job['dataset_hash'], 'model_id': job['model_id'],
            'config': job['config_json'], 'distributed_version': 1,
        })
        self.available = queue.Queue()
        for node in self.nodes:
            self.available.put(node)
        self.unavailable_nodes = {}
        self.node_lock = threading.Lock()
        self.used = {}
        self.lock = threading.Lock()
        self.stack = ExitStack()
        self.pool = None
        self.snapshot = self.snapshot_store = None

    def __enter__(self):
        self.stack.enter_context(self.repository.coordinator(self.parent))
        self.observation = DistributedProgress(self.repository.list(self.parent))
        self.pool = ThreadPoolExecutor(max_workers=len(self.nodes), thread_name_prefix='training-node')
        self.emit('distributed_started', 3, {'node_ids': [node.node_id for node in self.nodes], 'tasks_per_node': 1})
        return self

    def __exit__(self, kind, value, traceback):
        try:
            if self.pool:
                self.pool.shutdown(wait=True, cancel_futures=True)
            # Sub-tasks must not power-cycle a node between trials. Only finish a
            # user-configured auto-stop lifecycle after verifying the node is idle.
            for executor, original in self.used.values():
                if original.auto_stop:
                    try:
                        from factor_service.research.remote_cache import RemoteDatasetCache
                        RemoteDatasetCache(executor.node, executor.transport, None, self.parent)._idle()
                        executor.node = replace(executor.node, auto_stop=True)
                        executor._power_off_after_job(self.job, self.progress)
                    except Exception as exc:
                        self.emit('distributed_node_cleanup_pending', 89, {'node_id': original.node_id, 'error': str(exc)})
        finally:
            for executor, _ in self.used.values():
                close = getattr(getattr(executor, 'transport', None), 'close', None)
                if callable(close):
                    close()
            self.stack.close()

    def bind_snapshot(self, snapshot, store):
        self.snapshot, self.snapshot_store = snapshot, store

    def emit(self, stage, percent, details):
        with self.lock:
            snapshot = self.observation.update(stage, details)
            kind = details.get('kind')
            counts = snapshot['counts'].get(kind, {})
            if counts.get('total') and kind in ('optuna_trial', 'window'):
                base, width = (57, 7) if kind == 'optuna_trial' else (65, 19)
                percent = base + int(width * counts['completed'] / counts['total'])
            self.progress(stage, percent, {**details, 'distributed': snapshot})

    def plan(self, kind, key, payload):
        payload = json.loads(json.dumps(payload, default=str))
        return self.repository.plan(self.parent, key, self.plan_hash, kind, payload)

    def _take_node(self):
        while True:
            self.cancellation.checkpoint()
            with self.node_lock:
                if len(self.unavailable_nodes) == len(self.nodes):
                    memory_only = all(isinstance(error, (NodeMemoryBudgetExceeded, NodeOutOfMemory))
                                      for error in self.unavailable_nodes.values())
                    error_type = NodeMemoryBudgetExceeded if memory_only else NodeExecutionUnavailable
                    raise error_type('所选节点均不可用，暂无节点可承接；已保留完成的子任务。'
                                     + '；'.join(f'{node}: {error}' for node, error in self.unavailable_nodes.items()))
            try:
                node = self.available.get(timeout=0.2)
            except queue.Empty:
                continue  # A healthy selected node may still be finishing work.
            with self.node_lock:
                if node.node_id not in self.unavailable_nodes:
                    return node

    def _return_node(self, node):
        with self.node_lock:
            if node.node_id not in self.unavailable_nodes:
                self.available.put(node)

    def _run(self, row):
        if row['state'] in {'complete', 'pruned'}:
            return self._restore(row)
        transient_failures = 0
        previous_node = None
        # An attempt may move to another selected node; the params/window stay
        # frozen. The ledger's CAS token fences late results from older attempts.
        while True:
            self.cancellation.checkpoint()
            node = self._take_node()
            owner = uuid4().hex
            key, payload = row['task_key'], row['payload_json']
            active = None
            stop = threading.Event()
            heartbeat_errors = []

            def heartbeat():
                while not stop.wait(10):
                    try:
                        self.repository.heartbeat(self.parent, key, owner)
                    except Exception as exc:
                        heartbeat_errors.append(exc)
                        return

            monitor = threading.Thread(target=heartbeat, daemon=True)
            try:
                active = self.repository.start(self.parent, key, owner, node.node_id)
                if previous_node is not None:
                    self.emit('distributed_task_reassigned', 58 if row['kind'] == 'optuna_trial' else 65,
                              {'task_key': key, 'from_node_id': previous_node[0], 'node_id': node.node_id,
                               'reason': previous_node[1], 'error_code': previous_node[2],
                               'attempt': active['attempt_count']})
                    previous_node = None
                started_at = time.monotonic()
                monitor.start()
                work = self.root / key / f"attempt-{active['attempt_count']:04d}-{owner}"
                work.mkdir(parents=True)
                child = deepcopy(self.job)
                child['job_id'] = (self.parent if row['kind'] == 'final'
                                   else f"dist_{fingerprint(self.parent)[:16]}_{key}_{active['attempt_count']}")
                child['attempt_count'] = active['attempt_count']
                child['lease_token'] = 'snapshot-only-distributed-child'
                child['config_json']['execution'] = {
                    'node_id': node.node_id, 'mode': 'remote_ssh_docker',
                    'max_runtime_minutes': self.job['config_json']['execution']['max_runtime_minutes'],
                }
                child['config_json']['_distributed_task'] = {
                    **payload, 'kind': row['kind'], 'parent_job_id': self.parent,
                    'parent_execution': self.job['config_json']['execution'],
                }
                if row['kind'] != 'optuna_trial':
                    child['config_json']['optuna'] = {}
                    child['config_json']['model']['params'] = dict(payload['params'])
                # One transport/session per selected node and parent attempt,
                # so progress/cache probes don't reauthenticate every second.
                prior = self.used.get(node.node_id)
                executor = prior[0] if prior else self.factory(self.settings, replace(node, auto_stop=False))
                executor.recoverable_cache_jobs = self._previous_job_ids()
                # No repeated source queries, Parquet reloads, or per-trial
                # DataFrame copies in the controller; nodes reuse their disk slot.
                executor.snapshot_store = SimpleNamespace(
                    get_or_create=lambda *a, **k: self.snapshot,
                    artifacts=self.snapshot_store.artifacts, archive=self.snapshot_store.archive,
                )
                with self.lock:
                    self.used[node.node_id] = (executor, node)
                self.emit('distributed_task_started', 58 if row['kind'] == 'optuna_trial' else 65,
                          {'task_key': key, 'kind': row['kind'], 'node_id': node.node_id,
                           'attempt': active['attempt_count']})

                def progress(stage, percent, details):
                    if heartbeat_errors:
                        raise RetryableJobError('子任务心跳失效，停止当前执行') from heartbeat_errors[0]
                    self.emit('distributed_task_progress', 58 if row['kind'] == 'optuna_trial' else 65,
                              {**details, 'task_key': key, 'kind': row['kind'], 'node_id': node.node_id, 'child_stage': stage,
                               'child_percent': percent, 'task_elapsed_seconds': round(time.monotonic() - started_at, 3)})

                parent_token = self.cancellation

                class FencedCancellation:
                    def checkpoint(self):
                        parent_token.checkpoint()
                        if heartbeat_errors:
                            raise RetryableJobError('子任务心跳失效，停止当前执行') from heartbeat_errors[0]

                result = executor.train(child, work, cancellation=FencedCancellation(), progress=progress)
                self.cancellation.checkpoint()
                if heartbeat_errors:
                    raise RetryableJobError('子任务心跳失效，拒绝提交结果')
                if 'trial' in result.result:
                    result.result['trial'].setdefault('attrs', {}).update({
                        'distributed_node_id': node.node_id,
                        'distributed_attempt_count': active['attempt_count'],
                        'distributed_elapsed_seconds': time.monotonic() - started_at,
                    })
                durable = self._archive(key, owner, result, work)
                durable['execution_observation'] = {
                    'elapsed_seconds': round(time.monotonic() - started_at, 3),
                    'node_id': node.node_id,
                    'resources': self.observation.nodes.get(node.node_id, {}).get('resources', {}),
                }
                state = 'pruned' if result.result.get('trial', {}).get('state') == 'pruned' else 'complete'
                self.repository.finish(self.parent, key, owner, state, durable)
                self.emit('distributed_task_completed', 58 if row['kind'] == 'optuna_trial' else 72,
                          {'task_key': key, 'kind': row['kind'], 'node_id': node.node_id, 'state': state,
                           'task_elapsed_seconds': durable['execution_observation']['elapsed_seconds']})
                return result
            except Exception as exc:
                if active is not None:
                    try:
                        self.repository.finish(self.parent, key, owner, 'failed', error=str(exc))
                    except RetryableJobError:
                        pass  # A newer owner won; never overwrite its state.
                if isinstance(exc, (NodeMemoryBudgetExceeded, NodeOutOfMemory, NodeSSHError)):
                    # Executor cleanup has run or retained an uncertain lock. Retire
                    # this node for this parent attempt, then wait for another
                    # selected node without changing the frozen params/window.
                    # A disconnected node may still own work/cache: never send
                    # another task there or unlock it merely because it is old.
                    with self.node_lock:
                        self.unavailable_nodes[node.node_id] = exc
                    reason = 'ssh' if isinstance(exc, NodeSSHError) else 'memory'
                    previous_node = (node.node_id, reason, exc.code)
                    self.emit(f'distributed_node_{reason}_failed', 58 if row['kind'] == 'optuna_trial' else 65,
                              {'task_key': key, 'kind': row['kind'], 'node_id': node.node_id, 'error': str(exc),
                               'error_code': exc.code, 'attempt': active['attempt_count'] if active else None})
                else:
                    transient_failures += 1
                    if not isinstance(exc, (RetryableJobError, OSError, TimeoutError)) or transient_failures >= 2:
                        self.emit('distributed_task_failed', 65, {'task_key': key, 'kind': row['kind'], 'node_id': node.node_id, 'error': str(exc)})
                        raise
                    self.emit('distributed_task_retrying', 58 if row['kind'] == 'optuna_trial' else 65,
                              {'task_key': key, 'kind': row['kind'], 'node_id': node.node_id, 'error': str(exc),
                               'attempt': active['attempt_count'] if active else None})
            finally:
                stop.set()
                if monitor.is_alive():
                    monitor.join(timeout=2)
                self._return_node(node)

    def _previous_job_ids(self):
        identifiers = set()
        for row in self.repository.list(self.parent):
            for attempt in range(1, row['attempt_count'] + 1):
                identifiers.add(self.parent if row['kind'] == 'final' else
                                f"dist_{fingerprint(self.parent)[:16]}_{row['task_key']}_{attempt}")
        return identifiers

    def _archive(self, key, owner, result, work):
        files, artifacts = {}, []
        for kind, path in result.artifacts:
            if kind in {'dataset', 'dataset_raw', 'dataset_manifest'}:
                artifacts.append({'kind': kind, 'dataset_file': Path(path).name})
                continue
            relative = Path(path).relative_to(work).as_posix()
            files[relative] = Path(path)
            artifacts.append({'kind': kind, 'path': relative})
        prediction = Path(result.predictions_path).relative_to(work).as_posix()
        files[prediction] = Path(result.predictions_path)
        bundle = self.root / f'{key}-{owner}.tar.gz'
        with tarfile.open(bundle, 'w:gz') as target:
            for relative, path in files.items():
                if path.is_symlink() or not path.is_file():
                    raise ValueError('子任务制品必须为普通文件')
                target.add(path, arcname=relative, recursive=False)
        identity = self.objects.publish_file(
            job_id=self.parent, model_id=f'distributed_{fingerprint(self.parent)[:24]}', model_version=1,
            artifact_kind='bundle', source_path=bundle, digest=file_sha256(bundle), size_bytes=bundle.stat().st_size,
            checkpoint=self.cancellation.checkpoint,
        )
        if identity is None:
            raise ValueError('子任务制品未归档MinIO')
        self.objects.verify_file(identity, content=True, checkpoint=self.cancellation.checkpoint)
        return {'result': result.result, 'artifacts': artifacts, 'prediction': prediction, 'object': identity}

    def _restore(self, row):
        from factor_service.research.trainer import TrainingResult
        saved = row['result_json']
        self.emit('distributed_task_restoring', 65, {'kind': row['kind'], 'task_key': row['task_key']})
        root = self.root / 'restored' / f"{row['task_key']}-{uuid4().hex}"
        root.mkdir(parents=True)
        identity = saved['object']
        bundle = root / 'verified.tar.gz'
        self.objects.download_file(object_uri=identity['object_uri'], version_id=identity.get('version_id', ''),
                                   destination=bundle, digest=identity['sha256'], size_bytes=identity['size_bytes'],
                                   checkpoint=self.cancellation.checkpoint)
        extract_verified_archive(bundle, root / 'files')
        def resolve_file(name, base):
            relative = PurePosixPath(name)
            path = base.joinpath(*relative.parts)
            if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
                    or not path.is_file() or not path.resolve().is_relative_to(base.resolve())):
                raise ValueError('恢复制品路径不安全或不存在')
            return path
        artifacts = []
        for item in saved['artifacts']:
            path = (resolve_file(item['dataset_file'], self.snapshot.dataset_path.parent) if 'dataset_file' in item
                    else resolve_file(item['path'], root / 'files'))
            artifacts.append((item['kind'], path))
        return TrainingResult(saved['result'], artifacts, resolve_file(saved['prediction'], root / 'files'))

    def optimize(self, study, count, model_kind, report):
        import optuna
        from factor_service.research.trainer import _suggest_tree_hyperparameters
        existing = self.repository.list(self.parent, 'optuna_trial')
        self.emit('distributed_plan', 57, {'kind': 'optuna_trial', 'total': count})
        if existing:
            # Rehydrating FrozenTrials does not advance a TPE sampler's RNG.
            # Do not restart its startup sequence and repeat the first proposals.
            # Already-proposed params are durable; only future proposals reseed.
            study.sampler.reseed_rng()
        rows = {}
        for row in existing:
            payload = row['payload_json']
            number = payload['trial_number']
            if row['plan_hash'] != self.plan_hash or number != len(rows) or number >= count:
                raise ValueError('Optuna恢复计划或全局试验预算不一致')
            rows[number] = row
            self._add_frozen_trial(study, row)
        inflight, started, first_error = {}, set(), None
        while True:
            self.cancellation.checkpoint()
            while first_error is None and len(inflight) < len(self.nodes):
                pending = next((i for i, row in rows.items() if row['state'] not in {'complete', 'pruned'} and i not in started), None)
                if pending is None:
                    if len(rows) >= count:
                        break
                    trial = study.ask()
                    params = _suggest_tree_hyperparameters(trial, model_kind)
                    payload = {'trial_number': trial.number, 'params': params,
                               'distributions': {k: optuna.distributions.distribution_to_json(v)
                                                 for k, v in trial.distributions.items()}}
                    row = self.plan('optuna_trial', f'trial_{trial.number:05d}', payload)
                    rows[trial.number] = row
                    pending = trial.number
                started.add(pending)
                inflight[self.pool.submit(self._run, rows[pending])] = pending
            if not inflight:
                break
            done, _ = wait(inflight, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                number = inflight.pop(future)
                try:
                    result = future.result().result['trial']
                    state = optuna.trial.TrialState.COMPLETE if result['state'] == 'complete' else optuna.trial.TrialState.PRUNED
                    if result['params'] != rows[number]['payload_json']['params']:
                        raise ValueError('远程试验参数与冻结提案不一致')
                    study.tell(number, result['value'] if state == optuna.trial.TrialState.COMPLETE else None, state=state)
                    report(study, study.trials[number])
                except Exception as exc:
                    first_error = first_error or exc
        if first_error:
            raise first_error
        # Trial attributes (fold metrics, stability gate) are restored through
        # public Optuna APIs rather than private storage internals.
        complete = optuna.create_study(direction='maximize')
        for row in self.repository.list(self.parent, 'optuna_trial'):
            self._add_frozen_trial(complete, row)
        return complete

    @staticmethod
    def _add_frozen_trial(study, row):
        import optuna
        payload = row['payload_json']
        result = row.get('result_json', {}).get('result', {}).get('trial', {})
        state = {'complete': optuna.trial.TrialState.COMPLETE,
                 'pruned': optuna.trial.TrialState.PRUNED}.get(row['state'], optuna.trial.TrialState.RUNNING)
        study.add_trial(optuna.trial.create_trial(
            state=state, value=result.get('value') if state == optuna.trial.TrialState.COMPLETE else None,
            params=payload['params'], user_attrs=result.get('attrs', {}),
            distributions={k: optuna.distributions.json_to_distribution(v) for k, v in payload['distributions'].items()},
        ))

    def windows(self, tasks, params, series_root):
        pending = [self.plan('window', f"window_{task['alphablocks']['window']:05d}", {
            'window': task['alphablocks']['window'], 'segments': task['dataset']['kwargs']['segments'],
            'params': params,
        }) for task in tasks]
        self.emit('distributed_plan', 65, {'kind': 'window', 'total': len(tasks)})
        inflight, first_error = {}, None
        while pending or inflight:
            self.cancellation.checkpoint()
            while pending and len(inflight) < len(self.nodes) and first_error is None:
                row = pending.pop(0)
                inflight[self.pool.submit(self._run, row)] = row
            if not inflight:
                break
            done, _ = wait(inflight, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                row = inflight.pop(future)
                try:
                    result = future.result()
                    if result.result['window'] != row['payload_json']['window']:
                        raise ValueError('远程返回了错误的滚动窗口')
                    bundle = next(path for kind, path in result.artifacts if kind == 'window')
                    extract_verified_archive(bundle, series_root)
                except Exception as exc:
                    first_error = first_error or exc
        if first_error:
            raise first_error

    def final_model(self, params, optuna_result):
        row = self.plan('final', 'final', {'params': params, 'optuna_result': optuna_result})
        return self._run(row)
