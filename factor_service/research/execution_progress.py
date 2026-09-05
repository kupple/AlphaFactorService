"""Attempt-local observability. Never changes the frozen training plan."""
from copy import deepcopy
import time


PHASES = ('dataset', 'optuna', 'training', 'packaging', 'uploading', 'publishing')


class ExecutionProgress:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = self.changed = clock()
        self.phase = 'dataset'
        self.durations = dict.fromkeys(PHASES, 0.0)
        self.percent = 0
        self.distributed = {}

    def update(self, stage, percent, details):
        semantic_stage = stage.removeprefix('remote.')
        now = self.clock()
        self.durations[self.phase] += max(0, now - self.changed)
        self.changed = now
        if details.get('distributed') is not None:
            self.distributed = deepcopy(details['distributed'])
        if stage.startswith('uploading'):
            phase = 'uploading'
        elif stage in ('publishing_predictions', 'completing', 'succeeded'):
            phase = 'publishing'
        elif semantic_stage in ('packaging', 'packaged', 'rolling_series_ready'):
            phase = 'packaging'
        elif stage.startswith('distributed_') and details.get('kind') == 'optuna_trial':
            phase = 'optuna'
        elif stage.startswith('distributed_') and details.get('kind') in ('window', 'final'):
            phase = 'training'
        elif semantic_stage.startswith('optuna'):
            phase = 'optuna'
        elif semantic_stage in ('training', 'training_final_model', 'walk_forward_training', 'stacking_meta_training', 'predicting'):
            phase = 'training'
        else:
            phase = self.phase
        # Child heartbeats and restored files cannot take the parent backwards.
        if PHASES.index(phase) >= PHASES.index(self.phase):
            self.phase = phase
        self.percent = max(self.percent, min(99, max(0, int(percent))))
        return {
            **details, 'stage': stage, 'percent': self.percent,
            'phase': self.phase, 'attempt_elapsed_seconds': round(now - self.started, 3),
            'phase_seconds': {k: round(v, 3) for k, v in self.durations.items()},
            **({'distributed': deepcopy(self.distributed)} if self.distributed else {}),
        }


class DistributedProgress:
    """Updated under the coordinator's emission lock; no heartbeat DB scans."""
    def __init__(self, rows=()):
        self.tasks = {r['task_key']: {'kind': r['kind'], 'state': r['state'],
                      'reused': r['state'] in ('complete', 'pruned')} for r in rows}
        self.totals = {}
        self.nodes = {}

    def update(self, stage, details):
        kind, key, node = details.get('kind'), details.get('task_key'), details.get('node_id')
        if kind and 'total' in details:
            self.totals[kind] = int(details['total'])
        if key and kind:
            task = self.tasks.setdefault(key, {'kind': kind, 'state': 'pending', 'reused': False})
            if stage == 'distributed_task_started':
                task['state'] = 'running'
            elif stage == 'distributed_task_completed':
                task['state'] = details.get('state', 'complete')
            elif stage in ('distributed_task_retrying', 'distributed_node_memory_failed', 'distributed_node_ssh_failed', 'distributed_task_failed'):
                task['state'] = 'failed'
        if node:
            previous = self.nodes.get(node, {})
            current = {**previous, 'node_id': node, 'task_key': key or previous.get('task_key'),
                       'stage': details.get('child_stage') or stage, 'sampled_at': time.time()}
            for field in ('resources', 'resource_sampled_at', 'task_elapsed_seconds', 'effective_num_threads', 'error'):
                if field in details:
                    current[field] = deepcopy(details[field])
            if stage == 'distributed_task_started':
                # Never present a previous process's RSS as the new task's RSS.
                current = {k: v for k, v in current.items() if k not in ('resources', 'resource_sampled_at', 'error')}
                current['task_elapsed_seconds'] = 0
            self.nodes[node] = current
        counts = {}
        for task_kind in ('optuna_trial', 'window', 'final'):
            tasks = [t for t in self.tasks.values() if t['kind'] == task_kind]
            total = max(len(tasks), self.totals.get(task_kind, 0))
            if not total:
                continue
            complete = sum(t['state'] in ('complete', 'pruned') for t in tasks)
            running = sum(t['state'] == 'running' for t in tasks)
            failed = sum(t['state'] == 'failed' for t in tasks)
            counts[task_kind] = {'total': total, 'completed': complete, 'running': running,
                                'failed': failed, 'waiting': max(0, total - complete - running - failed),
                                'reused': sum(t['reused'] for t in tasks),
                                'pruned': sum(t['state'] == 'pruned' for t in tasks)}
        return {'counts': counts, 'nodes': deepcopy(self.nodes)}
