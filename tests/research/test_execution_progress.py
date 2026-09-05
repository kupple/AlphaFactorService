from factor_service.research.execution_progress import ExecutionProgress, DistributedProgress


def test_monotonic_parent_progress_and_attempt_phase_timings():
    now = [0.0]
    tracker = ExecutionProgress(lambda: now[0])
    tracker.update('dataset_ready', 56, {})
    now[0] = 10
    tracker.update('distributed_plan', 57, {'kind': 'optuna_trial'})
    now[0] = 20
    tracker.update('training_metrics_writing', 57, {})
    assert tracker.phase == 'optuna'
    now[0] = 30
    tracker.update('distributed_plan', 65, {'kind': 'window'})
    now[0] = 40
    tracker.update('distributed_task_completed', 72, {'kind': 'window'})
    now[0] = 45
    result = tracker.update('distributed_task_progress', 65, {'kind': 'window'})
    assert result['percent'] == 72
    assert result['phase_seconds'] == dict(dataset=10, optuna=20, training=15, packaging=0, uploading=0, publishing=0)
    tracker.update('packaging', 85, {})
    tracker.update('uploading_model_archive', 90, {})
    assert tracker.update('remote.training', 63, {})['phase'] == 'uploading'
    assert ExecutionProgress(lambda: 0).update('validating', 2, {})['percent'] == 2


def test_reused_counts_and_independent_node_samples():
    tracker = DistributedProgress([{'task_key': 'window_1', 'kind': 'window', 'state': 'complete'}])
    tracker.update('distributed_plan', {'kind': 'window', 'total': 3})
    tracker.update('distributed_task_started', {'kind': 'window', 'task_key': 'window_2', 'node_id': 'a'})
    tracker.update('distributed_task_progress', {'kind': 'window', 'task_key': 'window_2', 'node_id': 'a', 'resources': {'process_rss_bytes': 42}})
    view = tracker.update('distributed_task_started', {'kind': 'window', 'task_key': 'window_3', 'node_id': 'b'})
    assert view['counts']['window'] == dict(total=3, completed=1, running=2, failed=0, waiting=0, reused=1, pruned=0)
    assert view['nodes']['a']['resources']['process_rss_bytes'] == 42
    assert 'resources' not in view['nodes']['b']
    tracker.update('distributed_task_completed', {'kind': 'window', 'task_key': 'window_2', 'node_id': 'a'})
    snapshot = tracker.update('distributed_task_started', {'kind': 'window', 'task_key': 'window_4', 'node_id': 'a'})
    assert 'resources' not in snapshot['nodes']['a']
    assert view['counts']['window']['completed'] == 1  # returned snapshots are independent


def test_failover_counts_cannot_double_count_a_window():
    tracker = DistributedProgress()
    tracker.update('distributed_plan', {'kind': 'window', 'total': 1})
    task = {'kind': 'window', 'task_key': 'window_1', 'node_id': 'a'}
    tracker.update('distributed_task_started', task)
    failed = tracker.update('distributed_node_ssh_failed', task)
    assert failed['counts']['window']['failed'] == 1
    task['node_id'] = 'b'
    tracker.update('distributed_task_started', task)
    done = tracker.update('distributed_task_completed', task)
    assert done['counts']['window']['completed'] == 1
    assert done['counts']['window']['failed'] == 0


def test_completion_retains_distributed_observation():
    tracker = ExecutionProgress()
    tracker.update('distributed_task_completed', 84, {'kind': 'window', 'distributed': {'counts': {'window': {'completed': 37}}}})
    result = tracker.update('completing', 99, {})
    assert result['distributed']['counts']['window']['completed'] == 37
