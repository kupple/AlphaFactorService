"""Frozen execution contract for task-parallel training (not data parallelism)."""
import re


def execution_spec(source):
    if not isinstance(source, dict):
        raise ValueError("execution必须是对象")
    raw = source.get("node_ids")
    if raw is None:
        raw = [source.get("node_id") or "local"]
    if not isinstance(raw, list) or not raw or len(raw) > 16:
        raise ValueError("execution.node_ids必须包含1到16个节点")
    nodes = []
    for node in raw:
        if not isinstance(node, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", node):
            raise ValueError("execution.node_ids包含无效节点")
        if node in nodes or node == "distributed":
            raise ValueError("训练节点不能重复或使用保留ID")
        nodes.append(node)
    if len(nodes) > 1 and "local" in nodes:
        raise ValueError("多节点训练请选择远程执行节点；本机模式仍用于单节点训练")
    minutes = source.get("max_runtime_minutes", 720)
    if isinstance(minutes, bool) or float(minutes) != int(minutes) or not 60 <= int(minutes) <= 1440:
        raise ValueError("execution.max_runtime_minutes必须为60到1440之间的整数")
    result = {
        "node_id": nodes[0] if len(nodes) == 1 else "distributed",
        "mode": ("local" if nodes[0] == "local" else "remote_ssh_docker") if len(nodes) == 1 else "distributed",
        "max_runtime_minutes": int(minutes),
    }
    if len(nodes) > 1:
        result.update(node_ids=nodes, tasks_per_node=1,
                      optuna_parallelism="trial", walk_forward_parallelism="window")
    return result


def distributed_nodes(job):
    execution = (job.get("config_json") or {}).get("execution") or {}
    return list(execution.get("node_ids") or []) if execution.get("mode") == "distributed" else []
