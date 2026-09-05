"""PostgreSQL ledger for restartable, fenced distributed training sub-tasks."""
from contextlib import contextmanager
from hashlib import sha256

from psycopg.types.json import Jsonb

from factor_service.control_database import get_control_database
from factor_service.research.errors import RetryableJobError


class TrainingSubtaskRepository:
    def __init__(self, database=None):
        self.database = database or get_control_database()
        self._coordinator_connection = None

    @contextmanager
    def coordinator(self, parent_job_id):
        key = int.from_bytes(sha256(parent_job_id.encode()).digest()[:8], 'big', signed=True)
        with self.database.dedicated_connection(autocommit=True) as conn:
            if not conn.execute('SELECT pg_try_advisory_lock(%s) AS locked', (key,)).fetchone()['locked']:
                raise RetryableJobError('该训练任务已有分布式协调器，禁止重复派发')
            try:
                self._coordinator_connection = conn
                yield
            finally:
                self._coordinator_connection = None
                conn.execute('SELECT pg_advisory_unlock(%s)', (key,))

    def _check_coordinator(self):
        if self._coordinator_connection is not None:
            try:
                self._coordinator_connection.execute('SELECT 1')
            except Exception as exc:
                raise RetryableJobError('协调器数据库连接失效，停止派发及提交') from exc

    def list(self, parent_job_id, kind=None):
        with self.database.connection() as conn:
            rows = conn.execute(
                'SELECT * FROM model_training_subtasks WHERE parent_job_id=%s '
                'AND (%s::text IS NULL OR kind=%s) ORDER BY task_key',
                (parent_job_id, kind, kind),
            ).fetchall()
        return [dict(row) for row in rows]

    def plan(self, parent_job_id, key, plan_hash, kind, payload):
        with self.database.connection() as conn, conn.transaction():
            conn.execute(
                'INSERT INTO model_training_subtasks(parent_job_id,task_key,plan_hash,kind,payload_json) '
                'VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                (parent_job_id, key, plan_hash, kind, Jsonb(payload)),
            )
            row = dict(conn.execute(
                'SELECT * FROM model_training_subtasks WHERE parent_job_id=%s AND task_key=%s',
                (parent_job_id, key),
            ).fetchone())
            if row['plan_hash'] != plan_hash or row['payload_json'] != payload or row['kind'] != kind:
                raise ValueError('分布式子任务冻结配置不一致，拒绝复用结果')
            return row

    def start(self, parent, key, owner, node):
        self._check_coordinator()
        with self.database.connection() as conn:
            row = conn.execute(
                "UPDATE model_training_subtasks SET state='running',owner_token=%s,node_id=%s,"
                "attempt_count=attempt_count+1,error='',heartbeat_at=now(),updated_at=now() "
                "WHERE parent_job_id=%s AND task_key=%s AND state NOT IN ('complete','pruned') RETURNING *",
                (owner, node, parent, key),
            ).fetchone()
        if row is None:
            raise RetryableJobError('分布式子任务已经完成或状态发生变化')
        return dict(row)

    def heartbeat(self, parent, key, owner):
        self._check_coordinator()
        with self.database.connection() as conn:
            changed = conn.execute(
                "UPDATE model_training_subtasks SET heartbeat_at=now(),updated_at=now() "
                "WHERE parent_job_id=%s AND task_key=%s AND owner_token=%s AND state='running'",
                (parent, key, owner),
            ).rowcount
        if changed != 1:
            raise RetryableJobError('子任务执行权已失效，旧执行器不能继续提交')

    def finish(self, parent, key, owner, state, result=None, error=''):
        self._check_coordinator()
        if state not in {'complete', 'pruned', 'failed'}:
            raise ValueError('无效子任务终态')
        with self.database.connection() as conn:
            changed = conn.execute(
                'UPDATE model_training_subtasks SET state=%s,result_json=%s,error=%s,updated_at=now() '
                "WHERE parent_job_id=%s AND task_key=%s AND owner_token=%s AND state='running'",
                (state, Jsonb(result or {}), str(error)[:2000], parent, key, owner),
            ).rowcount
        if changed != 1:
            raise RetryableJobError('子任务结果执行批次已过期，拒绝重复或迟到提交')
