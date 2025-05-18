import threading
import queue
import uuid
from sqlalchemy.orm.query import Query
from sqlalchemy.engine.row import Row
from collections.abc import Iterable


class DBTaskExecutor:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.results = {}  # task_id -> result
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _serialize_result(self, result):
        # ORM对象列表或单个 ORM 对象
        if isinstance(result, list):
            return [self._serialize_obj(obj) for obj in result]
        elif hasattr(result, '__table__'):  # 是一个 ORM 实例
            return self._serialize_obj(result)
        elif isinstance(result, Row):
            return dict(result._mapping)
        else:
            return result

    def _serialize_obj(self, obj):
        # 将 ORM 实例转为字典（仅列字段）
        return {col.name: getattr(obj, col.name) for col in obj.__table__.columns}

    def _worker(self):
        while True:
            task_id, task_fn = self.task_queue.get()
            try:
                with DBContextManager() as db:
                    raw_result = task_fn(db)
                    result = self._serialize_result(raw_result)
                self.results[task_id] = result
            except Exception as e:
                self.results[task_id] = e
            self.task_queue.task_done()

    def run_db_task(self, task_fn):
        """传入一个 lambda db: ... 查询函数"""
        task_id = str(uuid.uuid4())
        self.task_queue.put((task_id, task_fn))
        while task_id not in self.results:
            pass  # 简化示例，生产可用 threading.Event 替代
        result = self.results.pop(task_id)
        if isinstance(result, Exception):
            raise result
        return result


# 全局实例
db_executor = DBTaskExecutor()
