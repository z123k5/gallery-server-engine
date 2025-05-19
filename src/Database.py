
import threading
from collections.abc import Iterable
from sqlalchemy.engine.row import Row
from sqlalchemy.orm.query import Query
import uuid
import queue
from contextlib import contextmanager
import os, sys
# sys.path.append("D:\Projects\IOSProjects\gallery-server-engine")
# sys.path.append("D:\Projects\IOSProjects\gallery-server-engine\src")
from .Logger import logger

from datetime import datetime
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base
from fastapi import Depends

SQLALCHEMY_DATABASE_URL = "sqlite:///F:/Projects/gallery-server-engine/db/data.db"

# 创建数据库引擎
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={
                    "check_same_thread": False, "timeout": 30}, pool_size=10, max_overflow=5, pool_recycle=3600)


# 创建Session工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Contex Manager


class DBContextManager:
    def __init__(self):
        self.db = SessionLocal()

    def __enter__(self):
        return self.db

    def __exit__(self, exc_type, exc_value, traceback):
        self.db.commit()
        self.db.close()


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

# `FastAPI` Dependency Injection


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Contex Manager With Commit


# `FastAPI` Context manager for database commit
def get_db_commit():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


if not os.path.exists("./db/data.db"):
    logger.info("Database not found, creating...")
    open("./db/data.db", "w").close()
    logger.info("Database created")
