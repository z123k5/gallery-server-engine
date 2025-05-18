from typing import Any, Callable
from .DatabaseQueue import get_db_task_queue
from .Database import SessionLocal
from sqlalchemy.orm import Session


async def run_db_task(task: Callable[[Session], Any]):
    async def wrapper():
        db = SessionLocal()
        try:
            result = task(db)
            db.commit()
            return result
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    # ⬇️ 动态获取已启动的队列
    return await get_db_task_queue().submit(wrapper)
