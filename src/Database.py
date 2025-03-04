from contextlib import contextmanager
import os
from datetime import datetime
from typing import Generator
from src.Logger import logger
import src.UserManager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base
from fastapi import Depends

SQLALCHEMY_DATABASE_URL = "sqlite:///./db/data.db"

# 创建数据库引擎
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={
                       "check_same_thread": False})

# 创建Session工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Contex Manager


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Contex Manager With Commit


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


# Check if db exists in path
if not os.path.exists("./db/data.db"):
    logger.info("Database not found, creating...")
    open("./db/data.db", "w").close()
    logger.info("Database created")
    with get_db_commit() as db:
        db.execute(
            '''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, email TEXT, full_name TEXT, disabled BOOLEAN, token_expire DATE)''')
        # If create table successful, insert a default user
        db.execute(
            "INSERT INTO users (username, password, email, full_name, disabled, token_expire) VALUES (?, ?, ?, ?, ?, ?)",
            ("admin", src.UserManager.hash_password("admin"), "admin@example.com", "Administrator", False, datetime.now()))
        db.commit()
