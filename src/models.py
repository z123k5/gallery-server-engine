from sqlalchemy import Boolean, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class UserScheme(Base):
    __tablename__ = 'users'  # 对应数据库中的 'users' 表

    username = Column(String, unique=True, primary_key=True, index=True)  # Username
    password = Column(String)  # Password
    email = Column(String)
    full_name = Column(String)
    disabled = Column(Boolean)
    token_expire = Column(DateTime)  # token expired time(minutes)

