from pydantic import BaseModel
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class UserOut(BaseModel):
    id: int
    name: str
    password: str
    email: str
    active: int
    token_expire: datetime
    is_audited: int
    is_admin: int
    upload_limit_a_day: int

class UserScheme(Base):
    __tablename__ = 'users'  # 对应数据库中的 'users' 表
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)  # 用户ID
    name = Column(String, unique=True, index=True)  # Username
    password = Column(String)  # Password
    email = Column(String)
    active = Column(Boolean)
    token_expire = Column(DateTime)  # token expired time(minutes)
    is_audited = Column(Boolean, default=0)  # 0:未审核 1:已审核
    is_admin = Column(Boolean, default=0)  # 0:普通用户 1:管理员
    upload_limit_a_day = Column(Integer, default=-1)  # 用户每天上传的图片数量限制

    media = relationship("MediaScheme", back_populates="user")
    classes = relationship("ClassesScheme", back_populates="user")
    media_classes = relationship("MediaClassesScheme", back_populates="user")
    logs = relationship("LogsScheme", back_populates="user",
                        cascade="all, delete-orphan")

class Media(BaseModel):
    user_id: int
    identifier: str
    name: str
    type: str
    created_at: datetime
    source: str
    processInfo: int = 0
    feature: str

class MediaScheme(Base):
    __tablename__ = 'media'
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'),)
    identifier = Column(String, primary_key=True, index=True)
    name = Column(String)
    type = Column(String)
    created_at = Column(DateTime, default=datetime.now())
    source = Column(String)
    processInfo = Column(Integer, default=0)
    # feature is blob
    feature = Column(String)

    user = relationship("UserScheme", back_populates="media")
    media_classes = relationship("MediaClassesScheme", back_populates="media")
    media_metadata = relationship(
        "MediaMetadataScheme",
        back_populates="media",
        uselist=False,
        primaryjoin="MediaScheme.identifier==MediaMetadataScheme.media_id"
    )



class Class(BaseModel):
    user_id: int
    id: int
    name: str
    artificial: int = 0
class ClassesScheme(Base):
    __tablename__ = 'classes'
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'),
                     index=True, nullable=False)
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String)
    artificial = Column(Integer, default=0)  # 0:自动分类 1:手动分类

    user = relationship("UserScheme", back_populates="classes")
    media_classes = relationship(
        "MediaClassesScheme", back_populates="classes")


class MediaClass(BaseModel):
    user_id: int
    media_id: str
    class_id: int
    artificial: int = 0
class MediaClassesScheme(Base):
    __tablename__ = 'media_classes'
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'),
                     primary_key=True, index=True, nullable=False)
    media_id = Column(String, ForeignKey('media.identifier', ondelete='CASCADE'),
                      primary_key=True, index=True)
    class_id = Column(Integer, ForeignKey('classes.id', ondelete='CASCADE'))
    artificial = Column(Integer, default=0)  # 0:自动分类 1:手动分类

    user = relationship("UserScheme", back_populates="media_classes")
    media = relationship("MediaScheme", back_populates="media_classes")
    classes = relationship("ClassesScheme", back_populates="media_classes")

class Log(BaseModel):
    user_id: int
    action: int
    valueBool: bool
    valueInt: int
    valueStr: str
    valueFloat: float
    valueInfo: str
class LogsScheme(Base):
    __tablename__ = 'logs'
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'),
                     primary_key=True, index=True)
    action = Column(Integer)  # Action performed by the user
    valueBool = Column(Boolean)  # Boolean value associated with the action
    valueInt = Column(Integer)  # Value associated with the action
    valueStr = Column(String)  # String value associated with the action
    valueFloat = Column(Float)  # Float value associated with the action
    valueInfo = Column(String)  # Additional information about the action
    timestamp = Column(DateTime, default=datetime.now()
                       )  # Timestamp of the action

    user = relationship("UserScheme", back_populates="logs",
                        passive_deletes=True)

class MediaMetadata(BaseModel):
    user_id: int
    media_id: str
    exif_lat: float
    exif_lon: float
    exif_dev: str
    location: str

class MediaMetadataScheme(Base):
    __tablename__ = 'media_metadata'
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'),
                    primary_key=True, index=True, nullable=False)
    media_id = Column(String, ForeignKey('media.identifier', ondelete='CASCADE'))
    exif_lat = Column(Float)
    exif_lon = Column(Float)
    exif_dev = Column(String)
    location = Column(String)

    # created_at = Column(DateTime, default=datetime.now())
    # modified_at = Column(DateTime, default=datetime.now())

    media = relationship("MediaScheme", back_populates="media_metadata")
