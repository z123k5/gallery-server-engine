from datetime import datetime, timedelta, timezone
from typing import Optional, Union

from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from src.models import UserScheme
from pydantic import BaseModel
import hashlib
import os
from jose import JWTError, jwt
import src.Database as Database


# User model
class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None
    token_expire: Union[datetime, None] = None


class UserInDB(User):
    password: str


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Database
salt = os.environ.get('SALT', '')
SECRET_KEY = os.environ.get('SECRET_KEY', '')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Get user from db



# Hash password


def hash_password(password: str):
    # Using md5 + salt to avoid rainbow table
    return hashlib.md5((salt + password).encode()).hexdigest()

# Create jwt token with expire time, default is 30 minutes


def create_jwt_token(data: dict, expire_delta: Optional[timedelta] = None):
    expire = datetime.now() + expire_delta if expire_delta else datetime.now() + \
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    data.update({'exp': expire})
    token = jwt.encode(claims=data, key=SECRET_KEY, algorithm=ALGORITHM)
    return token, expire

# Decode token


def decode_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token=token, key=SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get('sub')
        return sub
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Get user from db
def get_user(sub: str = Depends(decode_token), db: Session = Depends(Database.get_db)):
    # SQL query to get user from db
    user = db.query(UserScheme).filter(
        UserScheme.username == sub).first()
    return UserInDB(**user.__dict__)


# Get current user
async def get_current_user(token: str = Depends(oauth2_scheme), user = Depends(get_user)):
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# Get current active user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    # UTC
    if current_user.token_expire < datetime.now():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credential expired at: " + str(current_user.token_expire),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user
