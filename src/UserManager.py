from datetime import datetime, timedelta, timezone
from typing import Optional, Union
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import sqlite3
import hashlib
import os
from jose import JWTError, jwt



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
conn = sqlite3.connect('./db/database.db')
db = conn.cursor()
salt = os.environ.get('SALT', '')
SECRET_KEY = os.environ.get('SECRET_KEY', '')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Get user from db
def get_user(db, username: str):
    # SQL query to get user from db
    query = "SELECT * FROM users WHERE username = ?"
    db.execute(query, (username,))
    return UserInDB(**db.fetchone())

# Hash password
def hash_password(password: str):
    # Using md5 + salt to avoid rainbow table
    return hashlib.md5((salt + password).encode()).hexdigest()

# Create jwt token with expire time, default is 30 minutes
def create_jwt_token(data: dict, expire_delta: Optional[timedelta] = None):
    expire = datetime.now() + expire_delta if expire_delta else datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    data.update({'exp': expire})
    token = jwt.encode(claims=data, key=SECRET_KEY, algorithm=ALGORITHM)
    return token

# Decode token
def decode_token(token):
    payload = jwt.decode(token=token, key=SECRET_KEY, algorithms=[ALGORITHM])
    sub = payload.get('sub')
    return sub

# Get current user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    sub = decode_token(token)
    user = get_user(db, username=sub)
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
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

