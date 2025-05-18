from typing import List
import fastapi
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from .Database import get_db, get_db_commit, DBContextManager
from fastapi import APIRouter, Depends
from .shared import LOGI

from src.UserManager import User, get_current_active_user, hash_password
from src.models import LogsScheme, UserScheme

admin_router = APIRouter()


@admin_router.post("/reset_user_password", tags=["users_manage"])
async def reset_user_password(userId: int, userName: str, db: Session = Depends(get_db_commit), user: User = Depends(get_current_active_user)):
    if user.is_admin == 0:
        return {"status": "error", "message": "You are not authorized to perform this action."}
    else:
        user = db.query(UserScheme).filter(UserScheme.id == userId).first()
        if user is None:
            return {"status": "error", "message": "User not found."}
        else:
            user.password = hash_password(userName)


@admin_router.post("/audit_new_user", tags=["users_manage"])
async def audit_new_user(userId: int, db: Session = Depends(get_db_commit), user: User = Depends(get_current_active_user)):
    if user.is_admin == 0:
        return {"status": "error", "message": "You are not authorized to perform this action."}
    else:
        user = db.query(UserScheme).filter(UserScheme.id == userId).first()
        if user is None:
            return {"status": "error", "message": "User not found."}
        else:
            user.is_audited = 1


@admin_router.post("/delete_user", tags=["users_manage"])
async def delete_user(userId: int, db: Session = Depends(get_db_commit), user: User = Depends(get_current_active_user)):
    if user.is_admin == 0:
        return {"status": "error", "message": "You are not authorized to perform this action."}
    else:
        user = db.query(UserScheme).filter(UserScheme.id == userId).first()
        if user is None:
            return {"status": "error", "message": "User not found."}
        else:
            db.delete(user)


@admin_router.post("/limit_user_frequency", tags=["users_manage"])
async def limit_user_frequency(userId: int, upload_limit_a_day: int, db: Session = Depends(get_db_commit), user: User = Depends(get_current_active_user)):
    if user.is_admin == 0:
        return {"status": "error", "message": "You are not authorized to perform this action."}
    else:
        user = db.query(UserScheme).filter(UserScheme.id == userId).first()
        if user is None:
            return {"status": "error", "message": "User not found."}
        else:
            # TODO: implement limit user frequency
            if not upload_limit_a_day:
                return {"status": "error", "message": "upload_limit_a_day is required."}
            user.upload_limit_a_day = upload_limit_a_day
            try:
                db.execute(
                    f"UPDATE users SET upload_limit_a_day = {upload_limit_a_day} WHERE id = {userId}"
                )
                # insert log
                log = LogsScheme(
                    user_id=userId,
                    action=LOGI.USER_LIMIT,
                    valueInt=upload_limit_a_day,
                    valueStr=",".join([user.name, str(userId)]),
                    vaueInfo=f"User {user.name} upload limit set to {upload_limit_a_day}",
                )
                db.add(log)
            except Exception as e:
                return {"status": "error", "message": str(e)}


@admin_router.get("/get_user_list", tags=["users_manage"])
async def get_user_list(db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
    if user.is_admin == 0:
        return {"status": "error", "message": "You are not authorized to perform this action."}
    else:
        users = db.query(UserScheme).all()
        data = []
        for u in users:
            data.append({
                "id": u.id,
                "username": u.name,
                "email": u.email,
                "is_admin": u.is_admin,
                "is_audited": u.is_audited,
                "upload_limit_a_day": u.upload_limit_a_day,
            })
        return {"status": "success", "data": data}


@admin_router.get("/audit_user_upload", tags=["users_manage"])
async def audit_user_upload(userIds: List[User], db: Session = Depends(get_db_commit), user: User = Depends(get_current_active_user)):
    if user.is_admin == 0:
        return {"status": "error", "message": "You are not authorized to perform this action."}
    else:
        # for each user in userIds
        for userId in userIds:
            user = db.query(UserScheme).filter(
                UserScheme.id == userId).first()
            if user is None:
                return {"status": "error", "message": "User not found."}
            else:
                # query user upload log num with timestamp
                upload_log = db.query(LogsScheme).filter(
                    LogsScheme.user_id == user.id).filter(
                    LogsScheme.action == LOGI.USER_RESOLVE).all()
                if upload_log:
                    return {"status": "success", "data": upload_log}
                else:
                    return {"status": "error", "message": "用户未找到"}

@admin_router.get("/audit_user_query", tags=["users_manage"])
async def audit_user_upload(userIds: List[User], db: Session = Depends(get_db_commit), user: User = Depends(get_current_active_user)):
    if user.is_admin == 0:
        return {"status": "error", "message": "You are not authorized to perform this action."}
    else:
        for userId in userIds:
            user = db.query(UserScheme).filter(
                UserScheme.id == userId).first()
            if user is None:
                return {"status": "error", "message": "User not found."}
            else:
                # query user upload log num with timestamp
                query_log = db.query(LogsScheme).filter(
                    LogsScheme.user_id == user.id).filter(
                    LogsScheme.action == LOGI.USER_QUERY).all()
                if query_log:
                    return {"status": "success", "data": query_log}
                else:
                    return {"status": "error", "message": "用户未找到"}                
