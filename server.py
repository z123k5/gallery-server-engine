import base64
import io
import json
import os

from fastapi.responses import JSONResponse, StreamingResponse

import numpy as np
import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import clip
import PIL
from PIL import Image
from src.UserManager import User, UserInDB, create_jwt_token, decode_token, get_current_active_user, hash_password
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends, File, HTTPException, Request, Response, status
import uvicorn
import fastapi
from fastapi import FastAPI
# from fastapi.openapi.docs import (
#     get_redoc_html,
#     get_swagger_ui_html,
#     get_swagger_ui_oauth2_redirect_html,
# )
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import sys
from sqlalchemy.orm import Session
from src.Database import get_db, get_db_commit
from sqlalchemy import text
from src.models import UserScheme
from sqlite3 import Connection, Cursor
from src.Logger import logger

logger.info("Server Starting...")
logger.info("Importing Server Models...")


logger.info("Importing VL Models...")

logger.info("Creating Context...")
app = fastapi.FastAPI()


"""
Swagger UI
"""

"""
Backend
"""

"""CORS Middleware"""
origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def options_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        # 返回允许的 HTTP 方法和其他相关头部信息
        headers = {
            "allow": "GET, POST, PUT, DELETE",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE",
            "Access-Control-Allow-Headers": "*",
        }
        return JSONResponse(content=None, status_code=200, headers=headers)
    # 如果不是 OPTIONS 请求，则继续处理请求
    response = await call_next(request)
    return response
""""""


@app.get("/")
async def index(user: User = Depends(get_current_active_user)):
    return {"message": f'Hello {user.username} !'}


@app.post("/users/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db_commit)):
    user = db.query(UserScheme).filter(
        UserScheme.username == form_data.username).first()
    if user is None or user.password != hash_password(form_data.password):
        raise fastapi.HTTPException(
            status_code=400, detail="Incorrect username or password")
    else:
        token, expire = create_jwt_token({"sub": user.username})
        db.execute(
            text("UPDATE users SET token_expire = :expire WHERE username = :username"),
            {"expire": expire, "username": user.username})
        return {"access_token": token, "token_type": "bearer"}

"""接受token，如果token无效或过期，则返回401，如果有效，延长时间，并返回200，表示用户活跃，"""


@app.post("/users/active")
async def active(db_user: User = Depends(get_current_active_user), db: Session = Depends(get_db_commit)):
    token, expire = create_jwt_token({"sub": db_user.username})
    db.execute(
        text("UPDATE users SET token_expire = :expire WHERE username = :username"),
        {"expire": expire, "username": db_user.username})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/users/register")
async def register(form_data: OAuth2PasswordRequestForm = Depends(), db: Cursor = Depends(get_db_commit)):
    user = db.query(UserScheme).filter(
        UserScheme.username == form_data.username).first()
    if user is not None:
        raise fastapi.HTTPException(
            status_code=400, detail="User already exists")
    db.execute(
        text("""
        INSERT INTO users (username, password, email, full_name, disabled, token_expire)
        VALUES (:username, :password, :email, :full_name, :disabled, :token_expire)
    """),
        {
            "username": form_data.username,
            "password": hash_password(form_data.password),
            "email": form_data.email,
            "full_name": form_data.full_name,
            "disabled": False,
            "token_expire": datetime.now()
        }
    )


"""
    Engine Function
    1. Client upload photo thumbnails
    2. Server calculate the features of the thumbnails
    3. Server Respond the features to the client
"""


@app.post("/engine/resolve")
async def engine_resolve(
    file: fastapi.UploadFile = fastapi.File(...),
    current_user: User = Depends(get_current_active_user),
):
    img = Image.open(file.file).convert("RGB")

    inputs = image_processor(images=img, return_tensors="pt")

    with torch.no_grad():
        features = image_encoder.get_image_features(**inputs)
        features = features / features.norm(dim=1, keepdim=True)

    byteArray = features.cpu().numpy().tobytes()
    
    encoded_data = base64.b64encode(byteArray).decode("utf-8")
    # print(encoded_data)

    return {"feat": encoded_data}


@app.get("/engine/query")
async def engine_query(
    query: str,
    current_user: User = Depends(get_current_active_user),
):
    # 1. Encode the query
    inputs = text_tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        features = text_encoder(**inputs).logits
        # Normalize
        features = features / features.norm(dim=1, keepdim=True)

    ret = features.cpu().numpy().tobytes()
    # print("2:sizeof return tensor bytes = ", len(ret))

    return StreamingResponse(
        content=io.BytesIO(ret),
        media_type="application/octet-stream",
    )


# TODO: delte test
def test_procedure():
    global test_data
    # 1. Encode the query
    inputs = text_tokenizer("123", return_tensors="pt")
    with torch.no_grad():
        features = text_encoder(**inputs).logits
        # Normalize
        features = features / features.norm(dim=1, keepdim=True)

    ret = features.cpu().numpy().tobytes()
    print("2:sizeof return tensor bytes = ", len(ret))

    test_data = ret

#####


if __name__ == "__main__":
    logger.info("Starting server...")

    # Print System Info
    logger.info("System Info:")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"PIL Version: {PIL.__version__}")
    logger.info(f"CLIP Version: {clip.__version__}")
    logger.info(f"FastAPI Version: {fastapi.__version__}")
    logger.info(f"Uvicorn Version: {uvicorn.__version__}")

    logger.info("Loading Models...")
    text_tokenizer = BertTokenizer.from_pretrained(
        "./Taiyi-CLIP-Roberta-large-326M-Chinese")
    text_encoder = BertForSequenceClassification.from_pretrained(
        "./Taiyi-CLIP-Roberta-large-326M-Chinese").eval()
    logger.info(f"Text Model Encoder: Taiyi-CLIP-Roberta-large-326M-Chinese")

    image_encoder = CLIPModel.from_pretrained(
        "./clip-vit-large-patch14").eval()
    image_processor = CLIPProcessor.from_pretrained(
        "./clip-vit-large-patch14")
    logger.info(f"Image Model Encoder: clip-vit-large-patch14")

    logger.info("Loading DataBase...")

    test_procedure()

    uvicorn.run(app, host="0.0.0.0", port=8443) #,
                # ssl_certfile="./cert/certificate.crt", ssl_keyfile="./cert/private.key")
