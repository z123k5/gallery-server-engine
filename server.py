import sys
from datetime import datetime
import fastapi
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from src.UserManager import User, UserInDB, create_jwt_token, decode_token, get_current_active_user, hash_password
import uvicorn
import sqlite3

import PIL
import clip
from transformers import CLIPProcessor, CLIPModel
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
import torch


app = fastapi.FastAPI()
logger = fastapi.Logger(__name__, level="INFO")


def InitDatabase():
    db.execute(
        '''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, email TEXT, full_name TEXT, disabled BOOLEAN, token_expire DATE)''')
    conn.commit()

@app.get("/")
def index(user: User = Depends(get_current_active_user)):
    payload = decode_token(user.username)
    return {"message": f'Hello {payload['sub']} !'}


@app.post("/users/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db.execute("SELECT * FROM users WHERE username = ?", (form_data.username,))
    user = db.fetchone()
    if user is None or user[1] != form_data.password:
        raise fastapi.HTTPException(
            status_code=400, detail="Incorrect username or password")
    else:
        return {"access_token": create_jwt_token({"sub": user[0]}), "token_type": "bearer"}

@app.post("/users/register")
def register(form_data: OAuth2PasswordRequestForm = Depends()):
    db.execute("SELECT * FROM users WHERE username = ?", (form_data.username,))
    if db.fetchone() is not None:
        raise fastapi.HTTPException(
            status_code=400, detail="User already exists")
    db.execute(
        "INSERT INTO users (username, password, email, full_name, disabled, token_expire) VALUES (?, ?, ?, ?, ?, ?)",
        (form_data.username, hash_password(form_data.password), form_data.email, form_data.full_name, False, datetime.now()))
    conn.commit()

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/users/me/items")
def read_own_items(current_user: User = Depends(get_current_active_user)):
    pass

"""
    Engine Function
    1. Client upload photo thumbnails
    2. Server calculate the features of the thumbnails
    3. Server Respond the features to the client
"""
@app.get("/engine/handleThumbnails")
def uploadThumbnails(thumbnail_pack ,current_user: User = Depends(get_current_active_user)):
    pass



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
    conn = sqlite3.connect("./db/data.db")
    db = conn.cursor()
    InitDatabase()
    uvicorn.run(app, host="0.0.0.0", port=8000)
