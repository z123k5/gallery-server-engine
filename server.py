import base64
import io
import json
import os
from typing import List

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
from datetime import datetime, timedelta
import sys
from sqlalchemy.orm import Session
from src.Database import get_db, get_db_commit
from sqlalchemy import text
from src.models import UserScheme
from sqlite3 import Connection, Cursor
from src.Logger import logger

# Langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate


# Asyncio
import asyncio


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


@app.middleware("https")
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
    tags: str = fastapi.Form(...),
    current_user: User = Depends(get_current_active_user),
):
    img = Image.open(file.file).convert("RGB")

    inputs = image_processor(images=img, return_tensors="pt")

    with torch.no_grad():
        features = image_encoder.get_image_features(**inputs)
        features = features / features.norm(dim=1, keepdim=True)

    """
    客户端发送一个类别数组，服务端返回这些类别归一化后的特征向量
    客户端将用于对每个图片打上标签，标签个数根据相似度阈值确定
    """
    tags_list = json.loads(tags)
    # calculate the features from the text batch
    inputs = text_tokenizer(tags_list, return_tensors="pt", padding=True)['input_ids']
    with torch.no_grad():
        features_texts = text_encoder(inputs).logits
        # Normalize
        features_texts = features_texts / \
            features_texts.norm(dim=1, keepdim=True)
    
    # cosine similarity
    logit_scale = image_encoder.logit_scale.exp()
    logits_per_image = logit_scale * features @ features_texts.t()
    probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
    probs = probs[0]
    # select the classes with probability > 0.6
    selected_classes = []
    for i in range(len(tags_list)):
        if probs[i] > 0.6:
            selected_classes.append(tags_list[i])

    print("Selected Classes: ", selected_classes)

    byteArray = features.cpu().numpy().tobytes()
    
    encoded_data = base64.b64encode(byteArray).decode("utf-8")
    # print(encoded_data)

    return {"feat": encoded_data, "tags": selected_classes}

@app.get("/engine/query")
async def engine_query(
    query: str,
    current_user: User = Depends(get_current_active_user),
):
    global logger
    # 1. Encode the query
    inputs = text_tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        features = text_encoder(**inputs).logits
        # Normalize
        features = features / features.norm(dim=1, keepdim=True)

    ret = features.cpu().numpy().tobytes()
    # print("2:sizeof return tensor bytes = ", len(ret))
    logger.info(f"Query OK: {query}")

    return StreamingResponse(
        content=io.BytesIO(ret),
        media_type="application/octet-stream",
    )


### Preprocessing
model =  os.environ.get('MODEL', '')
api_key = os.environ.get('API_KEY', '')
llm = ChatOpenAI(model_name=model, temperature='0.7',
                openai_api_key=api_key)

metaParserTemplateStr = """你是一个json解析器，专门解析对查找的照片的信息，你将输入的文本通过理解，若文本没有呈现出某个键的信息，请不要解析这个键，否则解析出键对应的值，并以condensed json形式输出，可选的键有：[date, loc, device]，date:查找照片的日期时间(以今天{date}为参照日期，格式为YYYY-MM-DD HH:MM:SS或者区间YYYY-MM-DD HH:MM:SS~YYYY-MM-DD HH:MM:SS，例如：1."15年到16年的照片"解析为"2015-01-01 00:00:00~2016-01-01 00:00:00"2."去年以前的照片"解析为"1949-01-01 00:00:00~{dateLastYear}"，3."去年的照片"解析为"{dateLastYear}"，4."小时候的照片"由于日期不明确，故不解析date键)，loc:查找照片的地点(例如：1.北京市, 2.武功山)，device:查找照片的设备(如：1.iPhone 13, 2.Nikon zf)，输出格式如下:
{{date: "<date as you understand it>", loc: "<address as you understand it in china>", device: "<photo device as you understand it>",}}
输入的文本是："{query}"
"""

tagParserTemplateStr = """你是一个json解析器，专门解析输入文本中出现的标签，若文本没有呈现出标签，请以`无`作为结果，否则解析出标签，简洁回答，直接输出[tag1, tag2, ...]，可选的标签有：[{tags}]
输入的文本是："{query}"
"""

promptParserTemplateStr = """你是一个提示词生成器，专门解析输入文本对应的提示词，若输入文本中含有敏感信息如"时间、地点、拍摄设备"的信息，请做过滤处理，然后输出一个过滤后的信息，不包含以上敏感信息的文本提示词，你不能输出其他无关信息。
输入的文本是："{query}"
"""

#####

"""
客户端发送一个字符串查询，返回一个综合的查询结果
"""
@app.get("/engine/query_v2")
async def engine_query_v2(
    query: str,
    tagsAppended: str = "",
    current_user: User = Depends(get_current_active_user),
):
    global logger

    # 1. Search by Meta data
    # llm resolve metadata
    metaParserTemplate = PromptTemplate(template=metaParserTemplateStr, input_variables=["date", "dateLastYear", "query"])
    dateLastYear = datetime.now() - timedelta(days=365)
    metaParserTemplate = metaParserTemplate.partial(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dateLastYear=dateLastYear)
    metaParser = LLMChain(llm=llm, prompt=metaParserTemplate,
                        output_key="out")

    # 2. Get Tags
    # llm resolve tags
    tagParserTemplate = PromptTemplate(template=tagParserTemplateStr, input_variables=["tags", "query"])
    tagParserTemplate = tagParserTemplate.partial(tags=tagsAppended)
    tagParser = LLMChain(llm=llm, prompt=tagParserTemplate,
                        output_key="out")
    # new date, formate:YYYY-MM-DD HH:MM:SS

    # 3. calculate the features
    # llm resolve prompt
    promptParserTemplate = PromptTemplate(template=promptParserTemplateStr, input_variables=["query"])
    promptParser = LLMChain(llm=llm, prompt=promptParserTemplate,
                        output_key="out")
    responseMeta, responseTag, responsePrompt = await asyncio.gather(metaParser.ainvoke(query), tagParser.ainvoke(query), promptParser.ainvoke(query))


    inputs = text_tokenizer(responsePrompt.out, return_tensors="pt")
    with torch.no_grad():
        features = text_encoder(**inputs).logits
        # Normalize
        features = features / features.norm(dim=1, keepdim=True)

    responseFeat = features.cpu().numpy().tobytes()

    # print("2:sizeof return tensor bytes = ", len(ret))
    logger.info(f"Query OK: {query}")

    # 4. return to client
    return JSONResponse(content={
        "meta": responseMeta.out,
        "tag": responseTag.out,
        "feat": base64.b64encode(responseFeat).decode("utf-8")
    })


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

    uvicorn.run(app, host="0.0.0.0", port=8443,
                ssl_certfile="./cert/certificate.crt", ssl_keyfile="./cert/private.key")
