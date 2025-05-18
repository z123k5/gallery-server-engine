from src.Logger import logger           # noqa: E402
logger.info("Server Starting...")       # noqa: E402
logger.info("Importing Libraries...")   # noqa: E402

import asyncio
from src.admin import admin_router
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_deepseek import ChatDeepSeek
from sqlite3 import Connection, Cursor
from src.models import ClassesScheme, LogsScheme, MediaClassesScheme, MediaMetadataScheme, MediaScheme, UserScheme
from sqlalchemy import CursorResult, bindparam, select, text
from sqlalchemy.orm import joinedload
from src.UserManager import User, UserInDB, create_jwt_token, decode_token, get_current_active_user, hash_password
from sqlalchemy.orm import Session
import sys
from datetime import datetime, timedelta
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
import fastapi
import uvicorn
from fastapi import Depends, File, Form, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from src.shared import LOGI, globalVarUseLock, vlmModelLock
from src.Database import get_db, get_db_commit, DBContextManager, db_executor
from PIL import Image, UnidentifiedImageError
import PIL
import clip
from transformers import CLIPProcessor, CLIPModel
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
import torch
import requests
from pydantic import BaseModel
import numpy as np
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List
import re
import os
import json
import io
import base64



# from fastapi.openapi.docs import (
#     get_redoc_html,
#     get_swagger_ui_html,
#     get_swagger_ui_oauth2_redirect_html,
# )

# Langchain


# Asyncio


logger.info("Importing Server Models...")


logger.info("Importing VL Models...")

logger.info("Creating Context...")
app = fastapi.FastAPI()

app.include_router(admin_router,
                   prefix="/admin",
                   tags=["admin"])


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
    return {"message": f'Hello {user.name} !'}


@app.post("/users/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = None
    try:
        user = db_executor.run_db_task(lambda db: db.query(UserScheme).filter(
            UserScheme.name == form_data.username).first())
        user = UserScheme(**user)
    except Exception as e:
        logger.error(e)

    # insert log to db
    try:
        # sql = text("""
        # INSERT INTO logs (user_id, action, valueBool, valueInt, valueInfo)
        # VALUES (:user_id, :action, :valueBool, :valueInt, :valueInfo);""")
        # db.execute(sql, {
        #     "user_id": user.id,
        #     "action": LOGI.USER_LOGIN.value,    #  用户登录
        #     "valueBool": 1 if user is not None and user.is_audited else 0,  # 1:成功 0:失败
        #     "valueInt": user.id if user is not None else 0,   # 用户id
        #     "valueInfo": f"User {form_data.username} login"
        # })
        db_executor.run_db_task(lambda db: db.add(LogsScheme(
            user_id=user.id,
            action=LOGI.USER_LOGIN.value,  # 用户登录
            valueBool=1 if user is not None and user.is_audited else 0,  # 1:成功 0:失败
            valueInt=user.id if user is not None else 0,   # 用户id
            valueInfo=f"User {form_data.username} login"
        )))
    except Exception as e:
        logger.error(f"Insert log Error: {e}")

    if user is None or user.password != hash_password(form_data.password):
        raise fastapi.HTTPException(
            status_code=401, detail="用户名或密码错误")
    else:
        if user.is_audited == 0:
            raise fastapi.HTTPException(
                status_code=403, detail="用户未审核")

        token, expire = create_jwt_token({"sub": user.name})
        try:
            # db.execute(
            #     text("UPDATE users SET token_expire = :expire WHERE name = :name"),
            #     {"expire": expire, "name": user.name})
            db_executor.run_db_task(lambda db: db.query(UserScheme).filter(
                UserScheme.name == user.name).update({
                    "token_expire": expire
                }))
        except Exception as e:
            logger.error(e)
        return {"access_token": token, "token_type": "bearer"}

"""接受token，如果token无效或过期，则返回401，如果有效，延长时间，并返回200，表示用户活跃，"""


@app.post("/users/active")
async def active(db_user: User = Depends(get_current_active_user)):
    token, expire = create_jwt_token({"sub": db_user.name})
    try:
        db_executor.run_db_task(lambda db: db.query(UserScheme).filter(
            UserScheme.name == db_user.name).update({
                "token_expire": expire
            }))
    except Exception as e:
        logger.error(e)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/users/register")
async def register(username: str = Form(...),
                   password: str = Form(...),
                   email: str = Form(...)):
    # 检查用户是否已存在
    user = db_executor.run_db_task(lambda db: db.query(UserScheme).filter(
        UserScheme.name == username).first())
    if user is not None:
        raise fastapi.HTTPException(status_code=403, detail="用户已存在")

    try:
        def create_user_and_log(db):
            new_user = UserScheme(
                name=username,
                password=hash_password(password),
                email=email,
                active=True,
                token_expire=datetime.now()
            )
            db.add(new_user)
            db.flush()  # 刷新以获取新插入对象的主键

            db.add(LogsScheme(
                user_id=new_user.id,
                action=LOGI.USER_REGISTER.value,
                valueBool=1,
                valueInfo=f"User {username} register"
            ))

        db_executor.run_db_task(create_user_and_log)

    except Exception as e:
        logger.error(e)
        raise fastapi.HTTPException(
            status_code=500, detail="Internal Server Error")

    return {"message": "注册成功"}


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
    try:
        img = Image.open(file.file).convert("RGB")

        #
    except UnidentifiedImageError:
        raise fastapi.HTTPException(
            status_code=415, detail="Invalid image format")

    inputs = image_processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        features = image_encoder.get_image_features(**inputs)
        features = features / features.norm(dim=1, keepdim=True)

    """
    客户端发送一个类别数组，服务端返回这些类别归一化后的特征向量
    客户端将用于对每个图片打上标签，标签个数根据相似度阈值确定
    """
    tags_list = json.loads(tags)

    try:
        vlmModelLock.acquire()
        # calculate the features from the text batch
        inputs = text_tokenizer(tags_list, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            features_texts = text_encoder(
                input_ids=input_ids, attention_mask=attention_mask).logits
            # Normalize
            features_texts = features_texts / \
                features_texts.norm(dim=1, keepdim=True)
    except Exception as e:
        logger.error(e)
        raise fastapi.HTTPException(
            status_code=500, detail="Internal Server Error")
    finally:
        vlmModelLock.release()

    # cosine similarity
    logit_scale = image_encoder.logit_scale.exp()
    logits_per_image = logit_scale * features @ features_texts.t()
    probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
    probs = probs[0]
    # select the classes with probability > 0.3
    selected_classes = []
    for i in range(len(tags_list)):
        if probs[i] > 0.3:
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

    try:
        vlmModelLock.acquire()
        inputs = text_tokenizer(query, return_tensors="pt").to(device)
        with torch.no_grad():
            features = text_encoder(**inputs).logits
            # Normalize
            features = features / features.norm(dim=1, keepdim=True)
    except:
        logger.error("Error in encode_query")
    finally:
        vlmModelLock.release()

    ret = features.cpu().numpy().tobytes()
    # print("2:sizeof return tensor bytes = ", len(ret))
    logger.info(f"Query OK: {query}")

    return StreamingResponse(
        content=io.BytesIO(ret),
        media_type="application/octet-stream",
    )


# Preprocessing
model = os.environ.get('MODEL', '')
# api_key = os.environ.get('API_KEY', '')
# base_url = os.environ.get('BASE_URL', '')
llm = ChatDeepSeek(model_name=model, temperature='0.7')

metaParserTemplateStr = """你是一个json解析器，专门解析对查找的照片的信息，你将输入的文本通过理解，若文本没有呈现出某个键的信息，请不要解析这个键，否则解析出键对应的值，并以纯文本(花括号开头结尾)形式输出，可选的键有：[date, loc, device]，date:查找照片的日期时间(以今天{date}为参照日期，请以给出的时间精确到年、月或者日，如果只精确到月，那开始日期1号，结束日期月末；如果只精确到年，那开始日期1月1号，结束日期到年末，格式为区间YYYY-MM-DD HH:MM:SS~YYYY-MM-DD HH:MM:SS，例如：1."15年到16年的照片"解析为"2015-01-01 00:00:00~2016-01-01 00:00:00"2."去年以前的照片"解析为"1949-10-01 00:00:00~{dateLastYear}"，3."小时候的照片"由于日期不明确，故不解析date键)，loc:查找照片的地点(例如：1.北京市, 2.武功山)，device:拍摄用的设备，必须是相机类(如：1.iPhone 13, 2.Nikon zf)，输出格式如下:
{{date: "<date as you understand it>", loc: "<address as you understand it in china>", device: "<photo device as you understand it>",}}
输入的文本是："{query}"
"""

tagParserTemplateStr = """你是一个json解析器，专门解析输入文本中出现的标签，若文本没有呈现出标签，请以`无`作为结果，否则解析出标签，简洁回答，直接输出["tag1", "tag2", ...]，可选的标签有：[{tags}]
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
    metaParserTemplate = PromptTemplate(template=metaParserTemplateStr, input_variables=[
                                        "date", "dateLastYear", "query"])
    dateLastYear = datetime.now() - timedelta(days=365)
    metaParserTemplate = metaParserTemplate.partial(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dateLastYear=dateLastYear)
    metaParser = LLMChain(llm=llm, prompt=metaParserTemplate,
                          output_key="out")

    # 2. Get Tags
    # llm resolve tags
    tagParserTemplate = PromptTemplate(
        template=tagParserTemplateStr, input_variables=["tags", "query"])
    tagParserTemplate = tagParserTemplate.partial(tags=tagsAppended)
    tagParser = LLMChain(llm=llm, prompt=tagParserTemplate,
                         output_key="out")
    # new date, formate:YYYY-MM-DD HH:MM:SS

    # 3. calculate the features
    # llm resolve prompt
    promptParserTemplate = PromptTemplate(
        template=promptParserTemplateStr, input_variables=["query"])
    promptParser = LLMChain(llm=llm, prompt=promptParserTemplate,
                            output_key="out")
    responseMeta, responseTag, responsePrompt = await asyncio.gather(metaParser.ainvoke(query), tagParser.ainvoke(query), promptParser.ainvoke(query))

    # Meta data format:
    """
    {
    ?"date": "YYYY-MM-DD HH:MM:SS",
    ?"date_start": "YYYY-MM-DD HH:MM:SS",
    ?"date_end": "YYYY-MM-DD HH:MM:SS",
    ?"loc": "北京",
    ?"device": "iPhone 13"
    }
    """
    # filter meta data from llm output
    rspRaw: str = responseMeta["out"]
    rspJson = ""
    tl = rspRaw.find("{")
    tr = rspRaw.find("}")
    if tl != -1 and tr != -1 and tl < tr:
        rspJson = rspRaw[tl: tr + 1]
    else:
        rspJson = "{}"

    try:
        def safe_string_to_dict(raw_str):
            # 1. 去掉两边空白
            raw_str = raw_str.strip()

            # 2. 补齐 key 的引号：找到形如 key: 的模式，加上引号
            # 注意要避免破坏 value 中的 : 字符
            fixed_str = re.sub(r'(?<={|,)\s*(\w+)\s*:', r'"\1":', raw_str)

            # 3. 替换单引号为双引号（如果你输入里有单引号）
            fixed_str = fixed_str.replace("'", '"')

            # 4. 防止中文逗号（全角逗号）意外出现
            fixed_str = fixed_str.replace('，', ',')

            # 5. 解析
            try:
                result = json.loads(fixed_str)
            except json.JSONDecodeError as e:
                print("解析失败：", e)
                result = None

            return result
        rspJson = safe_string_to_dict(rspJson)
    except:
        rspJson = {}
        pass

    json_meta = rspJson
    meta_data = {}
    if "date" in json_meta.keys():
        date = json_meta["date"]
        try:
            if "~" in date:
                date_start, date_end = date.split("~")
                # date_start = datetime.strptime(date_start, "%Y-%m-%d %H:%M:%S")
                # date_end = datetime.strptime(date_end, "%Y-%m-%d %H:%M:%S")
                meta_data["date_start"] = date_start
                meta_data["date_end"] = date_end
            else:
                date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                meta_data["date"] = date.strftime("%Y-%m-%d %H:%M:%S")
        except:
            date = ""
    if "loc" in json_meta:
        loc = json_meta["loc"]
    else:
        loc = ""

    if "device" in json_meta:
        meta_data["device"] = json_meta["device"]

    if loc != "":
        meta_data["loc"] = loc

    with torch.no_grad():
        inputs = text_tokenizer(
            responsePrompt["out"], return_tensors="pt").to(device)
        features = text_encoder(**inputs).logits
        # Normalize
        features = features / features.norm(dim=1, keepdim=True)

    responseFeat = features.cpu().numpy().tobytes()

    # print("2:sizeof return tensor bytes = ", len(ret))
    logger.info(f"Query OK: {query} -> {responsePrompt['out']}")

    # 4. return to client
    return JSONResponse(content={
        "meta": meta_data,
        "prompt": responsePrompt["out"],
        "tags":  eval(responseTag["out"]) if responseTag["out"] != "" and responseTag["out"] != "无" else [],
        "feat": base64.b64encode(responseFeat).decode("utf-8")
    })


class MetaData(BaseModel):
    identifier: str = fastapi.Form(...)
    name: str = fastapi.Form(...)
    type: str = fastapi.Form(...)
    created_at: datetime = fastapi.Form(...)
    lat: float = fastapi.Form(...)
    lon: float = fastapi.Form(...)
    dev: str = fastapi.Form(...)


@app.post("/engine/resolve_v3")
async def engine_resolve_v3(
    file: fastapi.UploadFile = fastapi.File(...),
    identifier: str = fastapi.Form(...),
    name: str = fastapi.Form(...),
    type: str = fastapi.Form(...),
    created_at: int = fastapi.Form(...),
    lat: float = fastapi.Form(...),
    lon: float = fastapi.Form(...),
    dev: str = fastapi.Form(...),
    # tags: str = fastapi.Form(...),
    current_user: User = Depends(get_current_active_user)
):
    global features_tags, tags_list, logger

    meta_data = MetaData(identifier=identifier, name=name, type=type,
                         created_at=created_at, lat=lat, lon=lon, dev=dev)

    # 1. Image semantic analyze
    try:
        vlmModelLock.acquire()
        img = Image.open(file.file).convert("RGB")
        inputs = image_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            features = image_encoder.get_image_features(**inputs)
            features = features / features.norm(dim=1, keepdim=True)
    except UnidentifiedImageError:
        raise fastapi.HTTPException(
            status_code=415, detail="Invalid image format")
    finally:
        vlmModelLock.release()

    # 2. classification from cosine similarity
    try:
        globalVarUseLock.acquire()
        logit_scale = image_encoder.logit_scale.exp()
        logits_per_image = logit_scale * features @ features_tags.t()
        probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
        probs = probs[0]
        # select the classes with probability > 0.3
        selected_classes = []
        for i in range(len(tags_list)):
            if probs[i] > 0.3:
                selected_classes.append(tags_list[i])
    except Exception as e:
        logger.error("Error in resolve_v3, in tag select: {}".format(e))
    finally:
        globalVarUseLock.release()

    # 3. lat, lng to loc
    # request
    loc = ""
    try:
        url = f"http://api.tianditu.gov.cn/geocoder?postStr={{'lon':{meta_data.lon},'lat':{meta_data.lat},'ver':1}}&type=geocode&tk=6cc422bf3bab18d99e9a1be91b7b2afb"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            loc = data["result"]["formatted_address"]
    except Exception as e:
        logger.error(f"Error: {e}")

    # 3. Insert into DB
    """media table
    identifier TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    thumbnail TEXT,
    processInfo INTEGER DEFAULT 0,
    feature BLOB DEFAULT 0,
    meta_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    meta_loc TEXT,
    meta_device TEXT
    """

    try:
        # Insert into media table
        # sql = text("""
        #     INSERT OR IGNORE INTO media (user_id, identifier, name, type, created_at, source, processInfo, feature)
        #     VALUES (:user_id, :identifier, :name, :type, :created_at, :source, :processInfo, :feature);
        #     """)
        # db.execute(sql, {
        #     "user_id": current_user.id,
        #     "identifier": meta_data.identifier,
        #     "name": meta_data.name,
        #     "type": meta_data.type,
        #     "created_at": meta_data.created_at,
        #     "source": file.filename,
        #     "processInfo": 1,
        #     "feature": features.cpu().numpy().tobytes(),
        # })
        db_executor.run_db_task(lambda db: db.execute(text("""
                        INSERT OR IGNORE INTO media (user_id, identifier, name, type, created_at, source, processInfo, feature) VALUES (:user_id, :identifier, :name, :type, :created_at, :source, :processInfo, :feature);"""
        ), {
            "user_id": current_user.id,
            "identifier": meta_data.identifier,
            "name": meta_data.name,
            "type": meta_data.type,
            "created_at": meta_data.created_at,
            "source": file.filename,
            "processInfo": 1,
            "feature": features.cpu().numpy().tobytes(),
        }))


        # select tag names from db
        # sql = text("""
        #             SELECT id FROM classes WHERE user_id = :user_id AND name IN (:tags)"""
        #            )
        # tagIds = db.execute(sql, {
        #     "user_id": current_user.id,
        #     "tags": ",".join(selected_classes)
        # }).fetchall()
        tagIds = db_executor.run_db_task(lambda db: db.query(ClassesScheme).filter(
            ClassesScheme.user_id == current_user.id, ClassesScheme.name.in_(selected_classes)).all())

        # Insert into media_classes table
        if len(selected_classes) != 0:
            # sql = text("""
            #     INSERT OR IGNORE INTO media_classes (user_id, media_id, class_id)
            #     VALUES (:user_id, :media_id, :class_id);
            #     """)
            # for tag in tagIds:
            #     db.execute(sql, {
            #         "user_id": current_user.id,
            #         "media_id": meta_data.identifier,
            #         "class_id": tag.id,
            #     })
            # db_executor.run_db_task(lambda db: db.add_all([MediaClassesScheme(
            #     user_id=current_user.id, media_id=meta_data.identifier, class_id=tag['id']) for tag in tagIds]))
            
            # 先构造参数列表
            params = [
                {
                    "user_id": current_user.id,
                    "media_id": meta_data.identifier,
                    "class_id": tag['id']
                }
                for tag in tagIds
            ]

            # 执行批量插入
            if params:
                db_executor.run_db_task(lambda db: db.execute(text("""
                    INSERT OR IGNORE INTO media_classes (user_id, media_id, class_id)
                    VALUES (:user_id, :media_id, :class_id)
                """), params))

        # Insert into media_metadata table
        # sql = text("""
        #     INSERT OR IGNORE INTO media_metadata (user_id, media_id, exif_lat, exif_lon, exif_dev, location) VALUES (:user_id, :identifier, :exif_lat, :exif_lon, :exif_dev, :location)""")
        # db.execute(sql, {
        #     "user_id": current_user.id,
        #     "identifier": meta_data.identifier,
        #     "exif_lat": meta_data.lat,
        #     "exif_lon": meta_data.lon,
        #     "exif_dev": meta_data.dev,
        #     "location": loc
        # })
        # db_executor.run_db_task(lambda db: db.add(MediaMetadataScheme(user_id=current_user.id, media_id=meta_data.identifier,
        #                         exif_lat=meta_data.lat, exif_lon=meta_data.lon, exif_dev=meta_data.dev, location=loc)))
        # Core
        db_executor.run_db_task(lambda db: db.execute(text("""
            INSERT OR IGNORE INTO media_metadata (user_id, media_id, exif_lat, exif_lon, exif_dev, location) VALUES (:user_id, :identifier, :exif_lat, :exif_lon, :exif_dev, :location)"""), 
            { "user_id": current_user.id, "identifier": meta_data.identifier,
            "exif_lat": meta_data.lat, "exif_lon": meta_data.lon, "exif_dev": meta_data.dev, "location": loc}))
        


    except Exception as e:
        logger.error(f"Insert media Error: {e}")
        raise fastapi.HTTPException(
            status_code=500, detail="Insert media Error")

    logger.info(f"Resolve OK: {meta_data.identifier} -> {selected_classes}")

    # 日志表
    """CREATE TABLE IF NOT EXISTS logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(id),
        action INTEGER NOT NULL,
        valueBool TEXT,
        valueInt TEXT,
        valueStr TEXT,
        valueFloat TEXT,
        valueInfo TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    # insert log to db
    try:
        # sql = text("""
        # INSERT INTO logs (user_id, action, valueBool, valueInfo)
        # VALUES (:user_id, :action, :valueBool, :valueInfo);""")
        # db.execute(sql, {
        #     "user_id": current_user.id,
        #     "action": LOGI.USER_QUERY.value,
        #     "valueBool": 1,  # 1:成功 0:失败
        #     "valueInfo": f"Add media {meta_data.identifier} with tags {selected_classes}"
        # })
        db_executor.run_db_task(lambda db: db.add(LogsScheme(user_id=current_user.id, action=LOGI.USER_QUERY.value,
                                valueBool=1, valueInfo=f"Add media {meta_data.identifier} with tags {selected_classes}")))
    except Exception as e:
        logger.error(f"Insert log Error: {e}")

    return JSONResponse(content={
        "status": "OK",
        "identifier": meta_data.identifier,
        "tags": selected_classes,
        "location": loc,
        "feat": base64.b64encode(features.cpu().numpy().tobytes()).decode("utf-8")
    })


@app.get("/engine/query_v3")
async def engine_query_v3(
    query: str,
    tagsAppended: str = "",
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_commit)
):
    # 1. Get Meta data, tags and feat
    try:
        globalVarUseLock.acquire()
        if tagsAppended == "":
            nested_tags = tags_list
        else:
            nested_tags = tags_list + tagsAppended.split(",")
    except:
        logger.error("Error in getting tags")
        nested_tags = []
    finally:
        globalVarUseLock.release()

    rspv2 = await engine_query_v2(query, nested_tags, current_user)
    rspv2 = json.loads(rspv2.body.decode("utf-8"))

    meta_data = rspv2["meta"]
    feat_base64 = rspv2["feat"]
    feat = base64.b64decode(feat_base64)
    features_texts = torch.from_numpy(
        np.frombuffer(feat, dtype=np.float32).copy()).to(device)

    # 2. Filter criteria
    filter_criteria = []
    filter_criteria.append(UserScheme.id == current_user.id)
    if "date_start" in meta_data and "date_end" in meta_data:
        date_start = meta_data["date_start"]
        date_end = meta_data["date_end"]
        # 
        filter_criteria.append(
            MediaScheme.created_at.between(date_start, date_end))
    if "loc" in meta_data:
        loc = meta_data["loc"]
        filter_criteria.append(MediaMetadataScheme.location.like(f"%{loc}%"))
    if "device" in meta_data:
        meta_device = meta_data["device"]
        # device from media_metadata table
        filter_criteria.append(
            MediaMetadataScheme.exif_dev.like(f"%{meta_device}%"))
    try:
        if rspv2["tags"]:
            tagNames = rspv2["tags"]
            # stmt = text("""
            #     SELECT id
            #     FROM classes
            #     WHERE name IN :tagNames
            #     AND user_id = :user_id
            # """).bindparams(bindparam("tagNames", expanding=True))

            # tagIdsQueryResult = db.execute(
            #     stmt,
            #     {"tagNames": tagNames, "user_id": current_user.id}
            # ).all()
            tagIdsQueryResult = db_executor.run_db_task(lambda db: db.query(ClassesScheme).filter(
                ClassesScheme.name.in_(tagNames),
                ClassesScheme.user_id == current_user.id
            ).all())

            # tables: media, media_classes
            # 构建子查询
            subquery = (
                select(MediaClassesScheme.media_id)
                .where(MediaClassesScheme.class_id.in_([tag["id"] for tag in tagIdsQueryResult]))
                .subquery()
            )

            # 使用 select() 显式构造查询
            filter_criteria.append(
                MediaScheme.identifier.in_(select(subquery.c.media_id))
            )
        # query_result = db.query(MediaScheme).filter(
        #     *filter_criteria,
        #     MediaScheme.feature != None
        # ).all()
        query_result = db_executor.run_db_task(lambda db: db.query(MediaScheme)
            .join(UserScheme, MediaScheme.user_id == UserScheme.id)  # 显式 JOIN users 表
            .options(joinedload(MediaScheme.media_metadata))
            .join(MediaMetadataScheme, MediaScheme.identifier == MediaMetadataScheme.media_id)
            .filter(*filter_criteria, MediaScheme.feature != None)
            .all()
        )
    except Exception as e:
        logger.error(f"Query media Error: {e}")
        query_result = []

    # insert log to db
    try:
        # sql = text("""
        # INSERT INTO logs (user_id, action, valueBool, valueStr, valueInt, valueInfo)
        # VALUES (:user_id, :action, :valueBool, :valueStr, :valueInt, :valueInfo);""")
        # db.execute(sql, {
        #     "user_id": current_user.id,
        #     "action": LOGI.USER_QUERY.value,
        #     "valueBool": 1,  # 1:成功 0:失败
        #     "valueStr": query,  # 查询文本
        #     "valueInt": len(query_result),  # 查询结果数量
        #     "valueInfo": f"Query media with tags {rspv2['tags']} and meta data {meta_data}"
        # })
        db_executor.run_db_task(lambda db: db.add(LogsScheme(user_id=current_user.id, action=LOGI.USER_QUERY.value, valueBool=1,
                                valueStr=query, valueInt=len(query_result), valueInfo=f"Query media with tags {rspv2['tags']} and meta data {meta_data}")))
    except Exception as e:
        logger.error(f"Insert log Error: {e}")

    if len(query_result) == 0:
        raise fastapi.HTTPException(
            status_code=404, detail="No image found with the given criteria")

    image_data_ids = []
    for image in query_result:
        image_data_ids.append(image["identifier"])
    features = []
    for image in query_result:
        # image.feature is a blob, convert it to numpy array and convert it to torch tensor
        features.append(torch.from_numpy(
            np.frombuffer(image["feature"], dtype=np.float32)).to(device))
    features = torch.stack(features).to(device)

    # 4. calculate the similarity
    logit_scale = image_encoder.logit_scale.exp()
    logits_per_image = logit_scale * features @ features_texts.t()
    probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
    # print(
    #     "Top-10 search results:",
    #     [(id, prob, logits) for id, prob, logits in zip(image_data_ids, probs, logits_per_image)].sort(key=lambda x: x[1], reverse=True)[:10]
    # )

    # 5. fetch the top k images
    ret_images = []
    top_k = 100
    for i in range(top_k):
        max_index = int(np.argmax(probs))
        if probs[max_index] == 0:
            break
        ret_images.append(
            {"id": image_data_ids[max_index], "score": float(probs[max_index])})
        probs[max_index] = 0

    return JSONResponse(content={
        "meta": meta_data,
        "prompt": rspv2["prompt"],
        "tags": rspv2["tags"],
        "feat": feat_base64,
        "images": ret_images
    })


class MediaDeleteRequest(BaseModel):
    media_ids: List[str]
# handle a list of media_id in body request


@app.post("/users/media_delete")
async def user_media_delete(request: MediaDeleteRequest, user: User = Depends(get_current_active_user), db: Session = Depends(get_db_commit)):
    media_ids = request.media_ids
    if len(media_ids) == 0:
        raise fastapi.HTTPException(
            status_code=400, detail="No media id provided")
    rows_deleted = db.query(MediaScheme)\
        .filter(MediaScheme.identifier.in_(media_ids), MediaScheme.user_id == user.id)\
        .delete(synchronize_session=False)
    # insert log to db
    try:
        # sql = text("""
        # INSERT INTO logs (user_id, action, valueBool, valueStr, valueInt, valueInfo)
        # VALUES (:user_id, :action, :valueBool, :valueStr, :valueInt, :valueInfo);""")
        # db.execute(sql, {
        #     "user_id": user.id,
        #     "action": LOGI.USER_DELETE.value,
        #     "valueBool": 1,  # 1:成功 0:失败
        #     "valueInt": rows_deleted,  # 删除的图片数量
        #     "valueStr": ",".join(media_ids),  # 删除的图片id
        #     "valueInfo": f"Delete media {media_ids}"
        # })
        db_executor.run_db_task(lambda db: db.add(LogsScheme(user_id=user.id, action=LOGI.USER_DELETE.value,
                                valueBool=1, valueInt=rows_deleted, valueStr=",".join(media_ids), valueInfo=f"Delete media {media_ids}")))
    except Exception as e:
        logger.error(f"Insert log Error: {e}")
        return {"status": "error", "message": "Failed to delete media", "error": str(e)}

    return {"status": "success", "message": "Media deleted", "changes": rows_deleted}


def userInitProcess():
    global features_tags, tags_list

    # Init DB
    with DBContextManager() as db:
        db.execute(text("PRAGMA journal_mode=WAL;"))
        # Perform database upgrade statements
        from src.dbUpgradeStatements import dbUpgradeStatements

        # check sqlite db version
        db_version = 0
        try:
            # db_version = db.execute(text(
            #     "PRAGMA user_version;")).fetchone()[0]
            db_version = db_executor.run_db_task(lambda db: db.execute(text(
                "PRAGMA user_version;")).fetchone()[0])
            logger.info(f"System DB version: {db_version}")

            # check if db version is less than the latest version
            if db_version < dbUpgradeStatements[-1]["toVersion"]:
                for statement in dbUpgradeStatements[db_version:]:
                    db_version = statement["toVersion"]
                    logger.info(
                        f"Upgrading DB to version {statement['toVersion']}")
                    for sql in statement["statements"]:
                        # db.execute(text(sql))
                        db_executor.run_db_task(
                            lambda db: db.execute(text(sql)))
                    # db.execute(
                    #     text(f"PRAGMA user_version = {db_version};"))
                    db_executor.run_db_task(lambda db: db.execute(
                        text(f"PRAGMA user_version = {db_version};")))

                    if db_version == 1:
                        # Create default user
                        # db.execute(
                        #     text("INSERT OR IGNORE INTO users (name, password, email, active, token_expire) VALUES ('admin', :password, 'admin@example.com', 1, datetime('now', '+1 year'));"),
                        #     {"password": hash_password("admin")})
                        db_executor.run_db_task(lambda db: db.execute(
                            text("INSERT OR IGNORE INTO users (name, password, email, active, token_expire, is_audited, is_admin, upload_limit_a_day) VALUES ('admin', :password, 'admin@example.com', 1, datetime('now', '+1 year'), 1, 1, -1);"),
                            {"password": hash_password("admin")}))
        except Exception as e:
            logger.error(
                f"Error in DB upgrade: {e.with_traceback(None)}")

    # select the classes from the database
    tags_list = []
    # tags_list = ["人物", "动物", "植物", "食物", "建筑", "家具", "交通工具", "电子产品", "服装", "乐器", "屏幕截图"]
    with DBContextManager() as db:
        for tag in db.execute(text("SELECT name FROM classes")).all():
            tags_list.append(tag.name)
    # calculate the features from the text batch
    if (len(tags_list) == 0):
        logger.error("No tags found in the database, userInitProcess failed")
        features_tags = None
        return
    inputs = text_tokenizer(tags_list, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        features_tags = text_encoder(
            input_ids=input_ids, attention_mask=attention_mask).logits
        # Normalize
        features_tags = features_tags / \
            features_tags.norm(dim=1, keepdim=True)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    text_tokenizer = BertTokenizer.from_pretrained(
        "./Taiyi-CLIP-Roberta-large-326M-Chinese")
    text_encoder = BertForSequenceClassification.from_pretrained(
        "./Taiyi-CLIP-Roberta-large-326M-Chinese").eval()
    logger.info(f"Text Model Encoder: Taiyi-CLIP-Roberta-large-326M-Chinese")
    text_encoder.to(device)

    image_encoder = CLIPModel.from_pretrained(
        "./clip-vit-large-patch14").eval()
    image_encoder.to(device)
    image_processor = CLIPProcessor.from_pretrained(
        "./clip-vit-large-patch14")
    logger.info(f"Image Model Encoder: clip-vit-large-patch14")

    userInitProcess()

    # , ssl_certfile="./cert/certificate.crt", ssl_keyfile="./cert/private.key")
    uvicorn.run(app, host="0.0.0.0", port=8443)
