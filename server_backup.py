from src.Logger import logger

logger.info("Server Starting...")
logger.info("Importing Libraries...")

import base64
import io
import json
import os
import re
import threading
from typing import List

from fastapi.responses import JSONResponse, StreamingResponse

import numpy as np
from pydantic import BaseModel
import requests
import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import clip
import PIL
from PIL import Image, UnidentifiedImageError

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
from src.Database import get_db, get_db_commit, DBContextManager
from src.UserManager import User, UserInDB, create_jwt_token, decode_token, get_current_active_user, hash_password
from sqlalchemy import bindparam, text
from src.models import MediaClassesScheme, MediaScheme, UserScheme
from sqlite3 import Connection, Cursor

# Langchain
from langchain_deepseek import ChatDeepSeek
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate


# Asyncio
import asyncio


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
    return {"message": f'Hello {user.name} !'}


@app.post("/users/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db_commit)):
    user = None
    try:
        dbLock.acquire()
        user = db.query(UserScheme).filter(
            UserScheme.name == form_data.username).first()
    except Exception as e:
        logger.error(e)
    finally:
        dbLock.release()

    if user is None or user.password != hash_password(form_data.password):
        raise fastapi.HTTPException(
            status_code=401, detail="用户名或密码错误")
    else:
        token, expire = create_jwt_token({"sub": user.name})
        dbLock.acquire()
        try:
            db.execute(
                text("UPDATE users SET token_expire = :expire WHERE name = :name"),
                {"expire": expire, "name": user.name})
        except Exception as e:
            logger.error(e)
        finally:
            dbLock.release()
        return {"access_token": token, "token_type": "bearer"}

"""接受token，如果token无效或过期，则返回401，如果有效，延长时间，并返回200，表示用户活跃，"""


@app.post("/users/active")
async def active(db_user: User = Depends(get_current_active_user), db: Session = Depends(get_db_commit)):
    token, expire = create_jwt_token({"sub": db_user.name})
    try:
        dbLock.acquire()
        db.execute(
            text("UPDATE users SET token_expire = :expire WHERE name = :name"),
            {"expire": expire, "name": db_user.name})
    except Exception as e:
        logger.error(e)
    finally:
        dbLock.release()
    return {"access_token": token, "token_type": "bearer"}


@app.post("/users/register")
async def register(form_data: OAuth2PasswordRequestForm = Depends(), db: Cursor = Depends(get_db_commit)):
    user = db.query(UserScheme).filter(
        UserScheme.name == form_data.username).first()
    if user is not None:
        raise fastapi.HTTPException(
            status_code=403, detail="User already exists")
    try:
        dbLock.acquire()
        db.execute(
            text("""
            INSERT INTO users (name, password, email, full_name, active, token_expire)
            VALUES (:name, :password, :email, :full_name, :active, :token_expire)
        """),
            {
                "name": form_data.username,
                "password": hash_password(form_data.password),
                "email": form_data.email,
                "full_name": form_data.full_name,
                "active": True,
                "token_expire": datetime.now()
            }
        )
    except Exception as e:
        logger.error(e)
        raise fastapi.HTTPException(
            status_code=500, detail="Internal Server Error")
    finally:
        dbLock.release()


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
        inputs = text_tokenizer(tags_list, return_tensors="pt", padding=True)[
            'input_ids'].to(device)
        with torch.no_grad():
            features_texts = text_encoder(inputs).logits
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

metaParserTemplateStr = """你是一个json解析器，专门解析对查找的照片的信息，你将输入的文本通过理解，若文本没有呈现出某个键的信息，请不要解析这个键，否则解析出键对应的值，并以纯文本(花括号开头结尾)形式输出，可选的键有：[date, loc, device]，date:查找照片的日期时间(以今天{date}为参照日期，请以给出的时间精确到年、月或者日，如果只精确到月，那开始日期1号，结束日期月末；如果只精确到年，那开始日期1月1号，结束日期到年末，格式为区间YYYY-MM-DD HH:MM:SS~YYYY-MM-DD HH:MM:SS，例如：1."15年到16年的照片"解析为"2015-01-01 00:00:00~2016-01-01 00:00:00"2."去年以前的照片"解析为"1949-10-01 00:00:00~{dateLastYear}"，3."小时候的照片"由于日期不明确，故不解析date键)，loc:查找照片的地点(例如：1.北京市, 2.武功山)，device:查找照片的设备(如：1.iPhone 13, 2.Nikon zf)，输出格式如下:
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
    if "date" in json_meta:
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
        "tags":  eval(responseTag["out"]),
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


globalVarUseLock = threading.Lock()
dbLock = threading.Lock()
vlmModelLock = threading.Lock()
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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_commit),
):
    global features_tags, tags_list, logger

    meta_data = MetaData(identifier=identifier, name=name, type=type, created_at=created_at, lat=lat, lon=lon, dev=dev)

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
    processStep INTEGER DEFAULT 0,
    feature BLOB DEFAULT 0,
    meta_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    meta_loc TEXT,
    meta_device TEXT
    """

    try:
        sql = text("""
            INSERT OR IGNORE INTO media (identifier, name, type, processStep, feature, meta_date, meta_loc, meta_device)
            VALUES (:identifier, :name, :type, :processStep, :feature, :meta_date, :meta_loc, :meta_device);
            """)
        dbLock.acquire()
        db.execute(sql, {
            "identifier": meta_data.identifier,
            "name": meta_data.name,
            "type": meta_data.type,
            "processStep": 2,
            "feature": features.cpu().numpy().tobytes(),
            "meta_date": meta_data.created_at,
            "meta_loc": loc,
            "meta_device": meta_data.dev
        })

        if len(selected_classes) != 0:
            sql = text("""
                INSERT OR IGNORE INTO media_classes (media_id, class_id)
                VALUES (:identifier, :tags);
                """)
            for tag in selected_classes:
                db.execute(sql, {
                    "identifier": meta_data.identifier,
                    "tags": tag
                })
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        dbLock.release()
    
    logger.info(f"Resolve OK: {meta_data.identifier} -> {selected_classes}")
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
        np.frombuffer(feat, dtype=np.float32)).to(device)

    # 2. Filter criteria
    filter_criteria = []
    if "date_start" in meta_data and "date_end" in meta_data:
        date_start = meta_data["date_start"]
        date_end = meta_data["date_end"]
        filter_criteria.append(MediaScheme.meta_date.between(date_start, date_end))
    if "loc" in meta_data:
        loc = meta_data["loc"]
        filter_criteria.append(MediaScheme.meta_loc.like(f"%{loc}%"))
    if "device" in meta_data:
        meta_device = meta_data["device"]
        # MediaScheme.device like '%meta_device%'
        filter_criteria.append(MediaScheme.meta_device.like(f"%{meta_device}%"))
    try:
        dbLock.acquire()
        if "tags" in rspv2.keys():
            tagNames = rspv2["tags"]
            stmt = text("""
                SELECT id
                FROM classes
                WHERE name IN :tagNames
                AND user_id = :user_id
            """).bindparams(bindparam("tagNames", expanding=True))

            tagIdsQueryResult = db.execute(
                stmt,
                {"tagNames": tagNames, "user_id": current_user.id}
            ).all()

            # tables: media, media_classes
            # filter_criteria.append(
            #     MediaScheme.identifier.in_(
            #         db.query(MediaClassesScheme.media_id).filter(
            #             MediaClassesScheme.class_id.in_(
            #                 [tag.id for tag in tagIdsQueryResult]
            #             )
            #         ).subquery()
            #     )
            # )
        query = db.query(MediaScheme).filter(
            *filter_criteria,
            MediaScheme.feature != None
        ).all()
    except:
        query = []
    finally:
        dbLock.release()
    
    if len(query) == 0:
        raise fastapi.HTTPException(
            status_code=404, detail="No image found with the given criteria")

    image_data_ids = []
    for image in query:
        image_data_ids.append(image.identifier)
    features = []
    for image in query:
        # image.feature is a blob, convert it to numpy array and convert it to torch tensor
        features.append(torch.from_numpy(
            np.frombuffer(image.feature, dtype=np.float32)).to(device) )
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

def userInitProcess():
    global features_tags, tags_list

    # Init DB
    with DBContextManager() as db:
        # Perform database upgrade statements
        from src.dbUpgradeStatements import dbUpgradeStatements

        # check sqlite db version
        db_version = 0
        try:
            db.execute(text("SELECT sqlite_version()"))
            db_version = db.execute(text(
                "PRAGMA user_version;")).fetchone()[0]
            logger.info(f"System DB version: {db_version}")

            # check if db version is less than the latest version
            if db_version < dbUpgradeStatements[-1]["toVersion"]:
                for statement in dbUpgradeStatements[db_version:]:
                    db_version = statement["toVersion"]
                    logger.info(
                        f"Upgrading DB to version {statement['toVersion']}")
                    for sql in statement["statements"]:
                        db.execute(text(sql))
                        db.commit()
                    db.execute(
                        text(f"PRAGMA user_version = {db_version};"))

                    if db_version == 1:
                        # Create default user
                        db.execute(
                            text("INSERT OR IGNORE users (name, password, email, active, token_expire) VALUES ('admin', :password, 'admin@example.com', 1, datetime('now', '+1 year'));"),
                            {"password": hash_password("admin")})
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
    if(len(tags_list) == 0):
        logger.error("No tags found in the database, userInitProcess failed")
        features_tags = None
        return
    inputs = text_tokenizer(tags_list, return_tensors="pt", padding=True)[
        'input_ids'].to(device)
    with torch.no_grad():
        features_tags = text_encoder(inputs).logits
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

    uvicorn.run(app, host="0.0.0.0", port=8443, ssl_certfile="./cert/certificate.crt", ssl_keyfile="./cert/private.key")
