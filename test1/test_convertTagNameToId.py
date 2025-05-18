import sys
sys.path.append("D:\Projects\IOSProjects\gallery-server-engine")
from src.Database import DBContextManager, get_db_commit
import requests

from sqlalchemy import text


with DBContextManager() as db:

    # media_classes 表的class_id列内容错了，里面都是类的名称，请在classes表里找到一个对应的id，然后修改media_classes表
    sql = text("""UPDATE media_classes SET class_id = (SELECT id FROM classes WHERE name = media_classes.class_id)""")

    db.execute(sql)

