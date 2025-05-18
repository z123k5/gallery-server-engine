import sys

from sqlalchemy import text
sys.path.append("D:\Projects\IOSProjects\gallery-server-engine")
import requests

from src.Database import DBContextManager, get_db_commit

with DBContextManager() as db:
    # select from media_metadata where location is null and exif_lat is not null and exif_lon is not null
    result = db.execute(text(
        "SELECT user_id, media_id, exif_lat, exif_lon FROM media_metadata WHERE location = '' AND exif_lat IS NOT 0 AND exif_lon IS NOT 0;")).fetchall()
    # update location to 'lat, lon' where location is null and exif_lat is not null and exif_lon is not null

    for row in result:
        user_id = row[0]
        media_id = row[1]
        exif_lat = row[2]
        exif_lon = row[3]

        loc = ""
        try:
            url = f"http://api.tianditu.gov.cn/geocoder?postStr={{'lon':{exif_lon},'lat':{exif_lat},'ver':1}}&type=geocode&tk=6cc422bf3bab18d99e9a1be91b7b2afb"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                loc = data["result"]["formatted_address"]
        except Exception as e:
            print(f"Error: {e}")

        # update location to 'lat, lon' where location is null and exif_lat is not null and exif_lon is not null
        db.execute(text(
            f"UPDATE media_metadata SET location = '{loc}' WHERE user_id = {user_id} AND media_id = '{media_id}'"))

    
