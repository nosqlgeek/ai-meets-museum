from minio import Minio
import os
import pyminizip as zip

from dotenv import load_dotenv
load_dotenv()

client = Minio(
    os.getenv('minio_url'),
    access_key=os.getenv('minio_access_key'),
    secret_key=os.getenv('minio_secret_key'),
    secure=False
)

client.fget_object('museum', '11_0.jpg', file_path='download.zip')

zip.uncompress('download.zip', 'A1M33ts-Museum!', None, 1)

os.remove('download.zip')