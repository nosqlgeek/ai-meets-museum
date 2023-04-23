from minio import Minio
import os
import zipfile
import pyminizip as zip
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

client = Minio(
    os.getenv('minio_url'),
    access_key=os.getenv('minio_access_key'),
    secret_key=os.getenv('minio_secret_key'),
    secure=False
)

minio_bucket = 'museum'
# Download all objects that start with '10'
# can be disabled by removing the prefix parameter in line 21
object_prefix = '10'
objects = list(client.list_objects(minio_bucket, prefix=object_prefix)) #
downloadpath = '.\\src\\tools\\downloadImage\\img\\'
temppath = '.\\src\\tools\\downloadImage\\tmp\\'
original_dir = os.getcwd()

# Add each object to the zip archive
total_items = len(objects)
for obj in tqdm(objects, total=total_items, bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
    # Download the object data
    filenamejpg = obj.object_name.lower()
    filenamezip = filenamejpg.replace('.jpg', '.zip')

    if not os.path.isfile(downloadpath+filenamejpg):
        client.fget_object(minio_bucket, filenamejpg, file_path=temppath+filenamezip)
        zip.uncompress(temppath+filenamezip, os.getenv('ENCRYPTION_KEY'), temppath, 1)
        os.remove(filenamezip)

        os.chdir(original_dir)
        extracted_file_name = os.listdir(temppath)[1]
        
        os.rename(temppath+extracted_file_name, downloadpath+filenamejpg)
    else:
        print()
        print(f'{filenamejpg} already exists - skipping')