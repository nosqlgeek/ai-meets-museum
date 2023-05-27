from minio import Minio
import os
import zipfile
import pyminizip as zip
from tqdm import tqdm
import time

from dotenv import load_dotenv
load_dotenv()

client = Minio(
    os.getenv('minio_url'),
    access_key=os.getenv('minio_access_key'),
    secret_key=os.getenv('minio_secret_key'),
    secure=False
)

image_folder = os.getenv('image_folder')
uncompress_folder = os.getenv('uncompress_folder')
original_dir = os.getcwd()

# Download all objects that start with '10'
# can be disabled by removing the prefix parameter in line 21
object_prefix = '1030'
minio_bucket = 'museum'
objects = list(client.list_objects(minio_bucket)) #, prefix=object_prefix

def checkPathExistence():
    try:
        if not os.path.exists(os.getenv('image_folder')):
            os.makedirs(os.getenv('image_folder'))

        if not os.path.exists(os.getenv('uncompress_folder')):
            os.makedirs(os.getenv('uncompress_folder'))
    except Exception as e:
        print(e)
        exit()

checkPathExistence()

firstImagesOfObjects = [obj for obj in objects if "_0" in obj.object_name]
objects=firstImagesOfObjects

total_items = len(objects)
print('Downloading Items: ',total_items)

for obj in tqdm(objects, total=total_items, bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
    if "_0.jpg" in obj.object_name:
        # Download the object data
        filenamejpg = obj.object_name.lower()
        filenamezip = filenamejpg.replace('.jpg', '.zip')

        if not os.path.isfile(image_folder+filenamejpg):
            try:
                client.fget_object(minio_bucket, filenamejpg, file_path=uncompress_folder+filenamezip)
            except Exception as e:
                print(e)

            zip.uncompress(uncompress_folder+filenamezip, os.getenv('ENCRYPTION_KEY'), uncompress_folder, 1)
            os.remove(filenamezip)

            os.chdir(original_dir)
            extracted_file_name = os.listdir(uncompress_folder)[0]
            
            os.rename(uncompress_folder+extracted_file_name, image_folder+filenamejpg)
        else:
            print()
            print(f'{filenamejpg} already exists - skipping')
    if "_0.JPG" in obj.object_name:
        # Download the object data
        filenamejpg = obj.object_name
        filenamezip = filenamejpg.replace('.JPG', '.zip')
    
        if not os.path.isfile(image_folder+filenamejpg):
            try:
                client.fget_object(minio_bucket, filenamejpg, file_path=uncompress_folder+filenamezip)
            except Exception as e:
                print(e)
    
            zip.uncompress(uncompress_folder+filenamezip, os.getenv('ENCRYPTION_KEY'), uncompress_folder, 1)
            os.remove(filenamezip)
    
            os.chdir(original_dir)
            extracted_file_name = os.listdir(uncompress_folder)[0]
            
            os.rename(uncompress_folder+extracted_file_name, image_folder+filenamejpg.replace('.JPG', '.jpg'))
        else:
            print()
            print(f'{filenamejpg} already exists - skipping')