from minio import Minio
import os
import pyminizip as zip

URL = "Insert URL here"
ACCESS_KEY = "Insert ACCESS Key here"
SECRET_KEY = "Insert Secret Key here"

BUCKET_NAME = "Insert Bucket Name here"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TMP_FILE = f'{CURRENT_DIR}/tmp.zip'
TARGET_DIR = f'{CURRENT_DIR}/imgs'
ENCRYPTION_KEY="Insert Encryption Key here"

client = Minio(
    URL,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False
)

objects = client.list_objects(BUCKET_NAME)
count_objects = sum(1 for _ in client.list_objects(BUCKET_NAME))

downloaded_items = 0
skipped_items = 0

print(f'Downloading {count_objects} objects, this might take some time...')

for obj in objects:
    if not os.path.isfile(f'{TARGET_DIR}/{obj.object_name}'):
        downloaded_items = downloaded_items + 1
        client.fget_object(obj.bucket_name, obj.object_name, file_path=TMP_FILE)
        zip.uncompress(TMP_FILE, ENCRYPTION_KEY, TARGET_DIR, 1)
        os.remove(TMP_FILE)
    else:
        skipped_items = skipped_items + 1

    print(f'Processed {downloaded_items + skipped_items}nth {count_objects} objects')

print(f'Downloaded {downloaded_items} objects, skipped {skipped_items}.')
