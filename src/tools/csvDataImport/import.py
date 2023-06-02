import redis
import os
import json
import time
from tqdm import tqdm

import sys
sys.path.append('./src/tools/imageClassification')
from processImages import createTensor

from dotenv import load_dotenv
load_dotenv()

redis_client = redis.StrictRedis(host=os.getenv('redis_host'), port=os.getenv('redis_port'), password=os.getenv('redis_password'))
image_folder = os.getenv('image_folder')

"""
Uploads old data from a folder to Redis.

Returns:
    None
"""
def uploadOldDataToRedis(): 
    images = os.listdir(image_folder)    
    if len(images) == 0:
        print(f'No images found in {image_folder}')
        exit()

    f = open('./src/tools/csvDataImport/oldDataSet.json')
    myjson = json.load(f)
    t0 = time.time()

    for image in tqdm(images, bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
        try:            
            tensor = createTensor(image_folder+image)
            myjsonentry = None
            for entry in myjson:                           
                if entry['ObjektId'] == int(image.split('_')[0]):                        
                    entry['Tensor'] = tensor[0].tolist()
                    myjsonentry = entry
                    break
            
            redis_client.json().set(redis_client.incr('MyKey'), '$', myjsonentry)  
        except Exception as e:
            print(image)
            print(e)
            continue
    
    t1 = time.time()
    total = t1-t0
    print(f'Upload took: {total} seconds')

    #for item in jsonList:
    #    uploadTensorToRedis(createTensor(item), item)

uploadOldDataToRedis()