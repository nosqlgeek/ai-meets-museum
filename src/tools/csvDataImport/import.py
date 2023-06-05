import torch
from torchvision import models, transforms
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import os
import numpy as np
import redisearch
import tensorflow as tf
from tqdm import tqdm
import time
import json
import pickle

import redis
import redis.commands.search
import redis.commands.search.aggregation as aggregations
import redis.commands.search.reducers as reducers
from redis.commands.json.path import Path
from redis.commands.search import Search
from redis.commands.search.field import (
    GeoField,
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import GeoFilter, NumericFilter, Query
from redis.commands.search.result import Result
from redis.commands.search.suggestion import Suggestion

import sys
sys.path.append('./src/tools/imageClassification')
from processImages import createTensor

from dotenv import load_dotenv
load_dotenv()


redis_client = redis.StrictRedis(host=os.getenv('redis_host'), port=os.getenv('redis_port'), password=os.getenv('redis_password'))
#redis_client = redis.StrictRedis(host=os.getenv('redis_host_test'), port=os.getenv('redis_port_test'), password=os.getenv('redis_password_test'))
#redis_client = redis.StrictRedis(host=os.getenv('redis_host_manu'), port=os.getenv('redis_port_manu'), password=os.getenv('redis_password_manu'))
#redis_client = redis.StrictRedis(host=os.getenv('redis_host_markus'), port=os.getenv('redis_port_markus'), password=os.getenv('redis_password_markus'))
image_folder = os.getenv('image_folder')

target = 'C:/Users/steph/.cache/downloadImage/target_folder/'


def uploadOldDataToRedis():
    images = os.listdir(image_folder)
    if len(images) == 0:
        print(f'No images found in {image_folder}')
        exit()

    f = open('../oldDataSet.json')
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
            
            ObjektNr = redis_client.incr('ObjektNr')
            myjsonentry['BildNr'] = ObjektNr
            redis_client.json().set('art:'+str(ObjektNr), '$', myjsonentry)

            old_file = os.path.join(image_folder, image)
            new_file = os.path.join(target, str(ObjektNr)+'.jpg')
            os.rename(old_file, new_file)

        except Exception as e:
            print(e)
            continue
    
    t1 = time.time()
    total = t1-t0
    print(f'Upload took: {total} seconds')


uploadOldDataToRedis()