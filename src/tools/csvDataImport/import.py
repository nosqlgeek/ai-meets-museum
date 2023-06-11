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

from dotenv import load_dotenv
load_dotenv()

redis_client = redis.StrictRedis(host=os.getenv('redis_host'), port=os.getenv('redis_port'), password=os.getenv('redis_password'))

# folder with images
image_folder = os.getenv('image_folder')

# target folder for images after rename
target_folder = os.getenv('target_folder')

# json file with data
json_file = open('../oldDataSet.json')

def migrate_data_to_redis():
    images = os.listdir(image_folder)
    if len(images) == 0:
        print(f'No images found in {image_folder}')
        exit()

    myjson = json.load(json_file)
    t0 = time.time()

    for image in tqdm(images, bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
        try:            
            tensor = create_tensor(image_folder+image)
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
            new_file = os.path.join(target_folder, str(ObjektNr)+'.jpg')
            os.rename(old_file, new_file)

        except Exception as e:
            print(e)
            continue
    
    t1 = time.time()
    total = t1-t0
    print(f'Upload took: {total} seconds')

def create_tensor(image_path):
    # Set model to evaulation mode
    model.eval()

    # Image normalization
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Image import
    input_image = Image.open(image_path)
    input_image = input_image.convert('RGB')
    image_name = os.path.splitext(os.path.basename(image_path))[0] #input_image.filename

    # Preprocess image and prepare batch
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # use cuda cores on gpu
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        tensor = model(input_batch)

    return tensor


migrate_data_to_redis()