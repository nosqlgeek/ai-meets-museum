import torch
from torchvision import models, transforms
from PIL import Image
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
#redis_client2 = redis.StrictRedis(host=os.getenv('redis_host_manu'), port=os.getenv('redis_port_manu'), password=os.getenv('redis_password_manu'))
image_folder = os.getenv('image_folder')

#Model imports
 #mobilenet_v2 = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
 #mobilenet_v3_small = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")
mobilenet_v3_large = models.mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.DEFAULT")
 #alexnet = models.alexnet(weights="AlexNet_Weights.DEFAULT")
 #efficientnet = models.efficientnet_b0(weights="EfficientNet_B0_Weights.DEFAULT")
 #vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
 #ResNet50 = models.resnet50(weights="ResNet50_Weights.DEFAULT")

#Define the model to use in the code below:
model=mobilenet_v3_large

def checkPathExistence():
    try:
        if not os.path.exists(os.getenv('image_folder')):
            os.makedirs(os.getenv('image_folder'))
    except Exception as e:
        print(e)
        exit()

def createTensor(image_path):
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

def uploadTensorToRedis(tensor,objname):
    #Store the vector in Redis as HASH
     #tensor_bytes = tensor.numpy().astype(np.float32).tobytes(order='C')
     #redis_client.hset(objname, mapping={"vector_field": tensor_bytes})

    ###################################################################
    # WARNING !!!!!!!!!
    # THIS NEED TO BE CHANGED TO THE OTHER OPTION BELOW
    # BUT THIS IS ONLY POSSIBLE IF THERE ARE ENTRIES IN THE REDIS DB
    ###################################################################

    #Overwrite json in Redis DB
    redis_client.json().set(objname, '$', {"vector": tensor[0].tolist()})

    #Add Vector Data to JSON in Redis
     #redis_client.json().set(objname, '$.vector', {"vector": tensor[0].tolist()})    

def createIndex(index_name):
    # Create index
    t0 = time.time()
    redis_client.ft(index_name=index_name).create_index(
            fields=(
                VectorField(                    
                    "$.vector", "FLAT", {"TYPE": "FLOAT32", "DIM": 1000, "DISTANCE_METRIC": "L2"}, as_name="vectorfield"
                )
            ),
            definition=IndexDefinition(index_type=IndexType.JSON)
        )
    t1 = time.time()
    total = t1-t0
    print(f'Index created in {total} seconds')

def searchKNN(search_tensor, index_name):
    # vector_test for Similarity Search
    search_tensor_bytes = search_tensor[0].numpy().astype(np.float32).tobytes(order='C')

    # 10: is the number of nearest neighbors which we want to find
    # LIMIT 0 will disable that default limit
    # KNN 10 LIMIT 0 @vector_field $searchVector
    query = "*=>[KNN 10 @vectorfield $searchVector]"
    q = Query(query).dialect(2)
    
    result = redis_client.ft(index_name=index_name).search(
                query=q,
                query_params={'searchVector': search_tensor_bytes}
            )    
    #print(result)

    return result

def processImages(image_path):    
    checkPathExistence()
    print(f'Processing images in {image_path} - can take a while...')    
    images = os.listdir(image_path)    
    if len(images) == 0:
        print(f'No images found in {image_path}')
        exit()

    for image in tqdm(images, bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
        if not redis_client.exists(image):
            uploadTensorToRedis(createTensor(image_path+image), image)
        else:
            print()
            print(f'Tensor for {image} already exists - skipping')


################
# Test Section #
################

#Upload one Image Tensor to Redis:
 #uploadTensorToRedis(tensor1,'1402')

#Create Image Tensors of image_folder and Upload to Redis:
 #processImages(image_folder)

#Create index for KNN Search
 #createIndex('myindex1')

#KNN Search for given searchtensor
searchtensor = createTensor('C:/Users/steph/.cache/downloadImage/image_folder/1007_0.jpg')
result = searchKNN(searchtensor, "myindex1")
#Print the results of the searchKNN function
for doc in result.docs:  
    json_data = json.loads(doc.json)
    print('objname: ',doc.id,'| score: ',doc.__vectorfield_score)
    #print('vector in json: ',json_data['vector'])


# Compare different models and there scores
 #modellist = [ResNet50, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, alexnet, efficientnet, vgg]
def evauluatModels():
    # iterate over the models
    for model in modellist:
        t0 = time.time()
        processImages(model, image_folder)
        createIndex('myindex')
        tensor2 = createTensor(model, 'C:/Users/steph/.cache/downloadImage/image_folder/1018_0.jpg')
        result = searchKNN(tensor2, "myindex")
        print(f'Model: {model.__class__.__name__}')
        for doc in result.docs:
            print(doc.id, doc.__vector_field_score)# ,doc.vector

        redis_client.ft().execute_command('FLUSHALL')
        t1 = time.time()
        total = t1-t0
        print(f'Index created in {total} seconds')

#evauluatModels()

redis_client.close()