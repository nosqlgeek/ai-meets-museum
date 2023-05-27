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
redis_client_manu = redis.StrictRedis(host=os.getenv('redis_host_manu'), port=os.getenv('redis_port_manu'), password=os.getenv('redis_password_manu'))
redis_client_prod = redis.StrictRedis(host=os.getenv('redis_host_prod'), port=os.getenv('redis_port_prod'), password=os.getenv('redis_password_prod'))


#Model imports
 #mobilenet_v2 = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
 #mobilenet_v3_small = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")
 #mobilenet_v3_large = models.mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.DEFAULT")
 #alexnet = models.alexnet(weights="AlexNet_Weights.DEFAULT")
 #efficientnet = models.efficientnet_b0(weights="EfficientNet_B0_Weights.DEFAULT")
 #vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
ResNet50 = models.resnet50(weights="ResNet50_Weights.DEFAULT")



"""
# # # # # # # # # # # # # # # # # # # # # # # # # #
# Variable Deklaration for the rest of the code:  #
# # # # # # # # # # # # # # # # # # # # # # # # # #
"""
# Which Model should be used?
model = ResNet50
# Which Redis DB should be used?
redis_client = redis_client_prod
# Where are the iamges stored?
image_folder = os.getenv('image_folder')


"""
Creates a directory at the path specified by the 'image_folder' environment variable if it does not already exist.
If the directory already exists, nothing happens. If an error occurs when creating the directory, the program exits.

Returns:
    None
"""
def checkPathExistence():
    try:
        if not os.path.exists(os.getenv('image_folder')):
            os.makedirs(os.getenv('image_folder'))
    except Exception as e:
        print(e)
        exit()


"""
Create a tensor from an image file using the provided PyTorch model. The function takes in
a path to an image file as input and returns a PyTorch tensor. The model used in this
function should already be set to evaluation mode. The function uses PyTorch's transforms
module to preprocess the image by resizing it to 256x256, cropping the center to 224x224,
converting it to a tensor, and normalizing it using the ImageNet mean and standard deviation.
The function then prepares the input tensor to be used as a batch, and applies the model to
this batch either on CPU or GPU depending on availability. Finally, the function returns
the output tensor.
"""
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


"""
Uploads a tensor to Redis as a JSON object with a vector field.
:param tensor: A tensor to upload.
:type tensor: Tensor
"""
def uploadTensorToRedis(tensor):
    #Store the vector in Redis as HASH
     #tensor_bytes = tensor.numpy().astype(np.float32).tobytes(order='C')
     #redis_client.hset(objname, mapping={"vector_field": tensor_bytes})    

    ###################################################################
    # WARNING !!!!!!!!!
    # THIS NEED TO BE CHANGED TO THE OTHER OPTION BELOW
    # BUT THIS IS ONLY POSSIBLE IF THERE ARE ENTRIES IN THE REDIS DB
    ###################################################################

    #Overwrite json in Redis DB
     #redis_client.json().set(objname, '$', {"Tensor": tensor[0].tolist()})

    #Add Vector Data to JSON in Redis
    redis_client.json().set(redis_client.incr('MyKey'), '$.Tensor', {"Tensor": tensor[0].tolist()})


"""
Creates a new index in Redisearch with the specified index_name.
Args:
    index_name (str): The name of the index to create.
    recreate (bool): If set to True, drops the index if it already exists and creates a new one.
Returns:
    None
"""
def createIndex(index_name, recreate=False):
    def createIndexFunction():
        redis_client.ft(index_name=index_name).create_index(
            fields=(
                TextField(
                    "$.Bezeichnung", as_name="Bezeichnung"
                ),
                TextField(
                    "$.InventarNr", as_name="InventarNr"
                ),
                TextField(
                    "$.Material", as_name="Material"
                ),
                TextField(
                    "$.Beschreibung", as_name="Beschreibung"
                ),
                TextField(
                    "$.TrachslerNr", as_name="TrachslerNr"
                ),
                VectorField(                    
                    "$.Tensor", "FLAT", {"TYPE": "FLOAT32", "DIM": 1000, "DISTANCE_METRIC": "L2"}, as_name="vectorfield"
                )
            ),
            definition=IndexDefinition(index_type=IndexType.JSON)
        )
    
    t0 = time.time()
    try:
        if recreate:
            try:
                redis_client.ft(index_name=index_name).dropindex()
            except Exception as e:
                print('Index does not exist')
            finally:
                createIndexFunction()                
        else:
            if not redis_client.exists(index_name):
                createIndexFunction()
            else:
                print(f'Index {index_name} already exists')
                exit()
        t1 = time.time()
        total = t1-t0
        print(f'Index created in {total} seconds')
    except Exception as e:
        print(e)


"""
Performs a KNN search on an index using a given search tensor and index name.

Args:
    search_tensor: A tensor that will be used for the KNN search.
    index_name: The name of the index to perform the search on.

Returns:
    The results of the KNN search.
"""
def searchKNN(search_tensor, index_name):
    # vector_test for Similarity Search
    search_tensor_bytes = search_tensor[0].numpy().astype(np.float32).tobytes(order='C')

    # 10: is the number of nearest neighbors which we want to find
    # LIMIT 0 will disable that default limit
    # KNN 10 LIMIT 0 @vector_field $searchVector
    query = "*=>[KNN 10 @vectorfield $searchVector]"
    q = Query(query).sort_by('__vectorfield_score').dialect(2)
    
    result = redis_client.ft(index_name=index_name).search(
                query=q,
                query_params={'searchVector': search_tensor_bytes}                
            )

    return result


"""
Processes images in a given image folder by uploading their tensors to Redis using a Redis client. 
If a tensor for an image already exists in Redis, it is skipped. If no images are found in the folder, 
an error message is printed and the function exits. If an error occurs during processing, the function 
prints the name of the image and the error message, but continues processing the remaining images. 

Parameters:
None

Returns:
None
"""
def processImages():
    checkPathExistence()
    print(f'Processing images in {image_folder} - can take a while...')    
    images = os.listdir(image_folder)    
    if len(images) == 0:
        print(f'No images found in {image_folder}')
        exit()

    for image in tqdm(images, bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
        try:
            if not redis_client.exists(image):
                uploadTensorToRedis(createTensor(image_folder+image), image)
            else:
                print()
                print(f'Tensor for {image} already exists - skipping')
        except Exception as e:
            print(image)
            print(e)
            continue


"""
Searches a Redis index for documents containing the specified search keywords, and returns a list of search results.

Args:
    searchKeywords (str): The keywords to search for.
    index_name (str): The name of the Redis index to search.
    page (int): The page of results to return.

Returns:
    list: A list of dictionaries containing search result data.
"""
def fullTextSearch(searchKeywords, index_name, page):
    entries_per_page = 10
    start_entry = (page - 1) * entries_per_page

    query = searchKeywords
    q = Query(query).paging(start_entry,10).dialect(2)
    result = redis_client.ft(index_name=index_name).search(
                query=q
            )

    #return result

    for doc in result.docs:
        search = json.loads(doc.json)
        del search['Tensor']
        search = json.dumps(search, indent=4)
        search = json.loads(search)
        search_data.append(search)
    return search_data


"""
Iterates over a list of models and performs image processing, indexing, and searching
using each model. Prints the model name and search results for each model. Flushes
Redis at the end of each iteration. Returns nothing.
"""
def evauluatModels():
    #modellist = [ResNet50, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, alexnet, efficientnet, vgg]

    # iterate over the models
    for model in modellist:
        t0 = time.time()
        processImages()
        createIndex('myindex', recreate=True)
        tensor2 = createTensor(model, 'C:/Users/steph/.cache/downloadImage/image_folder/1018_0.jpg')
        result = searchKNN(tensor2, "myindex")
        print(f'Model: {model.__class__.__name__}')
        for doc in result.docs:
            print(doc.id, doc.__vector_field_score)# ,doc.vector

        redis_client.ft().execute_command('FLUSHALL')
        t1 = time.time()
        total = t1-t0
        print(f'Index created in {total} seconds')


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

    f = open('./src/ai-meets-museum/services/imageClassification/oldDataSet.json')
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