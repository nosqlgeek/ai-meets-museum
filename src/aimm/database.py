import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import redisearch
import tensorflow as tf
import json
import time
import pickle

import redis
import redis.commands.search
from redis.commands.json.path import Path
from redis.commands.search import Search
import redis.commands.search.reducers as reducers
from redis.commands.search.field import (
    TextField,
    VectorField,
)
from redis.commands.search.query import Query
from redis.commands.search.result import Result

from dotenv import load_dotenv
load_dotenv()

"""
Creates a tensor from an input image using the specified model.

Args:
    image_path (str): The file path to the input image.

Returns:
    tensor: A tensor representing the input image after preprocessing and running through the model.
"""
def create_tensor(REDIS_CLIENT, image_path):
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
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
Uploads a JSON object to Redis and returns the assigned ObjektNr. 

Args:
    json_data (dict): The JSON object to upload.
    objectClass (str): The object class to use as a prefix for the Redis key. Default is 'art:'.

Returns:
    ObjektNr (int): The assigned object number.
"""
def upload_object_to_redis(REDIS_CLIENT, json_data, object_class='art:'):
    #Get current ObjektNr
    objekt_nr = REDIS_CLIENT.incr('ObjektNr')

    #Add ObjektNr to json
    json_data['BildNr'] = objekt_nr

    #Upload to Redis
    REDIS_CLIENT.json().set(object_class+str(objekt_nr), '$', json_data)

    return objekt_nr


"""
Performs a KNN search on an index using a given search tensor and index name.

Args:
    search_tensor: A tensor that will be used for the KNN search.
    index_name: The name of the index to perform the search on.

Returns:
    The results of the KNN search.
"""
def search_knn(REDIS_CLIENT, search_tensor, index_name):
    # vector_test for Similarity Search
    search_tensor_bytes = search_tensor[0].numpy().astype(np.float32).tobytes(order='C')

    # 10: is the number of nearest neighbors which we want to find
    # LIMIT 0 will disable that default limit
    # KNN 10 LIMIT 0 @vector_field $searchVector
    query = "*=>[KNN 10 @vectorfield $searchVector]"
    q = Query(query).sort_by('__vectorfield_score').dialect(2)
    
    result = REDIS_CLIENT.ft(index_name=index_name).search(
                query=q,
                query_params={'searchVector': search_tensor_bytes}                
            )

    return result


"""
Searches a Redis index for documents containing the specified search keywords, and returns a list of search results.

Args:
    searchKeywords (str): The keywords to search for.
    index_name (str): The name of the Redis index to search.
    page (int): The page of results to return.

Returns:
    list: A list of dictionaries containing search result data.
"""
def full_text_search(REDIS_CLIENT, search_keywords, index_name, page):
    entries_per_page = 10
    start_entry = (page - 1) * entries_per_page

    query = search_keywords
    q = Query(query).paging(start_entry,10).dialect(2)
    result = REDIS_CLIENT.ft(index_name=index_name).search(
                query=q
            )

    search_data = []
    search_data.append(result.total)
    for doc in result.docs:
        search = json.loads(doc.json)
        del search['Tensor']
        search = json.dumps(search, indent=4)
        search = json.loads(search)
        search_data.append(search)
    return search_data


"""
Returns the total number of full text search results for the given search keywords
and Redisearch index name.

Args:
    searchKeywords (str): The search keywords to query the Redisearch index with.
    index_name (str): The name of the Redisearch index to query.

Returns:
    int: The total number of search results.
"""
def get_full_text_search_count(REDIS_CLIENT, search_keywords, index_name):
    query = search_keywords
    q = Query(query).paging(0,0).dialect(2)
    result = REDIS_CLIENT.ft(index_name=index_name).search(
                query=q
            )
            
    return result.total


"""
Given a JSON input and a count, returns a list of JSON objects representing the count
nearest neighbors of the input. 

Args:
    json_input: A JSON input string.
    count: An integer representing how many neighbors to retrieve.

Returns:
    list: A list of JSON objects representing the nearest neighbors of the input.
"""
def get_neighbours(REDIS_CLIENT, json_input, count):
    neighbours = []
    for i in range(count):
        myjson = json.loads(json_input.docs[i].json)
        del myjson['Tensor']
        myjson = json.dumps(myjson, indent=4)
        myjson = json.loads(myjson)
        neighbours.append(myjson)
    return neighbours