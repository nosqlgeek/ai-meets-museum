from torchvision import models, transforms
import torch
from PIL import Image
import os
from redis.commands.search.query import Query
import numpy as np
import json
import redis


redis_client = redis.Redis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"),
                           password=os.getenv("REDIS_PASSWORD"))


def createTensor(image_path):
    ResNet50 = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    model = ResNet50
    # Set model to evaluation mode
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
    image_name = os.path.splitext(os.path.basename(image_path))[0]  # input_image.filename

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


def searchKNN(search_tensor, index_name, redis_client):
    # vector_test for Similarity Search
    search_tensor_bytes = search_tensor[0].numpy().astype(np.float32).tobytes(order='C')

    # 10: is the number of nearest neighbors which we want to find
    # LIMIT 0 will disable that default limit
    # KNN 10 LIMIT 0 @vector_field $searchVector
    query = "*=>[KNN 10 @vectorfield $searchIndex]"
    q = Query(query).sort_by('__vectorfield_score').dialect(2)
    
    result = redis_client.ft(index_name=index_name).search(
                query=q,
                query_params={'searchIndex': search_tensor_bytes}
            )    
    # print(result)
    return result


def getNeighbours(json_input, count):
    neighbours = []
    for i in range(count):
        myjson = json.loads(json_input.docs[i].json)
        del myjson['Tensor']
        myjson = json.dumps(myjson, indent=4)
        myjson = json.loads(myjson)
        neighbours.append(myjson)
    return neighbours


def fullTextSearch(searchKeywords, index_name, page):
    entries_per_page = 10
    start_entry = (page - 1) * entries_per_page

    query = searchKeywords
    q = Query(query).paging(start_entry, 10).dialect(2)
    result = redis_client.ft(index_name=index_name).search(
                query=q
            )
    search_data = []
    for doc in result.docs:
        search = json.loads(doc.json)
        del search['Tensor']
        search = json.dumps(search, indent=4)
        search = json.loads(search)
        search_data.append(search)
    return search_data


def getFullTextSearchCount(searchKeywords, index_name):
    query = searchKeywords
    q = Query(query).paging(0, 0).dialect(2)
    result = redis_client.ft(index_name=index_name).search(
        query=q
    )
    return result.total
