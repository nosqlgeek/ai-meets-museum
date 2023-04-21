import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
import redisearch
import tensorflow as tf

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

 #mobilenet_v2 = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
 #mobilenet_v3_small = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")
 #mobilenet_v3_large = models.mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.DEFAULT")
 #alexnet = models.alexnet(weights="AlexNet_Weights.DEFAULT")
 #yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
 #resnet = models.resnet50(weights="ResNet50_Weights.DEFAULT")
 #efficientnet = models.efficientnet_b0(weights="EfficientNet_B0_Weights.DEFAULT")
 #vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
ResNet50 = models.resnet50(weights="ResNet50_Weights.DEFAULT")


def createTensor(model, image_path):
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
    input_image = Image.open(image_path) #'.\src\\tools\\imageClassification\\images\\000003_3.jpg'
    image_name = os.path.splitext(os.path.basename(input_image.filename))[0]

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
    # Convert tensor to bytes
    tensor_bytes = tensor.numpy().astype(np.float32).tobytes(order='C')

    # Store the vector in Redis
    redis_client.hset(objname, mapping={"vector_field": tensor_bytes})

def createIndex(index_name):
    # Create index
    redis_client.ft(index_name='myindex').create_index(
            fields=(
                VectorField(                    
                    "vector_field", "FLAT", {"TYPE": "FLOAT32", "DIM": 1000, "DISTANCE_METRIC": "L2"}
                )
            )
        )

def searchKNN(search_tensor, index_name):
    # vector_test for Similarity Search
    search_tensor_bytes = search_tensor.numpy().astype(np.float32).tobytes(order='C')

    # 2: is the number of nearest neighbors which we want to find
    query = "*=>[KNN 2 @vector_field $searchVector]"
    q = Query(query).dialect(2)
    
    result = redis_client.ft(index_name=index_name).search(
                query=q,
                query_params={'searchVector': search_tensor_bytes}
            )

    #print(result)

    # Retrieve the vector field for each matching document
    doc_ids = [doc.id for doc in result.docs]    
    vectors = []
    for doc_id in doc_ids:
        vector_bytes = redis_client.hget(doc_id, 'vector_field')
        vector = np.frombuffer(vector_bytes, dtype=np.float32)
        vectors.append(vector)

    return vectors



tensor1 = createTensor(ResNet50, '.\src\\tools\\imageClassification\\images\\000003_3.jpg')
tensor2 = createTensor(ResNet50, '.\src\\tools\\imageClassification\\images\\000002_1.jpg')

uploadTensorToRedis(tensor1,'1402')
#createIndex("myindex")
vectors = searchKNN(tensor2, "myindex")

print('original tensor:')
print(tensor1)

print('found tensor:')
print(vectors)