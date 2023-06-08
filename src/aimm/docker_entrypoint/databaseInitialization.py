import os
import time
import redis
import redis.commands.search
from redis.commands.search.field import (
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from dotenv import load_dotenv
load_dotenv()

REDIS_CLIENT = redis.StrictRedis(host=os.getenv('REDIS_HOST'), port=os.getenv('REDIS_PORT'), password=os.getenv('REDIS_PASSWORD'))

"""
Creates a new index in Redisearch with the specified index_name.
Args:
    index_name (str): The name of the index to create.
    recreate (bool): If set to True, drops the index if it already exists and creates a new one.
Returns:
    None
"""
def create_index(REDIS_CLIENT, index_name='searchIndex', recreate=False, object_class='art:'):
    def create_index_function():
        REDIS_CLIENT.ft(index_name=index_name).create_index(
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
            definition=IndexDefinition(prefix=[object_class], index_type=IndexType.JSON)
        )
    
    t0 = time.time()
    try:
        if recreate:
            try:
                REDIS_CLIENT.ft(index_name=index_name).dropindex()
            except Exception as e:
                print('Index does not exist')
            finally:
                create_index_function()                
        else:
            if not REDIS_CLIENT.exists(index_name):
                create_index_function()
            else:
                print(f'Index {index_name} already exists')
                exit()
        t1 = time.time()
        total = t1-t0
        print(f'Index created in {total} seconds')
    except Exception as e:
        print(e)

create_index(REDIS_CLIENT)