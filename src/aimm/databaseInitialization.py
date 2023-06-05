import redis
import redis.commands.search
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

redis_client = redis.StrictRedis(host=os.environ.get('REDIS_HOST'), port=os.environ.get('REDIS_PORT'), password=os.environ.get('REDIS_PASSWORD'))

"""
Creates a new index in Redisearch with the specified index_name.
Args:
    index_name (str): The name of the index to create.
    recreate (bool): If set to True, drops the index if it already exists and creates a new one.
Returns:
    None
"""
def createIndex(REDIS_CLIENT, index_name='searchIndex', recreate=False, objectClass='art:'):
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
            definition=IndexDefinition(prefix=[objectClass], index_type=IndexType.JSON)
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


createIndex(REDIS_CLIENT)