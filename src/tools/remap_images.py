import redis
import redis.commands.search.aggregation as aggregations
import redis.commands.search.reducers as reducers
from redis.commands.json.path import Path
from redis.commands.search.field import NumericField, TagField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import shutil

r = redis.Redis(host="<secret>", port=15763, password="<secret>", db=0, decode_responses=True)
index = r.ft("searchIndex")
res = index.search(Query("*").return_fields("$.ObjektId","$.BildNr").paging(0,10000))

for r in res.docs:
    mapping = r.__dict__
    objId = mapping.get('$.ObjektId')
    bildId = mapping.get('$.BildNr')
    print(str(objId) + ":" + str(bildId))
    try:
        shutil.copy(objId + ".jpg", "./clean/" + bildId + ".jpg")
    except:
        print("Skipping")
