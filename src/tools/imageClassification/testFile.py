from processImages import *

################
# Test Section #
################

#evauluatModels()


#uploadOldDataToRedis()


#Upload one Image Tensor to Redis:
 #uploadTensorToRedis(tensor1,'1402')


#Create Image Tensors of image_folder and Upload to Redis:
 #processImages()


#Create index for KNN Search
 #createIndex('searchIndex', recreate=True)


#KNN Search for given searchtensor
def testKNNsearch():
    searchtensor = createTensor('C:/Users/steph/.cache/downloadImage/image_folder/5_0.jpg')
    result = searchKNN(searchtensor, "searchIndex")
    #Print the results of the searchKNN function
    for doc in result.docs:  
        json_data = json.loads(doc.json)
        print(  'objname: ',doc.id,
                'score: ',doc.__vectorfield_score,
                'Bezeichnung: ',json_data['Bezeichnung'],
                'TrachslerNr: ',json_data['TrachslerNr'])
        #print('vector in json: ',json_data['vector'])

#testKNNsearch()


#Full-Text Search for given keyword
def testFullTextSearch(searchKeywords):
    result = fullTextSearch(searchKeywords=searchKeywords, index_name='searchIndex', page=1)
    print(result)
    print(result[0])



testFullTextSearch(searchKeywords='krippenfigur')


print(getFullTextSearchCount('krippenfigur', 'searchIndex'))