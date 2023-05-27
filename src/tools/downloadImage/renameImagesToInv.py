import os
import json

from dotenv import load_dotenv
load_dotenv()

image_folder = os.getenv('image_folder')
uncompress_folder = os.getenv('uncompress_folder')
json_file = "./src/tools/downloadImage/oldDataSet.json"

def checkPathExistence():
    try:
        if not os.path.exists(os.getenv('image_folder')):
            os.makedirs(os.getenv('image_folder'))

        if not os.path.exists(os.getenv('uncompress_folder')):
            os.makedirs(os.getenv('uncompress_folder'))
    except Exception as e:
        print(e)
        exit()

checkPathExistence()

def renameImagesWithJson(source, destination, json_file):

    if (source == destination):
        print("Source and destination are the same!")
        exit()

    with open(json_file, 'r') as f:
        data = json.load(f)

    for filename in os.listdir(source):
        if filename.endswith(".jpg"):
            current_name = filename.split('_')[0]
            for item in data:
                if str(item["ObjektId"]) == current_name:
                    new_name = item["InventarNr"].replace('/', '_')
                    extension = os.path.splitext(filename)[1]
                    new_filename = f"{new_name}{extension}"
                    old_path = os.path.join(source, filename)
                    new_path = os.path.join(destination, new_filename)
                    #os.move(old_path, new_path)
                    try:
                         os.rename(old_path, new_path)
                    except Exception as e:
                        print(e)
                        break
                    

renameImagesWithJson(uncompress_folder, image_folder, json_file)