# ImageClassification
Sammel Branch rund um AI 

## .env
Folgenden Environment Variablen würden über diesen Branch verwendet:
```
minio_url = ''
minio_access_key = ''
minio_secret_key = ''
ENCRYPTION_KEY = ''
redis_host = ''
redis_port = ''
redis_password = ''
```

## requirements.txt
Enthält alle Pakete die zur Auführung benötigt werden, mit folgendem Befehl kann sie mit pip installiert werden.
```
pip install -r .\src\tools\imageClassification\requirements.txt
```

## modelTest.py
Enthält Code um ein Image von verschiedene Modelle zu kategorisieren zu lassen, um so Wahrscheinlichkeiten/Kategorien über mehrere Modelle hinweg vergleichen zu können.

Im Code muss manuell angepasst werden, welches Bild klassifiziert werden soll, dies geschieht über folgende Codezeile:
```
input_image = Image.open('.\src\\tools\\imageClassification\\images\\000003_3.jpg')
```

Die Datei *imagenet_classes.txt* behinhaltet die Klassen, zu welchen ein Bild zugeordnet werden kann und stammt von der [Pytorch Seite](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt)


## processImages.py
Enthält jeglichen Code um ein Bild als Tensor in einer Redis DB zuspeichern.
Zudem kann mit einem input_tensor in der Redis DB mittels KNN ähnlich Vektoren ausgegeben werden, der hierfür benötigte Index wird ebenfalls erstellt.

Mittels processImages() kann nun ein kompletter Ordner von .jpg Dateien als Tensoren auf Redis hochgeladen werden.
```
def processImages():
    images = os.listdir(imagepath)
    images.pop(0)

    for image in tqdm(images, bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
        if not redis_client.exists(image):
            uploadTensorToRedis(createTensor(ResNet50, imagepath+image), image)
        else:
            print()
            print(f'Tensor for {image} already exists - skipping')
```


## splitDataset.py
Kann einen Datenbestand von einer .csv Datei in Trainings- und Testdaten aufteilen.

Input .csv Datei Aufbau:
```
filename,label
image1,jpg,dog
image2,jpg,dog
image3,jpg,cat
image4,jpg,cat
```