# ImageClassification
Sammel Branch rund um AI


## requirements.txt
Zu Beginn erstmal die benötigten Python Packages:
Für die einfachere Handhabung habe ich die beiden requirement Dateien in Eine konsolidiert. Diese ist in [diesem Verzeichnis](https://github.com/nosqlgeek/ai-meets-museum/tree/imageClassification/src/tools/imageClassification) zu finden.
Kann nach Clonen/Pullen des Branches wie folgt installiert werden:
```
pip install -r .\src\tools\imageClassification\requirements.txt
```


## .env
Eine Beispiel .env Datei liegt im [Hauptverzeichnis dieses Branches](https://github.com/nosqlgeek/ai-meets-museum/tree/imageClassification) --> example.env
Diese muss natürlich noch mit jeglichen Keys befüllt werden.

Für den Download von Bilder von Minio sind folgende Variablen notwendig:
```
minio_url = ''
minio_access_key = ''
minio_secret_key = ''
ENCRYPTION_KEY = ''
image_folder = ''
uncompress_folder = ''
```

Für die Erstellung und Upload von Tensoren zu Redis sind folgenden notwendig:
```
redis_host = ''
redis_port = ''
redis_password = ''
image_folder = ''
```
Für weitere Informationen bitte die example.env durchlesen.


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