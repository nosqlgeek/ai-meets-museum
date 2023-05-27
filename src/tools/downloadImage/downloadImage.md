# Image Access

Hiermit können Bilder aus der Minio Datenbank heruntergeladen und mit dem gesetzten Passwort in der .env Datei entschlüsselt werden.

# .env Datei
Die .env Datei müsst ihr bei euch lokal erstellen, da diese die *secrets* enthält mit denen man auf die Datenbank zugreifen kann.
Die Datei muss folgenden Variablen beinhalten:
```
minio_url = ''
minio_access_key = ''
minio_secret_key = ''
ENCRYPTION_KEY = ''
image_folder = 'C:/Users/steph/.cache/downloadImage/image_folder/'
uncompress_folder = 'C:/Users/steph/.cache/downloadImage/uncompress_folder/'
```

## downloadImage.py
Hiermit können die besagten Bilder aus dem Minio Bucket heruntergeladen und entpackt werden.

## renameImagesToInv.py
Mit dieser Funktion können die Bilder zur dazugehörgigen InventarNr umbenannt werden.
Wichtig ist hierbei, dass 'source' und 'destination' nicht der selbe Ordner ist.

Hierzu ist weiterhin die JSON File notwendig, die die Beziehungen enthält.
Diese kann jedoch wegen Datenschutzrechtlichen Gründen hier ebenfalls nicht hochgeladen werden.