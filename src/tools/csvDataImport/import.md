# Data Migration to Redis

Hiermit können Bild Vektoren und die dazugehörigen Daten aus der JSON File nach Redis importiert werden.

# .env Datei
Die .env Datei müsst ihr bei euch lokal erstellen, da diese die *secrets* enthält mit denen man auf die Datenbank zugreifen kann.
Die Datei muss folgenden Variablen beinhalten:
```
redis_host = ''
redis_port = ''
redis_password = ''
image_folder = 'C:/Users/steph/.cache/downloadImage/image_folder/'
target_folder = 'C:/Users/steph/.cache/downloadImage/target_folder/'
```