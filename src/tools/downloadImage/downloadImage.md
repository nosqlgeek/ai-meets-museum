# Image Access

Hiermit können Bilder aus der Minio Datenbank heruntergeladen und mit dem gesetzten Passwort in der .env Datei entschlüsselt werden.

# requirements.txt
Enthält alle Pakete die zur Auführung benötigt werden, mit folgendem Befehl kann sie mit pip installiert werden.
```
pip install -r .\requirements.txt
```

# .env Datei
Die .env Datei müsst ihr bei euch lokal erstellen, da diese die *secrets* enthält mit denen man auf die Datenbank zugreifen kann.
Die Datei muss folgenden Variablen beinhalten:
```
minio_url = ''
minio_access_key = ''
minio_secret_key = ''
ENCRYPTION_KEY = ''
```