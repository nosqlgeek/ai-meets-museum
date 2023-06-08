# Docker Containerized Setup

Wir haben uns dazu entschieden die App in einem Docker Container bereitzustellen, damit ist die Installation für den Endanwender deutlich vereinfacht und reduziert.
Hierfür sind jedoch ein paar Schritte notwendig:

## Schritt 1: Docker Umgebung installieren
Angenommen das Endgerät basiert auf Windows, sind folgende Schritte notwendig.

### Voraussetungen
* Windows Version:  
Windows 11 64-bit: Home or Pro version 21H2 or higher, or Enterprise or Education version 21H2 or higher.   
Windows 10 64-bit: Home or Pro 21H2 (build 19044) or higher, or Enterprise or Education 21H2 (build 19044) or higher.

* WSL 2 aktiviert:  
https://learn.microsoft.com/en-us/windows/wsl/install

### Docker Desktop Client
Für die Installation einfach den Client auf folgender Seite herunterladen und installieren  
https://docs.docker.com/desktop/install/windows-install/

## Schritt 2: Github Branch klonen
Hierzu auf der Startseite des Repositories über den Button "Code" --> "Download ZIP" den main Branch herunterladen.  
Im Anschluss im gewünschtem Ordner entpacken.

## Schritt 3: Config Dateien anpassen
Nun sollte sich im zuvor entpacktem Ordner folgende Ordnerstruktur befinden:    
```
src/aimm/*app files*
```

### docker-compose.yaml
Im Ordner aimm befindet sich eine docker-compose.yaml Datei, welche genau Instruktionen für die Docker Umgebung enthält.    
Diese muss mit den entsprechenden Credentials befüllt und auf die lokalen Gegebenheiten angepasst werden.
```
version: '3'
services:
  aimm:
    container_name: aimm
    ports:
      - 5000:5000
    environment:
      - REDIS_Host=""
      - REDIS_PORT=""
      - REDIS_PASSWORD=""
    volumes:
      - C:/Users/xyz/path/to/image/folder:/app/static/ImgStore
    image: aimm
```

## Schritt 4: Build Docker Image
Dieser Schritt ist wohl der zeitaufwendigste und kann je nach Endgerät einige Minuten dauern.   
Zuerst muss eine Eingabeaufforderung (aka CMD) im Verzeichnis *src/aimm/* geöfffnet werden.

Der Befehl *dir* sollte nun den Inhalt des Ordners auflisten:
```
08.06.2023  13:34    <DIR>          .
08.06.2023  13:34    <DIR>          ..
08.06.2023  12:01                25 .env
08.06.2023  13:29             6.811 app.py
08.06.2023  12:50             3.194 database.py
08.06.2023  13:29               376 docker-compose.yaml
08.06.2023  12:01               297 Dockerfile
08.06.2023  13:34    <DIR>          docker_entrypoint
08.06.2023  12:48               206 entrypoint.sh
08.06.2023  13:29               111 requirements.txt
08.06.2023  13:34    <DIR>          static
08.06.2023  13:34    <DIR>          templates
```


Anschließend kann das Docker Image mit folgendem Befehl gebaut werden:  
(Der Parameter -t gibt dem Image den Namen "aimm" mit)

```
docker build -t aimm .
```

## Schritt 5: Docker Container erstellen
Sollten alle vorherigen Schritte erfolgreich durchgelaufen sein, kann nun der Container erstellt werden.    
Dies geht dank Docker ebenfalls recht einfach, hierzu ist nur dieser Command notwendig:     
(Wichtig: die CMD ist weiterhin im *src/aimm/* Verzeichnis)
```
docker compose up -d
```

## Schritt 6: App Zugriff
Zuvor wurde in der docker-compose Datei der Port 5000 auf dem Host Gerät in den Container gebunden.     
Deshalb kann nun über *localhost:5000* auf die App zugegriffen werden.