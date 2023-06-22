# Docker Containerized Setup

We decided to deploy the app in a Docker container, this way the installation is much simplified and reduced for the end user.
However, a few steps are necessary for this:

## Step 1: Install Docker environment
Assuming the end device is based on Windows, the following steps are necessary.

### Prerequisites
* Windows Version:  
Windows 11 64-bit: Home or Pro version 21H2 or higher, or Enterprise or Education version 21H2 or higher.   
Windows 10 64-bit: Home or Pro 21H2 (build 19044) or higher, or Enterprise or Education 21H2 (build 19044) or higher.

* WSL 2 activated:  
https://learn.microsoft.com/en-us/windows/wsl/install

### Docker Desktop Client
For the installation simply download and install the client on the following page  
https://docs.docker.com/desktop/install/windows-install/

## Step 2: Clone Github Branch
To do this, download the main branch on the start page of the repository via the button "Code" --> "Download ZIP".  
Then unpack it in the desired folder.

## Step 3: Customize config files
Now the following folder structure should be in the previously unzipped folder:    
```
src/aimm/*app files*
```

### docker-compose.yaml
The aimm folder contains a docker-compose.yaml file which contains exact instructions for the Docker environment.    
This file must be filled with the appropriate Redis credentials. In addition, the images (which have already been renamed correctly) must be stored in a local directory, which is then referenced in the docker-compose.yaml file.
```
version: '3'
services:
  aimm:
    container_name: aimm
    ports:
      - 5000:5000
    environment:
      - REDIS_HOST=redis-99999.redis.url.com
      - REDIS_PORT=99999
      - REDIS_PASSWORD=password
    volumes:
      - C:/Users/steph/.cache/downloadImage/target_folder:/app/static/ImgStore
    image: aimm
```

## Step 4: Change End of Line Sequence
The file *entrypoint.sh* is copied to the Docker image in [step 5](#step-5-build-docker-image) and is executed at every container start.
Since Windows uses *CRLF* as the line end by default, it is imperative to change this to *LF* for the *entrypoint.sh* file beforehand.
This is necessary because our Docker image is based on Linux and Linux can't do anything with *CRLF* and accordingly errors occur at container startup.
This should be possible in any text editor (e.g. Notepad++). 


## Step 5: Build Docker Image
This step is probably the most time-consuming and can take a few minutes depending on the end device.   
First, a command prompt (aka CMD) must be opened in the *src/aimm/* directory.

The *dir* command should now list the contents of the folder:
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


After that, the Docker image can be built with the following command:  
(The parameter -t gives the image the name "aimm").

```
docker build -t aimm .
```

## Step 6: Create Docker Container
If all previous steps have been successfully completed, the container can now be created.    
This is also quite easy thanks to Docker, only this command is necessary:     
(Important: the CMD is still in the *src/aimm/* directory).
```
docker compose up -d
```

## Step 7: App access
Previously, in the docker-compose file, port 5000 on the host device was bound into the container.     
Therefore, the app can now be accessed via *http://localhost:5000*.
