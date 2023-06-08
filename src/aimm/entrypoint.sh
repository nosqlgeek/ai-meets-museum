#!/bin/bash
# Initialization steps
echo "Initialization of Database..."
python /app/docker_entrypoint/databaseInitialization.py

# Start the application
echo "Starting the application..."
python /app/app.py