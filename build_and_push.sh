#!/bin/bash

# Set variables
DOCKER_USERNAME="lindanciko"
IMAGE_NAME="demographic-predictor"
TAG="latest"

# Build the image
echo "Building Docker image..."
docker build -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG} .

# Tag the image
echo "Tagging image..."
docker tag ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG} ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}

# Push to Docker Hub
echo "Pushing to Docker Hub..."
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}

echo "Done!" 