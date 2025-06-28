# Base image 
FROM python:3.10.17-slim-bullseye AS Base

# Install packages
RUN apt-get update -y \
    && apt-get dist-upgrate -y \
    && apt-get install curl lsb-release -y

# Install Docker CLI
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && ehco "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyrin.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/source.list.d/docker/list > /dev/null \
    && apt-get update -y\
    && apt-get install -y docker-ce-CLI