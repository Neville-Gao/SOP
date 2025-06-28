# docker build -f Dockerfile -t sop_image .
# docker run -d --name sop_app -p 8080:8080 sop_image:latest 

# curl -X POST http://localhost:8080/train \
# -H "Content-Type: application/json" \
# -d '{"width": 100, "height": 100, "iterations": 1000, "input_size": 10, "seed": 42}' \
# --output som_output.png

# Base image 
FROM python:3.10.17-slim-bullseye AS base

# Install packages
RUN apt-get update -y \
    && apt-get dist-upgrade -y \
    && apt-get install curl lsb-release -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/list/*

# Environment varibles
ENV USERNAME='sop_user' \
    USER_UID='1000' \
    USER_GID='1000'

# Create user and group using ENV values
RUN groupadd -g ${USER_GID} ${USERNAME} \
    && useradd -u ${USER_UID} -g ${USER_GID} -m ${USERNAME}

# Meta data
LABEL org.label-schema.schema-version=1.0 \
      org.label-schema.vcs-url=https://github.com/Neville-Gao/SOP \
      org.label-schema.name='Self Organizing Map' \
      org.label-schema.description='Self-Organizing Map (SOM) for unsupervised learning of input vectors.'

USER ${USERNAME}
WORKDIR /home/${USERNAME}/sop_app

COPY --chown=${USERNAME}:${USERNAME} ./requirements.txt ./requirements.txt 

# Create virtual environment, activate it and install dependencies
RUN python3 -m venv sop_venv \
    && . sop_venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

# Update PATH to include the virtual environment's bin directory
ENV PATH="/home/${USERNAME}/sop_app/sop_venv/bin:$PATH"

COPY --chown=${USERNAME}:${USERNAME} ./utils ./utils
COPY --chown=${USERNAME}:${USERNAME} ./scripts ./scripts
COPY --chown=${USERNAME}:${USERNAME} ./main.py ./main.py

RUN chmod +x ./scripts/start-server.sh 

EXPOSE 8080

CMD ["./scripts/start-server.sh"]

