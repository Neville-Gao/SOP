# Base image 
FROM python:3.10.17-slim-bullseye AS Base

# Install packages
RUN apt-get update -y \
    && apt-get dist-upgrate -y \
    && apt-get install curl lsb-release -y

# Environment varibles
ENV APP_ENVIRONMENT = 'prod' \
    USERNAME = 'sop_user' \
    USER_ID = '1000' \
    USER_GID = '1000'

# Meta data
LABEL org.label-schema.schema-version = 1.0
LABEL org.label-schema.vcs-url = https://github.com/Neville-Gao/SOP
LABEL org.label-schema.name = 'Self Organizing Map'
LABEL org.label-schema.description = 'Self-Organizing Map (SOM) for unsupervised learning of input vectors.'

# Add/change user
RUN groupadd -g ${USER_GID} ${USERNAME} \
    && useradd -u ${USER_UID} -g ${USER_GID} -m ${USERNAME}

USER ${USERNAME}

WORKDIR /home/${USERNAME}/sop_app

COPY --chown=${USERNAME}:${USERNAME} ./requirements.txt ./requirements.txt 

# Create virtual environment, activate it and install dependencies
RUN python3 -m venv sop_venv \
    && . sop_venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

# Update PATH to include the virtual environment's bin directory
ENV PATH = "/home/${USERNAME}/sop_app/sop_venv/bin:${PATH}"

COPY --chown=${USERNAME}:${USERNAME} ./utils ./utils
COPY --chown=${USERNAME}:${USERNAME} ./scripts ./scripts
COPY --chown=${USERNAME}:${USERNAME} ./main.py ./main.py

RUN chmod +x ./scripts/start-server.sh 

EXPOSE 8080

CMD ["./scripts/start-server.sh"]

