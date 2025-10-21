#===========================================
#Dockerfile for NewsClipping Fine-Tuning
#===========================================

FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y \ 
    git wget curl unzip build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt 
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir ray[default]==2.49.2


WORKDIR /preprocessing

COPY . /preprocessing







