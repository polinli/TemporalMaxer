# Use the official Docker image for CUDA 11.8 as the base image
FROM nvidia/cuda:11.3.0-base-ubuntu20.04
FROM python:3.9

# SYSTEM
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    wget curl vim git \
    gcc

RUN pip install --upgrade pip

# PyTorch, torchvision, torchaudio, and cudatoolkit
RUN pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1

# install pre-requisites
WORKDIR /installer

# install requirements from temporalmaxer repo
COPY ./requirements.txt /installer/requirements.txt
RUN python -m pip install -r requirements.txt

WORKDIR /temporalmaxer
