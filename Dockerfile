#python 3.8, pytorch 1.13
FROM nvcr.io/nvidia/pytorch:22.11-py3

# SYSTEM, python
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    wget curl vim git \
    gcc

RUN pip install --upgrade pip

# install pre-requisites
WORKDIR /installer

# install requirements from temporalmaxer repo
COPY ./requirements.txt /installer/requirements.txt
RUN python -m pip install -r requirements.txt

WORKDIR /temporalmaxer
