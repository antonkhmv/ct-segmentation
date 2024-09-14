FROM nvcr.io/nvidia/pytorch:22.12-py3

ARG PYTHON="python3"
ARG UBUNTU_NAME="focal"
ARG PIP="$PYTHON -m pip"

WORKDIR /root

COPY ./requirements.txt .

RUN $PIP install --upgrade pip
RUN $PIP install -r requirements.txt
