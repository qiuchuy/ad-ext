FROM gcr.io/iree-oss/base:latest

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \ 
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN conda init bash

RUN apt-get update && apt-get install -y git ninja-build cmake

WORKDIR /workspace

