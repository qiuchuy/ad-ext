# use IREE image as base image
FROM gcr.io/iree-oss/base:latest

# install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \ 
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN /root/miniconda3/bin/conda init
RUN /root/miniconda3/bin/activate

WORKDIR /root
