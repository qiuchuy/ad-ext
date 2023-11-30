# use IREE image as base image
FROM gcr.io/iree-oss/base:latest

# install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \ 
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN /root/miniconda3/bin/conda init
RUN /root/miniconda3/bin/activate

# install AINL python deps
RUN /root/miniconda3/bin/conda install pybind11
RUN /root/miniconda3/bin/conda install cmake
RUN /root/miniconda3/bin/conda install pytest
RUN /root/miniconda3/bin/conda install gast
RUN /root/miniconda3/bin/conda install black

# install AINL package deps
RUN apt-get update
RUN apt-get install clang-format




