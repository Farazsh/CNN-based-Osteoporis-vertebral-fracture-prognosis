FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04


# BUILD
# docker build . -t cnw --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f docker/Dockerfile

# RUN
# docker run --rm --gpus=all --shm-size 8G -v $(pwd):/opt/project -ti cnw
# docker run --rm --gpus device=0 --shm-size 8G -v $(pwd):/opt/project -ti cnw
# docker run --rm --gpus '"device=2,3"' --shm-size 8G -v $(pwd):/opt/project -ti cnw
# docker run --rm --shm-size 64G --user "$(id -u):$(id -g)" -v $(pwd):/opt/project --gpus '"device=0,1,2,3"' -v /home/jlaue/checkpoints:/opt/checkpoints -v /home/jlaue/data:/data -ti cnw
# docker run --rm --gpus '"device=0,1,2,3"' --shm-size 64G --user "$(id -u):$(id -g)" -v $(pwd):/opt/project -v /home/hkruse/rescaled_data_julian:/data -ti cnw


# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

ARG USER_ID
ARG GROUP_ID

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin


COPY requirements.txt .

# Dependencies
RUN apt-get -y update
RUN apt-get -y upgrade && \
    apt-get -y install texlive-latex-extra texlive-fonts-recommended dvipng cm-super && \
    apt-get -y install tzdata &&\
    apt-get -y install python3-pip && \
    pip3 install --upgrade pip install

# create user
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

# install all packages
RUN pip3 install -r requirements.txt --user




