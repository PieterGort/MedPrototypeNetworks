FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get update --fix-missing -y

RUN apt-get install -y ffmpeg libsm6 libxrender1 libxtst6 zip 

RUN apt-get install -y p7zip-full

# Library components for av
RUN apt-get install -y \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev

RUN apt-get install -y python3 python3-pip git python3-dev pkg-config htop

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install torch-uncertainty
RUN pip3 install pytorch-lightning lightly
RUN pip3 install project-lighter --pre
RUN pip3 install numpy pandas scikit-image scikit-learn scipy opencv-python
RUN pip3 install matplotlib seaborn
RUN pip3 install nibabel SimpleITK monai wandb
RUN pip3 install pytz python-dateutil h5py openpyxl

WORKDIR /app/script

