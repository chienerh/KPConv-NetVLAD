FROM nvcr.io/nvidia/pytorch:21.12-py3


LABEL version="0.5"

USER root

ENV DEBIAN_FRONTEND noninteractive

# scipy, tensorboard
RUN pip install scipy
RUN pip install -U scikit-learn
RUN pip install pykitti
RUN pip install opencv-python
RUN pip install wandb -qqq

# RUN pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu102_pyt190/download.html
# RUN pip install --no-index --no-cache-dir -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu112_pyt1110/download.html pytorch3d
# RUN pip install pytorch3d

# build essential, cmake, vim, git
RUN apt-get update && apt-get -y install cmake
RUN apt-get update && apt-get install -y vim
RUN apt-get install -y build-essential
RUN apt-get update && apt-get install -y git-all

# for epn
RUN pip install numpy
RUN pip install torch
RUN pip install torchvision
RUN pip install scikit-image
RUN pip install scikit-learn
RUN pip install open3d
RUN pip install tqdm
RUN pip install plyfile
RUN pip install parse
RUN pip install colour

# for KPConv
RUN pip install PyYAML
RUN pip install matplotlib
RUN pip install mayavi
# RUN pip install PyQt5==5.15.2
