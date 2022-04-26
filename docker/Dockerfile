ARG CUDA_VERSION
ARG UBUNTU_VERSION

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION

# Install some basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends\
     meshlab \
     xvfb \
     libglew-dev \
     freeglut3-dev \ 
     build-essential \ 
     cmake \
     curl \
     ca-certificates \
     git \
     vim \
 && rm -rf /var/lib/apt/lists/*


# Install miniconda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \ 
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
     /opt/conda/bin/conda clean -ya


ENV PATH /opt/conda/bin:$PATH

COPY docker/environment.yaml .

RUN conda env update -f environment.yaml && conda clean -afy

# Mesh fusion dependency installations
COPY mesh_fusion mesh_fusion_simple/mesh_fusion
COPY setup.py README.md mesh_fusion_simple/ 
RUN cd mesh_fusion_simple && python setup.py build_ext --inplace && pip install -e .
COPY scripts mesh_fusion_simple/scripts
