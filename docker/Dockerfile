ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

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
COPY watertight_transformer mesh_fusion_simple/watertight_transformer
COPY setup.py README.md mesh_fusion_simple/ 
RUN cd mesh_fusion_simple && python setup.py build_ext --inplace && pip install -e .
RUN cd mesh_fusion_simple && git clone https://github.com/hjwdzh/ManifoldPlus.git && \
    cd ManifoldPlus && git submodule update --init --recursive && mkdir build && \
    cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 && mkdir ../../scripts/ && \
    mv manifold ../../scripts/manifoldplus && cd ../../ && rm -rf ManifoldPlus
COPY scripts mesh_fusion_simple/scripts
