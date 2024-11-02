FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu20.04

# Prevent timezone prompt during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set POSE_ROOT environment variable
ENV POSE_ROOT=/workspace/clothes-lm

# Install basic development tools and libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    build-essential \
    cmake \
    git \
    curl \
    libglib2.0-0 \
    python3 \
    python3-pip \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone the repository
RUN git clone https://github.com/JoniRozen/FashionTest.git ${POSE_ROOT}

# Setup Python environment
RUN cd clothes-lm && pip3 install -r requirements.txt

# Execute setup commands
RUN cd ${POSE_ROOT}/lib && \
    make && \
    cd .. && \
    mkdir -p output && \
    mkdir -p log

# Set default working directory to POSE_ROOT
WORKDIR ${POSE_ROOT}

# Set default command to bash
CMD ["/bin/bash"]