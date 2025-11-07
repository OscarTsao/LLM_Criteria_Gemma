# Multi-stage build for LLM_Criteria_Gemma
# Stage 1: Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Stage 2: Dependencies
FROM base AS dependencies

WORKDIR /tmp

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Stage 3: Application
FROM dependencies AS application

# Set working directory
WORKDIR /workspace

# Copy application code
COPY . /workspace/

# Install package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /workspace/data \
    /workspace/outputs \
    /workspace/logs \
    /workspace/benchmarks/results

# Set permissions
RUN chmod -R 755 /workspace

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default command
CMD ["/bin/bash"]

# Stage 4: Training (for production training)
FROM application AS training

WORKDIR /workspace

# Set entrypoint for training
ENTRYPOINT ["python", "src/training/train_gemma_hydra.py"]

# Stage 5: Development (with Jupyter)
FROM application AS development

WORKDIR /workspace

# Install JupyterLab extensions
RUN pip install --no-cache-dir \
    jupyterlab-git \
    jupyterlab-github

# Expose Jupyter port
EXPOSE 8888

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# Stage 6: CPU-only version (smaller image)
FROM python:3.10-slim AS cpu-only

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest pytest-cov

# Copy application
COPY . /workspace/
RUN pip install -e .

# Create directories
RUN mkdir -p /workspace/data /workspace/outputs /workspace/logs

CMD ["/bin/bash"]
