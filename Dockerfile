FROM python:3.10-slim-bullseye

LABEL maintainer="Samuel Appiah Kubi <samuel.appiah-kubi@student.utwente.nl>"
LABEL description="Mars Gully Digital Twin - Multi-Sensor Deep Learning Pipeline"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    wget \
    curl \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment
ENV GDAL_VERSION=3.4.3
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Set working directory
WORKDIR /workspace

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir GDAL==$(gdal-config --version) && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create data directories
RUN mkdir -p data/raw/{hirise,ctx,crism,hrsc,mola,themis} \
             data/processed/{hirise,ctx,feature_stacks,masks} \
             data/tiles/{images,masks} \
             data/models \
             data/outputs/{probability_maps,binary_maps,change_detection,stac,dashboard,reports,logs}

# Environment variables
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Default command
CMD ["python", "main.py", "--help"]

# Usage:
# Build: docker build -t mars-gully-twin .
# Run:   docker run --gpus all -v $(pwd)/data:/workspace/data mars-gully-twin python main.py --step all
