#!/bin/bash

set -e  # Exit immediately if a command fails

INSTALL_DIR="$PWD"
echo "Installing to $INSTALL_DIR"

# Update package list and upgrade system
sudo apt-get update
sudo apt-get upgrade -y

# Install required dependencies
sudo apt-get install -y \
  tmux vim cmake \
  python3-dev python3-venv python3-pip python3-pil python3-numpy \
  imagemagick git git-lfs \
  libopencv-dev python3-opencv \
  spidev

# Create and activate virtual environment
cd "$INSTALL_DIR"
python3 -m venv venv
. venv/bin/activate

# Upgrade pip and install Python packages
python -m pip install --upgrade pip
python -m pip install opencv_contrib_python
python -m pip install inky[rpi]==1.5.0
python -m pip install pillow
python -m pip install spidev

# Clone and build XNNPACK
cd "$INSTALL_DIR"
git clone https://github.com/google/XNNPACK.git
cd XNNPACK
git checkout 1c8ee1b68f3a3e0847ec3c53c186c5909fa3fbd3
mkdir build
cd build
cmake -DXNNPACK_BUILD_TESTS=OFF -DXNNPACK_BUILD_BENCHMARKS=OFF ..
cmake --build . --config Release

# Clone and build OnnxStream
cd "$INSTALL_DIR"
git clone https://github.com/vitoplantamura/OnnxStream.git
cd OnnxStream/src
mkdir build
cd build
cmake -DMAX_SPEED=ON -DOS_LLM=OFF -OS_CUDA=OFF -DXNNPACK_DIR="${INSTALL_DIR}/XNNPACK" ..
cmake --build . --config Release

# Download model
cd "$INSTALL_DIR"
mkdir -p models
cd models
git clone --depth=1 https://huggingface.co/vitoplantamura/stable-diffusion-xl-turbo-1.0-anyshape-onnxstream

echo "Installation complete!"
