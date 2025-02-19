# 1_system_setup.sh
#!/bin/bash
set -e
INSTALL_DIR="$PWD"
echo "Installing system dependencies..."

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y \
  tmux vim cmake \
  python3-dev python3-venv python3-pip python3-pil python3-numpy \
  python-gpiozero \
  imagemagick git git-lfs \
  libopencv-dev python3-opencv \
  spidev

echo "System dependencies installed!"

# 2_python_setup.sh
#!/bin/bash
set -e
INSTALL_DIR="$PWD"
echo "Setting up Python environment..."

cd "$INSTALL_DIR"
python3 -m venv --system-site-packages venv
. venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv_contrib_python
python -m pip install inky[rpi]==1.5.0
python -m pip install pillow
python -m pip install spidev

echo "Python environment setup complete!"

# 3_build_xnnpack.sh
#!/bin/bash
set -e
INSTALL_DIR="$PWD"
echo "Starting XNNPACK build (this will take several hours)..."

cd "$INSTALL_DIR"
git clone https://github.com/google/XNNPACK.git
cd XNNPACK
git checkout 1c8ee1b68f3a3e0847ec3c53c186c5909fa3fbd3
mkdir build
cd build
cmake -DXNNPACK_BUILD_TESTS=OFF -DXNNPACK_BUILD_BENCHMARKS=OFF ..
if [ $? -ne 0 ]; then
    echo "CMAKE configuration for XNNPACK failed"
    exit 1
fi

echo "Building XNNPACK using 2 cores (be patient, this is a long process)..."
cmake --build . --config Release -- -j2
if [ $? -ne 0 ]; then
    echo "XNNPACK build failed"
    exit 1
fi

echo "XNNPACK build complete!"

# 4_build_onnxstream.sh
#!/bin/bash
set -e
INSTALL_DIR="$PWD"
echo "Starting OnnxStream build (this will take several hours)..."

cd "$INSTALL_DIR"
git clone https://github.com/vitoplantamura/OnnxStream.git
cd OnnxStream/src
mkdir build
cd build
cmake -DMAX_SPEED=ON -DOS_LLM=OFF -DOS_CUDA=OFF -DXNNPACK_DIR="${INSTALL_DIR}/XNNPACK" ..
if [ $? -ne 0 ]; then
    echo "CMAKE configuration for OnnxStream failed"
    exit 1
fi

echo "Building OnnxStream using 2 cores (be patient, this is a long process)..."
cmake --build . --config Release -- -j2
if [ $? -ne 0 ]; then
    echo "OnnxStream build failed"
    exit 1
fi

echo "OnnxStream build complete!"

# 5_download_model.sh
#!/bin/bash
set -e
INSTALL_DIR="$PWD"
echo "Downloading model..."

cd "$INSTALL_DIR"
mkdir -p models
cd models
git clone --depth=1 https://huggingface.co/vitoplantamura/stable-diffusion-xl-turbo-1.0-anyshape-onnxstream

echo "Model download complete!"