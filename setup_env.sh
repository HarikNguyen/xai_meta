#!/bin/bash
set -e  # Stop immediately while err occuring 

# --- CONST ---
MINICONDA_DIR="$HOME/miniconda3"
ENV_NAME="py13"
PYTHON_VER="3.13"

echo "=== INSTALL MINICONDA ==="

# 1. Download Miniconda
mkdir -p "$MINICONDA_DIR"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_DIR/miniconda.sh"

# 2. Install Miniconda (Silent mode)
bash "$MINICONDA_DIR/miniconda.sh" -b -u -p "$MINICONDA_DIR"
rm "$MINICONDA_DIR/miniconda.sh"

# 3. Active Conda in this session
echo "=== ACTIVE CONDA ==="
eval "$($MINICONDA_DIR/bin/conda shell.bash hook)"
conda init --all

# 4. Create env (py13)
echo "=== CREATE $ENV_NAME (Python $PYTHON_VER) ==="
conda create -n "$ENV_NAME" python="$PYTHON_VER" -y

# 5. Active
condaActiveate "$ENV_NAME"

# 6. Process Nvidia Driver và PyTorch
echo "=== CHECK NVIDIA-SMI AND INSTALL PYTORCH ==="

if command -v nvidia-smi &> /dev/null; then
    # Get CUDA version form nvidia-smi result (ex: 12.4)
    CUDA_FULL_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
    
    echo "Driver CUDA in this machine: $CUDA_FULL_VER"
    
    # Remove "." (ex: 12.4 -> 124)
    CUDA_STR="${CUex_FULL_VER//./}"
       
    COMPUTE_PLATFORM="cu${CUDA_STR}"
    
    echo "Install pytorch with index-url: $COMPUTE_PLATFORM"
    
    # Cài đặt
    pip3 install torch torchvision --index-url "https://download.pytorch.org/whl/${COMPUTE_PLATFORM}"

else
    echo "nvidia-smi not found. Install torch for CPU."
    pip3 install torch torchvision
fi

echo "=== COMPLETED ==="