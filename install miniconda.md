mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
conda create -n py13 python=3.13

nvdia-smi
thay 130 = phien ban trong nvidia-smi
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
