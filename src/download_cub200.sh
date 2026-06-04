#!/usr/bin/env bash
set -e

wget -O miniImagenet.zip "https://huggingface.co/datasets/hariknguyen2419/FSL_datasets/resolve/main/CUB_200_2011.zip?download=true"
unzip -q cup200.zip
rm -rf cub200.zip


