#!/usr/bin/env bash
set -e

wget -O miniImagenet.zip "https://huggingface.co/datasets/hariknguyen2419/FSL_datasets/resolve/main/miniImagenet.zip?download=true"
unzip -q miniImagenet.zip
rm -rf miniImagenet.zip


