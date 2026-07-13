#!/bin/bash

REPO_ID="hariknguyen/tiered_imagenet"
BASE_URL="https://huggingface.co/datasets/$REPO_ID/resolve/main"
TARGET_DIR="tiered_imagenet"

mkdir -p "$TARGET_DIR"

# Function: Download -> Extract -> Delete
download_and_extract() {
    local file_name=$1
    local file_url="$BASE_URL/$file_name?download=true"
    local target_file="$TARGET_DIR/$file_name" # Cập nhật đường dẫn đích

    echo "Downloading: $file_name..."
    wget -q --show-progress -O "$target_file" "$file_url"

    if [[ "$file_name" == *.tar ]]; then
        echo "📦 Extracting: $file_name..."
        tar -xf "$target_file" -C "$TARGET_DIR"
        
        echo "Extraction complete! Deleting $file_name to free up disk space..."
        rm "$target_file"
    fi

    echo "Done: $file_name!"
    echo "---------------------------------------------------"
    return 0
}

echo "=== STARTING AUTOMATED DOWNLOAD AND EXTRACTION PROCESS ==="

# 1. Download CSV files first
echo "Downloading CSV files..."
for csv in "train.csv" "test.csv" "val.csv"; do
    download_and_extract "$csv"
done

# 2. Sequentially download and extract parts for train, test, and val
declare -A categories
categories=( ["test"]=5 ["train"]=9 ["val"]=3 )

for category in "train" "test" "val"; do
    total_parts=${categories[$category]}
    echo "---------------------------------------------------"
    echo "Starting $category set (Total parts: $total_parts)..."

    for ((part_num=1; part_num<=total_parts; part_num++)); do
        file_name="${category}_part_${part_num}.tar"
        download_and_extract "$file_name"
    done

    echo "--> All $total_parts parts for $category downloaded and extracted."
done

echo "PROCESS COMPLETED!"
