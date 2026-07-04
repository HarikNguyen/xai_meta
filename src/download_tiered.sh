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
for category in "train" "test" "val"; do
    part_num=1
    while true; do
        file_name="${category}_part_${part_num}.tar"
        
        # Call the function. If it returns 1 (file not found), break the category loop
        download_and_extract "$file_name"
        if [ $? -eq 1 ]; then
            echo "--> All parts for $category downloaded and extracted."
            echo "==================================================="
            break
        fi
        
        ((part_num++))
    done
done

echo "PROCESS COMPLETED!"
