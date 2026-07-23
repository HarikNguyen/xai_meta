#!/bin/bash
set -e

REPO_ID="hariknguyen/tiered_imagenet"
BASE_URL="https://huggingface.co/datasets/$REPO_ID/resolve/main"
TARGET_DIR="tiered_imagenet"

# Number of files downloaded+extracted at the same time. Each job downloads ONE
# file, extracts it, then deletes the .tar right away -- so at most balance_num
# .tar files ever sit on disk unextracted at once, regardless of how many files
# there are in total. Lower this if disk space is tight (e.g. 2-3 for a 32GB
# disk); raise it if you have more headroom and want more parallel connections.
balance_num=${1:-4}

mkdir -p "$TARGET_DIR"

# Function: Download -> Extract -> Delete (one file, one job)
download_and_extract() {
    local file_name=$1
    local file_url="$BASE_URL/$file_name?download=true"
    local target_file="$TARGET_DIR/$file_name"

    echo "Downloading: $file_name..."
    if ! wget -q -O "$target_file" "$file_url"; then
        echo "ERROR: failed to download $file_name" >&2
        rm -f "$target_file"
        return 1
    fi

    if [[ "$file_name" == *.tar ]]; then
        echo "Extracting: $file_name..."
        if ! tar -xf "$target_file" -C "$TARGET_DIR"; then
            echo "ERROR: failed to extract $file_name" >&2
            return 1
        fi
        rm "$target_file" # free up disk space right away, before the next file finishes
    fi

    echo "Done: $file_name!"
    echo "---------------------------------------------------"
}
export -f download_and_extract
export BASE_URL TARGET_DIR

echo "=== STARTING PARALLEL DOWNLOAD AND EXTRACTION PROCESS (balance_num=$balance_num) ==="

declare -A categories
categories=( ["test"]=5 ["train"]=9 ["val"]=3 )

all_files=("train.csv" "test.csv" "val.csv")
for category in "train" "test" "val"; do
    for ((part_num=1; part_num<=${categories[$category]}; part_num++)); do
        all_files+=("${category}_part_${part_num}.tar")
    done
done

if printf "%s\n" "${all_files[@]}" | xargs -I{} -P "$balance_num" bash -c 'download_and_extract "$@"' _ {}; then
    echo "PROCESS COMPLETED!"
else
    echo "ERROR: one or more files failed to download/extract. Check the log above." >&2
    exit 1
fi
