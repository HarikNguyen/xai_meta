import os
import pandas as pd

SRC_FOLDER = "cub_200"
DST_FOLDER = "cub_200_new"

# Create the new directory structure
os.makedirs(f"{DST_FOLDER}/val", exist_ok=True)
os.makedirs(f"{DST_FOLDER}/test", exist_ok=True)

print("Processing CSV files...")

# Merge old test and val dataframes into the new test dataframe
df_val = pd.read_csv(f"{SRC_FOLDER}/val.csv")
df_test = pd.read_csv(f"{SRC_FOLDER}/test.csv")
df_test_new = pd.concat([df_test, df_val], ignore_index=True)
df_test_new.to_csv(f"{DST_FOLDER}/test.csv", index=False)

# Save the old train dataframe as the new val dataframe
df_train = pd.read_csv(f"{SRC_FOLDER}/train.csv")
df_train.to_csv(f"{DST_FOLDER}/val.csv", index=False)


print("Creating symbolic links for images...")

# Map the old train directory to the new val directory
src_train_dir = os.path.abspath(f"{SRC_FOLDER}/train")
dst_val_dir = f"{DST_FOLDER}/val"

if os.path.exists(src_train_dir):
    for class_name in os.listdir(src_train_dir):
        src_class_path = os.path.join(src_train_dir, class_name)
        dst_class_path = os.path.join(dst_val_dir, class_name)
        
        # Link the entire class directory directly for the validation set
        if os.path.isdir(src_class_path) and not os.path.exists(dst_class_path):
            os.symlink(src_class_path, dst_class_path)

# Merge images from old val and old test into the new test directory
dst_test_dir = f"{DST_FOLDER}/test"

for split in ["val", "test"]:
    src_split_dir = os.path.abspath(f"{SRC_FOLDER}/{split}")
    if not os.path.exists(src_split_dir):
        continue
        
    for class_name in os.listdir(src_split_dir):
        src_class_path = os.path.join(src_split_dir, class_name)
        dst_class_path = os.path.join(dst_test_dir, class_name)
        
        if os.path.isdir(src_class_path):
            # Create the class directory in the destination since we are merging two sources
            os.makedirs(dst_class_path, exist_ok=True)
            
            # Iterate and create a symbolic link for each image file
            for filename in os.listdir(src_class_path):
                src_file_path = os.path.join(src_class_path, filename)
                dst_file_path = os.path.join(dst_class_path, filename)
                
                # Create the link only if it does not already exist
                if not os.path.exists(dst_file_path):
                    os.symlink(src_file_path, dst_file_path)

print(f"Completed dataset generation at '{DST_FOLDER}'")
