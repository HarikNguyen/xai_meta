import os
from PIL import Image

# Resize all images in the subdirectories of train/, val/, test/
def resize_all_images(dataset_name, size=(84, 84)):
    # Default data splits
    splits = ["train", "val", "test"]
    
    for split in splits:
        # Construct the path (e.g., miniImagenet/train)
        root_dir = os.path.join(dataset_name, split)
        
        # Skip if the directory does not exist
        if not os.path.isdir(root_dir):
            print(f"Warning: Directory {root_dir} not found, skipping...")
            continue
            
        print(f"Processing: {root_dir}")
        for label in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label)
            
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    with Image.open(img_path) as im:
                        # Note: From Pillow 10.0.0 onwards, use Image.Resampling.LANCZOS
                        # For backwards compatibility, getattr is used here.
                        im = im.resize(size, getattr(Image, 'Resampling', Image).LANCZOS)
                        im.save(img_path)
                except Exception as e:
                    print(f"Error with image: {img_path} - {e}")
                    
        print(f"Finished {root_dir}")

# Call the function
resize_all_images("miniImagenet")
resize_all_images("cub_200")
resize_all_images("tiered_imagenet")
