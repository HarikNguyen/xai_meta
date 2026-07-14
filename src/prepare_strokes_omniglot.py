import os
import urllib.request
import zipfile
import shutil
import csv
import io
import numpy as np
from PIL import Image


import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

BASE_DIR = "strokes_omniglot"
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")
VAL_CSV = os.path.join(BASE_DIR, "val.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
TEMP_DIR = "temp_omniglot"

URLS = [
    "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip",
    "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip",
    "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/strokes_background.zip",
    "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/strokes_evaluation.zip"
]

def get_color(k):    
    scol = ['r','g','b','m','c']
    if k < len(scol): return scol[k]
    return scol[-1]

def load_img(fn):
    I = plt.imread(fn)
    return np.array(I, dtype=bool)

def load_motor(fn):
    motor = []
    with open(fn, 'r') as fid:
        lines = [l.strip() for l in fid.readlines()]
    stk = []
    for myline in lines:
        if myline == 'START':
            stk = []
        elif myline == 'BREAK':
            motor.append(np.array(stk))
            stk = [] 
        else:
            arr = np.fromstring(myline, dtype=float, sep=',')
            stk.append(arr)
    return motor

def space_motor_to_img(pt):
    pt = np.copy(pt)
    pt[:,1] = -pt[:,1]
    return pt

def render_and_save_stroke_image(fn_img, fn_stk, dest_path, fig, ax):
    I = load_img(fn_img)
    motor = load_motor(fn_stk)
    
    ax.clear()
    ax.set_axis_off()
    
    # Process coordinates
    drawing = [d[:,0:2] for d in motor]
    drawing = [space_motor_to_img(d) for d in drawing]
    
    # Draw background
    ax.imshow(I, cmap='gray')
    
    # Draw strokes
    for sid in range(len(drawing)):
        stk = drawing[sid]
        color = get_color(sid)
        if stk.shape[0] > 1:
            ax.plot(stk[:,0], stk[:,1], color=color, linewidth=2)
        else:
            ax.plot(stk[0,0], stk[0,1], color=color, linewidth=2, marker='.')
            
    # Temporary buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # Convert to 84x84 and save
    with Image.open(buf) as img:
        img = img.resize((84, 84), Image.Resampling.LANCZOS)
        img.save(dest_path, 'JPEG', quality=95)

def download_and_extract():
    os.makedirs(TEMP_DIR, exist_ok=True)
    for url in URLS:
        zip_path = os.path.join(TEMP_DIR, os.path.basename(url))
        if not os.path.exists(zip_path):
            print(f"[*] Download {url}...")
            urllib.request.urlretrieve(url, zip_path)
        print(f"[*] Unzip {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)

def process_dataset(img_folder, stk_folder, dest_dir, csv_path):
    print(f"\n[*] Start processing {dest_dir}...")
    os.makedirs(dest_dir, exist_ok=True)
    records = []
    
    img_src = os.path.join(TEMP_DIR, img_folder)
    stk_src = os.path.join(TEMP_DIR, stk_folder)
    
    # Init Figure once to save memory
    fig = plt.figure(figsize=(2, 2), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    count = 0
    for alphabet in sorted(os.listdir(img_src)):
        alpha_path = os.path.join(img_src, alphabet)
        if not os.path.isdir(alpha_path): continue

        for char in sorted(os.listdir(alpha_path)):
            char_path = os.path.join(alpha_path, char)
            if not os.path.isdir(char_path): continue
            
            # Classname = alphabet + char
            class_name = f"{alphabet}_{char}"
            dest_class_dir = os.path.join(dest_dir, class_name)
            os.makedirs(dest_class_dir, exist_ok=True)
            
            for img_file in sorted(os.listdir(char_path)):
                if not img_file.endswith('.png'): continue
                
                # Get file path (img + stroke)
                fn_img = os.path.join(char_path, img_file)
                stk_file = img_file.replace('.png', '.txt')
                fn_stk = os.path.join(stk_src, alphabet, char, stk_file)
                
                if not os.path.exists(fn_stk): continue
                
                # Render and save
                out_img_name = img_file.replace('.png', '.jpg')
                dest_img_path = os.path.join(dest_class_dir, out_img_name)
                render_and_save_stroke_image(fn_img, fn_stk, dest_img_path, fig, ax)
                records.append([f"{class_name}/{out_img_name}", class_name])
                
                count += 1
                if count % 1000 == 0:
                    print(f"Rendered {count} images...")
                    
    plt.close(fig)
    
    # write csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(records)
    print(f"[+] Completed {dest_dir}: {len(records)} imgs.")

def main():
    download_and_extract()
    
    # Background => VAL set
    process_dataset("images_background", "strokes_background", VAL_DIR, VAL_CSV)
    
    # Evaluation => TEST set
    process_dataset("images_evaluation", "strokes_evaluation", TEST_DIR, TEST_CSV)
    
    print("\n[*] Clean up...")
    shutil.rmtree(TEMP_DIR)
    print("[+] COMPLETED!")

if __name__ == "__main__":
    main()
