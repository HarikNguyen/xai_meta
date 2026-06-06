import os
import shutil
import pandas as pd

FOLDER = "cub_200"

df_val = pd.read_csv(f"{FOLDER}/val.csv")
df_test = pd.read_csv(f"{FOLDER}/test.csv")

# val + test => test
df_test_new = pd.concat([df_test, df_val], ignore_index=True)
df_test_new.to_csv(f"{FOLDER}/test.csv", index=False)

# move val/* to test/
val_dir = f"{FOLDER}/val"
test_dir = f"{FOLDER}/test"
for filename in os.listdir(val_dir):
    src_path = os.path.join(val_dir, filename)
    dst_path = os.path.join(test_dir, filename)
    shutil.move(src_path, dst_path)

# rm val (old)
os.rmdir(val_dir)
os.remove('val.csv')

# rename train to val
os.rename('train', 'val')
os.rename('train.csv', 'val.csv')
