import os
import random
import shutil

img_train = "yolo/images/train"
lbl_train = "yolo/labels/train"

img_val = "yolo/images/val"
lbl_val = "yolo/labels/val"

os.makedirs(img_val, exist_ok=True)
os.makedirs(lbl_val, exist_ok=True)

files = os.listdir(img_train)
random.shuffle(files)

val_count = int(0.1 * len(files))
val_files = files[:val_count]

for f in val_files:
    shutil.move(os.path.join(img_train, f), os.path.join(img_val, f))
    label = f.rsplit(".",1)[0] + ".txt"
    shutil.move(os.path.join(lbl_train, label), os.path.join(lbl_val, label))

print("✅ Train/Val split done")
