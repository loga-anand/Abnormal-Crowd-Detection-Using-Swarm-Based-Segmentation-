import os
import shutil
import random

def split_dataset(src, train_dir, test_dir, ratio=0.8):
    files = os.listdir(src)
    random.shuffle(files)

    split = int(len(files) * ratio)
    train_files = files[:split]
    test_files = files[split:]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(src, f), train_dir)

    for f in test_files:
        shutil.copy(os.path.join(src, f), test_dir)
