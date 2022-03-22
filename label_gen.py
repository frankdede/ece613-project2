import os

from settings import *

if not os.path.exists(DATASET_META_DIR):
    os.mkdir(DATASET_META_DIR)

with open(TRAINING_LABEL_FILE_PATH, "w") as f:
    for root, dirs, files in os.walk(NEU_DET_DIR_TRAIN):
        for dirname in sorted(dirs):
            print(dirname)
