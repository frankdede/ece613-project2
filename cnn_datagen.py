import os.path
import re
import pandas as pd
from settings import *


output_cols = ["img_file", "class", "has_defect", "label_file", "is_training"]
df = pd.DataFrame(columns=output_cols)

class_regex = re.compile(r'(\d*)_mask.PNG')
for root, dirs, files in sorted(os.walk(MER_OUTPUT)):
    if root.endswith("Label") and label_txt_filename in files:
        num_class = int(class_regex.findall(root)[0]) - 1  # convert 1-based class to 0-based
        is_training = 1 if "Train" in root else 0
        print(f"processing label txt for class {num_class} - is_training {is_training}")
        label_txt_path = os.path.join(root, label_txt_filename)
        label_df = pd.read_csv(label_txt_path, delimiter="\\t", skiprows=1,
                               names=["sample_no", "has_defect", "img_file",
                                      "col", "label_file"], header=None, engine='python')
        img_dir = os.path.dirname(root)
        label_df["img_file"] = label_df["img_file"].apply(lambda im: os.path.join(img_dir, im))

        normal_samples = label_df[label_df["has_defect"] == 0]
        defect_samples = label_df[label_df["has_defect"] == 1]
        assert len(defect_samples) + len(normal_samples) == len(label_df), "Total number of samples should match"

        label_df.loc[label_df["has_defect"] == 0, "label_file"] = None
        label_df.loc[label_df["has_defect"] == 1, "label_file"] = defect_samples["label_file"].apply(
            lambda im: os.path.join(root, im))

        label_df["is_training"] = is_training
        label_df["class"] = num_class
        df = pd.concat([df, label_df[output_cols]])

df[df["is_training"] == 1].to_csv(os.path.join(DATASET_META_DIR, "training_set.csv"), index=False)
df[df["is_training"] == 0].to_csv(os.path.join(DATASET_META_DIR, "test_set.csv"), index=False)
