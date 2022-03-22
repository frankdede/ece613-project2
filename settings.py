import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NEU_DET_DIR_TRAIN = os.path.join(BASE_DIR, "NEU-DET/train")
NEU_DET_DIR_TEST = os.path.join(BASE_DIR, "NEU-DET/validation")

DATASET_META_DIR = os.path.join(BASE_DIR, "metadata")

TRAINING_LABEL_FILE_PATH = os.path.join(DATASET_META_DIR, "training_labels.jsonl")
TEST_LABEL_FILE_PATH = os.path.join(DATASET_META_DIR, "test_labels.jsonl")
