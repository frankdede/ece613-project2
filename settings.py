import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DAGM_DIR = os.path.join(BASE_DIR, "DAGM")

DATASET_META_DIR = os.path.join(BASE_DIR, "metadata")

MODELS_DIR = os.path.join(BASE_DIR, "models")

TRAINING_LABEL_FILE_PATH = os.path.join(DATASET_META_DIR, "training_set.csv")
TEST_LABEL_FILE_PATH = os.path.join(DATASET_META_DIR, "test_set.csv")

MER_OUTPUT = os.path.join(BASE_DIR, "MER_OUTPUT")

CNN_DICT = os.path.join(MODELS_DIR,  "cnn.pt")

