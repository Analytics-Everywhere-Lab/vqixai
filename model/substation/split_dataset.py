import cv2
import os
import json
from sklearn.model_selection import train_test_split

DATA_DIR = "../../data/substation/ds"
ANN_DIR = f"{DATA_DIR}/ann"
IMG_DIR = f"{DATA_DIR}/img"
TRAIN_DIR = f"../../data/substation/train"
VALIDATION_DIR = f"../../data/substation/validation"
TEST_DIR = f"../../data/substation/test"


def load_data():
    data = []
    for ann_file in os.listdir(ANN_DIR):
        ann_path = f"{ANN_DIR}/{ann_file}"
        img_path = f"{IMG_DIR}/{ann_file.replace('.json', '')}"
        with open(f"{ANN_DIR}/{ann_file}", "r") as f:
            ann = json.load(f)
        img = cv2.imread(f"{IMG_DIR}/{ann_file.replace('.json', '')}")
        data.append((img, ann, img_path, ann_path))
    return data


# Save the train_data and test_data to the train and test folder
def save_data(data, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for img, ann, img_path, ann_path in data:
        img_name = img_path.split("/")[-1]
        ann_name = ann_path.split("/")[-1]
        cv2.imwrite(f"{dir}/img/{img_name}", img)
        with open(f"{dir}/ann/{ann_name}", "w") as f:
            json.dump(ann, f)


def save_log(data, dir):
    # Save the name of the train and test data to log file
    with open(f"{dir}/log.txt", "w") as f:
        for img, ann, img_path, ann_path in data:
            img_name = img_path.split("/")[-1]
            ann_name = ann_path.split("/")[-1]
            f.write(f"{img_name} {ann_name}\n")


if __name__ == "__main__":
    data = load_data()
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    save_data(train_data, TRAIN_DIR)
    save_data(test_data, TEST_DIR)
    save_log(train_data, TRAIN_DIR)
    save_log(test_data, TEST_DIR)
