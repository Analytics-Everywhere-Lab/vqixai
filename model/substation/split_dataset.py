import cv2
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DATA_DIR = "../../data/substation/ds"
ANN_DIR = f"{DATA_DIR}/ann"
IMG_DIR = f"{DATA_DIR}/img"
TRAIN_DIR = f"../../data/substation/train"
VALIDATION_DIR = f"../../data/substation/validation"
TEST_DIR = f"../../data/substation/test"


def load_data():
    data = []
    for ann_file in tqdm(os.listdir(ANN_DIR), desc="Loading data"):
        ann_path = f"{ANN_DIR}/{ann_file}"
        img_path = f"{IMG_DIR}/{ann_file.replace('.json', '')}"
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Image not found or unable to read: {img_path}")
            continue

        with open(ann_path, "r") as f:
            ann = json.load(f)

        data.append((img, ann, img_path, ann_path))
    return data


# Save the train_data and test_data to the train and test folder
def save_data(data, dir):
    img_dir = f"{dir}/img"
    ann_dir = f"{dir}/ann"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    for img, ann, img_path, ann_path in tqdm(data, desc=f"Saving data to {dir}"):
        img_name = os.path.basename(img_path)
        ann_name = os.path.basename(ann_path)
        cv2.imwrite(f"{img_dir}/{img_name}", img)
        with open(f"{ann_dir}/{ann_name}", "w") as f:
            json.dump(ann, f)


def save_log(data, dir):
    # Save the name of the train and test data to log file
    with open(f"{dir}/log.txt", "w") as f:
        for img, ann, img_path, ann_path in data:
            img_name = os.path.basename(img_path)
            ann_name = os.path.basename(ann_path)
            f.write(f"{img_name} {ann_name}\n")


if __name__ == "__main__":
    data = load_data()
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    save_data(train_data, TRAIN_DIR)
    save_data(test_data, TEST_DIR)
    save_log(train_data, TRAIN_DIR)
    save_log(test_data, TEST_DIR)
