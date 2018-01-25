import cv2
import numpy as np
import os
import argparse
from random import shuffle
from tqdm import tqdm


TRAIN_DIR = 'Dataset/camall/train'
TEST_DIR = 'Dataset/camall/test_grouped'
IMAGE_HEIGHT = 32 # 32px height as default.


def create_label(image_name):
    # Create an one-hot encoded vector from image name
    word_label = image_name.split('_')[0]
    if word_label == 'empty':
        return np.array([1,0,0,0])
    elif word_label == 'low':
        return np.array([0,1,0,0])
    elif word_label == 'medium':
        return np.array([0,0,1,0])
    elif word_label == 'high':
        return np.array([0,0,0,1])


def create_training_dataset(to_height):
    training_data = []
    for dirpath, dirnames, filenames in os.walk(TRAIN_DIR):
        # Exclude hidden from list.
        files = [f for f in filenames if not f.startswith('.')]

        # Skip current loop of files is empty.
        if not files:
            continue

        for img in tqdm(files):
            path = os.path.join(dirpath, img)
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            new_width = to_height * int(img_data.shape[1] / img_data.shape[0])
            img_data = cv2.resize(img_data, (to_height, new_width))
            training_data.append([np.array(img_data), create_label(img)])

    shuffle(training_data)
    file_name = str(to_height) + '_training_data.npy'
    save_path = os.path.join('Dataset', file_name)
    np.save(save_path, training_data)


def create_testing_dataset(to_height):
    testing_data = []
    for dirpath, dirnames, filenames in os.walk(TEST_DIR):
        # Exclude hidden from list.
        files = [f for f in filenames if not f.startswith('.')]

        # Skip current loop of files is empty.
        if not files:
            continue

        for img in tqdm(files):
            path = os.path.join(dirpath,img)
            img_num = img.split('.')[0]
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            new_width = to_height * int(img_data.shape[1] / img_data.shape[0])
            img_data = cv2.resize(img_data, (to_height, new_width))
            testing_data.append([np.array(img_data), img_num])

    shuffle(testing_data)
    file_name = str(to_height) + '_testing_data.npy'
    save_path = os.path.join('Dataset', file_name)
    np.save(save_path, testing_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_height", help="height of image", type=int, required=True)
    args = parser.parse_args()

    if args.image_height:
        IMAGE_HEIGHT = args.image_height

    create_training_dataset(IMAGE_HEIGHT)
    create_testing_dataset(IMAGE_HEIGHT)
