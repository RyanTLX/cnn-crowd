import cv2
import numpy as np
import os
import argparse
from random import shuffle
from tqdm import tqdm


TRAIN_DIR = 'Dataset/train/camall'
VAL_DIR = 'Dataset/validate/camall'
TEST_DIR = 'Dataset/test/camall'
IMAGE_HEIGHT = int()


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


def create_training_dataset(to_height, set_type):
    data = []

    path = str()
    if set_type == 'train':
        path = TRAIN_DIR
    elif set_type == 'validate':
        path = VAL_DIR
    elif set_type == 'test':
        path = TEST_DIR

    for dirpath, dirnames, filenames in os.walk(path):
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
            data.append([np.array(img_data), create_label(img)])

    shuffle(data)
    file_name = str(to_height) + '_' + set_type + '_all_data.npy'
    save_path = os.path.join('Dataset', file_name)
    np.save(save_path, data)

    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_height", help="height of image", type=int, required=True)
    args = parser.parse_args()

    if args.image_height:
        IMAGE_HEIGHT = args.image_height

    print('Creating train dataset: ')
    print('Train data set saved at: ' + create_training_dataset(IMAGE_HEIGHT, 'train'))

    print('Creating validate dataset: ')
    print('validate data set saved at: ' + create_training_dataset(IMAGE_HEIGHT, 'validate'))

    print('Creating test dataset: ')
    print('Test data set saved at: ' + create_training_dataset(IMAGE_HEIGHT, 'test'))
