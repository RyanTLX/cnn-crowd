from django.shortcuts import render
from django.http import JsonResponse
# import os
# import random


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from numpy import array
import numpy as np
import argparse
import tensorflow as tf
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import random

# Default arguments.
image_height = 64
LR = 0.001
EPOCH = 10
CAM = 'camall'

image_width = None
X_train = None
y_train = None
X_val = None
y_val = None
X_test = None
y_test = None
filename_test = None
labels = None

model = None
model_name = None
model_path = None

def load_dataset():
    global image_width, X_train, y_train, X_val, y_val, X_test, y_test, filename_test, labels

    # Dataset paths.
    training_data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Dataset', '64_{}_train_dataset.npy'.format(CAM)))
    validate_data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Dataset', '64_{}_validate_dataset.npy'.format(CAM)))
    testing_data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Dataset', '64_{}_test_dataset.npy'.format(CAM)))

    # Check if datasets exists
    if not os.path.exists(training_data_path):
        print('Train dataset not found.')
        exit()
    if not os.path.exists(validate_data_path):
        print('Validate dataset not found.')
        exit()
    if not os.path.exists(testing_data_path):
        print('Test dataset not found.')
        exit()

    # Load datasets from paths.
    train_data = np.load(training_data_path)
    validate_data = np.load(validate_data_path)
    test_data = np.load(testing_data_path)

    image_width = len(train_data[0][0])

    X_train = np.array([i[0] for i in train_data]).reshape(-1, int(image_height), int(image_width), 1)
    y_train = [i[1] for i in train_data]

    X_val = np.array([i[0] for i in validate_data]).reshape(-1, int(image_height), int(image_width), 1)
    y_val = [i[1] for i in validate_data]

    X_test = np.array([i[0] for i in test_data]).reshape(-1, int(image_height), int(image_width), 1)
    y_test = [i[1] for i in test_data]
    filename_test = [i[2] for i in test_data]
    labels = ['empty', 'low', 'medium', 'high']


def predict_batch():
    images = []
    image_names = []

    # Generate 6 random images.
    for i in range(6):
        random_image_no = random.randint(0, len(X_test)-1)
        image_data = X_test[random_image_no]
        image_name = filename_test[random_image_no]

        images.append(image_data)
        image_names.append(image_name)

    # Prediction and results.
    results = np.round(model.predict(images), decimals=3)

    # Return predictions.
    predictions = []
    i = 0
    for res in results:
        top_k = res.argsort()[::-1]

        prediction = labels[top_k[0]]
        pred_image_name = image_names[i]

        i += 1
        color = "success"
        if prediction == "medium":
            color = "warning"
        elif prediction == "high":
            color = "danger"
        predictions.append({'cabin_image_name': pred_image_name, 'cabin_label': prediction, 'cabin_color': color})
    return predictions


def reload_data(request):
    if type(X_test) is type(None):
        index(request) # To initialise all global variables first. Occurs when starting server while page already running.

    params = predict_batch()

    return JsonResponse({'params': params})

def index(request):
    global LR, EPOCH, CAM, model_name, model_path, model
    # Hide TensorFlow deprecated errors.
    tf.logging.set_verbosity(tf.logging.ERROR)

    load_dataset()

    # Convolutional Neural Network
    tf.reset_default_graph()

    convnet = input_data(shape=[None, int(image_height), int(image_width), 1], name='input')
    convnet = max_pool_2d(convnet, 2) # Makes the network run faster. Get most interesting parts first?
    convnet = conv_2d(convnet, 16, 7, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 32, 7, activation='relu')
    convnet = conv_2d(convnet, 16, 5, activation='relu')
    convnet = fully_connected(convnet, 64, activation='relu')
    convnet = dropout(convnet, 0.5)
    convnet = fully_connected(convnet, 4, activation='softmax') # Because 4 categories
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    # Package network into a model.
    model = tflearn.DNN(convnet)

    # Initialise model name.
    model_name = str(image_height) + '_' + str(LR) + '_' + CAM + '_crowd_model'
    model_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Model', model_name))
    model.load(model_path)

    return render(request, 'CrowdEstimatorWeb/index.html')
