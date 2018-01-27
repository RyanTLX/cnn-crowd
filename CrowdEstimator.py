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

tf.logging.set_verbosity(tf.logging.ERROR)

IMAGE_HEIGHT = 64
LR = float()
EPOCH = int()

parser = argparse.ArgumentParser()
parser.add_argument('-t', help='run as training mode', action='store_true')
parser.add_argument('-p', help='run as predicting mode', action='store_true')
parser.add_argument('--lr', help='learning rate', type=float, required=True)
parser.add_argument('--epoch', help='number of epoch', type=int, required=True)
parser.add_argument('--cam', help='camera set', type=str, required=True)
args = parser.parse_args()

if args.t and args.p:
    print('-t and -p cannot be run together.')
    exit()
if not args.t and not args.p:
    print('Must state either -t or -p to run in train or predict mode.')
    exit()
if args.lr:
    LR = args.lr
if args.epoch:
    EPOCH = args.epoch
if args.cam:
    CAM = args.cam

training_data_path = os.path.join('Dataset', '64_{}_train_dataset.npy'.format(CAM))
validate_data_path = os.path.join('Dataset', '64_{}_validate_dataset.npy'.format(CAM))
testing_data_path = os.path.join('Dataset', '64_{}_test_dataset.npy'.format(CAM))
model_output_dir = 'Model'

if not os.path.exists(training_data_path):
    print('Train dataset not found.')
    exit()
if not os.path.exists(validate_data_path):
    print('Validate dataset not found.')
    exit()
if not os.path.exists(testing_data_path):
    print('Test dataset not found.')
    exit()
if not os.path.exists(model_output_dir):
    print('Created Models directory to store saved models.')
    os.makedirs(model_output_dir)

# Dataset
train_data = np.load(training_data_path)
validate_data = np.load(validate_data_path)
test_data = np.load(testing_data_path)

image_width = len(train_data[0][0])

X_train = np.array([i[0] for i in train_data]).reshape(-1, int(IMAGE_HEIGHT), int(image_width), 1)
y_train = [i[1] for i in train_data]

X_val = np.array([i[0] for i in validate_data]).reshape(-1, int(IMAGE_HEIGHT), int(image_width), 1)
y_val = [i[1] for i in validate_data]

X_test = np.array([i[0] for i in test_data]).reshape(-1, int(IMAGE_HEIGHT), int(image_width), 1)
y_test = [i[1] for i in test_data]
filename_test = [i[2] for i in test_data]
labels = ['empty', 'low', 'medium', 'high']

# Network
tf.reset_default_graph()

convnet = input_data(shape=[None, int(IMAGE_HEIGHT), int(image_width), 1], name='input')
convnet = max_pool_2d(convnet, 2) # Makes the network run faster. Get most interesting parts first?
convnet = conv_2d(convnet, 16, 7, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 32, 7, activation='relu')
convnet = conv_2d(convnet, 16, 5, activation='relu')
convnet = fully_connected(convnet, 64, activation='relu')
convnet = dropout(convnet, 0.5)
convnet = fully_connected(convnet, 4, activation='softmax') # Because 4 categories
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

model_name = str(IMAGE_HEIGHT) + '_' + str(LR) + '_' + CAM + '_crowd_model'
model_path = os.path.join('Model', model_name)

if args.t:
    # Train and save model
    starttime = datetime.now()
    print('\'{}\' started training at {}'.format(model_name, starttime))

    model.fit(
        {'input': X_train}, {'targets': y_train},
        n_epoch=EPOCH,
        validation_set=({'input': X_val}, {'targets': y_val}),
        show_metric=True,
        run_id=model_name)
    model.save(model_path)

    endtime = datetime.now()
    print('\'{}\' finished training at {}'.format(model_name, endtime))
    print('Training \'{}\' took {} seconds to complete.'.format(model_name, (endtime-starttime).seconds))

elif args.p:
    confusion_matrix_prediction = {'empty':0, 'low':0, 'medium':0, 'high':0}
    confusion_matrix_truth = {'empty':confusion_matrix_prediction.copy(),
                              'low':confusion_matrix_prediction.copy(),
                              'medium':confusion_matrix_prediction.copy(),
                              'high':confusion_matrix_prediction.copy()}

    model.load(model_path)

    accuracy = float()

    iteration = 0
    for x in tqdm(range(len(X_test)), total=len(X_test), unit='predictions'):
        results = np.round(model.predict([X_test[x]]), decimals=3)
        results = np.squeeze(results)
        top_k = results.argsort()[::-1] #[::-1] means reverse the order. Desc in this case.

        # for i in top_k:
            # print(labels[i], results[i])

        # print(filename_test[x])
        # print()

        truth = filename_test[x].split('_')[0]
        prediction = labels[top_k[0]]
        confusion_matrix_truth[truth][prediction] += 1


        if prediction == truth:
            accuracy += 1

    accuracy = (accuracy / len(X_test)) * 100

    print('Empty \t\t', 'Empty:', confusion_matrix_truth['empty']['empty'], '\t',
                        'Low:', confusion_matrix_truth['empty']['low'], '\t',
                        'Medium:', confusion_matrix_truth['empty']['medium'], '\t',
                        'High:', confusion_matrix_truth['empty']['high']
        )
    print('Low \t\t', 'Empty:', confusion_matrix_truth['low']['empty'], '\t',
                        'Low:', confusion_matrix_truth['low']['low'], '\t',
                        'Medium:', confusion_matrix_truth['low']['medium'], '\t',
                        'High:', confusion_matrix_truth['low']['high']
        )
    print('Medium \t\t', 'Empty:', confusion_matrix_truth['medium']['empty'], '\t',
                        'Low:', confusion_matrix_truth['medium']['low'], '\t',
                        'Medium:', confusion_matrix_truth['medium']['medium'], '\t',
                        'High:', confusion_matrix_truth['medium']['high']
        )
    print('High \t\t', 'Empty:', confusion_matrix_truth['high']['empty'], '\t',
                        'Low:', confusion_matrix_truth['high']['low'], '\t',
                        'Medium:', confusion_matrix_truth['high']['medium'], '\t',
                        'High:', confusion_matrix_truth['high']['high']
        )
    print('Accuracy:', accuracy)
