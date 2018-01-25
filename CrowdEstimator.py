import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
from numpy import array
import numpy as np
import argparse
import tensorflow as tf
import os

IMAGE_HEIGHT = int()
LR = float()
FS = int()

parser = argparse.ArgumentParser()
parser.add_argument("-t", help="run as training mode", action='store_true')
parser.add_argument("-p", help="run as predicting mode", action='store_true')
parser.add_argument("--image_height", help="height of image", type=int, required=True)
parser.add_argument("--lr", help="learning rate", type=float, required=True)
parser.add_argument("--fs", help="filter size", type=int ,required=True)
args = parser.parse_args()

if args.t and args.p:
    print('-t and -p cannot be run together.')
    exit()
if args.image_height:
    IMAGE_HEIGHT = args.image_height
if args.lr:
    LR = args.lr
if args.fs:
    FS = args.fs

training_data_path = os.path.join('Dataset', str(IMAGE_HEIGHT) + '_training_data.npy')
testing_data_path = os.path.join('Dataset', str(IMAGE_HEIGHT) + '_testing_data.npy')
model_output_dir = 'Model'

if not os.path.exists(training_data_path):
    print('Training dataset not found.')
    exit()
elif not os.path.exists(testing_data_path):
    print('Testing dataset not found.')
    exit()
elif not os.path.exists(model_output_dir):
    print('Created Models directory to store saved models.')
    os.makedirs(model_output_dir)

# Dataset
train_data = np.load(training_data_path)
test_data = np.load(testing_data_path)

image_width = len(train_data[0][0])

train = train_data[:-5000] # Everything up till the last 5000
test = train_data[-5000:] # Last 5000

X_train = np.array([i[0] for i in train]).reshape(-1, int(IMAGE_HEIGHT), int(image_width), 1)
y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, int(IMAGE_HEIGHT), int(image_width), 1)
y_test = [i[1] for i in test]

# Network
tf.reset_default_graph()

convnet = input_data(shape=[None, int(IMAGE_HEIGHT), int(image_width), 1], name='input')

convnet = conv_2d(convnet, 32, FS, activation='relu')
convnet = max_pool_2d(convnet, FS)

convnet = conv_2d(convnet, 64, FS, activation='relu')
convnet = max_pool_2d(convnet, FS)

convnet = conv_2d(convnet, 128, FS, activation='relu')
convnet = max_pool_2d(convnet, FS)

convnet = conv_2d(convnet, 64, FS, activation='relu')
convnet = max_pool_2d(convnet, FS)

convnet = conv_2d(convnet, 32, FS, activation='relu')
convnet = max_pool_2d(convnet, FS)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 4, activation='softmax') # Because 4 categories
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

model_name = str(IMAGE_HEIGHT) + '_' + str(FS) + '_' + str(LR) + '_crowd_model'

if args.t:
    # Train and save model
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10, validation_set=({'input': X_test}, {'targets': y_test}),
        snapshot_step=500, show_metric=True, run_id=model_name)
    model.save(os.path.join('Model', model_name))

elif args.p:
    model.load(model_name)
    import numpy as np
    for x in range(len(X_test)):
        print(np.round(model.predict([X_test[x]])[0], decimals=3))
        print(y_test[x])
        print()
