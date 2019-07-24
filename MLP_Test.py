import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow.python.lib.io import file_io
from pandas.compat import StringIO

print("Importing Data\n")
Testdata = pd.read_csv('./Test128.csv',header=None)

xt = Testdata.iloc[:,:].values

n_inputs = 4096
nh1 = 3223
nh2 = 3223
nh3 = 3223
nh4 = 3223
nh5 = 3223
n_outputs = 2350

weights = {
    'w1' : tf.Variable(tf.random_normal([n_inputs, nh1])),
    'w2' : tf.Variable(tf.random_normal([nh1, nh2])),
    'w3' : tf.Variable(tf.random_normal([nh2, nh3])),
    'w4' : tf.Variable(tf.random_normal([nh3, nh4])),
    'w5' : tf.Variable(tf.random_normal([nh4, nh5])),
    'out_w' : tf.Variable(tf.random_normal([nh5, n_outputs]))
}

biases = {
    'b1' : tf.Variable(tf.random_normal([nh1])),
    'b2' : tf.Variable(tf.random_normal([nh2])),
    'b3' : tf.Variable(tf.random_normal([nh3])),
    'b4' : tf.Variable(tf.random_normal([nh4])),
    'b5' : tf.Variable(tf.random_normal([nh5])),
    'out_b' : tf.Variable(tf.random_normal([n_outputs]))
}

init = tf.global_variables_initializer()

print("importing the Trained network")

with tf.Session() as sess:
    
    sess.run(init)
    
    saver = tf.train.Saver()
    saver.restore(sess,'./Save/ANN.ckpt')

    W = sess.run(weights)
    B = sess.run(biases)

def neural_network(x, weights, biases):
    layer1 = np.matmul(x, weights['w1']) + biases['b1']
    layer2 = np.matmul(layer1, weights['w2']) + biases['b2']
    layer3 = np.matmul(layer2, weights['w3']) + biases['b3']
    layer4 = np.matmul(layer3, weights['w4']) + biases['b4']
    layer5 = np.matmul(layer4, weights['w5']) + biases['b5']
    layer_out = np.matmul(layer5, weights['out_w']) + biases['out_b']
    return layer_out

def get_predictions(x, w, b):
    pred = neural_network(x, w, b)
    images, predictions = [], []
    for i in x:
        images.append(i.reshape(64, 64))
    for i in pred:
        predictions.append(list(i))
    predictions = [(int(i.index(max(i)))+1) for i in predictions]
    return (images, predictions)

images2, preds2 = get_predictions(xt, W, B)

with file_io.FileIO('./preds128.pkl', mode='w+') as f:
       pickle.dump(preds2, f)


