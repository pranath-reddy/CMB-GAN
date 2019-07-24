import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import pickle
from tensorflow.python.lib.io import file_io
import datetime

if not os.path.exists("./Save"):
    os.makedirs("./Save")

if not os.path.exists("./Plots"):
    os.makedirs("./Plots")

with h5py.File('Data.h5', 'r') as hf:
    x = hf['data'][:]

num_images = 60000

def next_batch(batch_size, data):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[: batch_size]
    data_shuffle = [data[i] for i in idx]
    return np.asarray(data_shuffle)

def generator(z,reuse=None):
    with tf.variable_scope('gen',reuse=reuse):

        hidden1 = tf.layers.dense(inputs=z,units=256,activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(inputs=hidden1,units=256,activation=tf.nn.leaky_relu)
        hidden3 = tf.layers.dense(inputs=hidden2,units=256,activation=tf.nn.leaky_relu)
        hidden4 = tf.layers.dense(inputs=hidden3,units=256,activation=tf.nn.leaky_relu)

        output = tf.layers.dense(hidden4,units=4096,activation=tf.nn.tanh)
        return output

def discriminator(X,reuse=None):
    with tf.variable_scope('dis',reuse=reuse):


        hidden1 = tf.layers.dense(inputs=X,units=256,activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(inputs=hidden1,units=256,activation=tf.nn.leaky_relu)
        hidden3 = tf.layers.dense(inputs=hidden2,units=256,activation=tf.nn.leaky_relu)
        hidden4 = tf.layers.dense(inputs=hidden3,units=256,activation=tf.nn.leaky_relu)
        
        logits = tf.layers.dense(hidden4,units=1)
        output = tf.sigmoid(logits)
    
        return output, logits

real_images = tf.placeholder(tf.float32,shape=[None,4096])
z = tf.placeholder(tf.float32,shape=[None,100])

G = generator(z)
D_output_real , D_logits_real = discriminator(real_images)
D_output_fake, D_logits_fake = discriminator(G,reuse=True)

def loss_func(logits_in,labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))

D_real_loss = loss_func(D_logits_real,tf.ones_like(D_logits_real)* (0.9))
D_fake_loss = loss_func(D_logits_fake,tf.zeros_like(D_logits_real))
D_loss = D_real_loss + D_fake_loss
G_loss = loss_func(D_logits_fake,tf.ones_like(D_logits_fake))

learning_rate = 0.001
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

print([v.name for v in d_vars])
print([v.name for v in g_vars])

D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_vars)

batch_size = 100
epochs = 50000
save_step = 100
init = tf.global_variables_initializer()
saver = tf.train.Saver()

samples = []
with tf.Session() as sess:

    loss_hist1, loss_hist2 = [], []

    saver.restore(sess,'./Save/GAN.ckpt')
    
    sess.run(init)
    
    # Recall an epoch is an entire run through the training data
    for e in range(epochs):
        # // indicates classic division
        num_batches = num_images // batch_size
        
        for i in range(num_batches):
            
            # Grab batch of images
            batch = next_batch(batch_size, x)
            
            # Get images, reshape and rescale to pass to D
            batch_images = batch[0].reshape((batch_size, 4096))
            batch_images = batch_images*2 - 1
            
            # Z (random latent noise data for Generator)
            # -1 to 1 because of tanh activation
            batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))
            
            # Run optimizers, no need to save outputs, we won't use them
            _ = sess.run(D_trainer, feed_dict={real_images: batch_images, z: batch_z})
            _ = sess.run(G_trainer, feed_dict={z: batch_z})

        l1 = sess.run(D_loss, feed_dict={real_images: batch_images, z: batch_z})
        l2 = sess.run(G_loss, feed_dict={z: batch_z})
        loss_hist1.append(l1)
        loss_hist2.append(l2)
        
            
        print("Currently on Epoch {} of {} total...".format(e+1, epochs))
        print(datetime.datetime.now().time())
        
        # Sample from generator as we're training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size=(1, 100))
        gen_sample = sess.run(generator(z ,reuse=True),feed_dict={z: sample_z})
        
        samples.append(gen_sample)

        if e % save_step == 0:
            saver.save(sess,'./Save/GAN.ckpt')
            print("Model Saved")

            with file_io.FileIO('./Plots/disc_loss.pkl', mode='w+') as f:
                pickle.dump(loss_hist1, f)

            with file_io.FileIO('./Plots/gen_loss.pkl', mode='w+') as f:
                pickle.dump(loss_hist2, f)







