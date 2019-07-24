import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

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
init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list=g_vars)

new_samples = []
with tf.Session() as sess:
    
    saver.restore(sess,'./Save/GAN.ckpt')
    
    for x in range(5):
        sample_z = np.random.uniform(-1,1,size=(1,100))
        gen_sample = sess.run(generator(z,reuse=True),feed_dict={z:sample_z})
        
        new_samples.append(gen_sample)

plt.imshow(new_samples[0].reshape(64,64),cmap='Greys')
plt.show()
print(new_samples[0])




