# coding = utf-8

import tensorflow as tf

## save to file

## define dtype and shape is important for saving
#
# W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name="weights")
# b = tf.Variable([[1,2,3]], dtype=tf.float32, name="biases")
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, "my_net/save_net.ckpt")  ## ckpt means checkpoint
#     print("saving success")
#

## restore variables

## redefine the same shape and same type for variables
import numpy as np

W = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name="biases")

# not need init step
saver = tf.train.Saver()

with tf.Session() as sess:
    # print("weights:", sess.run(W))
    # print("biases:", sess.run(b))
    saver.restore(sess, "my_net/save_net.ckpt")
    print("restore success")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))