from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import math
import numpy as np

from six.moves import xrange
import tensorflow as tf

import shelve

s = shelve.open('urban.db')
t_features = s['tr_features']
t_labels = s['tr_labels']
ts_features = s['ts_features']
ts_labels = s['ts_labels']
indices = np.random.rand(len(t_labels)) < 0.2
tr_features = t_features[indices]
tr_labels = t_labels[indices]

FRAMES = 41
BANDS = 60
N_CLASSES = 10
KERNEL_SIZE = 30
INPUT_UNITS = 2
FC_HEIGHT = 4
FC_WIDTH = 3
BLANK = -1
#depth = 20
batch_size = 100
KEEP_PROB = 0.5
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate')
flags.DEFINE_integer('max_steps', 2000, 'Number of training steps')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden1')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden2')
flags.DEFINE_integer('hidden3', 1024, 'Number of units in hidden3')
flags.DEFINE_float('stddev', 0.1, 'stddev')
flags.DEFINE_string('train_dir', 'data', 'The path of storing the model')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial, name='biases')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

#def apply_convolution(x, kernel_size, n_channels, depth):
#    weights = weight_variable([kernel_size, kernel_size, n_channels, depth])
#    biases = bias_variable([depth])
#    return tf.nn.relu(tf.add(conv2d(x, weights), biases))

def apply_max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



def placeholder_inputs():
    sounds_placeholder = tf.placeholder(tf.float32, [None, BANDS, FRAMES, INPUT_UNITS])
    labels_placeholder = tf.placeholder(tf.float32, [None, N_CLASSES])
    return sounds_placeholder, labels_placeholder

#def get_cov(sounds):
#    cov = apply_convolution(sounds, kernel_size, n_channels, depth)
#    shape = cov.get_shape().as_list()
#    cov_flat = tf.reshape(cov, [-1, shape[1]*shape[2]*shape[3]])
#    return cov_flat, shape


def inference(sounds,  hidden1_units, hidden2_units, hidden3_units):
    with tf.name_scope('hidden1'):
        weights = weight_variable([KERNEL_SIZE, KERNEL_SIZE, INPUT_UNITS, hidden1_units])
        biases = bias_variable([hidden1_units])
        conv1 = tf.nn.relu(conv2d(sounds, weights) + biases)
        hidden1 = apply_max_pool(conv1)

    with tf.name_scope('hidden2'):
        weights = weight_variable([KERNEL_SIZE, KERNEL_SIZE, hidden1_units, hidden2_units])
        biases = bias_variable([hidden2_units])
        conv2 = tf.nn.relu(conv2d(hidden1, weights) + biases)
        hidden2 = apply_max_pool(conv2)

    with tf.name_scope('hidden3'):
        weights = weight_variable([FC_HEIGHT*FC_WIDTH*hidden2_units, hidden3_units])
        biases = bias_variable([hidden3_units])
        sounds_flat = tf.reshape(hidden2, [-1, FC_HEIGHT*FC_WIDTH*hidden2_units])
        hidden3 = tf.nn.relu(tf.matmul(sounds_flat, weights) + biases)


    with tf.name_scope('softmax'):
        drop = tf.nn.dropout(hidden3, KEEP_PROB)
        weights = weight_variable([hidden3_units, N_CLASSES])
        biases = bias_variable([N_CLASSES])
        logits = tf.nn.softmax(tf.matmul(hidden3, weights) + biases)

    return logits

def cost(logits, labels):
    cost = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)), reduction_indices=[1]), name='loss')
    return cost

def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy



def run_training():
    with tf.Graph().as_default():
        sounds_placeholder, labels_placeholder = placeholder_inputs()
#        cov, shape = get_cov(sounds_placeholder)
        logits = inference(sounds_placeholder, FLAGS.hidden1, FLAGS.hidden2, FLAGS.hidden3)
        loss = cost(logits, labels_placeholder)
        train_op = training(loss, FLAGS.learning_rate)
        accuracy = evaluation(logits, labels_placeholder)
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        sess.run(init)
        for step in xrange(FLAGS.max_steps + 1):
            start_time = time.time()
#            offset = (step * batch_size) % (tr_features.shape[0] - batch_size)
#            tr_batch_features = tr_features[offset:(offset + batch_size), :, :, :]
#            tr_batch_labels = tr_labels[offset:(offset + batch_size), :]
            _, loss_value = sess.run([train_op, loss], feed_dict={sounds_placeholder:tr_features, labels_placeholder:tr_labels})
            duration = time.time() - start_time

            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' %(step,loss_value, duration))
                summary_str = sess.run(summary, feed_dict={sounds_placeholder:tr_features, labels_placeholder:tr_labels})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if step % 1000 == 0:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                eval_accuracy = sess.run(accuracy, feed_dict={sounds_placeholder:ts_features, labels_placeholder:ts_labels})
                print('Validation Data Eval:', eval_accuracy)


def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()







