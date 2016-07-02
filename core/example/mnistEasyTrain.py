#####
# The program is for training the mnist digital recognition(The Easy version)
# Version: 1.1(add annotation)
# Author: Whytin   # Mail: 583501947@qq.com
# Refer to: https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/tutorials/mnist_pros.md
#####

#import tensorflow and mnist data sets.
import tensorflow as tf
import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
#set up a interactive session to run the training model, you can insert other opreation graphs when the graph running.
#x represent any expand to 784 pixels pictures.
#y_ represent any expand to 10 dimensions labels.
#W reprensent every pixels occupy howmany weight in pictures,initialize zerovector
#b reprent every pictures bias to correct deviation,here also initialize zero vector
sess = tf.InteractiveSession()
x = tf.placeholder('float',shape=[None,784])
y_ = tf.placeholder('float',shape=[None,10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#you should initialize all variables before you use them.
sess.run(tf.initialize_all_variables())
#set up Softmax regression model to caculate the probability of digital recognition. here tf.nn.softmax is for selecting a most possibly figure.
y = tf.nn.softmax(tf.matmul(x,W)+b)
#caculate the cross entropy to prove the low efficiency of training.
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#gradient descent optimizer of cross entropy for training model.
#exactly execute built-in tensorflow of back propagation, probabily change the weight.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#circulate training model to abtain the better recognitial model
for i in range(2000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0],y_:batch[1]})
#evaluate the training model.
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
print accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})

