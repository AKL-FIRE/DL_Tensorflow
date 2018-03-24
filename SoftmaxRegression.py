import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
image = mnist.train.images[0,:]
reshaped = tf.reshape(image, [28, 28])
sess = tf.Session()
reshaped = sess.run(reshaped)
sess.close()
plt.figure('show_image')
plt.imshow(reshaped)
plt.show()

x = tf.placeholder(tf.float32, [None, 784]) # None is the number of examples , 784 is the number of features
y_ = tf.placeholder(tf.float32, [None, 10]) # Labels

w = tf.Variable(tf.truncated_normal([784, 10])) # Define the net weights
b = tf.Variable(tf.constant([0.0], shape=[10])) # Define the net biases

y = tf.nn.softmax(tf.matmul(x, w) + b) # compute the net's output

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy) # train_step

# Calculate the accuracy
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
        if i % 1000 == 0:
            print('After %d iterations, the accuracy is %f' % (i, sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})))