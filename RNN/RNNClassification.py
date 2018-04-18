from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

TIME_STEPS = 28
INPUTS_SIZE = 28
BATCH_SIZE = 50

x = tf.placeholder(tf.float32, [None, TIME_STEPS * INPUTS_SIZE])
image = tf.reshape(x, [-1, TIME_STEPS, INPUTS_SIZE])
y = tf.placeholder(tf.float32, [None, 10])


rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(64)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,
    image,
    dtype=tf.float32,
    initial_state=None,
    time_major=False
)
output = tf.layers.dense(outputs[:, -1, :], 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

accuracy = tf.metrics.accuracy(
    labels=tf.argmax(y, axis=1),
    predictions=tf.argmax(output, axis=1)
)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

for step in range(1000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], feed_dict={x:b_x, y:b_y})
    if step % 50 == 0:
        accuracy_ = sess.run(accuracy, feed_dict={x:test_x, y:test_y})
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)