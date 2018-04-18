import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.001

def get_batch():
    global BATCH_START, TIME_STEPS
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

x = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
y = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE])

rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE)
init_s = rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
outputs, final_s = tf.nn.dynamic_rnn(
    rnn_cell,
    x,
    dtype=tf.float32,
    initial_state=init_s,
    time_major=False
)
outs2d = tf.reshape(outputs, [-1, CELL_SIZE])
net_outs2d = tf.layers.dense(outs2d, INPUT_SIZE)
outs = tf.reshape(net_outs2d, [-1, TIME_STEPS, INPUT_SIZE])

loss = tf.losses.mean_squared_error(labels=y, predictions=outs)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

for step in range(100):
    X_batch, Y_batch, xs = get_batch()
    if 'final_s_' not in globals():
        feed_dict = {x:X_batch, y:Y_batch}
    else:
        feed_dict = {x: X_batch, y: Y_batch, init_s: final_s_}
    _, pred_, final_s_ = sess.run([train_op, outs, final_s], feed_dict)
    plt.plot(xs[0, :], Y_batch[0].flatten(), 'r', xs[0, :], pred_.flatten()[:TIME_STEPS], 'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.1)
