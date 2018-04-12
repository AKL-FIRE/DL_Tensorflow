import tensorflow as tf

batch_size = 32
num_batches = 100

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def inference(images):
    parameters = []

    with tf.name_scope('conv1') as scope:
        kernal = tf.Variable(tf.truncated_normal([11, 11, 3, 64], tf.float32, 0.1), name='weights')
        conv = tf.nn.conv2d(images, kernal, strides=[1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernal, biases]
        #lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernal = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(pool1, kernal, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernal, biases]
        print_activations(conv2)
        #lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernal = tf.Variable(tf.truncated_normal([3, 3, 192, 384], tf.float32, 0.1), name='weights')
        conv = tf.nn.conv2d(pool2, kernal, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        print_activations(conv3)
        parameters += [kernal, biases]

    with tf.name_scope('conv4') as scope:
        kernal = tf.Variable(tf.truncated_normal([3, 3, 384, 256], tf.float32, 0.1), name='weights')
        conv = tf.nn.conv2d(conv3, kernal, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        print_activations(conv4)
        parameters += [kernal, biases]

    with tf.name_scope('conv5') as scope:
        kernal = tf.Variable(tf.truncated_normal([3, 3, 256, 256], tf.float32, 0.1), name='weights')
        conv = tf.nn.conv2d(conv4, kernal, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        print_activations(conv5)
        parameters += [kernal, biases]
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        print_activations(pool5)

    reshape = tf.reshape(pool5, [batch_size, -1])
    dim = reshape.shape[1]

    with tf.name_scope('fc1') as scope:
        weights = tf.Variable(tf.truncated_normal([dim, 4096], stddev=0.1, dtype=tf.float32), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), name='biases')
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope)

    with tf.name_scope('fc2') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1, dtype=tf.float32), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), name='biases')
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope)

    with tf.name_scope('fc3') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, 1000], stddev=0.1, dtype=tf.float32), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[1000], dtype=tf.float32), name='biases')
        logits = tf.nn.softmax(tf.matmul(fc2, weights) + biases, name=scope)

    return logits, parameters