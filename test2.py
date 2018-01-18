import numpy as np
import tensorflow as tf


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


log_dir = 'tf_CNN_display_log'

data = np.random.randint(0, 10, size=[2, 3, 2])
print(data)
a = tf.constant([1, -2, 56])
w2 = init_weights([3, 3, 32, 64])
b = tf.nn.relu(w2)
# tf.summary.scalar('my_input_a', a)
# variable_summaries(w2)
variable_summaries(b)
# tf.summary.scalar('relu_output', b)
merged = tf.summary.merge_all()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    # train_writer.add_summary(sess.run(merged))
    train_writer.close()
    # print(sess.run(b))
