# coding:utf-8
# time: 2018-1-5
# author: LiuZhi
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 超参数，是不和网络相关的参数
batch_size = 128
test_size = 256
log_dir = 'tf_CNN_display_log'


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


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


# 定义CNN网络模型
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding="SAME"))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding="SAME"))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding="SAME"))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


# input data
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)
# set var, X is images Y is labels
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

# 定义模型函数
# 神经网络模型的构建函数，传入以下参数
# X：输入数据
# w: 每一层权重
w = init_weights([3, 3, 1, 32])  # 大小为3*3，输入的维度为1 ，输出维度为32
w2 = init_weights([3, 3, 32, 64])  # 大小为3*3,输入维度为32，输出维度为64
w3 = init_weights([3, 3, 64, 128])  # 大小为3*3,输入维度为64，输出维度为128
w4 = init_weights([128 * 4 * 4, 625])  # 全连接层，输入维度为128*4*4,也就是上一层的输出，输出维度为625
w_o = init_weights([625, 10])  # 输出层，输入的维度为625， 输出10维，代表10类（labels）

# p_keep_conv,p_keep_hidden:dropout 保留神经元比例
# 定义dropout的占位符keep_conv，表示一层中有多少比例的神经元被保留，生成网络模型，得到预测数据
# 在训练的时候把设定比例的节点改为0，避免过拟合
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

# 定义损失函数，采用tf.nn.softmax_cross_entropy_with_logists，作为比较预测值和真实值的差距
# 定义训练操作(train_op) 采用RMSProp算法作为优化器,
# reduce_mean计算给定张量的平均值，可以指定axis，这里没有指定，输出是所有元素和的平均值
# 输入：logits and labels 均为[batch_size, num_classes]
# 输出：loss:[batch_size], 里面保存的是 batch 中每个样本的交叉熵
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)  # 训练cost
predict_op = tf.argmax(py_x, 1)  # 预测操作, argmax 返回py_x这个向量中最大值的索引值


# Launch the graph in a session
with tf.Session() as sess:

    # summaries合并
    merged = tf.summary.merge_all()
    # 写到指定的磁盘路径中
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    # you need to initialize all variabels
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX) + 1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})
        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)  # np.random.shuffle 打乱test_indices的排序
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
        # 预测的时候设置为1 即对全部样本进行迭代训练
