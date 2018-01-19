# 贡献者：{沙舟}
# 源代码出处：在本地文件  /tensorflow-master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py


# 训练过程的可视化 ，TensorBoard的应用
# 运行 python mnist_with_summaries.py
# 导入模块并下载数据集
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 设置超参数
max_step = 1000
learning_rate = 0.001
dropout = 0.9

# 用logdir明确标明日志文件储存路径
# 训练过程中的数据储存在/tem/tensorflow/mnist 目录中，通过这个路径指定--log_dir

data_dir = '/tmp/tensorflow/mnist/input_data'
log_dir = '/tmp/tensorflow/mnis/logs/mnist_with_summaries'
mnist = input_data.read_data_sets(data_dir, one_hot=True)
sess = tf.InteractiveSession()

# 本句的含义是使图可视化，sess.graph是对图的定义
# 使用以上指定的路径创建摘要的文件写入符(FileWrite)
file_write = tf.summary.FileWriter('/tmp/tensorflow/mnis/logs/', sess.graph)


def variable_summaries(var, name):
    """对每一个张量添加多个摘要描述"""
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)  # 均值
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        """为确保计算图中各个层的分组，给每一层添加一个name_scope"""
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)  # 激活前的直方图
        activations = act(preactivate, name='activation')

        # 记录神经网络节点输出在经过激活函数之后的分布。
        tf.summary.histogram(layer_name + '/activations', activations)  # 激活后的直方图
        return activations

# 运行tensorboard命令，打开浏览器,查看模型训练过程中的可视化结果，
# 在终端输入下命令：
# tensorboard --logdir=/tmp/tensorflow/mnist/logs/mnist_with_summaries
