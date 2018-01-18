import tensorflow as tf
import numpy as np

with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.zeros([1, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, [None, 1])
    cost = tf.reduce_sum(tf.pow((y_ - y), 2))
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)
init = tf.initialize_all_variables()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

for i in range(100):
    # xs = np.array([[i]])
    xs = np.array([[20]])
    # ys = np.array([[3 * i]])
    ys = np.array([[40]])
    print(xs)
    feed = {x: xs, y_: ys}
    # 训练会通过上次的结果进行调整，可以使用同样的数据进行多次训练，得到符合预期的结果
    # 例如用同样的数据进行400次训练，得到了预期的结果，100次训练与预期相差甚多。如果是不同的数据，100次就可以得到预期结果（改变注释使用不同的数据）
    sess.run(train_step, feed_dict=feed)
    print("After%diteration:" % i)
    print("W:%f" % sess.run(W))
    print("b:%f" % sess.run(b))

# 这次测试中，梯度下降采用的是每次送入一个值
# 可以有多种梯度下降：1.stochastic gradient descent（每次一个值,随机梯度下降）2.mini-batch(送入一堆数据) 3.batch(送入所有数据)
