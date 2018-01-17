import tensorflow as tf
import numpy as np

a = [[2, 9], [3, 6]]
# bb = [[[2, 3, 5], [1, 2, 7], [9, 8, 90]]]
b = [[[7, 1, 4, 1], [7, 5, 6], [3, 1, 8]],
     [[4, 1, 9], [5, 4, 6], [4, 3, 6]],
     [[8, 3, 2], [6, 8, 8], [2, 9, 5]]]
b = np.random.randint(1, 10, size=[3, 3, 2])

b = np.array(b)
print('a:%s' % b)
print(b.shape)
# print('b:%s' % b)

# print('np model')
# print(np.argmax(a))
res = np.argmax(b, axis=0)
res1 = np.argmax(b, axis=0)
print(res)
print(res.shape)
print(res1)
# result 6 ，默认情况下，是把数组平铺了，就是不管你是几维度的，把所有元素拿出来合成1维数组，元素9 对应index为6
# 不用平铺就要用到axis参数。0代表行，1代表列

# ta = tf.argmax(a, 1)
# tb = tf.argmax(b, 1)
#
# with tf.Session() as sess:
#     print(sess.run(ta))
#     print(sess.run(tb))
