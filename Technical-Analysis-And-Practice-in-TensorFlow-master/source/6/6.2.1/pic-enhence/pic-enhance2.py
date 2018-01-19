# 贡献者：{吴翔 QQ：99456786}
# 源代码出处：
# 程序将实现图片翻转


import matplotlib.pyplot as plt;
import tensorflow as tf;

image_raw_data_jpg = tf.gfile.FastGFile('mountain.jpg', 'r').read()

with tf.Session() as sess:
	img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
	img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
	img_1 = tf.image.flip_up_down(img_data_jpg)
	img_2 = tf.image.flip_left_right(img_data_jpg)
	img_3 = tf.image.transpose_image(img_data_jpg)
  # 结果展示
	f,a= plt.subplots(2,2,figsize=(50,10))
	a[0][0].imshow(img_data_jpg.eval())
	a[0][1].imshow(img_1.eval())
	a[1][0].imshow(img_2.eval())
	a[1][1].imshow(img_3.eval())
	f.show()
	plt.draw()
	plt.waitforbuttonpress()
