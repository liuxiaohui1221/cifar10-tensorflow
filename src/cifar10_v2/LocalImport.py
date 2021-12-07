import os
import tensorflow as tf
import numpy as np
import pandas as ps

# 预处理，把数据压缩在小区间，保证梯度
# def preprocess(x, y):
# 	x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
# 	y = tf.cast(y, dtype=tf.int32)
# 	return x, y


def get_files(base_path,data_path,label_file):
	# two list to save train data and train label
	class_train = []
	# train_class is the train image direcotry
	file_path=os.path.join(base_path,data_path)
	for pic_name in os.listdir(file_path):
		image_raw = tf.io.read_file(file_path + '/' + pic_name)
		image = tf.image.decode_image(contents=image_raw, channels=3)
		image = tf.image.resize(image, size=[32, 32])
		image = tf.image.convert_image_dtype(image, tf.float32)
		#print(np.array(image).shape)
		image = np.reshape(image, (3, 32, 32))
		#print(np.array(image).shape)
		image = np.transpose(image, (1, 2, 0))
		#print(np.array(image).shape)
		image = image.astype(float)
		class_train.append(image.tolist())
	picName_label = np.array(ps.read_csv(os.path.join(base_path,label_file), sep=' '))
	#print(picName_label)
	# transpose temp to (n,2)
	train_db = np.array([class_train, picName_label[:, 1]])
	train_db = train_db.transpose()
	print(train_db.shape)

	#
	#print("before ",train_db)
	np.random.shuffle(train_db)
	#print("after ",train_db)
	num_train_samples = 50000
	x_train = train_db[:num_train_samples, 0]
	y_train = train_db[:num_train_samples, 1]
	x_test = train_db[num_train_samples:, 0]
	y_test = train_db[num_train_samples:, 1]

	x_train=np.array([name for name in x_train])
	y_train = np.array([name for name in y_train])
	x_test = np.array([name for name in x_test])
	y_test = np.array([name for name in y_test])

	print(x_train.shape,y_train.shape)
	return (x_train, y_train), (x_test, y_test)
