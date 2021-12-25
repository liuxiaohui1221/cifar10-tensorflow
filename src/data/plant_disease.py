
import cv2 as cv2
import src.data.datagenerator as dg
import numpy as np
import random

class Plant:

	def __init__(self):
		self.load_plant_disease('C:\\Downloads\\plant\\train')
		self._split_train_valid(valid_rate=0.9)
		self.n_train = self.train_images.shape[0]
		self.n_valid = self.valid_images.shape[0]
		self.n_test = self.test_images.shape[0]


	def _split_train_valid(self, valid_rate=0.9):
		images, labels = self.train_images, self.train_labels
		thresh = int(images.shape[0] * valid_rate)
		self.train_images, self.train_labels = images[0:thresh, :, :, :], labels[0:thresh]
		self.valid_images, self.valid_labels = images[thresh:, :, :, :], labels[thresh:]

	def load_plant_disease(self, directory):
		# 读取训练集
		plantdata = dg.get_data(directory)
		images, labels = plantdata.train.images, plantdata.train.cls
		print('plant disease shapes: ', images,labels)
		self.train_images, self.train_labels = images, labels

		# 读取测试集
		images, labels = plantdata.valid.images, plantdata.valid.cls
		images = np.array(images, dtype='float')
		labels = np.array(labels, dtype='int')
		self.test_images, self.test_labels = images, labels

	def data_augmentation(self, images, mode='train', flip=False,
						  crop=False, crop_shape=(24, 24, 3), whiten=False,
						  noise=False, noise_mean=0, noise_std=0.01):
		# 图像切割
		if crop:
			if mode == 'train':
				images = self._image_crop(images, shape=crop_shape)
			elif mode == 'test':
				images = self._image_crop_test(images, shape=crop_shape)
		# 图像翻转
		if flip:
			images = self._image_flip(images)
		# 图像白化
		if whiten:
			images = self._image_whitening(images)
		# 图像噪声
		if noise:
			images = self._image_noise(images, mean=noise_mean, std=noise_std)

		return images

	def _image_crop(self, images, shape):
		# 图像切割
		new_images = []
		for i in range(images.shape[0]):
			old_image = images[i, :, :, :]
			old_image = np.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
			left = np.random.randint(old_image.shape[0] - shape[0] + 1)
			top = np.random.randint(old_image.shape[1] - shape[1] + 1)
			new_image = old_image[left: left + shape[0], top: top + shape[1], :]
			new_images.append(new_image)

		return np.array(new_images)

	def _image_crop_test(self, images, shape):
		# 图像切割
		new_images = []
		for i in range(images.shape[0]):
			old_image = images[i, :, :, :]
			old_image = np.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
			left = int((old_image.shape[0] - shape[0]) / 2)
			top = int((old_image.shape[1] - shape[1]) / 2)
			new_image = old_image[left: left + shape[0], top: top + shape[1], :]
			new_images.append(new_image)

		return np.array(new_images)

	def _image_flip(self, images):
		# 图像翻转
		for i in range(images.shape[0]):
			old_image = images[i, :, :, :]
			if np.random.random() < 0.5:
				new_image = cv2.flip(old_image, 1)
			else:
				new_image = old_image
			images[i, :, :, :] = new_image

		return images

	def _image_whitening(self, images):
		# 图像白化
		for i in range(images.shape[0]):
			old_image = images[i, :, :, :]
			new_image = (old_image - np.mean(old_image)) / np.std(old_image)
			images[i, :, :, :] = new_image

		return images

	def _image_noise(self, images, mean=0, std=0.01):
		# 图像噪声
		for i in range(images.shape[0]):
			old_image = images[i, :, :, :]
			new_image = old_image
			for i in range(old_image.shape[0]):
				for j in range(old_image.shape[1]):
					for k in range(old_image.shape[2]):
						new_image[i, j, k] += random.gauss(mean, std)
			images[i, :, :, :] = new_image

		return images
