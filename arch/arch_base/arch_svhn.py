from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from .resnet_ops import *

class ARCH_svhn():
	def __init__(self):
		print("Creating SVHN architectures for SubGAN case")
		return


	def generator_model_resnet_svhn(self):
		G_num_channels = 1024

		inputs = tf.keras.Input(shape = (self.noise_dims,))
		enc_res = tf.keras.layers.Reshape([1,1,int(self.noise_dims)])(inputs) #1x1xlatent

		x = ResBlockUp(enc_res, G_num_channels)  # 2*2*G_num_channels
		G_num_channels = G_num_channels // 2

		x = ResBlockUp(x, G_num_channels)  # 4*4*G_num_channels
		G_num_channels = G_num_channels // 2

		x = ResBlockUp(x, G_num_channels)  # 8*8*G_num_channels
		G_num_channels = G_num_channels // 2

		x = ResBlockUp(x, G_num_channels // 2)  # 16*16*G_num_channels
		G_num_channels = G_num_channels // 2

		x = ResBlockUp(x, G_num_channels)  # 32*32*G_num_channels

		x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
		x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
		x = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1, padding = 'SAME')(x)
		out =  tf.keras.layers.Activation( activation = 'tanh')(x)

		model = tf.keras.Model(inputs=inputs, outputs=out)

		return model

	def discriminator_model_resnet_svhn(self):
		D_num_channels = 64

		inputs = tf.keras.Input(shape = (self.output_size, self.output_size, 3,))

		x = ResBlockDown(inputs, D_num_channels)  # 64*64
		D_num_channels = D_num_channels * 2

		x = ResBlockDown(x, D_num_channels)  # 32*32
		D_num_channels = D_num_channels * 2

		# x = attention(x, ch, is_training=self.is_training, scope="attention", reuse=reuse)  # 32*32*128

		x = ResBlockDown(x, D_num_channels)  # 16*16
		D_num_channels = D_num_channels * 2

		x = ResBlockDown(x, D_num_channels)  # 8*8
		D_num_channels = D_num_channels * 2

		# x = ResBlockDown(x, D_num_channels)  # 4*4

		flat = tf.keras.layers.Flatten()(x)
		out = tf.keras.layers.Dense(1)(flat)

		model = tf.keras.Model(inputs=inputs, outputs=out)

		return model



	def generator_model_dcgan_svhn(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape = (self.noise_dims,))

		enc_res = tf.keras.layers.Reshape([1,1,int(self.noise_dims)])(inputs) #1x1xlatent

		denc4 = tf.keras.layers.Conv2DTranspose(512, 5, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True)(enc_res) #2x2x128
		denc4 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc4)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc4)
		denc4 = tf.keras.layers.LeakyReLU(alpha=0.1)(denc4)

		denc3 = tf.keras.layers.Conv2DTranspose(256, 5, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True)(denc4) #4x4x256
		denc3 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc3)
		# denc3 = tf.keras.layers.Dropout(0.5)(denc3)
		denc3 = tf.keras.layers.LeakyReLU(alpha=0.1)(denc3)


		denc2 = tf.keras.layers.Conv2DTranspose(128, 5, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True)(denc3) #8x8x128
		denc2 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc2)
		# denc2 = tf.keras.layers.Dropout(0.5)(denc2)
		denc2 = tf.keras.layers.LeakyReLU(alpha=0.1)(denc2)


		denc1 = tf.keras.layers.Conv2DTranspose(64, 5, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True)(denc2) #16x16x64
		denc1 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc1)
		# denc1 = tf.keras.layers.Dropout(0.5)(denc1)
		denc1 = tf.keras.layers.LeakyReLU(alpha=0.1)(denc1)

		out = tf.keras.layers.Conv2DTranspose(3, 5,strides=2,padding='same', kernel_initializer=init_fn)(denc1) #32x32x3
		out =  tf.keras.layers.Activation( activation = 'tanh')(out)

		
		model = tf.keras.Model(inputs=inputs, outputs=out)
		return model

	def discriminator_model_dcgan_svhn(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential() #64x64x3
		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn,input_shape=[self.output_size, self.output_size, 3])) #32x32x64
		model.add(layers.BatchNormalization(momentum=0.9))
		model.add(layers.LeakyReLU(alpha=0.1))

		model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)) #16x16x128
		model.add(layers.BatchNormalization(momentum=0.9))
		model.add(layers.LeakyReLU(alpha=0.1))

		model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)) #8x8x256
		model.add(layers.BatchNormalization(momentum=0.9))
		model.add(layers.LeakyReLU(alpha=0.1))

		model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)) #4x4x512
		model.add(layers.BatchNormalization(momentum=0.9))
		model.add(layers.LeakyReLU(alpha=0.1))

		model.add(layers.Flatten()) #8192x1
		model.add(layers.Dense(512))	
		model.add(layers.Dense(1)) #1x1


		return model

	def generator_model_aedense_svhn(self):
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.5, seed=None)
		# else:
		# 	tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc1 = tf.keras.layers.Dense(1024, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(inputs)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Dense(512, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc1)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(256, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc2)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)

		enc4 = tf.keras.layers.Dense(128, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc3)
		enc4 = tf.keras.layers.LeakyReLU()(enc4)

		enc5 = tf.keras.layers.Dense(64, kernel_initializer=init_fn, use_bias = False)(enc4)
		enc5 = tf.keras.layers.LeakyReLU()(enc4)

		enc6 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = False)(enc5)
		# enc6 = tf.keras.layers.LeakyReLU()(enc6)

		model = tf.keras.Model(inputs = inputs, outputs = enc6)

		return model


	def discriminator_model_aedense_svhn(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)


		model = tf.keras.Sequential()
		model.add(layers.Dense(512, use_bias=False, input_shape=(self.latent_dims,), kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())


		model.add(layers.Dense(256, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(64, use_bias=True, kernel_initializer=init_fn))
		# # model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(32, use_bias=True, kernel_initializer=init_fn))

		model.add(layers.Dense(1))
		# model.add(layers.Softmax())

		return model



	def same_images_FID(self):
		import glob
		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	
			random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			if self.FID_num_samples <50000:
				self.fid_train_images = self.fid_train_images[random_points]


			## self.fid_train_images has the names to be read. Make a dataset with it
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images)
			# self.fid_image_dataset = self.fid_image_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.FID_num_samples)

		with tf.device(self.device):
			for i in range(self.FID_num_samples):	
				preds = self.generator(self.get_noise([2, self.noise_dims]), training=False)
				preds = preds.numpy()
				fake = preds[0,:,:,:]
				tf.keras.preprocessing.image.save_img(self.FIDFakespath+str(i)+'.png', fake,  scale=True)

			cur_num_reals = len(glob.glob(self.FIDRealspath))
			if cur_num_reals < self.FID_num_samples:
				for image_batch in self.fid_image_dataset:
					for i in range(self.FID_num_samples):
						real = image_batch[i,:,:,:]
						tf.keras.preprocessing.image.save_img(self.FIDRealspath+str(i)+'.png', real,  scale=True)
		return

	def save_interpol_figs(self):
		num_interps = 11
		from scipy.interpolate import interp1d
		with tf.device(self.device):
			for i in range(self.FID_num_samples):
				# self.FID_num_samples
				start = self.get_noise([1, self.noise_dims])#current_batch[i:1+i].numpy()
				end = self.get_noise([1, self.noise_dims]) #current_batch[num_interps+i:num_interps+1+i].numpy()
				stack = np.vstack([start, end])

				linfit = interp1d([1,num_interps+1], stack, axis=0)
				interp_latents = linfit(list(range(1,num_interps+1)))

				# print(interp_latents.shape)
				mid = interp_latents[5:6]
				mid_img = self.generator(mid)
				mid_img = (mid_img + 1.0)/2.0
				mid_img = mid_img.numpy()
				mid_img = mid_img[0,:,:,:]
				tf.keras.preprocessing.image.save_img(self.FIDInterpolpath+str(i)+'.png', mid_img,  scale=True)
		return
		
	def SVHN_Classifier(self):
		self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(299,299,3), classes=1000)

	def FID_svhn(self):

		def data_preprocess(image):
			with tf.device('/CPU'):
				image = tf.image.resize(image,[299,299])
			return image


		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	
			random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)

			self.fid_train_images_names = self.fid_train_images[random_points]

			## self.fid_train_images has the names to be read. Make a dataset with it
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images_names)
			self.fid_image_dataset = self.fid_image_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.fid_batch_size)

			self.SVHN_Classifier()


		if self.mode == 'fid':
			print(self.checkpoint_dir)
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			print('Models Loaded Successfully')

		with tf.device(self.device):
			for image_batch in self.fid_image_dataset:
				# noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
				noise = self.get_noise([self.fid_batch_size, self.noise_dims])
				preds = self.generator(noise, training=False)
				preds = tf.image.resize(preds, [299,299])
				preds = preds.numpy()

				act1 = self.FID_model.predict(image_batch)
				act2 = self.FID_model.predict(preds)
				try:
					self.act1 = np.concatenate((self.act1,act1), axis = 0)
					self.act2 = np.concatenate((self.act2,act2), axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
			self.eval_FID()
			return

