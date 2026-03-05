from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
from matplotlib.backends.backend_pgf import PdfPages

######## OLD NEED CLEANUP ########
class ARCH_gmmN():	
	def __init__(self):
		return

	def generator_model_dcgan_gmmN(self):

		# init_fn = tf.keras.initializers.Identity()
		# init_fn = tf.function(init_fn, autograph=False)
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None)
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		# bias_init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None)
		# bias_init_fn = tf.function(bias_init_fn, autograph=False)
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None)
		# init_fn = tf.function(init_fn, autograph=False)
		
		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc0 = tf.keras.layers.Dense(32*32*3, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(inputs)
		enc0 = tf.keras.layers.LeakyReLU()(enc0)

		enc_res = tf.keras.layers.Reshape([32,32,3])(enc0)

		enc1 = tf.keras.layers.Conv2D(1024, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc_res) #16x16x64
		enc1 = tf.keras.layers.BatchNormalization()(enc1)
		# enc1 = tf.keras.layers.Dropout(0.1)(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc1) #8x8x128
		enc2 = tf.keras.layers.BatchNormalization()(enc2)
		# enc2 = tf.keras.layers.Dropout(0.1)(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)


		enc3 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc2) #4x4x256
		enc3 = tf.keras.layers.BatchNormalization()(enc3)
		# enc3 = tf.keras.layers.Dropout(0.1)(enc3)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)


		enc4 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc3) #2x2x128
		enc4 = tf.keras.layers.BatchNormalization()(enc4)
		# enc4 = tf.keras.layers.Dropout(0.5)(enc4)
		enc4 = tf.keras.layers.LeakyReLU()(enc4)

		enc5 = tf.keras.layers.Conv2D(int(self.latent_dims), 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc4) #1x1xlatent
		# enc5 = tf.keras.layers.BatchNormalization()(enc5)


		enc = tf.keras.layers.Flatten()(enc5)
		# enc0 = tf.keras.layers.Dense(32*32, kernel_initializer=init_fn, use_bias = True)(inputs)
		# enc0 = tf.keras.layers.LeakyReLU()(enc0)

		# enc1 = tf.keras.layers.Dense(256, kernel_initializer=init_fn, use_bias = True, activation = 'sigmoid')(enc0)
		# # enc1 = tf.keras.layers.LeakyReLU()(enc1)
		
		# # enc12 = tf.keras.layers.Dense(int(self.latent_dims*10), kernel_initializer=init_fn, use_bias = True, activation = 'sigmoid')(enc11)
		# # enc12 = tf.keras.layers.ReLU()(enc12)

		# enc2 = tf.keras.layers.Dense(128, kernel_initializer=init_fn, use_bias = True)(enc1)
		# enc2 = tf.keras.layers.Activation( activation = 'sigmoid')(enc2)
		# # enc2 = tf.keras.layers.LeakyReLU()(enc2)

		# enc3 = tf.keras.layers.Dense(64, kernel_initializer=init_fn, use_bias = False)(enc2)
		# # enc3 = tf.keras.layers.Activation( activation = 'sigmoid')(enc3)
		# # enc3 = tf.keras.layers.LeakyReLU()(enc3)

		enc = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc)
		# enc = tf.clip_by_value(enc, clip_value_min = 0., clip_value_max = 6.)
		# enc = tf.keras.layers.Activation( activation = tf.nn.relu6)(enc)
		# enc = tf.keras.layers.Activation( activation = 'tanh')(enc)
		# enc = tf.keras.layers.Activation( activation = 'sigmoid')(enc)
		# enc = tf.math.scalar_mul(3., enc)
		# enc = tf.keras.layers.ReLU(max_value = 2.)(enc)
		# enc = tf.keras.layers.ReLU()(enc)

		model = tf.keras.Model(inputs = inputs, outputs = enc)

		return model

	def show_result_gmmN(self, images = None, num_epoch = 0, show = False, save = False, path = 'result.png'):

		print("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals,axis = 0), np.cov(self.reals,rowvar = False), np.mean(self.fakes, axis = 0), np.cov(self.fakes,rowvar = False) ))
		if self.res_flag:
			self.res_file.write("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals, axis = 0), np.cov(self.reals,rowvar = False), np.mean(self.fakes, axis = 0), np.cov(self.fakes,rowvar = False) ))

	def save_for_paper_gmmN(self, images = None, num_epoch = 0, path = 'result.png'):

		print("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals,axis = 0), np.cov(self.reals,rowvar = False), np.mean(self.fakes, axis = 0), np.cov(self.fakes,rowvar = False) ))
		if self.res_flag:
			self.res_file.write("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals, axis = 0), np.cov(self.reals,rowvar = False), np.mean(self.fakes, axis = 0), np.cov(self.fakes,rowvar = False) ))

