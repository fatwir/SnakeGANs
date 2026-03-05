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
class ARCH_gN():	
	def __init__(self):
		return

	def generator_model_nodense_gN(self):
		# init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.075, seed=None)
		# init_fn = tf.keras.initializers.Identity()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(self.GaussN, use_bias=True, input_shape=(self.noise_dims,), kernel_initializer=init_fn))
		model.add(layers.Activation(activation = 'sigmoid'))
		# model.add(layers.ReLU())
		model.add(layers.Dense(self.GaussN, use_bias=True, kernel_initializer=init_fn))
		# model.add(layers.ReLU())
		# model.add(layers.Dense(2, use_bias=True,kernel_initializer=init_fn))
		# model.add(layers.ReLU())

		return model


	def discriminator_model_nodense_gN(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(512, use_bias=False, input_shape=(self.GaussN,), kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		# model.add(layers.LeakyReLU())
		model.add(tf.keras.layers.Activation( activation = 'tanh'))

		model.add(layers.Dense(256, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		# model.add(layers.LeakyReLU())
		model.add(tf.keras.layers.Activation( activation = 'tanh'))

		model.add(layers.Dense(64, use_bias=True, kernel_initializer=init_fn))
		# # model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(32, use_bias=True, kernel_initializer=init_fn))

		model.add(layers.Dense(1))
		# model.add(layers.Softmax())

		return model



	def generator_model_dcgan_gN(self):

		# init_fn = tf.keras.initializers.Identity()
		# init_fn = tf.function(init_fn, autograph=False)
		# FOr regular comarisons when 2m-n = 0
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.075, seed=None) 
		# FOr comparisons when 2m-n > 0 
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
		# FOr comparisons when 2m-n < 0 

		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None)
		# init_fn = tf.function(init_fn, autograph=False)
		# bias_init_fn = tf.keras.initializers.Zeros()
		# bias_init_fn = tf.function(bias_init_fn, autograph=False)

		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.075, seed=None)
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.glorot_uniform()#tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)
		
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

	def discriminator_model_gN_base(self):
		inputs = tf.keras.Input(shape=(self.output_size,))
		w0_nt_x = tf.keras.layers.Dense(self.L, activation=None, use_bias = False)(inputs)
		w0_nt_x2 = tf.math.scalar_mul(2., w0_nt_x)
		cos_terms = tf.keras.layers.Activation( activation = tf.math.cos)(w0_nt_x)
		sin_terms = tf.keras.layers.Activation( activation = tf.math.sin)(w0_nt_x)
		cos2_terms  = tf.keras.layers.Activation( activation = tf.math.cos)(w0_nt_x2)

		cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms)
		sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)

		cos2_c_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) #Tau_c weights
		cos2_s_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) #Tau_s weights

		lambda_x_term = tf.keras.layers.Subtract()([cos2_s_sum, cos2_c_sum]) #(tau_s  - tau_r)
		if self.homo_flag:
			if self.latent_dims == 1:
				Out = tf.keras.layers.Add()([cos_sum, sin_sum, tf.divide(tf.abs(inputs),2.)])
			elif self.latent_dims >= 2:
				Out = tf.keras.layers.Add()([cos_sum, sin_sum, tf.divide(tf.reduce_sum(inputs,axis=-1,keepdims=True),self.latent_dims)])
		else:
			Out = tf.keras.layers.Add()([cos_sum, sin_sum])
		model = tf.keras.Model(inputs=inputs, outputs=[Out,lambda_x_term])
		return model

	def show_result_gN(self, images = None, num_epoch = 0, show = False, save = False, path = 'result.png'):

		print("\n Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals,axis = 0), np.cov(self.reals,rowvar = False), np.mean(self.fakes, axis = 0), np.cov(self.fakes,rowvar = False) ))
		if self.res_flag:
			self.res_file.write("\n Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals, axis = 0), np.cov(self.reals,rowvar = False), np.mean(self.fakes, axis = 0), np.cov(self.fakes,rowvar = False) ))

	def save_for_paper_gN(self, images = None, num_epoch = 0, path = 'result.png'):

		print("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals,axis = 0), np.cov(self.reals,rowvar = False), np.mean(self.fakes, axis = 0), np.cov(self.fakes,rowvar = False) ))
		if self.res_flag:
			self.res_file.write("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals, axis = 0), np.cov(self.reals,rowvar = False), np.mean(self.fakes, axis = 0), np.cov(self.fakes,rowvar = False) ))

	def FID_gN(self):


		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	

			self.fid_image_dataset = self.train_dataset


		if self.mode == 'fid':
			print(self.checkpoint_dir)
			# print('logs/130919_ELeGANt_mnist_lsgan_base_01/130919_ELeGANt_mnist_lsgan_base_Results_checkpoints')
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			print('Models Loaded Successfully')

		with tf.device(self.device):
			for data_batch in self.fid_image_dataset:
				# print(self.fid_train_images.shape)
				noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
				preds = self.generator(noise, training=False)
				preds = preds.numpy()

				try:
					self.act1 = np.concatenate([self.act1,data_batch], axis = 0)
					self.act2 = np.concatenate([self.act2,preds], axis = 0)
				except:
					self.act1 = data_batch
					self.act2 = preds
			# print(self.act1.shape, self.act2.shape)
			self.eval_FID()
			return
