from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



import tensorflow_probability as tfp
tfd = tfp.distributions


class ARCH_g1():
	def __init__(self):
		print("Creating 1-D Gaussian architectures for base cases ")
		return

	def generator_model_dense_g1(self):
		init_fn_bias = tf.keras.initializers.glorot_uniform() #tf.keras.initializers.Identity()#
		init_fn_bias = tf.function(init_fn_bias, autograph=False)
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.5, seed=None)
		init_fn = tf.keras.initializers.Identity()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(1, use_bias=True, input_shape=(self.noise_dims,),kernel_initializer=init_fn, bias_initializer = bias_init_fn))
		# model.add(layers.ReLU())
		return model


	def discriminator_model_dense_g1(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		# bias_init_fn = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
		# bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(5, use_bias=True, input_shape=(1,), kernel_initializer=init_fn))# bias_initializer = bias_init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(10, use_bias=True, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(2, use_bias=True, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		# model.add(layers.LeakyReLU())

		model.add(layers.Dense(1, use_bias = True))
		# model.add(layers.Softmax())

		return model

	def generator_model_denseTanh_g1(self):
		init_fn_bias = tf.keras.initializers.glorot_uniform() #tf.keras.initializers.Identity()#
		init_fn_bias = tf.function(init_fn_bias, autograph=False)
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.5, seed=None)
		init_fn = tf.keras.initializers.Identity()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(1, use_bias=True, input_shape=(self.noise_dims,),kernel_initializer=init_fn, bias_initializer = bias_init_fn))
		# model.add(layers.ReLU())
		return model

	def discriminator_model_denseTanh_g1(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		# bias_init_fn = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
		# bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(5, use_bias=True, input_shape=(1,), kernel_initializer=init_fn))# bias_initializer = bias_init_fn))
		# model.add(layers.BatchNormalization())

		model.add(layers.Dense(10, use_bias=True, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.Activation( activation = 'tanh')) 

		model.add(layers.Dense(2, use_bias=True, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		# model.add(layers.Activation( activation = 'tanh')) 
		
		model.add(layers.Dense(1, use_bias = True))

		return model


	def generator_model_linSig_g1(self):
		init_fn_bias = tf.keras.initializers.glorot_uniform() #tf.keras.initializers.Identity()#
		init_fn_bias = tf.function(init_fn_bias, autograph=False)
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.5, seed=None)
		init_fn = tf.keras.initializers.Identity()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(1, use_bias=True, input_shape=(self.noise_dims,),kernel_initializer=init_fn, bias_initializer = bias_init_fn))
		# model.add(layers.ReLU())
		return model

	def discriminator_model_linSig_g1(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		# bias_init_fn = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
		# bias_init_fn = tf.function(bias_init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(1,))
		den = tf.keras.layers.Dense(1, kernel_initializer=init_fn, use_bias = True)(inputs)
		out = tf.keras.layers.Activation( activation = 'sigmoid')(den)

		model = tf.keras.Model(inputs = inputs, outputs = out)

		self.linNoSig = tf.keras.Model(inputs = inputs, outputs = den)

		return model

	def generator_model_siren_g1(self):
		# init_fn_bias = tf.keras.initializers.glorot_uniform() #tf.keras.initializers.Identity()#
		# init_fn_bias = tf.function(init_fn_bias, autograph=False)
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.5, seed=None)
		init_fn = tf.keras.initializers.Identity()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(1, use_bias=True, input_shape=(self.noise_dims,),kernel_initializer=init_fn, bias_initializer = bias_init_fn))
		# model.add(layers.ReLU())
		return model

	def discriminator_model_siren_g1(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		# bias_init_fn = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
		# bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(5, use_bias=True, input_shape=(1,), kernel_initializer=init_fn))# bias_initializer = bias_init_fn))
		model.add(layers.Activation( activation = tf.math.sin)) 
		# model.add(layers.BatchNormalization())

		model.add(layers.Dense(10, use_bias=True, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.Activation( activation = tf.math.sin)) 

		model.add(layers.Dense(2, use_bias=True, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.Activation( activation = tf.math.sin)) 
		# model.add(layers.Activation( activation = 'tanh')) 
		
		model.add(layers.Dense(1, use_bias = True))

		return model


	def generator_model_sirenFS_g1(self):
		# init_fn_bias = tf.keras.initializers.glorot_uniform() #tf.keras.initializers.Identity()#
		# init_fn_bias = tf.function(init_fn_bias, autograph=False)
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.5, seed=None)
		init_fn = tf.keras.initializers.Identity()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(1, use_bias=True, input_shape=(self.noise_dims,),kernel_initializer=init_fn, bias_initializer = bias_init_fn))
		# model.add(layers.ReLU())
		return model

	def discriminator_model_sirenFS_g1(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		# bias_init_fn = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
		# bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(50, use_bias=True, input_shape=(1,), kernel_initializer=init_fn))# bias_initializer = bias_init_fn))
		model.add(layers.Activation( activation = tf.math.sin)) 
		# model.add(layers.BatchNormalization())
		
		model.add(layers.Dense(1, use_bias = True))

		return model

	def show_result_g1(self, images = None, num_epoch = 0, path = 'result.png', show = False, save = True):

		# Define a single scalar Normal distribution.
		pd_dist = tfd.Normal(loc=np.mean(self.reals), scale=np.std(self.reals))
		pg_dist = tfd.Normal(loc=np.mean(self.fakes), scale=np.std(self.fakes))

		basis = np.expand_dims(np.linspace(self.MIN, self.MAX, int(1e4), dtype=np.float32), axis = 1)
		pd_vals = pd_dist.prob(basis)
		pg_vals = pg_dist.prob(basis)

		basis_ip = tf.cast(basis, dtype = tf.float16)
		with tf.GradientTape() as l:
			l.watch(basis_ip)
			with tf.GradientTape() as t:
				t.watch(basis_ip)
				pred = self.discriminator(basis_ip, training = True)
			grad = t.gradient(pred, [basis_ip])
			# print(grad)
		gradgrad = l.gradient(grad, [basis_ip])
		# print(gradgrad)
		if self.data != 'g1':
			self.autodiff_lap = tf.squeeze(tf.reduce_sum(gradgrad, axis = 1))
		else:
			self.autodiff_lap = tf.squeeze(gradgrad)

		if self.arch != 'linSig':
			self.autodiff_lap = self.autodiff_lap - min(self.autodiff_lap)
			self.autodiff_lap = (self.autodiff_lap/max(self.autodiff_lap))*1.0
			self.autodiff_lap -= 0.5


		# print(basis, self.autodiff_lap)

		disc = self.discriminator(basis,training = False)
		if self.arch != 'linSig':
			disc = disc - min(disc)
			disc = (disc/max(disc))*1.0
			disc -= 0.50
			RHS = pg_vals - pd_vals
		else:
			Wt,b = self.discriminator.get_weights()
			self.linNoSig.set_weights([Wt,b])
			y = self.linNoSig(basis,training = False)
			LapAct = disc*(1.-disc)*(1.-2.*disc)
			LapD = LapAct * tf.math.square(tf.norm(Wt,ord='euclidean'))
			RHS = pg_vals - pd_vals

		true_classifier = np.ones_like(basis)
		# true_classifier[pd_vals > pg_vals] = 0

		with tf.GradientTape() as t:
			basis = tf.cast(basis, dtype = 'float32')
			t.watch(basis)

			scale = 1.0
			disc = self.discriminator(basis,training = False)
			scaled_disc = disc - min(disc)
			scaled_disc = (scaled_disc/max(abs(scaled_disc)))*scale*2.0
			scaled_disc -= scale

			grads = t.gradient(disc, basis)
			scale = 0.50
			grads = (grads/max(abs(grads)))*scale*2.0


		if self.paper and (self.total_count.numpy() == 1 or self.total_count.numpy() % 1000):
			np.save(path+'_disc_'+str(self.total_count.numpy())+'.npy',np.array(disc))
			np.save(path+'_reals_'+str(self.total_count.numpy())+'.npy',np.array(self.reals))
			np.save(path+'_fakes_'+str(self.total_count.numpy())+'.npy',np.array(self.fakes))
			np.save(path+'_grads_'+str(self.total_count.numpy())+'.npy',np.array(grads))
		
		if self.colab == 1:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
			
			from matplotlib.backends.backend_pgf import FigureCanvasPgf
			matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "serif",  # use serif/main font for text elements
				"font.size":10,	
				"font.serif": [], 
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
				# "pgf.preamble": [
				# 	 r"\usepackage[utf8x]{inputenc}",
				# 	 r"\usepackage[T1]{fontenc}",
				# 	 r"\usepackage{cmbright}",
				# 	 ]
			})


		with PdfPages(path+'_Classifier.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim(bottom=-0.5,top=1.8)
			ax1.plot(basis,pd_vals, linewidth = 1.5, c='r')
			ax1.plot(basis,pg_vals, linewidth = 1.5, c='g')
			ax1.scatter(self.reals, np.zeros_like(self.reals), c='r', linewidth = 1.5, label='Real Data', marker = '.')
			ax1.scatter(self.fakes, np.zeros_like(self.fakes), c='g', linewidth = 1.5, label='Fake Data', marker = '.')
			ax1.plot(basis,disc, c='b', linewidth = 1.5, label='Discriminator')

			ax1.plot(basis,grads, c='teal', linewidth = 1.5, label='Discriminator gradient')

			if self.total_count < 20:
				ax1.plot(basis,true_classifier,'c--', linewidth = 1.5, label='True Classifier')
			ax1.legend(loc = 'upper right')
			fig1.tight_layout()
			pdf.savefig(fig1)
			plt.close(fig1)

		with PdfPages(path+'_PoissonD.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.set_xlim([self.MIN,self.MAX])
			# ax1.set_ylim(bottom=-0.5,top=1.8)
			ax1.plot(basis,pd_vals, linewidth = 1.5, c='r', label='Pd')
			ax1.plot(basis,pg_vals, linewidth = 1.5, c='g', label='Pg')
			if self.arch == 'linSig':
				ax1.plot(basis,LapD, linewidth = 1.5, c='m', label='LHS')
			# ax1.plot(basis,y, 'm:', linewidth = 1.5, label='y = Wx+b')
			ax1.plot(basis,self.autodiff_lap, 'k:', linewidth = 1.5, label='Lap(D(x)): AutoDiff')
			ax1.plot(basis,RHS, linewidth = 1.5, c='y', label='RHS - No Lambda')
			ax1.plot(basis,disc, c='b', linewidth = 1.5, label='Discriminator')
			ax1.legend(loc = 'upper right')
			fig1.tight_layout()
			pdf.savefig(fig1)
			plt.close(fig1)


			# if self.total_count > 10:
			# 	exit(0)