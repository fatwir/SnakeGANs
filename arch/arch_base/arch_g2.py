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


class ARCH_g2():
	def __init__(self):
		print("Creating 2-D Gaussian architectures for base cases")
		return

	def generator_model_dense_g2(self):
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None)
		# init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc1 = tf.keras.layers.Dense(64, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(inputs)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Dense(32, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc1)
		enc2 = tf.keras.layers.Activation( activation = 'tanh')(enc2)
		# enc2 = tf.keras.layers.Activation( activation = 'sigmoid')(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(16, kernel_initializer=init_fn, use_bias = False)(enc2)
		enc3 = tf.keras.layers.Activation( activation = 'tanh')(enc3)
		# enc3 = tf.keras.layers.Activation( activation = 'sigmoid')(enc3)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)

		enc4 = tf.keras.layers.Dense(self.output_size, kernel_initializer=init_fn, use_bias = False)(enc3)
		# enc4 =  tf.keras.layers.Activation( activation = 'tanh')(enc4)
		# enc4 = tf.math.scalar_mul(1.2, enc4)
		# enc4 =  tf.keras.layers.Activation( activation = 'sigmoid')(enc4)
		# enc4 = tf.keras.layers.ReLU(max_value = 1.)(enc4)

		model = tf.keras.Model(inputs = inputs, outputs = enc4)

		return model

	def discriminator_model_dense_g2(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(10, use_bias=False, input_shape=(2,), kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(20, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(5, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))
		# model.add(layers.Softmax())

		return model

	def generator_model_nodense_g2(self):
		# init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.keras.initializers.Identity()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		# bias_init_fn = tf.keras.initializers.glorot_uniform()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(2, use_bias=True, input_shape=(self.noise_dims,), kernel_initializer=init_fn, bias_initializer = bias_init_fn))
		# model.add(layers.ReLU())
		# model.add(layers.Dense(2, use_bias=True,kernel_initializer=init_fn))
		# model.add(layers.ReLU())

		return model


	def discriminator_model_nodense_g2(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(10, use_bias=False, input_shape=(2,), kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
		# model.add(tf.keras.layers.Activation( activation = 'tanh'))

		model.add(layers.Dense(20, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(5, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))
		# model.add(layers.Softmax())

		return model

	def generator_model_nodentanh_g2(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(2, use_bias=True, input_shape=(self.noise_dims,), kernel_initializer=init_fn))
		# model.add(layers.ReLU())
		model.add(layers.Dense(2, use_bias=True,kernel_initializer=init_fn))
		# model.add(layers.ReLU())

		return model


	def discriminator_model_nodentanh_g2(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(10, use_bias=False, input_shape=(2,), kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		# model.add(layers.LeakyReLU())
		model.add(tf.keras.layers.Activation( activation = 'tanh'))

		model.add(layers.Dense(20, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(tf.keras.layers.Activation( activation = 'tanh'))
		# model.add(layers.LeakyReLU())

		model.add(layers.Dense(5, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		# model.add(tf.keras.layers.Activation( activation = 'tanh'))
		# model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))
		# model.add(layers.Softmax())
		return model

	def generator_model_siren_g2(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(2, use_bias=True, input_shape=(self.noise_dims,), kernel_initializer=init_fn))
		# model.add(layers.ReLU())
		model.add(layers.Dense(2, use_bias=True,kernel_initializer=init_fn))
		# model.add(layers.ReLU())

		return model


	def discriminator_model_siren_g2(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(10, use_bias=False, input_shape=(2,), kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		# model.add(layers.LeakyReLU())
		model.add(tf.keras.layers.Activation( activation = tf.math.sin))

		model.add(layers.Dense(20, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(tf.keras.layers.Activation( activation = tf.math.sin))
		# model.add(layers.LeakyReLU())

		model.add(layers.Dense(5, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(tf.keras.layers.Activation( activation = tf.math.sin))
		# model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))
		# model.add(layers.Softmax())
		return model


	def generator_model_sirenFS_g2(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(2, use_bias=True, input_shape=(self.noise_dims,), kernel_initializer=init_fn))
		# model.add(layers.ReLU())
		model.add(layers.Dense(2, use_bias=True,kernel_initializer=init_fn))
		# model.add(layers.ReLU())

		return model


	def discriminator_model_sirenFS_g2(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(700, use_bias=True, input_shape=(2,), kernel_initializer=init_fn))
		model.add(tf.keras.layers.Activation( activation = tf.math.sin))
		model.add(layers.Dense(1))
		return model

	def show_result_g2(self, images = None, num_epoch = 0, path = 'result.png', show = False, save = True):

		def calculate_squared_distances(a, b):
			'''returns the squared distances between all elements in a and in b as a matrix
			of shape #a * #b'''
			na = tf.shape(a)[0]
			nb = tf.shape(b)[0]
			nas, nbs = list(a.shape), list(b.shape)
			a = tf.reshape(a, [na, 1, -1])
			b = tf.reshape(b, [1, nb, -1])
			a.set_shape([nas[0], 1, np.prod(nas[1:])])
			b.set_shape([1, nbs[0], np.prod(nbs[1:])])
			a = tf.tile(a, [1, nb, 1]) #a_i repeated on axis 1
			b = tf.tile(b, [na, 1, 1]) #b_i repeated on axis 0
			d = a-b

			return tf.reduce_sum(tf.square(d), axis=2)

		def plummer_kernel(a, b, dimension, epsilon):
			m = 1
			r = calculate_squared_distances(a, b)
			r += epsilon*epsilon
			f1 = dimension-2*m
			return tf.multiply(tf.pow(r, -f1 / 2),tf.math.log(r))


		def get_potentials(x, y, dimension, cur_epsilon):
			'''
			This is alsmost the same `calculate_potential`, but
				px, py = get_potentials(x, y)
			is faster than:
				px = calculate_potential(x, y, x)
				py = calculate_potential(x, y, y)
			because we calculate the cross terms only once.
			'''
			# x_fixed = tf.stop_gradient(x)
			# y_fixed = tf.stop_gradient(y)
			nx = tf.cast(tf.shape(x)[0], x.dtype)
			ny = tf.cast(tf.shape(y)[0], y.dtype)
			pk_xx = plummer_kernel(x_fixed, x, dimension, cur_epsilon)
			pk_yx = plummer_kernel(y, x, dimension, cur_epsilon)
			pk_yy = plummer_kernel(y_fixed, y, dimension, cur_epsilon)
			#pk_xx = tf.matrix_set_diag(pk_xx, tf.ones(shape=x.get_shape()[0], dtype=pk_xx.dtype))
			#pk_yy = tf.matrix_set_diag(pk_yy, tf.ones(shape=y.get_shape()[0], dtype=pk_yy.dtype))
			kxx = tf.reduce_sum(pk_xx, axis=0) / (nx)
			kyx = tf.reduce_sum(pk_yx, axis=0) / ny
			kxy = tf.reduce_sum(pk_yx, axis=1) / (nx)
			kyy = tf.reduce_sum(pk_yy, axis=0) / ny
			pot_x = kxx - kyx
			pot_y = kxy - kyy
			pot_x = tf.reshape(pot_x, [-1])
			pot_y = tf.reshape(pot_y, [-1])
			return pot_x, pot_y

		def calculate_potential(x, y, a, dimension, epsilon):
			# x = tf.stop_gradient(x)
			# y = tf.stop_gradient(y)
			nx = tf.cast(tf.shape(x)[0], x.dtype)
			ny = tf.cast(tf.shape(y)[0], y.dtype)
			kxa = plummer_kernel(x, a, dimension, epsilon)
			kxa = tf.reduce_sum(kxa, axis=0) / nx
			kya = plummer_kernel(y, a, dimension, epsilon)
			kya = tf.reduce_sum(kya, axis=0) / ny
			p = kxa - kya
			p = tf.reshape(p, [-1])
			return p

		

		basis = np.expand_dims(np.linspace(-10., 10., int(1e4), dtype=np.float32), axis = 1)

		if self.total_count.numpy() == 1 or self.total_count.numpy()%10000 == 0:
			np.save(path+'_reals.npy',np.array(self.reals))
			np.save(path+'_fakes.npy',np.array(self.fakes))
		
		if self.colab == 1:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
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
			})

		with PdfPages(path+'_Classifier.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim(bottom=self.MIN,top=self.MAX)
			if (self.topic not in ["ScoreGAN", "SnakeGAN"]) or (self.topic == "ScoreGAN" and self.gan == 'SGAN'):

				ax1.scatter(self.reals[::1,0], self.reals[::1,1], c='r', linewidth = 0.5, s = 15,  marker = '.', alpha = 0.8)
				ax1.scatter(self.fakes[::1,0], self.fakes[::1,1], c='g', linewidth = 0.5, s = 15,  marker = '.', alpha = 0.8)


			elif self.topic in ["ScoreGAN"] and self.gan in ['WGAN']:

				from itertools import product as cart_prod

				# x = np.arange(self.MIN,self.MAX+0.5,0.5)
				# y = np.arange(self.MIN,self.MAX+0.5,0.5)
				x = np.arange(self.MIN,self.MAX,1.)
				y = np.arange(self.MIN,self.MAX,1.)

				# X, Y = np.meshgrid(x, y)
				prod = np.array([p for p in cart_prod(x,repeat = 2)])

				X = prod[:,0]
				Y = prod[:,1]

				# fake_grad,real_grad = self.grad_kernel(tf.cast(prod,dtype='float32'),self.reals,self.fakes)
				
				# # grad_vals = fake_grad - real_grad
				# grad_vals = tf.math.square(fake_grad - real_grad)
				# # grad_vals = tf.abs(fake_grad - real_grad)
				# # grad_vals = 0.*fake_grad - real_grad

				# grad_vals = self.discriminator_RBF(prod, training = False)
				self.real_output,self.fake_output = self.discriminator_RBF(prod, training = False)

				grad_vals = self.fake_output - self.real_output

				# print(grad_vals,grad_vals.shape)

				# grad_norm = tf.linalg.norm(grad_vals, axis = 1, ord = 2, keepdims=True)
				# grad_norm = grad_norm - min(grad_norm[:,0])
				# grad_norm = grad_norm/max(grad_norm[:,0])
				# grad_norm -= 0.5
				# grad_norm *= 3
				# grad_norm = np.reshape(grad_norm,(x.shape[0],y.shape[0])).transpose()
			
				dy = 5*grad_vals[:,1]
				dx = 5*grad_vals[:,0]
				n = -1
				color_array = np.sqrt(((dx-n)/2)**2 + ((dy-n)/2)**2)

				# print(X.shape,Y.shape,dx.shape,dy.shape)

				ax1.quiver(X,Y,dx,dy,color_array)

				# ax1.contour(x,y,grad_norm,5,linewidths = 1.2, alpha = 0.5 )
				# cs = ax1.contourf(x,y,grad_norm,alpha = 0.5, levels = 5, extend = 'both')

				ax1.scatter(self.reals[::1,0], self.reals[::1,1], c='r', linewidth = 0.5, s = 15,  marker = '.', alpha = 0.68)
				ax1.scatter(self.fakes[::1,0], self.fakes[::1,1], c='g', linewidth = 0.5, s = 15,  marker = '.', alpha = 0.68)


			else:

				pd_mean = tf.cast(np.mean(self.reals, axis = 0), dtype = 'float32')
				pg_mean = tf.cast(np.mean(self.fakes, axis = 0), dtype = 'float32')
				pd_dist = tfd.MultivariateNormalFullCovariance(loc=pd_mean, covariance_matrix=tf.cast(np.cov(self.reals,rowvar = False), dtype = 'float32'))
				pg_dist = tfd.MultivariateNormalFullCovariance(loc=pg_mean, covariance_matrix=tf.cast(np.cov(self.fakes,rowvar = False), dtype = 'float32'))


				# pd_min = pd_mean - 1.5
				# pd_max = pd_mean + 1.5
				# pd_x = np.arange(pd_min,pd_max,0.1)
				# pd_y = np.arange(pd_min,pd_max,0.1)
				# pd_prod = np.array([p for p in cart_prod(pd_x,repeat = 2)])
				



				from itertools import product as cart_prod

				x = np.arange(self.MIN,self.MAX+0.5,0.5)
				y = np.arange(self.MIN,self.MAX+0.5,0.5)

				# X, Y = np.meshgrid(x, y)
				prod = np.array([p for p in cart_prod(x,repeat = 2)])

				pd_vals = tf.expand_dims(pd_dist.prob(prod),axis =1)
				pg_vals = tf.expand_dims(pg_dist.prob(prod),axis =1)

				pd_vals = np.reshape(pd_vals,(x.shape[0],y.shape[0])).transpose()
				pg_vals = np.reshape(pg_vals,(x.shape[0],y.shape[0])).transpose()

				# print(pd_vals.shape)


				# print(x,prod)

				X = prod[:,0]
				Y = prod[:,1]

				# print(prod,X,Y)
				# print(XXX)

				with tf.GradientTape() as disc_tape:
					prod = tf.cast(prod, dtype = 'float32')
					disc_tape.watch(prod)
					# d_vals = calculate_potential(prod,  self.fakes, self.reals, 2, epsilon=0.001)
					d_vals = self.discriminator_RBF(prod,training = False)
					# reals_term = calculate_squared_distances(prod,self.reals)
					# fakes_term = calculate_squared_distances(prod,self.fakes)
					# d_vals = fakes_term - reals_term
				grad_vals = disc_tape.gradient(d_vals, [prod])[0]


				dy = 5*grad_vals[:,1]
				dx = 5*grad_vals[:,0]				

				n = -1
				color_array = np.sqrt(((dx-n)/2)**2 + ((dy-n)/2)**2)

				# cs = ax1.contourf(x,y,- pd_vals + pg_vals,alpha = 0.25, levels = 15, extend = 'both', cmap = 'Spectral')
				# ax1.contour(x,y,- pd_vals + pg_vals,15,linewidths = 0.95, alpha = 0.9, cmap ='Spectral')
				
				### cs alpha 0.125 and contour alphs 0.75 for fig T
				### cs alpha 0.25 and contour alphs 0.95 for fig Method
				ax1.quiver(X,Y,dx,dy,color_array)

				indexing_rate = 2

				ax1.scatter(self.reals[::,0], self.reals[::,1], c='r', linewidth = 0.5, s = 15,  marker = '.', alpha = 0.9, label = 'DATA SAMPLES')
				ax1.scatter(self.fakes[::indexing_rate,0], self.fakes[::indexing_rate,1], c='g', linewidth = 0.5, s = 35,  marker = '.', alpha = 0.9, label = 'GENERATOR OUTPUTS') ### s=15 for FIG T, 35 for other

				# ax1.scatter(self.snakes2[::indexing_rate,0], self.snakes2[::indexing_rate,1], c='b', s = 5,  marker = 'd', alpha = 0.15, label = 'CONVERGED SNAKE CENTROID') 
				# ax1.scatter(self.snakes3[::indexing_rate,0], self.snakes3[::indexing_rate,1], c='b', s = 5,  marker = 'D', alpha = 0.25, label = 'CONVERGED SNAKE CENTROID') 
				# ax1.scatter(self.snakes4[::indexing_rate,0], self.snakes4[::indexing_rate,1], c='b', s = 5,  marker = 'h', alpha = 0.5, label = 'CONVERGED SNAKE CENTROID') 

				ax1.scatter(self.fakes_target[::indexing_rate,0], self.fakes_target[::indexing_rate,1], c='b', s = 5,  marker = 'X', alpha = 0.75, label = 'CONVERGED SNAKE CENTROID') ### s=5 for FIG T, 25 for other
				# ax1.legend(loc = 'upper right')

				## Snake Trajectory Plotting -- Decripit now. 


				# j = -1
				# c = ['purple', 'blueviolet', 'xkcd:cerulean', 'navy', 'teal',]
				# for i in range(0,self.batch_size,indexing_rate):
				# 	j+=1
				# 	circle1 = plt.Circle((self.fakes[i,0], self.fakes[i,1]), 0.2, alpha = 0.5, linewidth = 0.5, color=c[j], fill=False)
				# 	ax1.add_patch(circle1)
				# 	circle2 = plt.Circle((self.snakes2[i,0], self.snakes2[i,1]), 0.2, alpha = 0.5, linewidth = 0.5, color=c[j], fill=False)
				# 	ax1.add_patch(circle2)
				# 	circle3 = plt.Circle((self.snakes3[i,0], self.snakes3[i,1]), 0.2, alpha = 0.5, linewidth = 0.5, color=c[j], fill=False)
				# 	ax1.add_patch(circle3)
				# 	circle4 = plt.Circle((self.snakes4[i,0], self.snakes4[i,1]), 0.2, alpha = 0.5, linewidth = 0.5, color=c[j], fill=False)
				# 	ax1.add_patch(circle4)
				# 	circle5 = plt.Circle((self.fakes_target[i,0], self.fakes_target[i,1]), 0.2, alpha = 0.5, linewidth = 0.5, color=c[j], fill=False)
				# 	ax1.add_patch(circle5)
					
				# 	ax1.plot([self.fakes[i,0],self.snakes2[i,0]], [self.fakes[i,1],self.snakes2[i,1]],  c = c[j], alpha=0.85, linewidth = 0.75)
				# 	ax1.plot([self.snakes2[i,0],self.snakes3[i,0]], [self.snakes2[i,1],self.snakes3[i,1]], c = c[j], alpha=0.85, linewidth = 0.75)
				# 	ax1.plot([self.snakes3[i,0],self.snakes4[i,0]], [self.snakes3[i,1],self.snakes4[i,1]], c = c[j], alpha=0.85, linewidth = 0.75)
				# 	ax1.plot([self.snakes4[i,0],self.fakes_target[i,0]], [self.snakes4[i,1],self.fakes_target[i,1]], c = c[j], alpha=0.85, linewidth = 0.75)


					# ax1.plot([self.fakes[i,0],self.fakes_target[i,0]], [self.fakes[i,1],self.fakes_target[i,1]], c = 'k', alpha=0.85, linewidth = 0.75)

			pdf.savefig(fig1)
			plt.close(fig1)
			# if self.total_count.numpy() == 15:
			# 	exit(0)