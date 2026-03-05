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
import tensorflow_probability as tfp
tfd = tfp.distributions


######## OLD NEED CLEANUP ########
class ARCH_gmm2():

	def __init__(self):
		print("CREATING ARCH_deq_gmm2 CLASS")
		return

	def generator_model_gmm2_base(self):

		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn1 = tf.random_normal_initializer(mean = 3.10, stddev = 0.01)
		init_fn = tf.function(init_fn, autograph=False)
		init_fn1 = tf.function(init_fn1, autograph=False)


		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc1 = tf.keras.layers.Dense(int(self.latent_dims*5), kernel_initializer=init_fn, use_bias = True, bias_initializer = init_fn1)(inputs)
		enc1 = tf.keras.layers.ReLU()(enc1)

		# enc11 = tf.keras.layers.Dense(int(self.latent_dims*256), kernel_initializer=init_fn, use_bias = True, activation = 'sigmoid')(enc1)
		# enc11 = tf.keras.layers.ReLU()(enc11)
		
		# enc12 = tf.keras.layers.Dense(int(self.latent_dims*64), kernel_initializer=init_fn, use_bias = True, activation = 'sigmoid')(enc11)
		# enc12 = tf.keras.layers.ReLU()(enc12)

		enc2 = tf.keras.layers.Dense(int(self.latent_dims*10), kernel_initializer=init_fn, use_bias = True, bias_initializer = init_fn1)(enc1)
		enc2 = tf.keras.layers.ReLU()(enc2)

		enc3 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn, use_bias = True, bias_initializer = init_fn1)(enc2)
		# enc3 = tf.keras.layers.ReLU()(enc3)

		enc4 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True, bias_initializer = init_fn1)(enc3)
		# enc4 = tf.keras.layers.ReLU(max_value = 1.)(enc4)

		model = tf.keras.Model(inputs = inputs, outputs = enc4)

		return model


	def generator_model_gmm2_AE(self):

		init_fn = tf.keras.initializers.glorot_uniform()
		# init_fn = tf.random_normal_initializer(mean = 1.0, stddev = 5.0)
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc1 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn, use_bias = True)(inputs)
		# enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn, use_bias = True)(enc1)
		# enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn, use_bias = True)(enc2)
		# enc3 = tf.keras.layers.LeakyReLU()(enc3)

		enc4 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True)(enc3)
		# enc3 = tf.keras.layers.ReLU()(enc3)

		model = tf.keras.Model(inputs = inputs, outputs = enc4)

		return model

	def generator_model_gmm2_Cycle(self):

		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc1 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(inputs)
		# enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn)(enc1)
		# enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(enc2)
		# enc3 = tf.keras.layers.LeakyReLU()(enc3)

		enc4 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True)(enc3)
		# enc3 = tf.keras.layers.ReLU()(enc3)


		encoded = tf.keras.Input(shape=(self.latent_dims,))

		dec1 = tf.keras.layers.Dense(int((self.latent_dims)*2), kernel_initializer=init_fn)(encoded)
		# dec1 = tf.keras.layers.LeakyReLU()(dec1)

		dec2 = tf.keras.layers.Dense(int((self.latent_dims)*4), kernel_initializer=init_fn)(dec1)
		# dec2 = tf.keras.layers.LeakyReLU()(dec2)

		dec3 = tf.keras.layers.Dense(int((self.latent_dims)*2), kernel_initializer=init_fn)(dec2)
		# dec3 = tf.keras.layers.LeakyReLU()(dec3)

		out = tf.keras.layers.Dense(self.noise_dims, kernel_initializer=init_fn)(dec3)

		model = tf.keras.Model(inputs = inputs, outputs = enc4)
		self.generator_dec = tf.keras.Model(inputs = encoded, outputs = out)

		print("\n\n GENERATOR DECODER MODEL: \n\n")
		print(self.generator_dec.summary())

		return model


	def encoder_model_gmm2_AE(self):  # FOR BASE AE

		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.output_size,))

		enc0 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn, activation ='tanh')(inputs)
		# enc0 = tf.keras.layers.Dropout(0.5)(enc0)
		enc0 = tf.keras.layers.LeakyReLU()(enc0)

		enc1 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn, activation ='tanh')(enc0)
		# enc1 = tf.keras.layers.Dropout(0.5)(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn, activation ='tanh')(enc1)
		# enc2 = tf.keras.layers.Dropout(0.5)(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn)(enc2)
		# enc3 = tf.keras.layers.LeakyReLU()(enc3)

		encoded = tf.keras.Input(shape=(self.latent_dims,))

		dec1 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(encoded)
		# dec1 = tf.keras.layers.Dropout(0.5)(dec1)
		dec1 = tf.keras.layers.LeakyReLU()(dec1)

		dec2 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn)(dec1)
		# dec2 = tf.keras.layers.Dropout(0.5)(dec2)
		dec2 = tf.keras.layers.LeakyReLU()(dec2)

		dec3 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(dec2)
		# dec3 = tf.keras.layers.Dropout(0.5)(dec3)
		dec3 = tf.keras.layers.LeakyReLU()(dec3)

		out = tf.keras.layers.Dense(int(self.output_size), kernel_initializer=init_fn)(dec3)
		# out_enc = tf.keras.layers.LeakyReLU()(out_enc)

		self.Encoder = tf.keras.Model(inputs = inputs, outputs = enc3)
		self.Decoder = tf.keras.Model(inputs = encoded, outputs = out)
		
		return self.Encoder

	def encoder_model_gmm2_Cycle(self):  # FOR BASE AE

		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.output_size,))

		enc0 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(inputs)
		# enc0 = tf.keras.layers.Dropout(0.5)(enc0)
		enc0 = tf.keras.layers.LeakyReLU()(enc0)

		enc1 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn)(enc0)
		# enc1 = tf.keras.layers.Dropout(0.5)(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(enc1)
		# enc2 = tf.keras.layers.Dropout(0.5)(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn)(enc2)
		# enc3 = tf.keras.layers.LeakyReLU()(enc3)

		encoded = tf.keras.Input(shape=(self.latent_dims,))

		dec1 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(encoded)
		# dec1 = tf.keras.layers.Dropout(0.5)(dec1)
		dec1 = tf.keras.layers.LeakyReLU()(dec1)

		dec2 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn)(dec1)
		# dec2 = tf.keras.layers.Dropout(0.5)(dec2)
		dec2 = tf.keras.layers.LeakyReLU()(dec2)

		dec3 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(dec2)
		# dec3 = tf.keras.layers.Dropout(0.5)(dec3)
		dec3 = tf.keras.layers.LeakyReLU()(dec3)

		out = tf.keras.layers.Dense(int(self.output_size), kernel_initializer=init_fn, use_bias=False)(dec3)
		# out_enc = tf.keras.layers.LeakyReLU()(out_enc)

		self.Encoder = tf.keras.Model(inputs = inputs, outputs = enc3)
		self.Decoder = tf.keras.Model(inputs = encoded, outputs = out)
		
		return self.Encoder

	def discriminator_model_gmm2_base(self):
		inputs = tf.keras.Input(shape=(self.latent_dims,)) #used to be self.N
		cos_terms = tf.keras.layers.Dense(self.L, activation=tf.math.cos, use_bias = False)(inputs)
		sin_terms = tf.keras.layers.Dense(self.L, activation=tf.math.sin, use_bias = False)(inputs)
		cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms)
		sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)
		Out = tf.keras.layers.Add()([cos_sum, sin_sum])
		# Out = tf.keras.layers.Activation(activation = 'sigmoid')(Out)
		model = tf.keras.Model(inputs=inputs, outputs= Out)
		return model

	def discriminator_model_gmm2_AE(self):
		inputs = tf.keras.Input(shape=(self.latent_dims,)) #used to be self.N
		cos_terms = tf.keras.layers.Dense(self.L, activation=tf.math.cos, use_bias = False)(inputs)
		sin_terms = tf.keras.layers.Dense(self.L, activation=tf.math.sin, use_bias = False)(inputs)
		cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms)
		sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)
		Out = tf.keras.layers.Add()([cos_sum, sin_sum])
		# Out = tf.keras.layers.Activation(activation = 'sigmoid')(Out)
		model = tf.keras.Model(inputs=inputs, outputs= Out)
		return model

	def discriminator_model_gmm2_Cycle(self):
		inputs = tf.keras.Input(shape=(self.latent_dims,)) #used to be self.N
		cos_terms = tf.keras.layers.Dense(self.L, activation=tf.math.cos, use_bias = False)(inputs)
		sin_terms = tf.keras.layers.Dense(self.L, activation=tf.math.sin, use_bias = False)(inputs)
		cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms)
		sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)
		Out = tf.keras.layers.Add()([cos_sum, sin_sum])
		model = tf.keras.Model(inputs=inputs, outputs= Out)
		return model


	def show_result_gmm2(self, images = None, num_epoch = 0, path = 'result.png', show = False, save = True):

		beta_c,beta_s = eval('self.Fourier_Series_Comp(self.fakes)')

		self.pdf.set_weights([self.Coeffs, self.Coeffs, self.alpha_c,np.array([0]), self.alpha_s])
		self.pgf.set_weights([self.Coeffs, self.Coeffs, beta_c ,np.array([0]), beta_s])

		basis = np.expand_dims(np.linspace(self.MIN, self.MAX, int(1e4), dtype=np.float32), axis = 1)
		disc = self.discriminator(basis,training = False)

		pd_vals_FS = self.pdf(basis,training=False)
		pg_vals_FS = self.pgf(basis,training=False)
		
		true_classifier = np.ones_like(basis)
		true_classifier[pd_vals_FS > pg_vals_FS] = 0

		if self.paper and (self.total_count.numpy() == 1 or self.total_count.numpy()%1000 == 0):
			np.save(path+'_disc.npy',np.array(disc))
			np.save(path+'_reals.npy',np.array(self.reals))
			np.save(path+'_fakes.npy',np.array(self.fakes))
			np.save(path+'_pd_FS.npy',np.array(pd_vals_FS))
			np.save(path+'_pg_FS.npy',np.array(pg_vals_FS))

		plt.rcParams.update({
			"pgf.texsystem": "pdflatex",
			"font.family": "DejaVu Sans",  
			"font.size":9,
			"font.serif": [], 
			"text.usetex": True,     # use inline math for ticks
			"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			"pgf.preamble": [
				 r"\usepackage[utf8x]{inputenc}",
				 r"\usepackage[T1]{fontenc}",
				 r"\usepackage{cmbright}",
				 ]
		})

		with PdfPages(path+'_Classifier.pdf', metadata={'author': 'ANON'}) as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim(bottom=-0.5,top=1.8)
			ax1.scatter(self.reals, np.zeros_like(self.reals), c='r', linewidth = 1.5, label='Real Data', marker = '.')
			ax1.scatter(images, np.zeros_like(images), c='g', linewidth = 1.5, label='Fake Data', marker = '.')
			ax1.plot(basis,disc, c='b', linewidth = 1.5, label='Discriminator')
			if self.total_count < 20:
				ax1.plot(basis,true_classifier,'c--', linewidth = 1.5, label='True Classifier')
			ax1.legend(loc = 'upper right')
			fig1.tight_layout()
			pdf.savefig(fig1)
			plt.close(fig1)

