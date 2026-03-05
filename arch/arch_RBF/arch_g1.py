from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pgf import FigureCanvasPgf
# matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
# from matplotlib.backends.backend_pgf import PdfPages
import tensorflow_probability as tfp
tfd = tfp.distributions


class ARCH_g1():	
	def __init__(self):
		return
		
	def generator_model_linear_g1(self):
		iden_init_fn = tf.keras.initializers.Identity()
		iden_init_fn = tf.function(iden_init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(1, use_bias=True, input_shape=(self.noise_dims,),kernel_initializer=iden_init_fn, bias_initializer = bias_init_fn))
		# model.add(layers.ReLU())
		return model

	def show_result_g1(self, images = None, num_epoch = 0, path = 'result.png', show = False, save = False):
		# print("Gaussian Stats : True mean {} True Sigma {}, Fake mean {} Fake Sigma {}".format(np.mean(self.reals), np.std(self.reals), np.mean(self.fakes), np.std(self.fakes) ))
		# Define a single scalar Normal distribution.
		pd_dist = tfd.TruncatedNormal(loc=np.mean(self.reals), scale=np.std(self.reals), low = -10., high = 20.)
		pg_dist = tfd.TruncatedNormal(loc=np.mean(self.fakes), scale=np.std(self.fakes), low = -10., high = 20.)

		# beta_c,beta_s = eval('self.Fourier_Series_Comp(self.fakes)')


		basis = np.expand_dims(np.linspace(self.MIN, self.MAX, int(1e4), dtype=np.float32), axis = 1)
		pd_vals = pd_dist.prob(basis)
		pg_vals = pg_dist.prob(basis)

		with tf.GradientTape() as t:
			basis = tf.cast(basis, dtype = 'float32')
			t.watch(basis)

			scale = 1.0
			disc,_ = self.discriminator_RBF(basis,training = False)
			scaled_disc = disc - min(disc)
			scaled_disc = (scaled_disc/max(abs(scaled_disc)))*scale*2.0
			scaled_disc -= scale

			grads = t.gradient(disc, basis)
			if self.rbf_m == 1:
				scale = 1.0
				# grads = grads - min(grads)
				grads = (grads/max(abs(grads)))*scale*2.0
				# grads -= scale

			if self.rbf_m == 2:
				scale = 2.5
				# grads = grads - min(grads)
				grads = (grads/max(abs(grads)))*scale*2.0
				# grads -= scale

			if self.rbf_m == 3:
				scale = 5.0
				# grads = grads - min(grads)
				grads = (grads/max(abs(grads)))*scale*2.0
				# grads -= scale

		# print(grads.shape)

		true_classifier = np.ones_like(basis)
		true_classifier[pd_vals < pg_vals] = 0

		if self.paper and (self.total_count.numpy() == 1 or self.total_count.numpy()%10 == 0):
			np.save(path+'_disc.npy',np.array(disc))
			np.save(path+'_grads.npy',np.array(grads))
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
			# ax1.set_ylim(bottom=-1.1,top=2.1)
			ax1.plot(basis,pd_vals, linewidth = 1.5, c='r')
			ax1.plot(basis,pg_vals, linewidth = 1.5, c='g')
			ax1.scatter(self.reals, np.zeros_like(self.reals), c='r', linewidth = 1.5, label=r'TARGET: $p_d(\mathbf{x})$', marker = '.')
			ax1.scatter(self.fakes, np.zeros_like(self.fakes), c='g', linewidth = 1.5, label=r'GENERATOR: $p_g(\mathbf{x})$', marker = '.')
			# ax1.fill_between(basis, integrand_KL, y2=0, where=None, interpolate=False, step=None)
			ax1.plot(basis,scaled_disc, c='b', linewidth = 1.5, label='Discriminator')
			ax1.plot(basis,grads, c='teal', linewidth = 1.5, label='Discriminator gradient')
			if self.total_count < 20:
				ax1.plot(basis,true_classifier,'c--', linewidth = 1.5, label='True Classifier')
			ax1.legend(loc = 'upper right')
			fig1.tight_layout()
			pdf.savefig(fig1)
			plt.close(fig1)


		if self.total_count.numpy()>=2:
			exit(0)

			