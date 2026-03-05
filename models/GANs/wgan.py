from __future__ import print_function
import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
import tensorflow_probability as tfp
from matplotlib.backends.backend_pgf import PdfPages
tfd = tfp.distributions

import matplotlib.pyplot as plt
import math
import tensorflow as tf
from absl import app
from absl import flags

from gan_topics import *
# tf.keras.backend.set_floatx('float64')

'''***********************************************************************************
********** Baseline WGANs ************************************************************
***********************************************************************************'''
class WGAN_Base(GAN_Base):

	def __init__(self,FLAGS_dict):

		# self.KLD_flag = KLD_flag
		# self.KLD = []
		GAN_Base.__init__(self,FLAGS_dict)

		self.lambda_GP = 0.1 #100 for normal data, 0.1 for synth
		self.lambda_ALP = 10.0 #100 for normal data, 0.1 for synth
		self.lambda_LP = 0.1 #10 for normal? 0.1 for synth
		self.lambda_R1 = 0.1 # 0.1 for gmm. For rest, 0.5
		self.lambda_R2 = 0.1 # 0.1 for gmm. for rest, 0.5

		self.sobolev_lambda = tf.Variable((0.))

		self.sobolev_rho = 0.005

	#################################################################

	def create_optimizer(self):
		# with tf.device(self.device):
		if self.loss == 'GP' :
			self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=200, decay_rate=0.9, staircase=True)
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2) #Had it for JMLR. Now for ICML, no.
		elif self.loss == 'ALP' :
			self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=100, decay_rate=0.9, staircase=True)
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
		else:
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
		self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)

		if self.loss == 'sobolev':
			self.lambda_optimizer = tf.keras.optimizers.SGD(self.sobolev_rho)
		print("Optimizers Successfully made")	
		return	

	#################################################################

	def save_epoch_h5models(self):
		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
		self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return

	#################################################################

	# def test(self):
	# 	self.impath += '_Testing_'
	# 	for img_batch in self.train_dataset:
	# 		self.reals = img_batch
	# 		self.generate_and_save_batch(0)
	# 		return

	#################################################################

	def train_step(self,reals_all):
		for i in tf.range(self.Dloop):
			# with tf.device(self.device):
			noise = self.get_noise([self.batch_size, self.noise_dims])
			
			# print(self.charmap)
			if self.data == 'words':
				self.reals = np.array([[self.charmap[c.decode()] for c in l] for l in reals_all],dtype='int32')
			else:
				self.reals = reals_all

			# print(self.reals.shape)

			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as lambda_tape:
				lambda_tape.watch(self.sobolev_lambda)
				
				self.fakes = self.generator(noise, training=True)

				# print(self.fakes.shape)
				# exit(0)


				self.real_output = self.discriminator(self.reals, training = True)
				self.fake_output = self.discriminator(self.fakes, training = True)
				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.Disc_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))

			if self.loss == 'base':
				wt = []
				for w in self.discriminator.get_weights():
					w = tf.clip_by_value(w, -0.1,0.1) #0.01 for [0,1] data, 0.1 for [0,10]
					wt.append(w)
				self.discriminator.set_weights(wt)

			if self.arch == 'linSig' and self.loss == 'ELGP':
				w,b = self.discriminator.get_weights()
				w = w/tf.norm(w, ord = 'euclidean') #0.01 for [0,1] data, 0.1 for [0,10]
				self.discriminator.set_weights([w,b])

			if self.loss == 'sobolev':
				self.lambda_grad = lambda_tape.gradient(self.D_loss, self.sobolev_lambda)
				self.sobolev_lambda.assign_sub(-self.sobolev_rho * self.lambda_grad)

			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	# def train_step(self,reals_all):
	# 	assert tf.distribute.get_replica_context() is None
	# 	self.strategy.run(self.actual_train_step, args=(reals_all,))
	# 	return 

	#################################################################

	def loss_base(self):

		loss_fake = tf.reduce_mean(self.fake_output)

		loss_real = tf.reduce_mean(self.real_output) 

		self.D_loss = 1 * (-loss_real + loss_fake)

		self.G_loss = 1 * (loss_real - loss_fake)

	#################################################################

	def loss_ELGP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.ELgradient_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_GP * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def ELgradient_penalty(self):
		ip = tf.concat((self.reals,self.fakes),axis = 0)
		# with tf.GradientTape() as l:
			# l.watch(ip)
		with tf.GradientTape() as t:
			t.watch(ip)
			pred = self.discriminator(ip, training = True)
		grad = t.gradient(pred, [ip])
			# print(grad)
		# gradgrad = l.gradient(grad, [ip])
		# print(gradgrad)
		# self.autodiff_lap = self.gp = tf.reduce_sum(gradgrad, axis = 1)
		if self.data in ['g1']:
			slopes = tf.sqrt(tf.square(grad))
		elif self.data in [ 'g2', 'gmm8', 'gN']:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
		else:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
		self.gp = tf.reduce_mean((slopes**2 - 1.))
		# self.LapD_curr = tf.reduce_mean(pred*(1.-pred)*(1.-2*pred))
		return 

	#################################################################

	def loss_GP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_GP * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty(self):
		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		else:
			alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
		diff = tf.cast(self.fakes,dtype='float32') - tf.cast(self.reals,dtype='float32')
		inter = tf.cast(self.reals,dtype='float32') + (alpha * diff)
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])
		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
		else:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
		self.gp = tf.reduce_mean((slopes - 1.)**2)
		return 

	#################################################################

	def loss_sobolev(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.sobolev_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def sobolev_penalty(self):
		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		else:
			alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
		# diff = tf.cast(self.fakes,dtype='float32') - tf.cast(self.reals,dtype='float32')
		# inter = tf.cast(self.reals,dtype='float32') + (alpha * diff)
		with tf.GradientTape() as r_tape, tf.GradientTape() as f_tape :
			reals = tf.cast(self.reals,dtype='float32')
			fakes = tf.cast(self.fakes,dtype='float32')
			r_tape.watch(reals)
			f_tape.watch(fakes)
			pred_reals = self.discriminator(reals, training = True)
			pred_fakes = self.discriminator(fakes, training = True)
		grad_reals = r_tape.gradient(pred_reals, [reals])
		grad_fakes = f_tape.gradient(pred_fakes, [fakes])
		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			slopes_reals = tf.sqrt(tf.reduce_sum(tf.square(grad_reals), axis=[1]))
			slopes_fakes = tf.sqrt(tf.reduce_sum(tf.square(grad_fakes), axis=[1]))
		else:
			slopes_reals = tf.sqrt(tf.reduce_sum(tf.square(grad_reals), axis=[1, 2, 3]))
			slopes_fakes = tf.sqrt(tf.reduce_sum(tf.square(grad_fakes), axis=[1, 2, 3]))

		self.first_order_gp = 1. - (0.5*tf.reduce_mean(slopes_reals)+0.5*tf.reduce_mean(slopes_fakes))
		self.second_order_gp = 0.5*((0.5*tf.reduce_mean(slopes_reals)+0.5*tf.reduce_mean(slopes_fakes))-1.)**2

		self.gp = self.sobolev_lambda*self.first_order_gp + self.sobolev_rho*self.second_order_gp
		return 
	#################################################################

	def loss_R1(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty_R1()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_R1 * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty_R1(self):
		inter = tf.cast(self.reals,dtype='float32')
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])
		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])
		self.gp = tf.reduce_mean(slopes)
		return 

	#################################################################

	def loss_m1(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty_m1()

		self.D_loss = 1 * (-loss_real + loss_fake) + 0.001 * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty_m1(self):
		inter = tf.cast(tf.concat([self.reals,self.fakes],axis = 0),dtype='float32')
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])
		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])
		self.gp = tf.reduce_mean(slopes)
		return 

	#################################################################

	def loss_m2(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty_m2()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_R1 * 0.001 * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty_m2(self):
		inter = tf.cast(tf.concat([self.reals,self.fakes],axis = 0),dtype='float32')
		with tf.GradientTape() as t1:
			t1.watch(inter)
			with tf.GradientTape() as t2:
				t2.watch(inter)
				pred = self.discriminator(inter, training = True)
			grad1 = t2.gradient(pred, [inter])
		grad2 = t1.gradient(grad1, [inter])

		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			slopes = tf.reduce_sum(tf.square(grad2), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad2), axis=[1, 2, 3])
		self.gp = tf.reduce_mean(slopes)
		return 

	#################################################################

	def loss_m3(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty_m3()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_R1 * 0.005 * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty_m3(self):
		inter = tf.cast(tf.concat([self.reals,self.fakes],axis = 0),dtype='float32')
		with tf.GradientTape() as t1:
			t1.watch(inter)
			with tf.GradientTape() as t2:
				t2.watch(inter)
				with tf.GradientTape() as t3:
					t3.watch(inter)
					pred = self.discriminator(inter, training = True)
				grad1 = t3.gradient(pred, [inter])
			grad2 = t2.gradient(grad1, [inter])
		grad3 = t1.gradient(grad2, [inter])

		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			slopes = tf.reduce_sum(tf.square(grad3), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad3), axis=[1, 2, 3])
		self.gp = tf.reduce_mean(slopes)
		return 

	#################################################################

	def loss_m4(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty_m4()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_R1 * 0.001 * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty_m4(self):
		inter = tf.cast(tf.concat([self.reals,self.fakes],axis = 0),dtype='float32')
		with tf.GradientTape() as t1:
			t1.watch(inter)
			with tf.GradientTape() as t2:
				t2.watch(inter)
				with tf.GradientTape() as t3:
					t3.watch(inter)
					with tf.GradientTape() as t4:
						t4.watch(inter)
						pred = self.discriminator(inter, training = True)
					grad1 = t4.gradient(pred, [inter])
				grad2 = t3.gradient(grad1, [inter])
			grad3 = t2.gradient(grad2, [inter])
		grad4 = t1.gradient(grad3, [inter])

		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			slopes = tf.reduce_sum(tf.square(grad4), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad4), axis=[1, 2, 3])
		self.gp = tf.reduce_mean(slopes)
		return 


	#################################################################

	def loss_R2(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty_R2()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_R2 * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty_R2(self):
		inter = tf.cast(self.fakes,dtype='float32')
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])
		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])
		self.gp = tf.reduce_mean(slopes)
		return 

	#################################################################

	def loss_LP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.lipschitz_penalty()

		self.D_loss = -loss_real + loss_fake + self.lambda_LP * self.lp 
		self.G_loss = loss_real - loss_fake

	def lipschitz_penalty(self):
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py

		self.K = 1
		self.p = 2

		if self.data in ['g1', 'g2', 'gmm2', 'gmm8', 'u1', 'gN']:
			epsilon = tf.random.uniform([tf.shape(self.reals)[0], 1], 0.0, 1.0)
		else:
			epsilon = tf.random.uniform([tf.shape(self.reals)[0], 1, 1, 1], 0.0, 1.0)
		x_hat = epsilon * self.fakes + (1 - epsilon) * self.reals

		with tf.GradientTape() as t:
			t.watch(x_hat)
			D_vals = self.discriminator(x_hat, training = False)
		grad_vals = t.gradient(D_vals, [x_hat])

		# print(grad_vals)

		#### args.p taken from github as default p=2
		dual_p = 1 / (1 - 1 / self.p) if self.p != 1 else np.inf

		#gradient_norms = stable_norm(gradients, ord=dual_p)
		grad_norms = tf.norm(grad_vals, ord=dual_p, axis=1, keepdims=True)

		#### Default K = 1
		# lp = tf.maximum(gradient_norms - args.K, 0)
		self.lp = tf.reduce_mean(tf.maximum(grad_norms - self.K, 0)**2)
		# lp_loss = args.lambda_lp * reduce_fn(lp ** 2)

	#################################################################

	def loss_ALP(self):
		
		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.adversarial_lipschitz_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_ALP * self.alp 
		self.G_loss = 1 * (loss_real - loss_fake)


	def adversarial_lipschitz_penalty(self):
		def normalize(x, ord):
			return x / tf.maximum(tf.norm(x, ord=ord, axis=1, keepdims=True), 1e-10)
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py
		self.eps_min = 0.1
		self.eps_max = 10.0
		self.xi = 10.0
		self.ip = 1
		self.p = 2
		self.K = 5 #was 1. made 5 for G2 compares

		samples = tf.concat([self.reals, self.fakes], axis=0)
		if self.data in ['g1', 'g2', 'gmm2', 'gmm8', 'u1', 'gN']:
			noise = tf.random.uniform([tf.shape(samples)[0], 1], 0, 1, dtype=tf.dtypes.float32)
		else:
			noise = tf.random.uniform([tf.shape(samples)[0], 1, 1, 1], 0, 1, dtype=tf.dtypes.float32)

		eps = self.eps_min + (self.eps_max - self.eps_min) * noise

		with tf.GradientTape(persistent = True) as t:
			t.watch(samples)
			validity = self.discriminator(samples, training = False)

			d = tf.random.uniform(tf.shape(samples), 0, 1) - 0.5
			d = normalize(d, ord=2)
			t.watch(d)
			for _ in range(self.ip):
				samples_hat = tf.clip_by_value(samples + self.xi * d, clip_value_min=-1, clip_value_max=1)
				validity_hat = self.discriminator(samples_hat, training = False)
				dist = tf.reduce_mean(tf.abs(validity - validity_hat))
				grad = t.gradient(dist, [d])[0]
				# print(grad)
				d = normalize(grad, ord=2)
			r_adv = d * eps

		# print(r_adv, samples)

		samples_hat = tf.clip_by_value(samples + r_adv, clip_value_min=-1, clip_value_max=1)

		d_lp                   = lambda x, x_hat: tf.norm(x - x_hat, ord=self.p, axis=1, keepdims=True)
		d_x                    = d_lp

		samples_diff = d_x(samples, samples_hat)
		samples_diff = tf.maximum(samples_diff, 1e-10)

		validity      = self.discriminator(samples    , training = False)
		validity_hat  = self.discriminator(samples_hat, training = False)
		validity_diff = tf.abs(validity - validity_hat)

		alp = tf.maximum(validity_diff / samples_diff - self.K, 0)
		# alp = tf.abs(validity_diff / samples_diff - args.K)

		nonzeros = tf.greater(alp, 0)
		count = tf.reduce_sum(tf.cast(nonzeros, tf.float32))

		self.alp = tf.reduce_mean(alp**2)
		# alp_loss = args.lambda_lp * reduce_fn(alp ** 2)

	#####################################################################

	def loss_AE(self):
		mse = tf.keras.losses.MeanSquaredError()
		loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))#mse(self.reals, self.reals_dec)
		self.AE_loss =  loss_AE_reals  


'''***********************************************************************************
********** Baseline WGANs ************************************************************
***********************************************************************************'''
class WGAN_AE_Base(GAN_AE_Base):

	def __init__(self,FLAGS_dict):

		# self.KLD_flag = KLD_flag
		# self.KLD = []
		GAN_AE_Base.__init__(self,FLAGS_dict)

		self.lambda_GP = 0.1 #100 for normal data, 0.1 for synth
		self.lambda_ALP = 10.0 #100 for normal data, 0.1 for synth
		self.lambda_LP = 0.1 #10 for normal? 0.1 for synth
		self.lambda_R1 = 0.1 # 0.1 for gmm. For rest, 0.5
		self.lambda_R2 = 0.1 # 0.1 for gmm. for rest, 0.5

		self.kernel_dimension = 3
		self.epsilon = 1.0
	#################################################################

	def create_optimizer(self):
		# with tf.device(self.device):
		if self.loss == 'GP' :
			self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=200, decay_rate=0.9, staircase=True)
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2) #Had it for JMLR. Now for ICML, no.
		elif self.loss == 'ALP' :
			self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=100, decay_rate=0.9, staircase=True)
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
		else:
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
		self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)
		print("Optimizers Successfully made")	
		return	

	#################################################################

	def save_epoch_h5models(self):
		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
		self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return


	def train_step(self,reals_all):
		for i in tf.range(self.Dloop):
			# with tf.device(self.device):
			noise = self.get_noise([self.batch_size, self.noise_dims])
			
			self.reals = reals_all


			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
				
				self.reals_enc = self.Encoder(self.reals,training = False)
				self.fakes_enc = self.generator(noise, training=True)


				if self.loss == 'coulomb':
					with gen_tape.stop_recording(), disc_tape.stop_recording():
						self.pot_reals, self.pot_fakes = self.get_potentials(self.reals_enc, self.fakes_enc, self.kernel_dimension, self.epsilon)


					self.D_output_net_reals = self.discriminator(self.reals_enc, training = True)
					self.D_output_net_fakes = self.discriminator(self.fakes_enc, training = True)


				self.real_output = self.discriminator(self.reals_enc, training = True)
				self.fake_output = self.discriminator(self.fakes_enc, training = True)


				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.Disc_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))

			if self.loss == 'base':
				wt = []
				for w in self.discriminator.get_weights():
					w = tf.clip_by_value(w, -0.1,0.1) #0.01 for [0,1] data, 0.1 for [0,10]
					wt.append(w)
				self.discriminator.set_weights(wt)

			if self.arch == 'linSig' and self.loss == 'ELGP':
				w,b = self.discriminator.get_weights()
				w = w/tf.norm(w, ord = 'euclidean') #0.01 for [0,1] data, 0.1 for [0,10]
				self.discriminator.set_weights([w,b])

			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	# def train_step(self,reals_all):
	# 	assert tf.distribute.get_replica_context() is None
	# 	self.strategy.run(self.actual_train_step, args=(reals_all,))
	# 	return 

	#################################################################

	def loss_base(self):

		loss_fake = tf.reduce_mean(self.fake_output)

		loss_real = tf.reduce_mean(self.real_output) 

		self.D_loss = 1 * (-loss_real + loss_fake)

		self.G_loss = 1 * (loss_real - loss_fake)

	#################################################################

	def loss_ELGP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.ELgradient_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_GP * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def ELgradient_penalty(self):
		ip = tf.concat((self.reals_enc,self.fakes_enc),axis = 0)
		# with tf.GradientTape() as l:
			# l.watch(ip)
		with tf.GradientTape() as t:
			t.watch(ip)
			pred = self.discriminator(ip, training = True)
		grad = t.gradient(pred, [ip])
			# print(grad)
		# gradgrad = l.gradient(grad, [ip])
		# print(gradgrad)
		# self.autodiff_lap = self.gp = tf.reduce_sum(gradgrad, axis = 1)
		if self.data in ['g1']:
			slopes = tf.sqrt(tf.square(grad))
		elif self.data in [ 'g2', 'gmm8', 'gN']:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
		else:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
		self.gp = tf.reduce_mean((slopes**2 - 1.))
		# self.LapD_curr = tf.reduce_mean(pred*(1.-pred)*(1.-2*pred))
		return 

	#################################################################

	def loss_GP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_GP * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty(self):
		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		else:
			alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		diff = tf.cast(self.fakes_enc,dtype='float32') - tf.cast(self.reals_enc,dtype='float32')
		inter = tf.cast(self.reals_enc,dtype='float32') + (alpha * diff)
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])
		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
		else:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
		self.gp = tf.reduce_mean((slopes - 1.)**2)
		return 


	#################################################################

	def calculate_squared_distances(self,a, b):
		'''returns the squared distances between all elements in a and in b as a matrix
		of shape #a * #b'''
		na = tf.shape(a)[0]
		nb = tf.shape(b)[0]
		nas, nbs = list(a.shape), list(b.shape)
		a = tf.reshape(a, [na, 1, -1])
		b = tf.reshape(b, [1, nb, -1])
		a.set_shape([nas[0], 1, np.prod(nas[1:])])
		b.set_shape([1, nbs[0], np.prod(nbs[1:])])
		a = tf.tile(a, [1, nb, 1])
		b = tf.tile(b, [na, 1, 1])
		d = a-b
		return tf.reduce_sum(tf.square(d), axis=2)



	def plummer_kernel(self, a, b, dimension, epsilon):
		r = self.calculate_squared_distances(a, b)
		r += epsilon*epsilon
		f1 = dimension-2
		return tf.pow(r, -f1 / 2)


	def get_potentials(self, x, y, dimension, cur_epsilon):
		'''
		This is alsmost the same `calculate_potential`, but
			px, py = get_potentials(x, y)
		is faster than:
			px = calculate_potential(x, y, x)
			py = calculate_potential(x, y, y)
		because we calculate the cross terms only once.
		'''
		x_fixed = x
		y_fixed = y
		nx = tf.cast(tf.shape(x)[0], x.dtype)
		ny = tf.cast(tf.shape(y)[0], y.dtype)
		pk_xx = self.plummer_kernel(x_fixed, x, dimension, cur_epsilon)
		pk_yx = self.plummer_kernel(y, x, dimension, cur_epsilon)
		pk_yy = self.plummer_kernel(y_fixed, y, dimension, cur_epsilon)
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

	def loss_coulomb(self):

		mse = tf.keras.losses.MeanSquaredError()

		# mse_real = mse(self.real_output_net, self.real_output)
		# mse_fake = mse(self.fake_output_net, self.fake_output)

		self.D_loss = mse(self.pot_reals, self.D_output_net_reals) + mse(self.pot_fakes, self.D_output_net_fakes) 
		# loss_fake = tf.reduce_mean(tf.minimum(self.fake_output,0))
		# loss_real = tf.reduce_mean(tf.maximum(self.real_output,0))
		loss_fake_G = tf.reduce_mean(self.fake_output)
		loss_real_G = tf.reduce_mean(self.real_output)

		# print(self.fake_output,loss_fake)
		# print(self.real_output,loss_real)

		self.G_loss = -1*loss_fake_G


		# self.D_loss = -1 * (-loss_real + self.alpha*loss_fake)
		# self.G_loss = -1 * (self.beta*loss_real - loss_fake)

	#################################################################

	def loss_R1(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty_R1()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_R1 * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty_R1(self):
		inter = tf.cast(self.reals_enc,dtype='float32')
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])
		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		self.gp = tf.reduce_mean(slopes)
		return 

	#################################################################

	def loss_R2(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty_R2()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_R2 * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty_R2(self):
		inter = tf.cast(self.fakes_enc,dtype='float32')
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])
		if self.data in ['g1', 'g2', 'gmm8', 'gN']:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		self.gp = tf.reduce_mean(slopes)
		return 

	#################################################################

	def loss_LP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.lipschitz_penalty()

		self.D_loss = -loss_real + loss_fake + self.lambda_LP * self.lp 
		self.G_loss = loss_real - loss_fake

	def lipschitz_penalty(self):
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py

		self.K = 1
		self.p = 2

		if self.data in ['g1', 'g2', 'gmm2', 'gmm8', 'u1', 'gN']:
			epsilon = tf.random.uniform([tf.shape(self.reals_enc)[0], 1], 0.0, 1.0)
		else:
			epsilon = tf.random.uniform([tf.shape(self.reals_enc)[0], 1,], 0.0, 1.0)
		x_hat = epsilon * self.fakes_enc + (1 - epsilon) * self.reals_enc

		with tf.GradientTape() as t:
			t.watch(x_hat)
			D_vals = self.discriminator(x_hat, training = False)
		grad_vals = t.gradient(D_vals, [x_hat])

		# print(grad_vals)

		#### args.p taken from github as default p=2
		dual_p = 1 / (1 - 1 / self.p) if self.p != 1 else np.inf

		#gradient_norms = stable_norm(gradients, ord=dual_p)
		grad_norms = tf.norm(grad_vals, ord=dual_p, axis=1, keepdims=True)

		#### Default K = 1
		# lp = tf.maximum(gradient_norms - args.K, 0)
		self.lp = tf.reduce_mean(tf.maximum(grad_norms - self.K, 0)**2)
		# lp_loss = args.lambda_lp * reduce_fn(lp ** 2)

	#################################################################

	def loss_ALP(self):
		
		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.adversarial_lipschitz_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_ALP * self.alp 
		self.G_loss = 1 * (loss_real - loss_fake)


	def adversarial_lipschitz_penalty(self):
		def normalize(x, ord):
			return x / tf.maximum(tf.norm(x, ord=ord, axis=1, keepdims=True), 1e-10)
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py
		self.eps_min = 0.1
		self.eps_max = 10.0
		self.xi = 10.0
		self.ip = 1
		self.p = 2
		self.K = 5 #was 1. made 5 for G2 compares

		samples = tf.concat([self.reals_enc, self.fakes_enc], axis=0)
		if self.data in ['g1', 'g2', 'gmm2', 'gmm8', 'u1', 'gN']:
			noise = tf.random.uniform([tf.shape(samples)[0], 1], 0, 1, dtype=tf.dtypes.float32)
		else:
			noise = tf.random.uniform([tf.shape(samples)[0], 1], 0, 1, dtype=tf.dtypes.float32)

		eps = self.eps_min + (self.eps_max - self.eps_min) * noise

		with tf.GradientTape(persistent = True) as t:
			t.watch(samples)
			validity = self.discriminator(samples, training = False)

			d = tf.random.uniform(tf.shape(samples), 0, 1) - 0.5
			d = normalize(d, ord=2)
			t.watch(d)
			for _ in range(self.ip):
				samples_hat = tf.clip_by_value(samples + self.xi * d, clip_value_min=-1, clip_value_max=1)
				validity_hat = self.discriminator(samples_hat, training = False)
				dist = tf.reduce_mean(tf.abs(validity - validity_hat))
				grad = t.gradient(dist, [d])[0]
				# print(grad)
				d = normalize(grad, ord=2)
			r_adv = d * eps

		# print(r_adv, samples)

		samples_hat = tf.clip_by_value(samples + r_adv, clip_value_min=-1, clip_value_max=1)

		d_lp                   = lambda x, x_hat: tf.norm(x - x_hat, ord=self.p, axis=1, keepdims=True)
		d_x                    = d_lp

		samples_diff = d_x(samples, samples_hat)
		samples_diff = tf.maximum(samples_diff, 1e-10)

		validity      = self.discriminator(samples    , training = False)
		validity_hat  = self.discriminator(samples_hat, training = False)
		validity_diff = tf.abs(validity - validity_hat)

		alp = tf.maximum(validity_diff / samples_diff - self.K, 0)
		# alp = tf.abs(validity_diff / samples_diff - args.K)

		nonzeros = tf.greater(alp, 0)
		count = tf.reduce_sum(tf.cast(nonzeros, tf.float32))

		self.alp = tf.reduce_mean(alp**2)
		# alp_loss = args.lambda_lp * reduce_fn(alp ** 2)



'''***********************************************************************************
********** Baseline GMMNs ************************************************************
***********************************************************************************'''
class WGAN_GMMN(GAN_Base):

	def __init__(self,FLAGS_dict):

		# self.KLD_flag = KLD_flag
		# self.KLD = []
		GAN_Base.__init__(self,FLAGS_dict)

	#################################################################

	def create_optimizer(self):
		# with tf.device(self.device):
		self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
		print("Optimizers Successfully made")	
		return	

	#################################################################

	def save_epoch_h5models(self):
		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
		return

	#################################################################

	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer, \
							generator = self.generator, \
							total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return

	#################################################################

	def train_step(self,reals_all):
		# with tf.device(self.device):
		noise = self.get_noise([self.batch_size, self.noise_dims])
		self.reals = reals_all

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			
			self.fakes = self.generator(noise, training=True)

			if self.data in ['mnist', 'svhn']:
				self.reals = tf.reshape(self.reals, [self.reals.shape[0], self.reals.shape[1]*self.reals.shape[2]*self.reals.shape[3]])
				self.fakes = tf.reshape(self.fakes, [self.fakes.shape[0], self.fakes.shape[1]*self.fakes.shape[2]*self.fakes.shape[3]])

			eval(self.loss_func)

			self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	#################################################################

	def loss_RBFG(self):
		#### Code Courtest: https://github.com/siddharth-agrawal/Generative-Moment-Matching-Networks

		def makeScaleMatrix(num_gen, num_orig):

			# first 'N' entries have '1/N', next 'M' entries have '-1/M'
			s1 =  tf.constant(1.0 / num_gen, shape = [self.batch_size, 1])
			s2 = -tf.constant(1.0 / num_orig, shape = [self.batch_size, 1])
			return tf.concat([s1, s2], axis = 0)

		sigma = [1]#[1, 5,10,20,50]

		X = tf.concat([self.reals, self.fakes], axis = 0)
		# print(X)

		# dot product between all combinations of rows in 'X'
		XX = tf.matmul(X, tf.transpose(X))

		# dot product of rows with themselves
		X2 = tf.reduce_sum(X * X, 1, keepdims = True)

		# exponent entries of the RBF kernel (without the sigma) for each
		# combination of the rows in 'X'
		# -0.5 * (x^Tx - 2*x^Ty + y^Ty)
		exponent = XX - 0.5 * X2 - 0.5 * tf.transpose(X2)
		# print(exponent)

		# scaling constants for each of the rows in 'X'
		s = makeScaleMatrix(tf.cast(self.batch_size,'float32'), tf.cast(self.batch_size,'float32'))

		# scaling factors of each of the kernel values, corresponding to the
		# exponent values
		S = tf.matmul(s, tf.transpose(s))

		loss = 0

		# for each bandwidth parameter, compute the MMD value and add them all
		for i in range(len(sigma)):

			# kernel values for each combination of the rows in 'X' 
			kernel_val = tf.exp((1.0 / sigma[i]) * exponent)
			loss += tf.reduce_sum(S * kernel_val)
			# print(kernel_val)
			# print(loss)

		self.G_loss = tf.sqrt(loss)
		self.D_loss = tf.constant(0.0)

		return 

	#################################################################

	def loss_IMQ(self):
		###3 Code Courtesy https://github.com/hiwonjoon/wae-wgan/blob/master/wae_mmd.py

		n = tf.cast(self.batch_size,tf.float32)
		C_base = 2.*self.reals.shape[1]

		z = self.reals
		z_tilde = self.fakes

		z_dot_z = tf.matmul(z,z,transpose_b=True) #[B,B} matrix where its (i,j) element is z_i \dot z_j.
		z_tilde_dot_z_tilde = tf.matmul(z_tilde,z_tilde,transpose_b=True)
		z_dot_z_tilde = tf.matmul(z,z_tilde,transpose_b=True)

		dist_z_z = \
			(tf.expand_dims(tf.linalg.diag_part(z_dot_z),axis=1)\
				+ tf.expand_dims(tf.linalg.diag_part(z_dot_z),axis=0))\
			- 2*z_dot_z
		dist_z_tilde_z_tilde = \
			(tf.expand_dims(tf.linalg.diag_part(z_tilde_dot_z_tilde),axis=1)\
				+ tf.expand_dims(tf.linalg.diag_part(z_tilde_dot_z_tilde),axis=0))\
			- 2*z_tilde_dot_z_tilde
		dist_z_z_tilde = \
			(tf.expand_dims(tf.linalg.diag_part(z_dot_z),axis=1)\
				+ tf.expand_dims(tf.linalg.diag_part(z_tilde_dot_z_tilde),axis=0))\
			- 2*z_dot_z_tilde

		L_D = 0.
		#with tf.control_dependencies([
		#    tf.assert_non_negative(dist_z_z),
		#    tf.assert_non_negative(dist_z_tilde_z_tilde),
		#    tf.assert_non_negative(dist_z_z_tilde)]):

		for scale in [1.0]:
			C = tf.cast(C_base*scale,tf.float32)

			k_z = C / (C + dist_z_z + 1e-8)
			k_z_tilde = C / (C + dist_z_tilde_z_tilde + 1e-8)
			k_z_z_tilde = C / (C + dist_z_z_tilde + 1e-8)

			loss = 1/(n*(n-1))*tf.reduce_sum(k_z)\
				+ 1/(n*(n-1))*tf.reduce_sum(k_z_tilde)\
				- 2/(n*n)*tf.reduce_sum(k_z_z_tilde)

			L_D += loss

		self.G_loss = L_D
		self.D_loss = tf.constant(0.0)

	#####################################################################

	def loss_AE(self):
		mse = tf.keras.losses.MeanSquaredError()
		loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))#mse(self.reals, self.reals_dec)
		self.AE_loss =  loss_AE_reals  



'''***********************************************************************************
********** PolyWGAN (RBF-CoulombGAN) **************************************************
***********************************************************************************'''
class WGAN_PolyGAN(GAN_Base, RBFSolver):

	def __init__(self,FLAGS_dict):

		GAN_Base.__init__(self,FLAGS_dict)

		''' Set up the Fourier Series Solver common to WAEFR and WGAN-FS'''
		RBFSolver.__init__(self)

		self.postfix = {0: f'{0:3.0f}', 1: f'{0:2.4e}', 2: f'{0:2.4e}'}
		self.bar_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  Batch: {postfix[0]} ETA: {remaining}  Elapsed Time: {elapsed}  D_Loss: {postfix[1]}  G_Loss: {postfix[2]}'

		self.first_iteration_flag = 1

	def create_models(self):
		# with tf.device(self.device):
		self.total_count = tf.Variable(0,dtype='int64')
		self.generator = eval(self.gen_model)
		self.discriminator_RBF = self.discriminator_model_RBF()

		print("Model Successfully made")
		print("\n\n GENERATOR MODEL: \n\n")
		print(self.generator.summary())
		print("\n\n DISCRIMINATOR RBF: \n\n")
		print(self.discriminator_RBF.summary())

		if self.res_flag == 1 and self.resume != 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n GENERATOR MODEL: \n\n")
				self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR RBF: \n\n")
				self.discriminator_RBF.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return

	def create_optimizer(self):
		# with tf.device(self.device):
		self.lr_G_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=500, decay_rate=0.9, staircase=True)
		# self.G_optimizer = tf.keras.optimizers.SGD(self.lr_G) #Nadam
		self.G_optimizer = tf.keras.optimizers.RMSprop(self.lr_G) #Nadam
		self.D_optimizer = tf.keras.optimizers.RMSprop(self.lr_D) #Nadam
		print("Optimizers Successfully made")
		return


	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer, \
							generator = self.generator, \
							discriminator_RBF = self.discriminator_RBF, \
							total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator_RBF = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_RBF.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return


	def train_step(self,reals_all):

		for Diter in tf.range(self.Dloop):
			# with tf.device(self.device):
			noise = self.get_noise([self.batch_size, self.noise_dims])
			self.reals = reals_all
			if self.data in ['mnist']:
				self.reals += tfp.distributions.TruncatedNormal(loc=0., scale=0.1, low=-1.,high=1.).sample([self.batch_size, self.output_size, self.output_size, 1])

			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

				# print(self.reals.shape)
				# print(noise.shape)
				# exit(0)
				self.fakes = self.generator(noise, training=True)

				# print("Discriminator weighs list")
				# for w in self.discriminator_RBF.get_weights():
				# 	print(w.shape)
					
				self.real_output,self.lambda_x_terms_1 = self.discriminator_RBF(self.reals, training = True)
				self.fake_output,self.lambda_x_terms_2 = self.discriminator_RBF(self.fakes, training = True)

				# print(self.real_output, self.fake_output)
				with gen_tape.stop_recording():
					if self.total_count.numpy()%self.ODE_step == 0 or self.total_count.numpy() <= 2 :

						Centres, Weights, Lamb_Weights = self.find_rbf_centres_weights()
						# print(Centres, Weights, Lamb_Weights)
						self.discriminator_RBF.set_weights([Centres,-Weights,Centres,-Lamb_Weights])

						if self.first_iteration_flag:
							self.first_iteration_flag = 0 
							self.lamb = tf.constant(0.1)
							self.D_loss = self.G_loss = tf.constant(0)
							return

						self.find_lambda()

				self.divide_by_lambda()
				
				eval(self.loss_func)

				# if self.G_loss_counter == 20 and self.data not in ['mnist', 'g2', 'gmm8']:
				# 	self.G_loss_counter = 0
				# 	opt_m = self.rbf_n//2 if self.rbf_n%2==0 else (self.rbf_n//2)+1
				# 	if self.poly_case == 0 or self.poly_case == 2:
				# 		self.rbf_m = min(self.rbf_m+1,opt_m)
				# 	else:
				# 		self.rbf_m = max(self.rbf_m-1,opt_m)
				# 	print("dropping to m=",self.rbf_m)
				# 	self.discriminator_RBF = self.discriminator_model_RBF()
				# 	# self.rbf_m = self.rbf_n//2 if self.rbf_n%2==0 else self.rbf_n//2+1

				# self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator_RBF.trainable_variables)
				# # print(self.D_grads)
				# self.D_optimizer.apply_gradients(zip(self.D_grads, self.discriminator_RBF.trainable_variables))

				if Diter >= (self.Dloop - self.Gloop):
					self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
					# print(self.G_grads)
					self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))




	##### Brought here for Supplementary Experiments. Maybe comment out afterwards
	def train_centers(self):
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)   
			start = time.time()
			batch_count = tf.Variable(0,dtype='int64')
			start_time =0

			for image_batch,center_batch in zip(self.train_dataset,self.center_dataset):
				# print(image_batch.shape)
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_time = time.time()
				# with tf.device(self.device):
				self.train_step(image_batch,center_batch)
				self.eval_metrics()
						

				train_time = time.time()-start_time

				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():6.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())
				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy()))

				self.print_batch_outputs(epoch)

				# Save the model every SAVE_ITERS iterations
				if (self.total_count.numpy() % self.save_step.numpy()) == 0:
					if self.save_all:
						self.checkpoint.save(file_prefix = self.checkpoint_prefix)
					else:
						self.manager.save()

			if self.pbar_flag:
				bar.close()
				del bar
			tf.print('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
			self.save_epoch_h5models()


	def train_step_New(self,reals_all,centers_all):
		# with tf.device(self.device):
		noise = self.get_noise([self.batch_size, self.noise_dims])
		noise_centers = self.get_noise([self.N_centers, self.noise_dims])
		self.reals = reals_all
		self.real_centers = centers_all
		# if self.data in ['mnist'] and self.arch in ['dcgan']:
		# 	self.reals += tfp.distributions.TruncatedNormal(loc=0., scale=0.316, low=-1.,high=1.).sample([self.batch_size, self.output_size, self.output_size, 1])
		# 	self.real_centers += tfp.distributions.TruncatedNormal(loc=0., scale=0.316, low=-1.,high=1.).sample([self.N_centers, self.output_size, self.output_size, 1])

		with tf.GradientTape() as gen_tape:

			self.fakes = self.generator(noise, training=True)
			

			# print("Discriminator weighs list")
			# for w in self.discriminator_RBF.get_weights():
			# 	print(w.shape)
				
			self.real_output,self.lambda_x_terms_1 = self.discriminator_RBF(self.reals, training = True)
			self.fake_output,self.lambda_x_terms_2 = self.discriminator_RBF(self.fakes, training = True)

			# print(self.real_output, self.fake_output)
			with gen_tape.stop_recording():
				# _, Weights, Lamb_Weights = self.find_rbf_centres_weights()
				self.fake_centers = self.generator(noise_centers, training = False)
				C_d = self.real_centers
				C_g = self.fake_centers
				# print(C_d,C_g)
				if self.data not in ['g2','gmm8', 'gN', 'gmmN']:
					C_d = tf.reshape(C_d, [C_d.shape[0], C_d.shape[1]*C_d.shape[2]*C_d.shape[3]])
					C_g = tf.reshape(C_g, [C_g.shape[0], C_g.shape[1]*C_g.shape[2]*C_g.shape[3]])
				Centres = np.concatenate((C_d,C_g), axis = 0)

				self.alpha = self.beta = 1
				D_d = (-1/C_d.shape[0])*np.ones([self.N_centers])
				D_g = (1/(C_g.shape[0]))*np.ones([self.N_centers])
				W_lamb = 1*tf.ones_like(D_d)

				Weights = np.concatenate((D_d,D_g), axis = 0)
				Lamb_Weights = np.concatenate((W_lamb,W_lamb), axis = 0)

				self.discriminator_RBF.set_weights([Centres,Weights,Centres,Lamb_Weights])
				self.find_lambda()

			if self.data not in ['gmm8','g2'] or self.rbf_m>1:
				self.divide_by_lambda()
			
			eval(self.loss_func)

			if self.G_loss_counter == 20 and self.data not in ['mnist', 'g2', 'gmm8']:
				self.G_loss_counter = 0
				opt_m = self.rbf_n//2 if self.rbf_n%2==0 else (self.rbf_n//2)+1
				if self.poly_case == 0 or self.poly_case == 2:
					self.rbf_m = min(self.rbf_m+1,opt_m)
				else:
					self.rbf_m = max(self.rbf_m-1,opt_m)
				print("dropping to m=",self.rbf_m)
				self.discriminator_RBF = self.discriminator_model_RBF()
				# self.rbf_m = self.rbf_n//2 if self.rbf_n%2==0 else self.rbf_n//2+1

			self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
			# print(self.G_grads)
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))

	def loss_RBF(self):
		# loss_fake = tf.reduce_mean(tf.minimum(self.fake_output,0))
		# loss_real = tf.reduce_mean(tf.maximum(self.real_output,0))
		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)

		# print(self.fake_output,loss_fake)
		# print(self.real_output,loss_real)

		### s=Used to be. Not for TMLR Resub
		self.D_loss = 1 * (-loss_real + self.alpha*loss_fake)
		if (2*self.rbf_m - self.rbf_n) == 1 or (2*self.rbf_m - self.rbf_n) == 0:
			self.G_loss = -1*loss_fake
		elif (2*self.rbf_m - self.rbf_n) < 0:
			self.G_loss = -1*(loss_real - loss_fake)
		else:
			# self.G_loss = tf.maximum(-1*loss_fake,0.)
			self.G_loss = tf.nn.leaky_relu(-1*loss_fake,alpha=0.02) #was 0.02 for g2 and gmm8
			if self.G_loss <= 10**(-50):
				self.G_loss_counter += 1

		self.D_loss = -1 * (-loss_real + loss_fake)
		self.G_loss = -1 * (loss_real - loss_fake)


		# self.D_loss = -1 * (-loss_real + self.alpha*loss_fake)
		# self.G_loss = -1 * (self.beta*loss_real - loss_fake)


'''***********************************************************************************
********** WGAN ELEGANT WITH LATENT **************************************************
***********************************************************************************'''
class WGAN_AE_SnakeGAN(GAN_AE_Base, SnakeSolver):

	def __init__(self,FLAGS_dict):

		GAN_AE_Base.__init__(self,FLAGS_dict)

		''' Set up the Fourier Series Solver common to WAEFR and WGAN-FS'''
		# RBFSolver.__init__(self)
		SnakeSolver.__init__(self)

		self.postfix = {0: f'{0:3.0f}', 1: f'{0:2.4e}', 2: f'{0:2.4e}'}
		self.bar_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  Batch: {postfix[0]} ETA: {remaining}  Elapsed Time: {elapsed}  D_Loss: {postfix[1]}  G_Loss: {postfix[2]}'

		self.first_iteration_flag = 1

	def create_models(self):
		# with tf.device(self.device):
		self.total_count = tf.Variable(0,dtype='int64')
		self.generator = eval(self.gen_model)
		self.discriminator_RBF = self.discriminator_model_RBF()
		self.Encoder = tf.keras.models.load_model("logs/AE_Models/"+self.data+"_Encoder.h5")
		self.Decoder = tf.keras.models.load_model("logs/AE_Models/"+self.data+"_Decoder.h5")


		print("Model Successfully made")
		print("\n\n GENERATOR MODEL: \n\n")
		print(self.generator.summary())
		print("\n\n DISCRIMINATOR RBF: \n\n")
		print(self.discriminator_RBF.summary())
		print("\n\n Encoder: \n\n")
		print(self.Encoder.summary())
		print("\n\n Decoder: \n\n")
		print(self.Decoder.summary())

		if self.res_flag == 1 and self.resume != 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n GENERATOR MODEL: \n\n")
				self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR RBF: \n\n")
				self.discriminator_RBF.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n ENCODER MODEL: \n\n")
				self.Encoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DECODER MODEL: \n\n")
				self.Decoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return

	def create_optimizer(self):
		# with tf.device(self.device):
		self.lr_G_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=500, decay_rate=0.9, staircase=True)
		self.G_optimizer = tf.keras.optimizers.SGD(self.lr_G) #Nadam
		print("Optimizers Successfully made")
		return


	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer, \
							generator = self.generator, \
							Encoder = self.Encoder, \
							Decoder = self.Decoder, \
							total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					# self.discriminator_RBF = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_RBF.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return


	def train_step(self,reals_all):


		# with tf.device(self.device):
		noise = self.get_noise([self.batch_size, self.noise_dims])
		reference_noise = self.get_noise([self.batch_size, self.noise_dims])
		self.reals = reals_all

		with tf.GradientTape() as gen_tape:

			self.reals_enc = self.Encoder(self.reals,training = False)
			self.fakes_enc = self.generator(noise, training=True)

			self.reference_fakes_enc = self.generator(reference_noise, training = False)

			with gen_tape.stop_recording():
				if self.snake_kind == 'o':
					self.fakes_target = self.snake_flow_o(self.reals_enc,self.fakes_enc,self.reference_fakes_enc)
				else:
					self.fakes_target = self.snake_flow_uo(self.reals_enc,self.fakes_enc,self.reference_fakes_enc)

			
			eval(self.loss_func)

			self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
			# print(self.G_grads)
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	def loss_snake(self):
		mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM)
		# print(self.fakes.shape, self.fakes_target.shape)
		L1 = 0.75*tf.reduce_mean(tf.abs(self.fakes_enc - self.fakes_target))
		L2 = 0.25*(mse(self.fakes_target, self.fakes_enc))
		self.D_loss = self.G_loss = 1*(L1 + L2)



'''***********************************************************************************
********** Snake GANs **************************************************
***********************************************************************************'''
class WGAN_SnakeGAN(GAN_Base, SnakeSolver):

	def __init__(self,FLAGS_dict):

		GAN_Base.__init__(self,FLAGS_dict)

		''' Set up the Fourier Series Solver common to WAEFR and WGAN-FS'''
		# RBFSolver.__init__(self)
		SnakeSolver.__init__(self)

		self.postfix = {0: f'{0:3.0f}', 1: f'{0:2.4e}', 2: f'{0:2.4e}'}
		self.bar_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  Batch: {postfix[0]} ETA: {remaining}  Elapsed Time: {elapsed}  D_Loss: {postfix[1]}  G_Loss: {postfix[2]}'

		self.first_iteration_flag = 1

	def create_models(self):
		# with tf.device(self.device):
		self.total_count = tf.Variable(0,dtype='int64')
		self.generator = eval(self.gen_model)
		self.discriminator_RBF = self.discriminator_model_RBF()

		print("Model Successfully made")
		print("\n\n GENERATOR MODEL: \n\n")
		print(self.generator.summary())
		print("\n\n DISCRIMINATOR RBF: \n\n")
		print(self.discriminator_RBF.summary())

		if self.res_flag == 1 and self.resume != 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n GENERATOR MODEL: \n\n")
				self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR RBF: \n\n")
				self.discriminator_RBF.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return

	def create_optimizer(self):
		# with tf.device(self.device):
		self.lr_G_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=500, decay_rate=0.9, staircase=True)
		self.G_optimizer = tf.keras.optimizers.SGD(self.lr_G) #Nadam
		print("Optimizers Successfully made")
		return


	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer, \
							generator = self.generator, \
							total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					# self.discriminator_RBF = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_RBF.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return


	def train_step(self,reals_all):


		# with tf.device(self.device):
		noise = self.get_noise([self.batch_size, self.noise_dims])
		reference_noise = self.get_noise([self.batch_size, self.noise_dims])
		self.reals = reals_all
		if self.data in ['mnist']:
			self.reals += tfp.distributions.TruncatedNormal(loc=0., scale=0.1, low=-1.,high=1.).sample([self.batch_size, self.output_size, self.output_size, 1])

		with tf.GradientTape() as gen_tape:

			self.fakes = self.generator(noise, training=True)
			self.reference_fakes = self.generator(reference_noise, training = False)

			# print("Discriminator weighs list")
			# for w in self.discriminator_RBF.get_weights():
			# 	print(w.shape)
				
			# self.real_output,self.lambda_x_terms_1 = self.discriminator_RBF(self.reals, training = True)
			with gen_tape.stop_recording():
				if self.snake_kind == 'o':
					self.fakes_target = self.snake_flow_o(self.reals,self.fakes,self.reference_fakes)
				else:
					self.fakes_target = self.snake_flow_uo(self.reals,self.fakes,self.reference_fakes)

			# print(self.real_output, self.fake_output)
			# with gen_tape.stop_recording():
			# 	Centres, Weights, Lamb_Weights = self.find_rbf_centres_weights()
			# 	self.discriminator_RBF.set_weights([Centres,Weights,Centres,Lamb_Weights])

			# 	if self.first_iteration_flag:
			# 		self.first_iteration_flag = 0 
			# 		self.lamb = tf.constant(0.1)
			# 		self.D_loss = self.G_loss = tf.constant(0)
			# 		return

			# 		self.find_lambda()

			# self.divide_by_lambda()
			
			eval(self.loss_func)

			# if self.G_loss_counter == 20 and self.data not in ['mnist', 'g2', 'gmm8']:
			# 	self.G_loss_counter = 0
			# 	opt_m = self.rbf_n//2 if self.rbf_n%2==0 else (self.rbf_n//2)+1
			# 	if self.poly_case == 0 or self.poly_case == 2:
			# 		self.rbf_m = min(self.rbf_m+1,opt_m)
			# 	else:
			# 		self.rbf_m = max(self.rbf_m-1,opt_m)
			# 	print("dropping to m=",self.rbf_m)
			# 	self.discriminator_RBF = self.discriminator_model_RBF()
			# 	# self.rbf_m = self.rbf_n//2 if self.rbf_n%2==0 else self.rbf_n//2+1

			self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
			# print(self.G_grads)
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	def loss_snake(self):
		mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM)
		# print(self.fakes.shape, self.fakes_target.shape)
		L1 = 0.75*tf.reduce_mean(tf.abs(self.fakes - self.fakes_target))
		L2 = 0.25*(mse(self.fakes_target, self.fakes))
		self.D_loss = self.G_loss = 1*(L1 + L2)



'''***********************************************************************************
********** Coulomb GANs **************************************************
***********************************************************************************'''
class WGAN_CoulombGAN(GAN_Base, RBFSolver):
	def __init__(self,FLAGS_dict):

		GAN_Base.__init__(self,FLAGS_dict)

		''' Set up the Fourier Series Solver common to WAEFR and WGAN-FS'''
		RBFSolver.__init__(self)

		self.postfix = {0: f'{0:3.0f}', 1: f'{0:2.4e}', 2: f'{0:2.4e}'}
		self.bar_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  Batch: {postfix[0]} ETA: {remaining}  Elapsed Time: {elapsed}  D_Loss: {postfix[1]}  G_Loss: {postfix[2]}'

		self.first_iteration_flag = 1

		self.kernel_dimension = 3
		self.epsilon = 1.0

	def create_models(self):
		# with tf.device(self.device):
		self.total_count = tf.Variable(0,dtype='int64')
		self.generator = eval(self.gen_model)
		self.discriminator = eval(self.disc_model)
		self.discriminator_RBF = self.discriminator_model_RBF()

		print("Model Successfully made")
		print("\n\n GENERATOR MODEL: \n\n")
		print(self.generator.summary())
		print("\n\n DISCRIMINATOR MODEL: \n\n")
		print(self.discriminator.summary())
		print("\n\n DISCRIMINATOR RBF: \n\n")
		print(self.discriminator_RBF.summary())

		if self.res_flag == 1 and self.resume != 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n GENERATOR MODEL: \n\n")
				self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
				self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR RBF: \n\n")
				self.discriminator_RBF.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return

	def create_optimizer(self):
		# with tf.device(self.device):
		self.lr_G_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=500, decay_rate=0.9, staircase=True)
		self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G) #Nadam or #SGD
		self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D) #Nadam or #SGD
		print("Optimizers Successfully made")
		return


	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer, \
							generator = self.generator, \
							discriminator_RBF = self.discriminator_RBF, \
							total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator_RBF = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_RBF.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return





	def train_step(self,reals_all):

		for i in tf.range(self.Dloop):
			# with tf.device(self.device):
			noise = self.get_noise([self.batch_size, self.noise_dims])
			self.reals = reals_all
			if self.data in ['mnist']:
				self.reals += tfp.distributions.TruncatedNormal(loc=0., scale=0.1, low=-1.,high=1.).sample([self.batch_size, self.output_size, self.output_size, 1])

			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

				self.fakes = self.generator(noise, training=True)

				with gen_tape.stop_recording(), disc_tape.stop_recording():
					self.pot_reals, self.pot_fakes = self.get_potentials(self.reals, self.fakes, self.kernel_dimension, self.epsilon)
		

				self.D_output_net_reals = self.discriminator(self.reals, training = True)
				self.D_output_net_fakes = self.discriminator(self.fakes, training = True)

				self.real_output_G = self.discriminator(self.reals, training = True)
				self.fake_output_G = self.discriminator(self.fakes, training = True)
				
				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.Disc_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))


			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	def calculate_squared_distances(self,a, b):
		'''returns the squared distances between all elements in a and in b as a matrix
		of shape #a * #b'''
		na = tf.shape(a)[0]
		nb = tf.shape(b)[0]
		nas, nbs = list(a.shape), list(b.shape)
		a = tf.reshape(a, [na, 1, -1])
		b = tf.reshape(b, [1, nb, -1])
		a.set_shape([nas[0], 1, np.prod(nas[1:])])
		b.set_shape([1, nbs[0], np.prod(nbs[1:])])
		a = tf.tile(a, [1, nb, 1])
		b = tf.tile(b, [na, 1, 1])
		d = a-b
		return tf.reduce_sum(tf.square(d), axis=2)



	def plummer_kernel(self, a, b, dimension, epsilon):
		r = self.calculate_squared_distances(a, b)
		r += epsilon*epsilon
		f1 = dimension-2
		return tf.pow(r, -f1 / 2)


	def get_potentials(self, x, y, dimension, cur_epsilon):
		'''
		This is alsmost the same `calculate_potential`, but
			px, py = get_potentials(x, y)
		is faster than:
			px = calculate_potential(x, y, x)
			py = calculate_potential(x, y, y)
		because we calculate the cross terms only once.
		'''
		x_fixed = x
		y_fixed = y
		nx = tf.cast(tf.shape(x)[0], x.dtype)
		ny = tf.cast(tf.shape(y)[0], y.dtype)
		pk_xx = self.plummer_kernel(x_fixed, x, dimension, cur_epsilon)
		pk_yx = self.plummer_kernel(y, x, dimension, cur_epsilon)
		pk_yy = self.plummer_kernel(y_fixed, y, dimension, cur_epsilon)
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



	def train_step_V1(self,reals_all):

		for i in tf.range(self.Dloop):
			# with tf.device(self.device):
			noise = self.get_noise([self.batch_size, self.noise_dims])
			self.reals = reals_all
			if self.data in ['mnist']:
				self.reals += tfp.distributions.TruncatedNormal(loc=0., scale=0.1, low=-1.,high=1.).sample([self.batch_size, self.output_size, self.output_size, 1])

			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

				self.fakes = self.generator(noise, training=True)

				# print("Discriminator weighs list")
				# for w in self.discriminator_RBF.get_weights():
				# 	print(w.shape)
				
				perturb_reals = tf.random.normal(self.reals.shape, mean = 0, stddev = 0.01)
				perturb_fakes = tf.random.normal(self.fakes.shape, mean = 0, stddev = 0.01)

				self.real_output,self.lambda_x_terms_1=self.discriminator_RBF(self.reals, training = True)
				self.fake_output,self.lambda_x_terms_2=self.discriminator_RBF(self.fakes, training = True)


				self.D_output_reals,_=self.discriminator_RBF(self.reals+perturb_reals, training = True)
				self.D_output_fakes,_=self.discriminator_RBF(self.fakes+perturb_fakes, training = True)
				# self.fake_output,self.lambda_x_terms_2=self.discriminator_RBF(samples_a training = True)

				# print(self.real_output, self.fake_output)
				with gen_tape.stop_recording():
					if self.total_count.numpy()%self.ODE_step == 0 or self.total_count.numpy() <= 2 :

						Centres, Weights, Lamb_Weights = self.find_rbf_centres_weights()
						self.discriminator_RBF.set_weights([Centres,Weights,Centres,Lamb_Weights])

						if self.first_iteration_flag:
							self.first_iteration_flag = 0 
							self.lamb = tf.constant(0.1)
							self.D_loss = self.G_loss = tf.constant(0)
							return

						self.find_lambda()

				self.divide_by_lambda()

				self.D_output_net_reals = self.discriminator(self.reals+perturb_reals, training = True)
				self.D_output_net_fakes = self.discriminator(self.fakes+perturb_fakes, training = True)

				self.real_output_G = self.discriminator(self.reals, training = True)
				self.fake_output_G = self.discriminator(self.fakes, training = True)
				
				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.Disc_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))


			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))



	def loss_RBF(self):

		mse = tf.keras.losses.MeanSquaredError()

		# mse_real = mse(self.real_output_net, self.real_output)
		# mse_fake = mse(self.fake_output_net, self.fake_output)

		self.D_loss = mse(self.pot_reals, self.D_output_net_reals) + mse(self.pot_fakes, self.D_output_net_fakes) 
		# loss_fake = tf.reduce_mean(tf.minimum(self.fake_output,0))
		# loss_real = tf.reduce_mean(tf.maximum(self.real_output,0))
		loss_fake_G = tf.reduce_mean(self.fake_output_G)
		loss_real_G = tf.reduce_mean(self.real_output_G)

		# print(self.fake_output,loss_fake)
		# print(self.real_output,loss_real)

		self.G_loss = -1*loss_fake_G


		# self.D_loss = -1 * (-loss_real + self.alpha*loss_fake)
		# self.G_loss = -1 * (self.beta*loss_real - loss_fake)

