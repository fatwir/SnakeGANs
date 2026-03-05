from __future__ import print_function
import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
import tensorflow_probability as tfp
from matplotlib.backends.backend_pgf import PdfPages
tfd = tfp.distributions
from itertools import product as cart_prod

import matplotlib.pyplot as plt
import math
import tensorflow as tf
from absl import app
from absl import flags
from scipy.interpolate import interp1d

from gan_topics import *

###### NEEDS CLEANING #######
'''***********************************************************************************
********** WAEFeR ********************************************************************
***********************************************************************************'''
class WAE_ELeGANt(GAN_WAE, FourierSolver):

	def __init__(self,FLAGS_dict):

		GAN_WAE.__init__(self,FLAGS_dict)

		''' Set up the Fourier Series Solver common to WAEFR and WGAN-FS'''
		FourierSolver.__init__(self)
	

		if self.colab and (self.data in ['mnist', 'celeba', 'cifar10', 'svhn']):
			self.bar_flag = 0
		else:
			self.bar_flag = 1


	def create_models(self):
		# with self.strategy.scope():
		self.total_count = tf.Variable(0,dtype='int64')
		eval(self.encdec_model)
		self.discriminator_A = self.discriminator_model_FS_A()
		self.discriminator_A.set_weights([self.Coeffs])
		self.discriminator_B = self.discriminator_model_FS_B()
		

		print("Model Successfully made")

		print("\n\n ENCODER MODEL: \n\n")
		print(self.Encoder.summary())
		print("\n\n DECODER MODEL: \n\n")
		print(self.Decoder.summary())
		print("\n\n DISCRIMINATOR PART A MODEL: \n\n")
		print(self.discriminator_A.summary())
		print("\n\n DISCRIMINATOR PART B MODEL: \n\n")
		print(self.discriminator_B.summary())


		if self.res_flag == 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n ENCODER MODEL: \n\n")
				self.Encoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DECODER MODEL: \n\n")
				self.Decoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR PART A MODEL: \n\n")
				self.discriminator_A.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR PART B MODEL: \n\n")
				self.discriminator_B.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return

	def create_optimizer(self):

		# with self.strategy.scope():

		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=200, decay_rate=0.95, staircase=True)
		#### Added for IMCL rebuttal
		self.E_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Enc)
		self.D_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Dec)
		self.G_optimizer = tf.keras.optimizers.Nadam(self.lr_G)
		# self.G_optimizer = tf.keras.optimizers.Adam(lr_schedule)
		# self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D)
		print("Optimizers Successfully made")	
		return	

	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(E_optimizer = self.E_optimizer,
								 D_optimizer = self.D_optimizer,
								 G_optimizer = self.G_optimizer,
								 Encoder = self.Encoder,
								 Decoder = self.Decoder,
								 discriminator_A = self.discriminator_A,
								 discriminator_B = self.discriminator_B,
								 locs = self.locs,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					# self.Encoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Encoder.h5')
					self.Decoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Decoder.h5')
					# self.discriminator_A = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_A.h5')
					# self.discriminator_B = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_B.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
			


	def save_epoch_h5models(self):
		self.Encoder.save(self.checkpoint_dir + '/model_Encoder.h5', overwrite = True)
		self.Decoder.save(self.checkpoint_dir + '/model_Decoder.h5', overwrite = True)
		self.discriminator_A.save(self.checkpoint_dir + '/model_discriminator_A.h5', overwrite = True)
		self.discriminator_B.save(self.checkpoint_dir + '/model_discriminator_B.h5', overwrite = True)
		return


	def actual_pretrain_step_GAN(self,reals_all):

		## Actually Pretrain GAN. - Will make a sperate flag nd control if it does infact work out
		# with self.strategy.scope():
		self.fakes_enc = target_noise = self.get_noise(self.batch_size)
		self.reals = reals_all
		self.AE_loss = tf.constant(0)

		with tf.GradientTape() as G_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			# print(self.reals_enc.numpy())
			
			with G_tape.stop_recording():
				if self.total_count.numpy()%1 == 0:
					eval(self.DEQ_func)
					self.discriminator_B.set_weights([self.Gamma_c, self.Gamma_s, self.Tau_c, self.Tau_s])
					# self.discriminator_B.set_weights([self.Gamma_c,self.bias, self.Gamma_s, self.Tau_c, self.Tau_s])


			self.real_output, self.lambda_x_terms_1 = self.discriminator_B(self.discriminator_A(target_noise, training = True), training = True)
			self.fake_output, self.lambda_x_terms_2 = self.discriminator_B(self.discriminator_A(self.reals_enc, training = True), training = True)

			# self.find_and_divide_lambda()
			with G_tape.stop_recording():
				self.find_lambda()

			self.divide_by_lambda()
			
			eval(self.loss_func)

			self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
			# print("FS Grads",self.E_grads,"=========================================================")
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))

	# @tf.function
	def pretrain_step_GAN(self,reals_all):
		assert tf.distribute.get_replica_context() is None
		self.strategy.run(self.actual_pretrain_step_GAN, args=(reals_all,))
		return #self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
					

	def actual_pretrain_step_AE(self,reals_all):
		# with self.strategy.scope():
		self.reals = reals_all

		with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.reals_dec = self.Decoder(self.reals_enc, training = True)

			self.loss_AE()
			self.E_grads = enc_tape.gradient(self.AE_loss, self.Encoder.trainable_variables)
			self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
			self.E_optimizer.apply_gradients(zip(self.E_grads, self.Encoder.trainable_variables))
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))

			self.D_loss = self.G_loss = tf.constant(0.)

	# @tf.function
	def pretrain_step_AE(self,reals_all):
		assert tf.distribute.get_replica_context() is None
		self.strategy.run(self.actual_pretrain_step_AE, args=(reals_all,))
		return #self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
				

	#### Enc-Dec version added for ICML rebuttal. Uncomment only if needed
	def actual_train_step(self,reals_all):
		# with self.strategy.scope():
		self.fakes_enc = target_noise = self.get_noise(self.batch_size)
		self.reals = reals_all

		with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as G_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.reals_dec = self.Decoder(self.reals_enc, training = True)

			self.loss_AE()
			self.E_grads = enc_tape.gradient(self.AE_loss, self.Encoder.trainable_variables)
			self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
			# print("AE Grads",self.E_grads, "=================================================")
			self.E_optimizer.apply_gradients(zip(self.E_grads, self.Encoder.trainable_variables))
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))


			with G_tape.stop_recording():
				if self.total_count.numpy()%1 == 0:
					eval(self.DEQ_func)
					self.discriminator_B.set_weights([self.Gamma_c, self.Gamma_s, self.Tau_c, self.Tau_s])
					# self.discriminator_B.set_weights([self.Gamma_c,self.bias, self.Gamma_s, self.Tau_c, self.Tau_s])


			self.real_output, self.lambda_x_terms_1 = self.discriminator_B(self.discriminator_A(target_noise, training = True), training = True)
			self.fake_output, self.lambda_x_terms_2 = self.discriminator_B(self.discriminator_A(self.reals_enc, training = True), training = True)

			# self.find_and_divide_lambda()
			with G_tape.stop_recording():
				self.find_lambda()

			self.divide_by_lambda()
			eval(self.loss_func)

			self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
			# print("FS Grads",self.E_grads,"=========================================================")
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))

	# @tf.function
	def train_step(self,reals_all):
		assert tf.distribute.get_replica_context() is None
		self.strategy.run(self.actual_train_step, args=(reals_all,))
		return #self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
				

	def loss_FS(self):
		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output) 
		# used 0.1
		self.D_loss = (-loss_real + loss_fake) #+ self.AE_loss
		self.G_loss = (loss_real - loss_fake) #+ self.AE_loss


	def loss_AE(self):
		mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM)
		if self.data in ['celeba', 'bedroom', 'mnist', 'church', 'ukiyoe']:
			loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))
		elif self.data in [ 'svhn','cifar10']:
			loss_AE_reals = 0.75*tf.reduce_mean(tf.abs(self.reals - self.reals_dec)) + 0.25*(mse(self.reals, self.reals_dec))
		else:
			loss_AE_reals = mse(self.reals, self.reals_dec)

		self.AE_loss =  loss_AE_reals



###### NEEDS CLEANING #######
'''***********************************************************************************
********** WAEFeR ********************************************************************
***********************************************************************************'''
class WAE_PolyGAN(GAN_WAE, RBFSolver):

	def __init__(self,FLAGS_dict):

		GAN_WAE.__init__(self,FLAGS_dict)

		''' Set up the Fourier Series Solver common to WAEFR and WGAN-FS'''
		RBFSolver.__init__(self)

		self.first_iteration_flag = 1
	

		if self.colab and (self.data in ['mnist', 'celeba', 'cifar10', 'svhn', 'church', 'bedroom']):
			self.bar_flag = 0
		else:
			self.bar_flag = 1

		self.lambda_PGP = 100.


	def create_models(self):
		# with self.strategy.scope():
		self.total_count = tf.Variable(0,dtype='int64')
		eval(self.encdec_model)
		self.discriminator_RBF = self.discriminator_model_RBF()
		

		print("Model Successfully made")

		print("\n\n ENCODER MODEL: \n\n")
		print(self.Encoder.summary())
		print("\n\n DECODER MODEL: \n\n")
		print(self.Decoder.summary())
		print("\n\n DISCRIMINATOR RBF: \n\n")
		print(self.discriminator_RBF.summary())


		if self.res_flag == 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n ENCODER MODEL: \n\n")
				self.Encoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DECODER MODEL: \n\n")
				self.Decoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR RBF: \n\n")
				self.discriminator_RBF.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return

	def create_optimizer(self):

		# with self.strategy.scope():
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=100000, decay_rate=0.95, staircase=True)
		#### Added for IMCL rebuttal
		self.E_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Enc)
		self.D_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Dec)
		self.G_optimizer = tf.keras.optimizers.Nadam(self.lr_G)
		# self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D)
		if self.data in ['mnist']:
			decay_steps = 5000
			decay_rate = 0.95
		elif self.data in ['cifar10', 'svhn']:
			decay_steps = 5000
			decay_rate = 0.98
		elif self.data in ['celeba', 'church', 'bedroom', 'ukiyoe']:
			decay_steps = 20000
			decay_rate = 0.99



		if self.loss == 'RBF':

			self.Enc_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_AE_Enc, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
			self.Dec_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_AE_Dec, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
			self.G_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=5000, decay_rate=0.98, staircase=True)
			### Added for IMCL rebuttal
			if self.data == 'cifarrr10':
				self.E_optimizer = tf.keras.optimizers.Adam(self.Enc_lr_schedule)
				self.D_optimizer = tf.keras.optimizers.Adam(self.Dec_lr_schedule)
				self.G_optimizer = tf.keras.optimizers.Adam(self.G_lr_schedule)	
			else:
				self.E_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Enc)
				self.D_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Dec)
				self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G)			
		print("Optimizers Successfully made")	
		return	

	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(E_optimizer = self.E_optimizer,
								 D_optimizer = self.D_optimizer,
								 G_optimizer = self.G_optimizer,
								 Encoder = self.Encoder,
								 Decoder = self.Decoder,
								 discriminator_RBF = self.discriminator_RBF, \
								 locs = self.locs,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
				print("Model restored...")
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.Encoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Encoder.h5')
					self.Decoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Decoder.h5')
					# self.discriminator_RBF = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_RBF.h5')
					print("Model restored...")
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
			


	def save_epoch_h5models(self):
		self.Encoder.save(self.checkpoint_dir + '/model_Encoder.h5', overwrite = True)
		self.Decoder.save(self.checkpoint_dir + '/model_Decoder.h5', overwrite = True)
		self.discriminator_RBF.save(self.checkpoint_dir + '/model_discriminator_RBF.h5', overwrite = True)
		return


	def pretrain_step_GAN(self,reals_all):
		self.AE_loss = tf.constant(0.)
		## Actually Pretrain GAN. - Will make a sperate flag nd control if it does infact work out
		# with self.strategy.scope():
		self.fakes_enc = target_noise = self.get_noise(self.batch_size)
		self.reals = reals_all

		with tf.GradientTape() as G_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			# print(self.reals_enc.numpy())

			self.real_output,self.lambda_x_terms_1 = self.discriminator_RBF(target_noise,training = True)
			self.fake_output,self.lambda_x_terms_2 = self.discriminator_RBF(self.reals_enc,training=True)

			with G_tape.stop_recording():
				if self.total_count.numpy()%self.ODE_step == 0 or self.total_count.numpy() <= 2 :

					Centres, Weights, Lamb_Weights = self.find_rbf_centres_weights()
					self.discriminator_RBF.set_weights([Centres,Weights,Centres,Lamb_Weights])

					if self.first_iteration_flag:
						self.first_iteration_flag = 0 
						self.lamb = tf.constant(0.1)
						self.D_loss = self.G_loss = tf.constant(0.)
						return

					self.find_lambda()

			self.divide_by_lambda()

			eval(self.loss_func)

			if self.G_loss_counter == 20:
				self.G_loss_counter = 0
				opt_m = self.rbf_n//2 if self.rbf_n%2==0 else (self.rbf_n//2)+1
				if self.poly_case == 0 or self.poly_case == 2:
					self.rbf_m = min(self.rbf_m+1,opt_m)
				else:
					self.rbf_m = max(self.rbf_m-1,opt_m)
				print("dropping to m=",self.rbf_m)
				self.discriminator_RBF = self.discriminator_model_RBF()
				# self.rbf_m = self.rbf_n//2 if self.rbf_n%2==0 else self.rbf_n//2+1

			self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
			# print("FS Grads",self.E_grads,"=========================================================")
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))

	# # @tf.function
	# def pretrain_step_GAN(self,reals_all):
	# 	assert tf.distribute.get_replica_context() is None
	# 	self.strategy.run(self.actual_pretrain_step_GAN, args=(reals_all,))
	# 	return #self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
					

	def pretrain_step_AE(self,reals_all):
		# with self.strategy.scope():
		self.reals = reals_all

		with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.reals_dec = self.Decoder(self.reals_enc, training = True)

			self.loss_AE()
			self.E_grads = enc_tape.gradient(self.AE_loss, self.Encoder.trainable_variables)
			self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
			self.E_optimizer.apply_gradients(zip(self.E_grads, self.Encoder.trainable_variables))
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))

			self.D_loss = self.G_loss = tf.constant(0.)

	# @tf.function
	# def pretrain_step_AE(self,reals_all):
	# 	assert tf.distribute.get_replica_context() is None
	# 	self.strategy.run(self.actual_pretrain_step_AE, args=(reals_all,))
	# 	return #self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
					


	def train_step_joint(self,reals_all):

		# with self.strategy.scope():
		self.fakes_enc = target_noise = self.get_noise(self.batch_size)
		self.reals = reals_all

		with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as G_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.reals_dec = self.Decoder(self.reals_enc, training = True)

			self.real_output,self.lambda_x_terms_1 = self.discriminator_RBF(target_noise, training = True)
			self.fake_output,self.lambda_x_terms_2 = self.discriminator_RBF(self.reals_enc,training = True)

			with G_tape.stop_recording():

				if self.first_iteration_flag:
					self.first_iteration_flag = 0 
					self.lamb = tf.constant(0.1)
					self.D_loss = self.G_loss = self.AE_loss = tf.constant(0.)
					Centres, Weights, Lamb_Weights = self.find_rbf_centres_weights()
					self.discriminator_RBF.set_weights([Centres,Weights,Centres,Lamb_Weights])
					return

			self.find_lambda()
			self.divide_by_lambda()

			# if self.G_loss_counter == 20:
			# 	self.G_loss_counter = 0
			# 	opt_m = self.rbf_n//2 if self.rbf_n%2==0 else (self.rbf_n//2)+1
			# 	if self.poly_case == 0 or self.poly_case == 2:
			# 		self.rbf_m = min(self.rbf_m+1,opt_m)
			# 	else:
			# 		self.rbf_m = max(self.rbf_m-1,opt_m)
			# 	print("dropping to m=",self.rbf_m)
			# 	self.discriminator_RBF = self.discriminator_model_RBF()
				# self.rbf_m = self.rbf_n//2 if self.rbf_n%2==0 else self.rbf_n//2+1

			self.loss_AE()
			eval(self.loss_func)
			self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))
			self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))

		Centres, Weights, Lamb_Weights = self.find_rbf_centres_weights()
		self.discriminator_RBF.set_weights([Centres,Weights,Centres,Lamb_Weights])


	def train_step(self,reals_all):
		# with self.strategy.scope():
		self.fakes_enc = target_noise = self.get_noise(self.batch_size)
		self.reals = reals_all

		with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as G_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.reals_dec = self.Decoder(self.reals_enc, training = True)

			self.loss_AE()
			self.E_grads = enc_tape.gradient(self.AE_loss, self.Encoder.trainable_variables)
			self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
			# print("AE Grads",self.E_grads, "=================================================")
			self.E_optimizer.apply_gradients(zip(self.E_grads, self.Encoder.trainable_variables))
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))


			self.real_output,self.lambda_x_terms_1 = self.discriminator_RBF(target_noise, training = True)
			self.fake_output,self.lambda_x_terms_2 = self.discriminator_RBF(self.reals_enc,training =True)
						
			if self.first_iteration_flag:
				self.first_iteration_flag = 0 
				self.lamb = tf.constant(0.1)
				self.D_loss = self.G_loss = tf.constant(0)
				Centres, Weights, Lamb_Weights = self.find_rbf_centres_weights()
				self.discriminator_RBF.set_weights([Centres,Weights,Centres,Lamb_Weights])
				return

			self.find_lambda()
			self.divide_by_lambda()
			eval(self.loss_func)

			# if self.G_loss_counter == 20:
			# 	self.G_loss_counter = 0
			# 	opt_m = self.rbf_n//2 if self.rbf_n%2==0 else (self.rbf_n//2)+1
			# 	if self.poly_case == 0 or self.poly_case == 2:
			# 		self.rbf_m = min(self.rbf_m+1,opt_m)
			# 	else:
			# 		self.rbf_m = max(self.rbf_m-1,opt_m)
			# 	print("dropping to m=",self.rbf_m)
			# 	self.discriminator_RBF = self.discriminator_model_RBF()

			self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))

			Centres, Weights, Lamb_Weights = self.find_rbf_centres_weights()
			# print(Centres, Weights, Lamb_Weights)
			self.discriminator_RBF.set_weights([Centres,Weights,Centres,Lamb_Weights])

			return #self.AE_loss, self.G_loss, self.D_loss

	# @tf.function
	# def train_step(self,reals_all):
	# 	assert tf.distribute.get_replica_context() is None
	# 	self.strategy.run(self.actual_train_step, args=(reals_all,))
	# 	return #self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
					

	def loss_RBF(self):
		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output) 
		# used 0.1
		# self.D_loss = 1 * (-loss_real + self.alpha*loss_fake)
		# self.G_loss = 1 * (self.beta*loss_real - loss_fake)
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

		# self.G_loss = tf.nn.compute_average_loss(self.G_loss, global_batch_size=self.batch_size)
		# self.D_loss = tf.nn.compute_average_loss(self.D_loss, global_batch_size=self.batch_size)

		# self.G_loss = self.AE_loss + self.lambda_PGP*self.G_loss

	def loss_AE(self):
		mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM)
		if self.data in ['celeba', 'bedroom', 'mnist', 'church', 'ukiyoe']:
			loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))
		elif self.data in [ 'svhn','cifar10']:
			loss_AE_reals = 0.75*tf.reduce_mean(tf.abs(self.reals - self.reals_dec)) + 0.25*(mse(self.reals, self.reals_dec))
		else:
			loss_AE_reals = mse(self.reals, self.reals_dec)

		self.AE_loss =  loss_AE_reals


###### NEEDS CLEANING #######
'''***********************************************************************************
********** WAE ********************************************************************
***********************************************************************************'''
### self.gan = WAE and self.topic = GMMN
class WAE_ScoreGAN(GAN_WAE):

	def __init__(self,FLAGS_dict):

		GAN_WAE.__init__(self,FLAGS_dict)

		if self.colab and self.data in ['mnist', 'svhn', 'celeba','cifar10']:
			self.bar_flag = 0
		else:
			self.bar_flag = 1

		self.GAN_pretrain_epochs = 0
		self.AE_pretrain_epochs = 0

		self.D_loss = tf.constant(0.)

		self.target_score_flag = 1
		self.scaling_flag = 1
	
	def create_models(self):
		# with self.strategy.scope():
		self.total_count = tf.Variable(0,dtype='int64')
		eval(self.encdec_model)
		# self.discriminator = eval(self.disc_model)

		print("Model Successfully made")

		print("\n\n ENCODER MODEL: \n\n")
		print(self.Encoder.summary())
		print("\n\n DECODER MODEL: \n\n")
		print(self.Decoder.summary())
		# print("\n\n DISCRIMINATOR MODEL: \n\n")
		# print(self.discriminator.summary())

		if self.res_flag == 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n ENCODER MODEL: \n\n")
				self.Encoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DECODER MODEL: \n\n")
				self.Decoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				# fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
				# self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return


	def create_optimizer(self):
		# with self.strategy.scope():
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=1000000, decay_rate=0.95, staircase=True)

		if self.loss == 'SW':
			optimizer = tf.keras.optimizers.Adam
			self.E_optimizer = optimizer(self.lr_AE_Enc)
			self.D_optimizer = optimizer(self.lr_AE_Dec)
			self.G_optimizer = optimizer(self.lr_G,0.5,0.9)
		else:
			optimizer = tf.keras.optimizers.Adam
			self.E_optimizer = optimizer(self.lr_AE_Enc)
			self.D_optimizer = optimizer(self.lr_AE_Dec)
			self.G_optimizer = optimizer(self.lr_G)
		# self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D)

		# self.G_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=5000, decay_rate=1.1, staircase=True)
		# if self.data in ['church','cifar10', ] and self.loss == 'SW':
		# 	self.G_optimizer = tf.keras.optimizers.RMSprop(self.G_lr_schedule)	

		print("Optimizers Successfully made")
		return

	def create_load_checkpoint(self):
		self.checkpoint = tf.train.Checkpoint(E_optimizer = self.E_optimizer,
								 D_optimizer = self.D_optimizer,
								 Encoder = self.Encoder,
								 Decoder = self.Decoder,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.Encoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Encoder.h5')
					self.Decoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Decoder.h5')
					# self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
			

	def save_epoch_h5models(self):
		self.Encoder.save(self.checkpoint_dir + '/model_Encoder.h5', overwrite = True)
		self.Decoder.save(self.checkpoint_dir + '/model_Decoder.h5', overwrite = True)
		# self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return

	def train_step(self,reals_all):
		# with self.strategy.scope():
		self.fakes_enc = target_noise = self.get_noise(self.batch_size)
		self.reals = reals_all
		with tf.device(self.device):
			with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape(persistent=True) as G_tape:
				with tf.GradientTape(persistent=True) as JG_tape, tf.GradientTape(persistent=True) as JGinv_tape:

					JG_tape.watch(self.reals)
					JG_tape.watch(self.fakes_enc)
					JGinv_tape.watch(self.reals)
					JGinv_tape.watch(self.fakes_enc)
					G_tape.watch(self.reals)
					G_tape.watch(self.fakes_enc)

					self.reals_enc = self.Encoder(self.reals, training = True)
					self.reals_dec = self.Decoder(self.reals_enc, training = True)

					self.loss_AE()
					self.E_grads = enc_tape.gradient(self.AE_loss, self.Encoder.trainable_variables)
					self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
					# print("AE Grads",self.E_grads, "=================================================")
					self.E_optimizer.apply_gradients(zip(self.E_grads, self.Encoder.trainable_variables))
					self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))


					J_G_full = JG_tape.batch_jacobian(self.reals_enc,self.reals)
					# print(J_G_full)
					J_G = tf.reshape(J_G_full, (J_G_full.shape[0],J_G_full.shape[1], J_G_full.shape[2]*J_G_full.shape[3]*J_G_full.shape[4]))
					# print(J_G_rect)
					J_G_T = tf.transpose(J_G, perm = [0,2,1])
					M = tf.einsum('bij,bjk->bik',J_G,J_G_T)
					# M = tf.linalg.sqrtm(M_sqr)
					# M = tf.linalg.cholesky(M_sqr)
					# print('J_G',J_G.shape) ## (batch, latent_size, img_size)
					# print('M',M.shape) ## (batch, latent_size, latent_size)
					# exit(0)


					# J_G_Inv = tf.linalg.inv(J_G)
					# print('J_G_Inv',J_G_Inv,J_G_Inv.shape)

					J_Ginv_full = JGinv_tape.batch_jacobian(self.reals_dec,self.reals_enc)
					# print(J_G_full)
					J_G_Inv = tf.reshape(J_Ginv_full, (J_Ginv_full.shape[0],J_Ginv_full.shape[1]*J_Ginv_full.shape[2]*J_Ginv_full.shape[3], J_Ginv_full.shape[4]))
					# print('J_G_Inv',J_G_Inv.shape) ## (batch, img_size, latent_size)
					# J_G_Inv = tf.linalg.pinv(J_G)
					# I = tf.einsum('bij,bjk->bik',J_G,J_G_Inv)
					# print(I)
					# exit(0)
					# print('J_G_Inv',J_G_Inv,J_G_Inv.shape)

				# print('J_G',J_G)
				Det_M_Sqr = tf.linalg.det(M, name=None)
				Det_M = tf.math.sqrt(Det_M_Sqr)
				# print('Det_M',Det_M)
				AbsDet_M = tf.abs(Det_M)
				# print('AbsDet_M',AbsDet_M)
				Ones = tf.ones_like(AbsDet_M)
				Pre_T2 = LogAbsDet_M = tf.math.log(AbsDet_M + 0.1*Ones)
				# print('Pre_T2',Pre_T2)
				# exit(0)
				if self.total_count.numpy() <= 5 or self.total_count.numpy()%1 == 0:
					T2_full = G_tape.gradient(Pre_T2,self.reals)
					# print("T2_full",T2_full.shape)
					T2_rect = tf.reshape(T2_full, (T2_full.shape[0],T2_full.shape[1]*T2_full.shape[2]*T2_full.shape[3]))
					# print(M_rect)
					# T2_rect_T = tf.transpose(T2_rect, perm = [0,2,1])
					# T2 = tf.einsum('bij,bjk->bik',T2_rect,T2_rect_T)

				if self.scaling_flag:
					self.pow_val = int(max(np.ceil(np.log10(np.amax(np.abs(T2_rect.numpy()), axis = 0)))))
					self.scaling = 1#**(-self.pow_val)
					self.scaling_flag = 0

				self.reals_res = tf.reshape(self.reals, (self.reals.shape[0],self.reals.shape[1]*self.reals.shape[2]*self.reals.shape[3]))

				Diff_Term = tf.expand_dims(- self.reals_res - self.scaling*T2_rect,axis = 1)

				# print('Diff_Term',Diff_Term.shape)
				
				self.fakes_score_approx = tf.squeeze(tf.einsum('bij,bjk->bik',Diff_Term,J_G_Inv))		
				# print('fakes_score_approx',self.fakes_score_approx)	
				# exit(0)

				self.get_score()
				eval(self.loss_func)

				self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))
				
				if self.loss_norm == '1':
					self.D_loss =  self.G_loss = mse(0,tf.reduce_mean(self.G_loss, axis = 0))
				return 


	# @tf.function
	# def train_step(self,reals_all):
	# 	assert tf.distribute.get_replica_context() is None
	# 	self.strategy.run(self.actual_train_step, args=(reals_all,))
	# 	return #self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
					
	def get_score(self):
		# self.mu_d = tf.math.reduce_mean(self.fakes_enc, axis=0, keepdims=True, name='fakes_mean')
		# self.Cov_d = tfp.stats.covariance(self.fakes_enc)
		# self.Cov_d_inv = tf.linalg.inv(self.Cov_d)
		# diff_term_target = tf.expand_dims(self.fakes_enc - self.mu_d,axis =2)
		# self.target_score = tf.squeeze(tf.einsum('ij,bjk->bik',self.Cov_d_inv, diff_term_target))

		self.target_score = -self.fakes_enc




	# def train_step(self,reals_all):
	# 	assert tf.distribute.get_replica_context() is None
	# 	self.strategy.run(self.actual_train_step, args=(reals_all,))
	# 	return 


	def loss_score(self):
		##### V2 Good
		mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM)
		
		# else:
		# 	scaling = 1.0
		if self.loss_norm == '1':
			self.G_loss = tf.abs(self.target_score - self.fakes_score_approx)
		if self.loss_norm == '2':
			self.G_loss = mse(0,self.target_score - self.fakes_score_approx)
		if self.loss_norm == 'mix':
			self.G_loss = tf.abs(self.target_score - self.fakes_score_approx) + mse(0,self.target_score - self.fakes_score_approx)
		self.D_loss =  - self.G_loss



	#####################################################################

	def loss_AE(self):
		mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM)

		if self.data in ['celeba', 'bedroom', 'mnist', 'cifar10', 'church']:
			loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))
		elif self.data in ['svhn']:
			loss_AE_reals = 0.5*tf.reduce_mean(tf.abs(self.reals - self.reals_dec)) + 1.5*(mse(self.reals, self.reals_dec))
		else:
			loss_AE_reals = mse(self.reals, self.reals_dec)

		self.AE_loss =  loss_AE_reals 





###### NEEDS CLEANING #######
'''***********************************************************************************
********** WAE ********************************************************************
***********************************************************************************'''
class WAE_Base(GAN_WAE):

	def __init__(self,FLAGS_dict):

		self.lambda_GP = 1.
		GAN_WAE.__init__(self,FLAGS_dict)


		if self.colab and self.data in ['mnist', 'svhn', 'celeba','cifar10']:
			self.bar_flag = 0
		else:
			self.bar_flag = 1

	
	def create_models(self):
		# with self.strategy.scope():
		self.total_count = tf.Variable(0,dtype='int64')
		eval(self.encdec_model)
		self.discriminator = eval(self.disc_model)

		print("Model Successfully made")

		print("\n\n ENCODER MODEL: \n\n")
		print(self.Encoder.summary())
		print("\n\n DECODER MODEL: \n\n")
		print(self.Decoder.summary())
		print("\n\n DISCRIMINATOR MODEL: \n\n")
		print(self.discriminator.summary())

		if self.res_flag == 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n ENCODER MODEL: \n\n")
				self.Encoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DECODER MODEL: \n\n")
				self.Decoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
				self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return


	def create_optimizer(self):
		# with self.strategy.scope():
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=1000000, decay_rate=0.95, staircase=True)

		self.E_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Enc)
		self.D_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Dec)
		self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G)
		self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D)

		print("Optimizers Successfully made")
		return

	def create_load_checkpoint(self):
		self.checkpoint = tf.train.Checkpoint(E_optimizer = self.E_optimizer,
								 D_optimizer = self.D_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
								 Encoder = self.Encoder,
								 Decoder = self.Decoder,
								 discriminator = self.discriminator,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.Encoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Encoder.h5')
					self.Decoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Decoder.h5')
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
			

	def save_epoch_h5models(self):
		self.Encoder.save(self.checkpoint_dir + '/model_Encoder.h5', overwrite = True)
		self.Decoder.save(self.checkpoint_dir + '/model_Decoder.h5', overwrite = True)
		self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return
	

	def pretrain_step_GAN(self,reals_all):
		# with self.strategy.scope():
		self.fakes_enc = target_noise = self.get_noise(self.batch_size)
		self.reals = reals_all

		with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as G_tape, tf.GradientTape(persistent = True) as disc_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.AE_loss = tf.constant(0)

			for i in tf.range(self.Dloop):

				self.real_output = self.discriminator(target_noise, training = True)
				self.fake_output = self.discriminator(self.reals_enc, training = True)
				
				eval(self.loss_func)
				# self.D_loss = self.G_loss

				self.Disc_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
				self.Disc_optimizer.apply_gradients(zip(self.Disc_grads, self.discriminator.trainable_variables))

				if self.loss == 'base':
					wt = []
					for w in self.discriminator.get_weights():
						w = tf.clip_by_value(w, -0.01,0.01) #0.01 for [0,1] data
						wt.append(w)
					self.discriminator.set_weights(wt)

				if i >= (self.Dloop - self.Gloop):
					self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
					self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))


	# @tf.function
	# def pretrain_step_GAN(self,reals_all):
	# 	assert tf.distribute.get_replica_context() is None
	# 	self.strategy.run(self.actual_pretrain_step_GAN, args=(reals_all,))
	# 	return #self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
					



	def train_step(self,reals_all):
		# with self.strategy.scope():
		self.fakes_enc = target_noise = self.get_noise(self.batch_size)
		self.reals = reals_all

		with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as G_tape, tf.GradientTape(persistent = True) as disc_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.reals_dec = self.Decoder(self.reals_enc, training = True)

			self.loss_AE()
			self.E_grads = enc_tape.gradient(self.AE_loss, self.Encoder.trainable_variables)
			self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
			self.E_optimizer.apply_gradients(zip(self.E_grads, self.Encoder.trainable_variables))
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))

			for i in tf.range(self.Dloop):

				self.real_output = self.discriminator(target_noise, training = True)
				self.fake_output = self.discriminator(self.reals_enc, training = True)
				
				eval(self.loss_func)
				# self.D_loss = self.G_loss

				self.Disc_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
				self.Disc_optimizer.apply_gradients(zip(self.Disc_grads, self.discriminator.trainable_variables))

				if self.loss == 'base':
					wt = []
					for w in self.discriminator.get_weights():
						w = tf.clip_by_value(w, -0.1,0.1) #0.01 for [0,1] data
						wt.append(w)
					self.discriminator.set_weights(wt)

				if i >= (self.Dloop - self.Gloop):
					self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
					self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))


	# @tf.function
	# def train_step(self,reals_all):
	# 	assert tf.distribute.get_replica_context() is None
	# 	self.strategy.run(self.actual_train_step, args=(reals_all,))
	# 	return #self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
					


	def loss_base(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output) 

		self.D_loss = (-loss_real + loss_fake)
		self.G_loss = (loss_real - loss_fake)


	# def loss_CW(self):

	# 	def euclidean_norm_squared(X, axis=None):
	# 		return tf.reduce_sum(tf.square(X), axis=axis)
		
	# 	D = self.noise_dims#tf.cast(tf.shape(self.reals_enc)[1], tf.float32)
	# 	N = self.batch_size#tf.cast(tf.shape(self.reals_enc)[0], tf.float32)
	# 	y = tf.cast(tf.pow(4/(3*N), 0.4), dtype = 'float32')

	# 	K = tf.cast(1/(2*D-3), dtype = 'float32')

	# 	A1 = tf.cast(euclidean_norm_squared(tf.subtract(tf.expand_dims(self.reals_enc, 0), tf.expand_dims(self.reals_enc, 1)), axis=2), dtype = 'float32')
	# 	A = tf.cast((1/(N**2)), dtype = 'float32') * tf.reduce_sum((1/tf.sqrt(y + K*A1)))

	# 	B1 = tf.cast(euclidean_norm_squared(self.reals_enc, axis=1), dtype = 'float32')
	# 	B = tf.cast((2/N), dtype = 'float32')*tf.reduce_sum((1/tf.sqrt(y + tf.cast(0.5, dtype = 'float32') + K*B1)))

	# 	self.G_loss = (1/tf.sqrt(1+y)) + A - B
	# 	self.D_loss = tf.constant(0)
	# 	return 


	#################################################################

	def loss_KL(self):
		self.wae_lambda = 1.
		logits_Pz = self.real_output
		logits_Qz = self.fake_output
		loss_Pz = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				logits=logits_Pz, labels=tf.ones_like(logits_Pz)))
		loss_Qz = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				logits=logits_Qz, labels=tf.zeros_like(logits_Qz)))
		loss_Qz_trick = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				logits=logits_Qz, labels=tf.ones_like(logits_Qz)))
		self.D_loss = self.wae_lambda * (loss_Pz + loss_Qz)
		# Non-saturating loss trick
		self.G_loss = loss_Qz_trick

		# kl = tf.keras.losses.KLDivergence

		# logloss_D_fake = tf.math.log(1 - self.fake_output)
		# logloss_D_real = tf.math.log(self.real_output) 

		# logloss_G_fake = tf.math.log(self.fake_output)

		# self.D_loss = -tf.reduce_mean(logloss_D_fake + logloss_D_real)
		# self.G_loss = -tf.reduce_mean(logloss_G_fake)

	#################################################################

	def loss_JS(self):

		cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction = tf.keras.losses.Reduction.SUM)

		D_real_loss = cross_entropy(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = cross_entropy(tf.zeros_like(self.fake_output), self.fake_output)

		G_fake_loss = cross_entropy(tf.ones_like(self.fake_output), self.fake_output)

		self.D_loss = D_real_loss + D_fake_loss
		self.G_loss = G_fake_loss


	#################################################################

	def loss_GP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty()

		self.D_loss = (-loss_real + loss_fake + self.lambda_GP * self.gp )
		self.G_loss = (loss_real - loss_fake)


	def gradient_penalty(self):
		alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		diff = tf.cast(self.fakes_enc,dtype = 'float32') - tf.cast(self.reals_enc,dtype = 'float32')
		inter = tf.cast(self.reals_enc,dtype = 'float32') + (alpha * diff)
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])
		slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
		self.gp = tf.reduce_mean((slopes - 1.)**2)
		return 


	#################################################################

	def loss_LP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.lipschitz_penalty()

		self.D_loss = (-loss_real + loss_fake + self.lambda_GP * self.lp )
		self.G_loss = (loss_real - loss_fake)

	def lipschitz_penalty(self):
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py

		self.K = 1
		self.p = 2

		epsilon = tf.random.uniform([self.batch_size, 1], 0.0, 1.0)
		x_hat = epsilon * tf.cast(self.fakes_enc,dtype = 'float32') + (1 - epsilon) * tf.cast(self.reals_enc,dtype = 'float32')

		with tf.GradientTape() as t:
			t.watch(x_hat)
			D_vals = self.discriminator(x_hat, training = False)
		grad_vals = t.gradient(D_vals, [x_hat])

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

		self.D_loss = (-loss_real + loss_fake + self.lambda_GP * self.alp)
		self.G_loss = (loss_real - loss_fake)


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
		self.K = 1

		samples = tf.concat([tf.cast(self.reals_enc,dtype = 'float32'), tf.cast(self.fakes_enc,dtype = 'float32')], axis=0)
		noise = tf.random.uniform([tf.shape(samples)[0], 1], 0, 1)
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
		mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM)

		if self.data in ['celeba','church','cifar10', 'mnist']:
			loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))
		elif self.data in ['mnist', 'svhn']:
			loss_AE_reals = 0.5*tf.reduce_mean(tf.abs(self.reals - self.reals_dec)) + 1.5*(mse(self.reals, self.reals_dec))
		else:
			loss_AE_reals = mse(self.reals, self.reals_dec)

		self.AE_loss =  loss_AE_reals 


###### NEEDS CLEANING #######
'''***********************************************************************************
********** WAE ********************************************************************
***********************************************************************************'''
### self.gan = WAE and self.topic = GMMN
class WAE_WAEMMD(GAN_WAE):

	def __init__(self,FLAGS_dict):

		GAN_WAE.__init__(self,FLAGS_dict)

		if self.colab and self.data in ['mnist', 'svhn', 'celeba','cifar10']:
			self.bar_flag = 0
		else:
			self.bar_flag = 1

		self.GAN_pretrain_epochs = 0
		self.AE_pretrain_epochs = 0

		self.D_loss = tf.constant(0.)

		if self.loss == 'CW':
			if self.data in ['mnist', 'cifar10', 'svhn']:
				self.lambda_CW = 1.0
			elif self.data in ['fmnist']:
				self.lambda_CW = 10.0
			elif self.data in ['celeba', 'ukiyoe', 'church']:
				self.lambda_CW = 5.0 

		if self.loss == 'SW':

			# if self.data in ['mnist','cifar10']:
			# 	self.lambda_SW = 10
			# elif self.data in ['celeba', 'ukiyoe']: 
			# 	self.lambda_SW = 100.
			# elif self.data in ['church']:
			# 	self.lambda_SW = 50.

			if self.data in ['mnist',]:
				self.lambda_SW = 0.1#10
				L=500
			elif self.data in ['celeba', 'ukiyoe']: 
				self.lambda_SW = 0.1 ## Was 1.0
				L=500 #L=50 #Was 50 for NeurIPS21
			elif self.data in ['church']:
				self.lambda_SW = 0.1 ### Was 1.
				L=500
			elif self.data in ['cifar10']:
				self.lambda_SW = 0.1#1. ## was 10 for NeurIPS21
				L=500 ## was 200 for NeurIPS21

			
			self.theta=[w/np.sqrt((w**2).sum()) for w in np.random.normal(size=(L,self.latent_dims))]

	
	def create_models(self):
		# with self.strategy.scope():
		self.total_count = tf.Variable(0,dtype='int64')
		eval(self.encdec_model)
		self.discriminator = eval(self.disc_model)

		print("Model Successfully made")

		print("\n\n ENCODER MODEL: \n\n")
		print(self.Encoder.summary())
		print("\n\n DECODER MODEL: \n\n")
		print(self.Decoder.summary())
		# print("\n\n DISCRIMINATOR MODEL: \n\n")
		# print(self.discriminator.summary())

		if self.res_flag == 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n ENCODER MODEL: \n\n")
				self.Encoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DECODER MODEL: \n\n")
				self.Decoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				# fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
				# self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return


	def create_optimizer(self):
		# with self.strategy.scope():
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=1000000, decay_rate=0.95, staircase=True)

		if self.loss == 'SW':
			optimizer = tf.keras.optimizers.Adam
			self.E_optimizer = optimizer(self.lr_AE_Enc)
			self.D_optimizer = optimizer(self.lr_AE_Dec)
			self.G_optimizer = optimizer(self.lr_G,0.5,0.9)
		else:
			optimizer = tf.keras.optimizers.Adam
			self.E_optimizer = optimizer(self.lr_AE_Enc)
			self.D_optimizer = optimizer(self.lr_AE_Dec)
			self.G_optimizer = optimizer(self.lr_G)
		# self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D)

		# self.G_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=5000, decay_rate=1.1, staircase=True)
		# if self.data in ['church','cifar10', ] and self.loss == 'SW':
		# 	self.G_optimizer = tf.keras.optimizers.RMSprop(self.G_lr_schedule)	

		print("Optimizers Successfully made")
		return

	def create_load_checkpoint(self):
		self.checkpoint = tf.train.Checkpoint(E_optimizer = self.E_optimizer,
								 D_optimizer = self.D_optimizer,
								 Encoder = self.Encoder,
								 Decoder = self.Decoder,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.Encoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Encoder.h5')
					self.Decoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Decoder.h5')
					# self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
			

	def save_epoch_h5models(self):
		self.Encoder.save(self.checkpoint_dir + '/model_Encoder.h5', overwrite = True)
		self.Decoder.save(self.checkpoint_dir + '/model_Decoder.h5', overwrite = True)
		# self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return

	def actual_train_step(self,reals_all):
		# with self.strategy.scope():
		self.fakes_enc = target_noise = self.get_noise(self.batch_size)
		self.reals = reals_all

		with tf.GradientTape() as dec_tape, tf.GradientTape() as G_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.reals_dec = self.Decoder(self.reals_enc, training = True)

			self.loss_AE()
			eval(self.loss_func)

			self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
			self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))



	# @tf.function
	def train_step(self,reals_all):
		assert tf.distribute.get_replica_context() is None
		self.strategy.run(self.actual_train_step, args=(reals_all,))
		return #self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
					


	def loss_RBFG(self):
		#### Code Courtest: https://github.com/siddharth-agrawal/Generative-Moment-Matching-Networks

		def makeScaleMatrix(num_gen, num_orig):

			# first 'N' entries have '1/N', next 'M' entries have '-1/M'
			s1 =  tf.constant(1.0 / num_gen, shape = [self.batch_size, 1])
			s2 = -tf.constant(1.0 / num_orig, shape = [self.batch_size, 1])
			return tf.concat([s1, s2], axis = 0)

		sigma = [1,2,5,10,20,50]

		X = tf.concat([self.fakes_enc, self.reals_enc], axis = 0)
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

		self.G_loss = self.AE_loss + tf.sqrt(loss)
		self.D_loss = tf.constant(0.0)

		return 

	#################################################################

	def loss_CW(self):

		def euclidean_norm_squared(X, axis=None):
			return tf.reduce_sum(tf.square(X), axis=axis)
		
		D = tf.cast(self.latent_dims, tf.float32)#tf.cast(tf.shape(self.reals_enc)[1], tf.float32)
		N = tf.cast(self.batch_size, tf.float32)#tf.cast(tf.shape(self.reals_enc)[0], tf.float32)
		y = tf.pow(4/(3*N), 0.4)

		K = 1/(2*D-3)

		A1 = euclidean_norm_squared(tf.subtract(tf.expand_dims(self.reals_enc, 0), tf.expand_dims(self.reals_enc, 1)), axis=2)
		A = (1/(N**2)) * tf.reduce_sum((1/tf.sqrt(y + K*A1)))

		B1 = euclidean_norm_squared(self.reals_enc, axis=1)
		B = (2/N)*tf.reduce_sum((1/tf.sqrt(y + 0.5 + K*B1)))

		tensor_cw_distance = (1/tf.sqrt(1+y)) + A - B

		self.G_loss = self.AE_loss + self.lambda_CW*tf.math.log(tensor_cw_distance)
		return 


	#################################################################

	def loss_IMQ(self):
		###3 Code Courtesy https://github.com/hiwonjoon/wae-wgan/blob/master/wae_mmd.py

		n = tf.cast(self.batch_size,tf.float32)
		C_base = 2.*self.fakes_enc.shape[1]

		z = self.fakes_enc
		z_tilde = self.reals_enc

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

		self.G_loss = self.AE_loss + L_D
		self.D_loss = tf.constant(0.0)

	#####################################################################

	def loss_SW(self):

		# Let projae be the projection of the encoded samples
		# projae = tf.dot(self.reals_enc,tf.transpose(theta))
		# projae = tf.linalg.matmul(self.reals_enc, self.theta, transpose_a=False, transpose_b=True)
		projae = tf.keras.backend.dot(tf.cast(self.reals_enc,'float32'), tf.transpose(tf.cast(self.theta,'float32')))

		# Let projz be the projection of the $q_Z$ samples
		# projz = tf.dot(z,tf.transpose(theta))
		# projz = tf.linalg.matmul(self.fakes_enc, self.theta, transpose_a=False, transpose_b=True)
		projz = tf.keras.backend.dot(tf.cast(self.fakes_enc,'float32'), tf.transpose(tf.cast(self.theta,'float32')))

		# Calculate the Sliced Wasserstein distance by sorting 
		# the projections and calculating the L2 distance between
		W2=(tf.nn.top_k(tf.transpose(projae),k=tf.cast(self.batch_size, 'int32')).values-tf.nn.top_k(tf.transpose(projz),k=tf.cast(self.batch_size, 'int32')).values)**2

		SWLoss = self.lambda_SW * tf.reduce_mean(W2)

		bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)

		bce_loss = bce(self.reals_dec, self.reals)

		self.G_loss = self.AE_loss + SWLoss #+ bce_loss 


	#####################################################################

	def loss_AE(self):
		mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM)

		if self.data in ['celeba', 'bedroom', 'mnist', 'cifar10', 'church']:
			loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))
		elif self.data in ['svhn']:
			loss_AE_reals = 0.5*tf.reduce_mean(tf.abs(self.reals - self.reals_dec)) + 1.5*(mse(self.reals, self.reals_dec))
		else:
			loss_AE_reals = mse(self.reals, self.reals_dec)

		self.AE_loss =  loss_AE_reals 



