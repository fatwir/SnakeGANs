from __future__ import print_function
import os, sys, time, argparse
from datetime import date
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
from absl import app
from absl import flags
import json

from gan_data import *
from gan_src import *

import tensorflow_probability as tfp
tfd = tfp.distributions
from matplotlib.backends.backend_pgf import PdfPages
from scipy.interpolate import interp1d
mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM)
from ops import *


from itertools import product,combinations,combinations_with_replacement



# FLAGS(sys.argv)
# tf.keras.backend.set_floatx('float64')

'''
GAN_topic is the Overarching class file, where corresponding parents are instantialized, along with setting up the calling functions for these and files and folders for resutls, etc. data reading is also done from here. Sometimes display functions, architectures, etc may be modified here if needed (overloading parent classes)
'''


'''***********************************************************************************
********** GAN Baseline setup ********************************************************
***********************************************************************************'''
class GAN_Base(GAN_SRC, GAN_DATA_Base):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all fundamental ops and metric functions'''
		GAN_SRC.__init__(self,FLAGS_dict)
		''' Set up the GAN_DATA class'''
		GAN_DATA_Base.__init__(self)

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.arch+'_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.arch+'_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		self.noise_setup()


		''' Define dataset and tf.data function. batch sizing done'''
		# self.get_data()

		# self.create_models()

		# self.create_optimizer()

		# self.create_load_checkpoint()

	def get_data(self):
		with tf.device('/CPU'):
			self.train_data = eval(self.gen_func)

			self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
			''' Set PRINT and SAVE iters if 0'''
			self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
			self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

			self.train_dataset = eval(self.dataset_func)
			self.train_dataset_size = self.train_data.shape[0]

			# self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)

			print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size,  self.num_batches,self.print_step, self.save_step))


	###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	def create_models(self):
		# with self.strategy.scope():
		self.total_count = tf.Variable(0,dtype='int64')
		self.generator = eval(self.gen_model)
		self.discriminator = eval(self.disc_model)

		if self.res_flag == 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n GENERATOR MODEL: \n\n")
				self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
				self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

		print("Model Successfully made")

		print(self.generator.summary())
		print(self.discriminator.summary())
		return		


	###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
								 generator = self.generator,
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
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return

	def noise_setup(self):

		if self.noise_kind == 'non_para':
			self.non_para_dist_mat = sio.loadmat('data/non_param_pdf.mat')

		if self.noise_kind == 'gamma':
			self.gamma_shape = 0.5
			self.gamma_scale = 1.0

		if self.noise_kind == 'trip':
			self.num_latents_trip = 128
			self.num_components_trip = 10
			self.tt_int_trip = 40

		return

	def get_noise(self, shape):
		#shape = [self.batch_size, self.noise_dims]

		def random_gen(shape, pdf, points, epsilon):
			assert len(shape) == 2
			rn = np.random.choice(points, size = shape, p=pdf).astype(np.float32)
			for i in range(shape[0]):
				for j in range(shape[1]):
					rn[i,j] = np.random.uniform(rn[i,j], rn[i,j]+epsilon, 1).astype(np.float32)
			return rn

		def sample_spherical(npoints, ndim=3):
			vec = np.random.randn(ndim, npoints)
			vec /= np.linalg.norm(vec, axis=0)
			return vec

		# def TRIP()

		if self.noise_kind == 'non_para':
			pdf = np.reshape(self.non_para_dist_mat['X_final'], (1024))
			points = np.linspace(-2, 2, 1024)
			epsilon = 4./1024.
			noise = random_gen(shape, pdf, points, epsilon)

		elif self.noise_kind == 'gaussian':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = self.noise_stddev)

		elif self.noise_kind == 'gaussian075':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = 0.75)

		elif self.noise_kind == 'gamma':
			nu = sample_spherical(shape[0], shape[1]).transpose()
			r = np.random.gamma(self.gamma_shape, scale=self.gamma_scale, size=shape[0])
			root_r_mat = np.repeat(np.expand_dims(np.sqrt(r),axis=1), shape[1], axis = 1)

			noise = np.multiply(root_r_mat,nu)
			# print(root_r_mat.shape,nu.shape,noise.shape)
		elif self.noise_kind == 'cauchy':
			noise = np.random.standard_cauchy(size=shape)

		elif self.noise_kind == 'trip':
			prior = TRIP(self.num_latents_trip * (('c', self.num_components_trip),),tt_int=self.tt_int_trip, distr_init='uniform')

		elif self.noise_kind == 'ThesisGMM':

			locs = [[-5,5], \
					[5,-5], \
					]
			probs = [0.5,0.5]
			stddev_scale = [1., 1.]
			noise_gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))
			noise = noise_gmm.sample(sample_shape=shape[0])


			# n = tf.random.normal(shape, mean = self.noise_mean, stddev = self.noise_stddev)

			# print(noise.shape,n.shape)
			# exit(0)
		elif self.noise_kind == 'ThesisMoon':
			from sklearn.datasets import make_moons

			def make_moons_da(n_samples=100, rotation=90, noise=0.01, random_state=0):
				Xs, ys = make_moons(n_samples=n_samples,
									noise=noise,
									random_state=random_state)
				Xs[:, 0] -= 0.5
				theta = np.radians(-rotation)
				cos_theta, sin_theta = np.cos(theta), np.sin(theta)
				rot_matrix = np.array(
					((cos_theta, -sin_theta),
					 (sin_theta, cos_theta))
				)
				Xt = Xs.dot(rot_matrix)
				yt = ys
				return Xs, ys, Xt, yt


			Xs, ys, Xt, yt = make_moons_da(n_samples=2*shape[0])

			noise = Xt[ys==1]
			# print(noise.shape)
		return noise

	def train(self):
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)   
			start = time.perf_counter()
			batch_count = tf.Variable(0,dtype='int64')
			start_time =0

			for image_batch in self.train_dataset:
				# print(image_batch.shape)
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_time = time.perf_counter()
				# with self.strategy.scope():
				self.train_step(image_batch.numpy())
				self.eval_metrics()
						

				train_time = time.perf_counter()-start_time

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

				if self.iters_flag:
					if self.num_iters == self.total_count.numpy():
						tf.print("\n Training for {} Iterations completed".format( self.total_count.numpy()))
						if self.pbar_flag:
							bar.close()
							del bar
						tf.print('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
						self.save_epoch_h5models()
						return

			if self.pbar_flag:
				bar.close()
				del bar
			tf.print('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
			self.save_epoch_h5models()


	def save_epoch_h5models(self):

		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)

		if self.loss == 'FS':
			self.discriminator_A.save(self.checkpoint_dir + '/model_discriminator_A.h5', overwrite = True)
			self.discriminator_B.save(self.checkpoint_dir + '/model_discriminator_B.h5', overwrite = True)
		elif self.loss == 'RBF':
			self.discriminator_RBF.save(self.checkpoint_dir +'/model_discriminator_RBF.h5',overwrite=True)
		elif self.topic not in ['SnakeGAN', 'ScoreGAN']:
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return


	def print_batch_outputs(self,epoch):
		if ((self.total_count.numpy() % 5) == 0 and self.data in ['g1', 'g2']):### Was 10 - ICML22 plots
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() <= 5) and self.data in [ 'g1', 'g2', 'gmm2', 'gmm8']:
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 100) == 0 and self.data in ['gmm2', 'gmm8']):
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 10) == 0 and self.data in ['gmm2', 'gmm8'] and self.topic in ['ScoreGAN']):
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 100) == 0 and self.data in ['celeba']):
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def eval_sharpness(self):
		i = 0
		for train_batch in self.train_dataset:
			noise = self.get_noise([self.batch_size, self.noise_dims])
			preds = self.generator(noise, training = False)

			sharp = self.find_sharpness(preds)
			base_sharp = self.find_sharpness(train_batch)
			try:
				sharp_vec.append(sharp)
				base_sharp_vec.append(base_sharp)

			except:
				sharp_vec = [sharp]
				base_sharp_vec = [base_sharp]
			i += 1
			if i == 10:
				break
		###### Sharpness averaging measure
		sharpness = np.mean(np.array(sharp_vec))
		baseline_sharpness = np.mean(np.array(base_sharp_vec))

		return baseline_sharpness, sharpness



	def test(self):
		num_interps = 10
		if self.mode == 'test':
			num_figs = 20#int(400/(2*num_interps))
		else:
			num_figs = 9
		# there are 400 samples in the batch. to make 10x10 images, 
		for j in range(num_figs):
			# print("Interpolation Testing Image ",j)
			path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
			# noise = self.get_noise([20*num_figs, self.noise_dims])
			# current_batch = noise[2*num_interps*j:2*num_interps*(j+1)]
			# image_latents = self.generator(current_batch)
			for i in range(num_interps):
				# print("Pair ",i)
				# current_batch = self.get_noise([20*num_figs, self.noise_dims])
				start = self.get_noise([1, self.noise_dims])#current_batch[i:1+i].numpy()
				end = self.get_noise([1, self.noise_dims]) #current_batch[num_interps+i:num_interps+1+i].numpy()
				stack = np.vstack([start, end])
				# print(stack)



				linfit = interp1d([1,num_interps+1], stack, axis=0)
				interp_latents = linfit(list(range(1,num_interps+1)))

				# print(interp_latents)
				cur_interp_figs = self.generator(interp_latents)

				# print(cur_interp_figs)

				sharpness = self.find_sharpness(cur_interp_figs)

				try:
					sharpness_vec.append(sharpness)
				except:
					shaprpness_vec = [sharpness]
				# cur_interp_figs_with_ref = np.concatenate((current_batch[i:1+i],cur_interp_figs.numpy(),current_batch[num_interps+i:num_interps+1+i]), axis = 0)
				# print(cur_interp_figs_with_ref.shape)
				try:
					batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs),axis = 0)
				except:
					batch_interp_figs = cur_interp_figs

			images = (batch_interp_figs + 1.0)/2.0
			# print(images.shape)
			size_figure_grid = num_interps
			images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(num_interps,num_interps))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()
			del batch_interp_figs

		###### Interpol samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Interpolation Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("Interpolation Sharpness - "+str(overall_sharpness))

		# for i in range(self.num_test_images):

		# 	path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
		# 	label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())

		# 	size_figure_grid = self.num_to_print
		# 	test_batch_size = size_figure_grid*size_figure_grid
		# 	noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)

		# 	images = self.generator(noise, training=False)
		# 	if self.data != 'celeba':
		# 		images = (images + 1.0)/2.0

		# 	self.save_image_batch(images = images,label = label, path = path)

		# self.impath += '_Testing_'
		# for img_batch in self.train_dataset:
		# 	self.reals = img_batch
		# 	self.generate_and_save_batch(0)
		# 	return



'''***********************************************************************************
********** GAN Baseline setup ********************************************************
***********************************************************************************'''
class GAN_AE_Base(GAN_SRC, GAN_DATA_Base):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all fundamental ops and metric functions'''
		GAN_SRC.__init__(self,FLAGS_dict)
		''' Set up the GAN_DATA class'''
		GAN_DATA_Base.__init__(self)

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.arch+'_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.arch+'_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		self.noise_setup()


		''' Define dataset and tf.data function. batch sizing done'''
		# self.get_data()

		# self.create_models()

		# self.create_optimizer()

		# self.create_load_checkpoint()

	def get_data(self):
		with tf.device('/CPU'):
			self.train_data = eval(self.gen_func)

			self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
			''' Set PRINT and SAVE iters if 0'''
			self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
			self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

			self.train_dataset = eval(self.dataset_func)
			self.train_dataset_size = self.train_data.shape[0]

			# self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)

			print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size,  self.num_batches,self.print_step, self.save_step))


	###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	def create_models(self):
		# with self.strategy.scope():
		self.total_count = tf.Variable(0,dtype='int64')
		self.generator = eval(self.gen_model)
		self.discriminator = eval(self.disc_model)
		self.Encoder = tf.keras.models.load_model("logs/AE_Models/"+self.data+"_Encoder.h5", custom_objects={'Functional':tf.keras.models.Model})
		self.Decoder = tf.keras.models.load_model("logs/AE_Models/"+self.data+"_Decoder.h5", custom_objects={'Functional':tf.keras.models.Model})


		if self.res_flag == 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n GENERATOR MODEL: \n\n")
				self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
				self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n ENCODER MODEL: \n\n")
				self.Encoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DECODER MODEL: \n\n")
				self.Decoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

		print("Model Successfully made")

		print("\n\n Generator: \n\n")
		print(self.generator.summary())
		print("\n\n Discriminator: \n\n")
		print(self.discriminator.summary())
		print("\n\n Encoder: \n\n")
		print(self.Encoder.summary())
		print("\n\n Decoder: \n\n")
		print(self.Decoder.summary())
		return		


	###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
								 generator = self.generator,
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
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return

	def noise_setup(self):

		if self.noise_kind == 'non_para':
			self.non_para_dist_mat = sio.loadmat('data/non_param_pdf.mat')

		if self.noise_kind == 'gamma':
			self.gamma_shape = 0.5
			self.gamma_scale = 1.0

		if self.noise_kind == 'trip':
			self.num_latents_trip = 128
			self.num_components_trip = 10
			self.tt_int_trip = 40

		return

	def get_noise(self, shape):
		#shape = [self.batch_size, self.noise_dims]

		def random_gen(shape, pdf, points, epsilon):
			assert len(shape) == 2
			rn = np.random.choice(points, size = shape, p=pdf).astype(np.float32)
			for i in range(shape[0]):
				for j in range(shape[1]):
					rn[i,j] = np.random.uniform(rn[i,j], rn[i,j]+epsilon, 1).astype(np.float32)
			return rn

		def sample_spherical(npoints, ndim=3):
			vec = np.random.randn(ndim, npoints)
			vec /= np.linalg.norm(vec, axis=0)
			return vec

		# def TRIP()

		if self.noise_kind == 'non_para':
			pdf = np.reshape(self.non_para_dist_mat['X_final'], (1024))
			points = np.linspace(-2, 2, 1024)
			epsilon = 4./1024.
			noise = random_gen(shape, pdf, points, epsilon)

		elif self.noise_kind == 'gaussian':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = self.noise_stddev)

		elif self.noise_kind == 'gaussian075':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = 0.75)

		elif self.noise_kind == 'gamma':
			nu = sample_spherical(shape[0], shape[1]).transpose()
			r = np.random.gamma(self.gamma_shape, scale=self.gamma_scale, size=shape[0])
			root_r_mat = np.repeat(np.expand_dims(np.sqrt(r),axis=1), shape[1], axis = 1)

			noise = np.multiply(root_r_mat,nu)
			# print(root_r_mat.shape,nu.shape,noise.shape)
		elif self.noise_kind == 'cauchy':
			noise = np.random.standard_cauchy(size=shape)

		elif self.noise_kind == 'trip':
			prior = TRIP(self.num_latents_trip * (('c', self.num_components_trip),),tt_int=self.tt_int_trip, distr_init='uniform')

		return noise

	def train(self):
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)   
			start = time.perf_counter()
			batch_count = tf.Variable(0,dtype='int64')
			start_time =0

			for image_batch in self.train_dataset:
				# print(image_batch.shape)
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_time = time.perf_counter()
				# with self.strategy.scope():
				self.train_step(image_batch.numpy())
				self.eval_metrics()
						

				train_time = time.perf_counter()-start_time

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

				if self.iters_flag:
					if self.num_iters == self.total_count.numpy():
						tf.print("\n Training for {} Iterations completed".format( self.total_count.numpy()))
						if self.pbar_flag:
							bar.close()
							del bar
						tf.print('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
						self.save_epoch_h5models()
						return

			if self.pbar_flag:
				bar.close()
				del bar
			tf.print('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
			self.save_epoch_h5models()


	def save_epoch_h5models(self):

		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
		self.Encoder.save(self.checkpoint_dir + '/model_Encoder.h5', overwrite = True)
		self.Decoder.save(self.checkpoint_dir + '/model_Decoder.h5', overwrite = True)

		if self.loss == 'FS':
			self.discriminator_A.save(self.checkpoint_dir + '/model_discriminator_A.h5', overwrite = True)
			self.discriminator_B.save(self.checkpoint_dir + '/model_discriminator_B.h5', overwrite = True)
		elif self.loss == 'RBF':
			self.discriminator_RBF.save(self.checkpoint_dir +'/model_discriminator_RBF.h5',overwrite=True)
		elif self.topic not in ['SnakeGAN', 'ScoreGAN']:
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return


	def print_batch_outputs(self,epoch):
		if ((self.total_count.numpy() % 5) == 0 and self.data in ['g1', 'g2']):### Was 10 - ICML22 plots
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() <= 5) and self.data in [ 'g1', 'g2', 'gmm2', 'gmm8']:
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 100) == 0 and self.data in ['gmm2', 'gmm8']):
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 10) == 0 and self.data in ['gmm2', 'gmm8'] and self.topic in ['ScoreGAN']):
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 100) == 0 and self.data in ['celeba']):
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def eval_sharpness(self):
		i = 0
		for train_batch in self.train_dataset:
			noise = self.get_noise([self.batch_size, self.noise_dims])
			preds = self.generator(noise, training = False)

			sharp = self.find_sharpness(preds)
			base_sharp = self.find_sharpness(train_batch)
			try:
				sharp_vec.append(sharp)
				base_sharp_vec.append(base_sharp)

			except:
				sharp_vec = [sharp]
				base_sharp_vec = [base_sharp]
			i += 1
			if i == 10:
				break
		###### Sharpness averaging measure
		sharpness = np.mean(np.array(sharp_vec))
		baseline_sharpness = np.mean(np.array(base_sharp_vec))

		return baseline_sharpness, sharpness



	def test(self):
		return
		### NEED FIXXXXXXXXXXXXXXX
		num_interps = 10
		if self.mode == 'test':
			num_figs = 20#int(400/(2*num_interps))
		else:
			num_figs = 9
		# there are 400 samples in the batch. to make 10x10 images, 
		for j in range(num_figs):
			# print("Interpolation Testing Image ",j)
			path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
			# noise = self.get_noise([20*num_figs, self.noise_dims])
			# current_batch = noise[2*num_interps*j:2*num_interps*(j+1)]
			# image_latents = self.generator(current_batch)
			for i in range(num_interps):
				# print("Pair ",i)
				# current_batch = self.get_noise([20*num_figs, self.noise_dims])
				start = self.get_noise([1, self.noise_dims])#current_batch[i:1+i].numpy()
				end = self.get_noise([1, self.noise_dims]) #current_batch[num_interps+i:num_interps+1+i].numpy()
				stack = np.vstack([start, end])
				# print(stack)



				linfit = interp1d([1,num_interps+1], stack, axis=0)
				interp_latents = linfit(list(range(1,num_interps+1)))

				# print(interp_latents)
				cur_interp_figs = self.generator(interp_latents)

				# print(cur_interp_figs)

				sharpness = self.find_sharpness(cur_interp_figs)

				try:
					sharpness_vec.append(sharpness)
				except:
					shaprpness_vec = [sharpness]
				# cur_interp_figs_with_ref = np.concatenate((current_batch[i:1+i],cur_interp_figs.numpy(),current_batch[num_interps+i:num_interps+1+i]), axis = 0)
				# print(cur_interp_figs_with_ref.shape)
				try:
					batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs),axis = 0)
				except:
					batch_interp_figs = cur_interp_figs

			images = (batch_interp_figs + 1.0)/2.0
			# print(images.shape)
			size_figure_grid = num_interps
			images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(num_interps,num_interps))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()
			del batch_interp_figs

		###### Interpol samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Interpolation Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("Interpolation Sharpness - "+str(overall_sharpness))





'''***********************************************************************************
********** GAN Baseline setup ********************************************************
***********************************************************************************'''
class GAN_ScoreGAN(GAN_SRC, GAN_DATA_Base):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all fundamental ops and metric functions'''
		GAN_SRC.__init__(self,FLAGS_dict)
		''' Set up the GAN_DATA class'''
		GAN_DATA_Base.__init__(self)

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.arch+'_'+self.data+'()'
		if self.data not in ['g2','g1','gN']:
			self.score_model = 'self.score_model_'+self.arch+'_'+self.data+'()'
		else:
			#### Makes no difference. Dummy variable
			self.score_model = 'self.generator_model_'+self.arch+'_'+self.data+'()'

		self.disc_model = 'self.discriminator_model_'+self.arch+'_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		self.noise_setup()

	def get_data(self):
		with tf.device('/CPU'):
			self.train_data = eval(self.gen_func)

			self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
			''' Set PRINT and SAVE iters if 0'''
			self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
			self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

			self.train_dataset = eval(self.dataset_func)
			self.train_dataset_size = self.train_data.shape[0]

			# self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)

			print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size,  self.num_batches,self.print_step, self.save_step))


	###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	def create_models(self):
		# with self.strategy.scope():
		self.total_count = tf.Variable(0,dtype='int64')
		self.generator = eval(self.gen_model)
		self.ScoreNet = eval(self.score_model)
		self.discriminator = eval(self.disc_model)

		if self.res_flag == 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n GENERATOR MODEL: \n\n")
				self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
				self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n SCORE MODEL: \n\n")
				self.ScoreNet.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

		print("Model Successfully made")

		print(self.generator.summary())
		print(self.discriminator.summary())
		print(self.ScoreNet.summary())
		return		


	###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
								 S_optimizer = self.S_optimizer,
								 generator = self.generator,
								 ScoreNet = self.ScoreNet,
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
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return

	def noise_setup(self):

		if self.noise_kind == 'non_para':
			self.non_para_dist_mat = sio.loadmat('data/non_param_pdf.mat')

		if self.noise_kind == 'gamma':
			self.gamma_shape = 0.5
			self.gamma_scale = 1.0

		if self.noise_kind == 'trip':
			self.num_latents_trip = 128
			self.num_components_trip = 10
			self.tt_int_trip = 40

		return

	def get_noise(self, shape):
		#shape = [self.batch_size, self.noise_dims]

		def random_gen(shape, pdf, points, epsilon):
			assert len(shape) == 2
			rn = np.random.choice(points, size = shape, p=pdf).astype(np.float32)
			for i in range(shape[0]):
				for j in range(shape[1]):
					rn[i,j] = np.random.uniform(rn[i,j], rn[i,j]+epsilon, 1).astype(np.float32)
			return rn

		def sample_spherical(npoints, ndim=3):
			vec = np.random.randn(ndim, npoints)
			vec /= np.linalg.norm(vec, axis=0)
			return vec

		# def TRIP()

		if self.noise_kind == 'non_para':
			pdf = np.reshape(self.non_para_dist_mat['X_final'], (1024))
			points = np.linspace(-2, 2, 1024)
			epsilon = 4./1024.
			noise = random_gen(shape, pdf, points, epsilon)

		elif self.noise_kind == 'gaussian':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = self.noise_stddev)

		elif self.noise_kind == 'gaussian075':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = 0.75)

		elif self.noise_kind == 'gamma':
			nu = sample_spherical(shape[0], shape[1]).transpose()
			r = np.random.gamma(self.gamma_shape, scale=self.gamma_scale, size=shape[0])
			root_r_mat = np.repeat(np.expand_dims(np.sqrt(r),axis=1), shape[1], axis = 1)

			noise = np.multiply(root_r_mat,nu)
			# print(root_r_mat.shape,nu.shape,noise.shape)
		elif self.noise_kind == 'cauchy':
			noise = np.random.standard_cauchy(size=shape)

		elif self.noise_kind == 'trip':
			prior = TRIP(self.num_latents_trip * (('c', self.num_components_trip),),tt_int=self.tt_int_trip, distr_init='uniform')

		return noise

	def train(self):
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1

		if self.data not in ['g1','g2','gN']:	
			for score_epoch in range(start,self.Score_pretrain_epochs):
				if self.pbar_flag:
					bar = self.pbar(score_epoch)   
				start = time.perf_counter()
				batch_count = tf.Variable(0,dtype='int64')
				start_time =0
				for image_batch in self.train_dataset:
					# print(image_batch.shape)
					batch_count.assign_add(1)
					start_time = time.perf_counter()
					# with self.strategy.scope():
					self.ScoreNet_train_step(image_batch.numpy())
					# self.eval_metrics()
					if batch_count.numpy()%10 == 0:
						self.show_scores_gmm8(num_batch = batch_count.numpy(),num_epoch = score_epoch)
					train_time = time.perf_counter()-start_time

					if self.pbar_flag:
						bar.postfix[0] = f'{batch_count.numpy():6.0f}'
						bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
						bar.postfix[2] = f'{self.ScoreNet_loss.numpy():2.4e}'
						bar.update(self.batch_size.numpy())
					if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
						if self.res_flag:
							self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; ScoreNet_loss - {:>2.4f} \n".format(score_epoch,batch_count.numpy(),train_time,self.ScoreNet_loss.numpy()))
				if self.pbar_flag:
					bar.close()
					del bar
				tf.print('Time for Score net epoch {} is {} sec'.format(score_epoch, time.perf_counter()-start))

		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)   
			start = time.perf_counter()
			batch_count = tf.Variable(0,dtype='int64')
			start_time =0

			for image_batch in self.train_dataset:
				# print(image_batch.shape)
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_time = time.perf_counter()
				# with self.strategy.scope():
				self.train_step(image_batch.numpy())
				self.eval_metrics()
						

				train_time = time.perf_counter()-start_time

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

				if self.iters_flag:
					if self.num_iters == self.total_count.numpy():
						tf.print("\n Training for {} Iterations completed".format( self.total_count.numpy()))
						if self.pbar_flag:
							bar.close()
							del bar
						tf.print('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
						self.save_epoch_h5models()
						return

			if self.pbar_flag:
				bar.close()
				del bar
			tf.print('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
			self.save_epoch_h5models()


	def save_epoch_h5models(self):

		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)

		if self.loss == 'FS':
			self.discriminator_A.save(self.checkpoint_dir + '/model_discriminator_A.h5', overwrite = True)
			self.discriminator_B.save(self.checkpoint_dir + '/model_discriminator_B.h5', overwrite = True)
		elif self.loss == 'RBF':
			self.discriminator_RBF.save(self.checkpoint_dir +'/model_discriminator_RBF.h5',overwrite=True)
		elif self.topic not in ['SnakeGAN', 'ScoreGAN']:
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return


	def print_batch_outputs(self,epoch):
		if ((self.total_count.numpy() % 5) == 0 and self.data in ['g1', 'g2']):### Was 10 - ICML22 plots
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() <= 5) and self.data in [ 'g1', 'g2', 'gmm2', 'gmm8']:
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 100) == 0 and self.data in ['gmm2', 'gmm8']):
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 10) == 0 and self.data in ['gmm2', 'gmm8'] and self.topic in ['ScoreGAN']):
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 100) == 0 and self.data in ['celeba']):
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def eval_sharpness(self):
		i = 0
		for train_batch in self.train_dataset:
			noise = self.get_noise([self.batch_size, self.noise_dims])
			preds = self.generator(noise, training = False)

			sharp = self.find_sharpness(preds)
			base_sharp = self.find_sharpness(train_batch)
			try:
				sharp_vec.append(sharp)
				base_sharp_vec.append(base_sharp)

			except:
				sharp_vec = [sharp]
				base_sharp_vec = [base_sharp]
			i += 1
			if i == 10:
				break
		###### Sharpness averaging measure
		sharpness = np.mean(np.array(sharp_vec))
		baseline_sharpness = np.mean(np.array(base_sharp_vec))

		return baseline_sharpness, sharpness



	def test(self):
		num_interps = 10
		if self.mode == 'test':
			num_figs = 20#int(400/(2*num_interps))
		else:
			num_figs = 9
		# there are 400 samples in the batch. to make 10x10 images, 
		for j in range(num_figs):
			# print("Interpolation Testing Image ",j)
			path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
			# noise = self.get_noise([20*num_figs, self.noise_dims])
			# current_batch = noise[2*num_interps*j:2*num_interps*(j+1)]
			# image_latents = self.generator(current_batch)
			for i in range(num_interps):
				# print("Pair ",i)
				# current_batch = self.get_noise([20*num_figs, self.noise_dims])
				start = self.get_noise([1, self.noise_dims])#current_batch[i:1+i].numpy()
				end = self.get_noise([1, self.noise_dims]) #current_batch[num_interps+i:num_interps+1+i].numpy()
				stack = np.vstack([start, end])
				# print(stack)



				linfit = interp1d([1,num_interps+1], stack, axis=0)
				interp_latents = linfit(list(range(1,num_interps+1)))

				# print(interp_latents)
				cur_interp_figs = self.generator(interp_latents)

				# print(cur_interp_figs)

				sharpness = self.find_sharpness(cur_interp_figs)

				try:
					sharpness_vec.append(sharpness)
				except:
					shaprpness_vec = [sharpness]
				# cur_interp_figs_with_ref = np.concatenate((current_batch[i:1+i],cur_interp_figs.numpy(),current_batch[num_interps+i:num_interps+1+i]), axis = 0)
				# print(cur_interp_figs_with_ref.shape)
				try:
					batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs),axis = 0)
				except:
					batch_interp_figs = cur_interp_figs

			images = (batch_interp_figs + 1.0)/2.0
			# print(images.shape)
			size_figure_grid = num_interps
			images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(num_interps,num_interps))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()
			del batch_interp_figs

		###### Interpol samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Interpolation Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("Interpolation Sharpness - "+str(overall_sharpness))


'''***********************************************************************************
********** Conditional GAN (cGAN-PD, ACGAN, TACGAN) setup ****************************
***********************************************************************************'''
class GAN_CondGAN(GAN_SRC, GAN_DATA_CondGAN):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all GAN architectures'''

		GAN_SRC.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_CondGAN.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.train_labels, self.batch_size)'
		# self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		if self.loss == 'FS':
			self.gen_model = 'self.generator_model_'+self.data+'_'+self.latent_kind+'()'
			self.disc_model = 'self.discriminator_model_'+self.data+'_'+self.latent_kind+'()' 
			self.EncDec_func = 'self.encoder_model_'+self.data+'_'+self.latent_kind+'()'
			self.DEQ_func = 'self.discriminator_ODE()'

		''' Define dataset and tf.data function. batch sizing done'''
		# self.get_data()

		# self.create_models()

		# self.create_optimizer()

		# self.create_load_checkpoint()

	def get_data(self):
		with tf.device('/CPU'):
			self.train_data, self.train_labels = eval(self.gen_func)

			self.num_batches = int(np.floor((self.train_data.shape[0])/self.batch_size))
			''' Set PRINT and SAVE iters if 0'''
			self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
			self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

			self.train_dataset = eval(self.dataset_func)
			print("Dataset created - this is it")
			print(self.train_dataset)

			self.train_dataset_size = self.train_data.shape[0]

			print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, 
			self.num_batches,self.print_step, self.save_step))

	def get_noise(self,noise_case,batch_size):
		noise = tf.random.normal([batch_size, self.noise_dims], mean = self.noise_mean, stddev = self.noise_stddev)
		if noise_case == 'test':
			if self.data in ['mnist', 'cifar10']:
				if self.testcase in ['single', 'few']:
					noise_labels = self.number*np.ones((batch_size,1)).astype('int32')
				elif self.testcase in ['sharp']:
					noise_labels = np.expand_dims(np.random.choice([1,2,4,5,7,9], batch_size), axis = 1).astype('int32')
				elif self.testcase in ['even']:
					noise_labels = np.expand_dims(np.random.choice([0,2,4,6,8], batch_size), axis = 1).astype('int32')
				elif self.testcase in ['odd']:
					noise_labels = np.expand_dims(np.random.choice([1,3,5,7,9], batch_size), axis = 1).astype('int32')
				elif self.testcase in ['animals']:
					noise_labels = np.expand_dims(np.random.choice([2,3,4,5,6,7], batch_size), axis = 1).astype('int32')
			elif self.data in ['celeba']:
				if self.testcase in ['male', 'fewmale', 'bald', 'hat']:
					noise_labels = np.ones((batch_size,1)).astype('int32')
				elif self.testcase in ['female', 'fewfemale']:
					noise_labels = np.zeros((batch_size,1)).astype('int32')
		if noise_case == 'train':
			noise_labels = np.random.randint(0, self.num_classes, batch_size)
			if self.data == 'celeba':
				noise_labels = np.expand_dims(noise_labels, axis = 1)

		return noise, noise_labels

	def create_models(self):

		# with self.strategy.scope():
		self.total_count = tf.Variable(0,dtype='int64')
		self.generator = eval(self.gen_model)
		self.discriminator = eval(self.disc_model)

		if self.res_flag == 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n GENERATOR MODEL: \n\n")
				self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
				self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

		print("Model Successfully made")

		print(self.generator.summary())
		print(self.discriminator.summary())
		return		

	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
								 generator = self.generator,
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
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size_big) / (self.train_data.shape[0])) + 1))
		return

	def train(self):
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)   
			start = time.perf_counter()
			batch_count = tf.Variable(0, dtype='int64')
			start_time = 0

			for image_batch,labels_batch in self.train_dataset:
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_time = time.perf_counter()
				# with self.strategy.scope():
				self.train_step(image_batch,labels_batch)
				self.eval_metrics()
				train_time = time.perf_counter()-start_time

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

				if (self.total_count.numpy() % 1000) == 0:
					self.test()


			if self.pbar_flag:
				bar.close()
				del bar
			tf.print('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)

	def print_batch_outputs(self,epoch):
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def test(self):
		for i in range(10):

			path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())

			size_figure_grid = self.num_to_print
			test_batch_size = size_figure_grid*size_figure_grid
			noise, noise_labels = self.get_noise('test',test_batch_size)

			if self.label_style == 'base':
				#if base mode, ACGAN generator takes in one_hot labels
				noise_labels = tf.one_hot(np.squeeze(noise_labels), depth = self.num_classes)

			images = self.generator([noise,noise_labels] , training=False)
			if self.data != 'celeba':
				images = (images + 1.0)/2.0
			
			self.save_image_batch(images = images,label = label, path = path)

'''***********************************************************************************
********** GAN RumiGAN setup *********************************************************
***********************************************************************************'''
class GAN_RumiGAN(GAN_SRC, GAN_DATA_RumiGAN):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all GAN architectures'''
		GAN_SRC.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_RumiGAN.__init__(self)

		self.noise_setup()
		self.fixed_noise = self.get_noise([self.num_to_print*self.num_to_print, self.noise_dims])


	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data_pos, self.train_data_neg, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		''' Define dataset and tf.data function. batch sizing done'''
		# self.get_data()

		# self.create_models()

		# self.create_optimizer()

		# self.create_load_checkpoint()
	def noise_setup(self):

		if self.noise_kind == 'non_para':
			self.non_para_dist_mat = sio.loadmat('data/non_param_pdf.mat')

		if self.noise_kind == 'gamma':
			self.gamma_shape = 0.5
			self.gamma_scale = 1.0

		if self.noise_kind == 'trip':
			self.num_latents_trip = 128
			self.num_components_trip = 10
			self.tt_int_trip = 40

		return

	def get_noise(self, shape):
		#shape = [self.batch_size, self.noise_dims]

		def random_gen(shape, pdf, points, epsilon):
			assert len(shape) == 2
			rn = np.random.choice(points, size = shape, p=pdf).astype(np.float32)
			for i in range(shape[0]):
				for j in range(shape[1]):
					rn[i,j] = np.random.uniform(rn[i,j], rn[i,j]+epsilon, 1).astype(np.float32)
			return rn

		def sample_spherical(npoints, ndim=3):
			vec = np.random.randn(ndim, npoints)
			vec /= np.linalg.norm(vec, axis=0)
			return vec

		# def TRIP()

		if self.noise_kind == 'non_para':
			pdf = np.reshape(self.non_para_dist_mat['X_final'], (1024))
			points = np.linspace(-2, 2, 1024)
			epsilon = 4./1024.
			noise = random_gen(shape, pdf, points, epsilon)

		elif self.noise_kind == 'gaussian':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = self.noise_stddev)

		elif self.noise_kind == 'gamma':
			nu = sample_spherical(shape[0], shape[1]).transpose()
			r = np.random.gamma(self.gamma_shape, scale=self.gamma_scale, size=shape[0])
			root_r_mat = np.repeat(np.expand_dims(np.sqrt(r),axis=1), shape[1], axis = 1)

			noise = np.multiply(root_r_mat,nu)
			# print(root_r_mat.shape,nu.shape,noise.shape)
		elif self.noise_kind == 'cauchy':
			noise = np.random.standard_cauchy(size=shape)

		elif self.noise_kind == 'trip':
			prior = TRIP(self.num_latents_trip * (('c', self.num_components_trip),),tt_int=self.tt_int_trip, distr_init='uniform')

		return noise

	def get_data(self):
		
		with tf.device('/CPU'):
			self.train_data_pos, self.train_data_neg = eval(self.gen_func)
			self.max_data_size = max(self.train_data_pos.shape[0],self.train_data_neg.shape[0])

			self.num_batches = int(np.floor(self.max_data_size/self.batch_size))
			''' Set PRINT and SAVE iters if 0'''
			self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
			self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

			self.train_dataset_pos, self.train_dataset_neg = eval(self.dataset_func)

			self.train_dataset_size = self.max_data_size
		print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, 
		 self.num_batches,self.print_step, self.save_step))

	def create_models(self):

		# with self.strategy.scope():
		self.total_count = tf.Variable(0,dtype='int64')
		self.generator = eval(self.gen_model)
		self.discriminator = eval(self.disc_model)

		if self.res_flag == 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n GENERATOR MODEL: \n\n")
				self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
				self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

		print("Model Successfully made")

		print(self.generator.summary())
		print(self.discriminator.summary())
		return		

	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
								 generator = self.generator,
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
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size_big) / (self.train_data.shape[0])) + 1))
		return

	def train(self):    	    
		start = int((self.total_count.numpy() * self.batch_size) / (max(self.train_data_pos.shape[0],self.train_data_neg.shape[0]))) + 1
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)   
			start = time.perf_counter()
			batch_count = tf.Variable(0,dtype='int64')
			start_time =0

			for image_batch_pos,image_batch_neg in zip(self.train_dataset_pos,self.train_dataset_neg):

				self.total_count.assign_add(1)
				batch_count.assign_add(self.Dloop)
				start_time = time.perf_counter()
				# with self.strategy.scope():
				self.train_step(image_batch_pos,image_batch_neg)
				self.eval_metrics()
				train_time = time.perf_counter()-start_time

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
			tf.print('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)

	def print_batch_outputs(self,epoch):
		if self.total_count.numpy() <= 2 and 'g' not in self.data:
			predictions = self.reals_pos[0:self.num_to_print*self.num_to_print]
			if self.data!='celeba':
				predictions = (predictions + 1.0)/(2.0)
			path = self.impath + 'pos.png'
			label = 'POSITIVE CLASS SAMPLES'
			self.save_image_batch(images = predictions,label = label, path = path)
			# eval(self.show_result_func)
			predictions = self.reals_neg[0:self.num_to_print*self.num_to_print]
			if self.data!='celeba':
				predictions = (predictions + 1.0)/(2.0)
			path = self.impath + 'negs.png'
			label = "NEGATIVE CLASS SAMPLES"
			self.save_image_batch(images = predictions,label = label, path = path)
			# eval(self.show_result_func)
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 100) == 0:
			self.generate_and_save_batch(epoch)
		if self.update_fig == 1 and (self.total_count.numpy() % 10) == 0:
			self.generate_and_save_batch(51004)

	def test(self):
		for i in range(self.num_test_images):

			path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())

			size_figure_grid = self.num_to_print
			test_batch_size = size_figure_grid*size_figure_grid
			noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)

			images = self.generator(noise, training=False)
			if self.data != 'celeba':
				images = (images + 1.0)/2.0
			self.save_image_batch(images = images,label = label, path = path)


'''***********************************************************************************
********** WAE-GAN Setup *************************************************************
***********************************************************************************'''
class GAN_WAE(GAN_SRC, GAN_DATA_WAE):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all GAN architectures'''

		GAN_SRC.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_WAE.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')

		self.noise_setup()

	def noise_setup(self):
		self.num_of_components = 20

		probs = list((1/self.num_of_components)*np.ones([self.num_of_components]))
		stddev_scale = list(0.8*np.ones([self.num_of_components]))
		# locs = list(np.random.uniform(size = [10, self.latent_dims], low = 1., high = 8.))
		locs = np.random.uniform(size = [self.num_of_components, self.latent_dims], low = -3., high = 3.)
		self.locs = tf.Variable(locs)
		locs = [list(x) for x in list(locs)]
		
		# print(locs)       #[[7.5, 5], [5, 7.5], [2.5,5], [5,2.5], [7.5*0.7071, 7.5*0.7071], [2.5*0.7071, 7.5*0.7071], [7.5*0.7071, 2.5*0.7071], [2.5*0.7071, 2.5*0.7071] ]
		# stddev_scale = [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5]

		# self.gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))

		# probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
		# locs = [[0.75, 0.5], [0.5, 0.75], [0.25,0.5], [0.5,0.25], [0.5*1.7071, 0.5*1.7071], [0.5*0.2929, 0.5*1.7071], [0.5*1.7071, 0.5*0.2929], [0.5*0.2929, 0.5*0.2929] ]
		# stddev_scale = [.04, .04, .04, .04, .04, .04, .04, .04]
		self.gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))

		self.gN = tfd.Normal(loc=1.25, scale=1.)

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.encdec_model = 'self.encdec_model_'+self.arch+'_'+self.data+'()'
		# self.disc_model = 'self.discriminator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.arch+'_'+self.data+'()'  
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		if self.FID_kind == 'torch':
			self.FID_func = 'self.FID_torch_'+self.data+'()'
		else:
			self.FID_func = 'self.FID_'+self.data+'()'

		if self.loss == 'FS':
			self.disc_model = 'self.discriminator_model_FS()' 
			self.DEQ_func = 'self.discriminator_ODE()'

		''' Define dataset and tf.data function. batch sizing done'''
		# self.get_data()
		# print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.num_batches,self.print_step, self.save_step))

	def get_data(self):
		# with tf.device('/CPU'):
		self.train_data = eval(self.gen_func)

		# self.batch_size = self.batch_size * self.strategy.num_replicas_in_sync

		# self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
		self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)
		self.train_dataset_size = self.train_data.shape[0]

		# self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
		# self.recon_dataset = self.strategy.experimental_distribute_dataset(self.recon_dataset)

		print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size,  self.num_batches,self.print_step, self.save_step))
		### with was till here

	def get_noise(self,batch_size):
		###Uncomment for the continues CelebaCode on Vega
		if self.noise_kind == 'gaussian_trunc':
			noise = tfp.distributions.TruncatedNormal(loc=0., scale=0.3, low=-1., high=1.).sample([batch_size, self.latent_dims])

		###Uncomment for the continues CIFAR10Code on Vayu
		if self.noise_kind == 'gmm':
			noise = self.gmm.sample(sample_shape=(int(batch_size.numpy())))

		if self.noise_kind == 'gN':
			noise = self.gN.sample(sample_shape=(int(batch_size.numpy()),self.latent_dims))


		# tf.random.normal([100, self.latent_dims], mean = self.locs.numpy()[i], stddev = 1.)
		if self.noise_kind == 'gaussian':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = 1.0)

		if self.noise_kind == 'gaussian_s2':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = np.sqrt(2))

		if self.noise_kind == 'gaussian_s4':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = 2)

		if self.noise_kind == 'gaussian_1m1':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = 0.25)

		if self.noise_kind == 'gaussian_05':
			noise = tfp.distributions.TruncatedNormal(loc=2.5, scale=1., low=0., high=5.).sample([batch_size, self.latent_dims])

		if self.noise_kind == 'gaussian_02':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.7*np.ones((1,self.latent_dims)), stddev = 0.2*np.ones((1,self.latent_dims)))

		if self.noise_kind == 'gaussian_01':
			noise = tfp.distributions.TruncatedNormal(loc=0.5, scale=0.2, low=0., high=1.).sample([batch_size, self.latent_dims])
		return noise


	def train(self):    
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1 
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)
			start = time.perf_counter()
			batch_count = tf.Variable(0,dtype='int64')
			start_1 = 0

			# if epoch>10:
			# 	if self.loss == 'SW':
			# 		self.lambda_SW *= 1.1

			for image_batch in self.train_dataset:
				self.total_count.assign_add(1)
				batch_count.assign_add(self.Dloop)
				start_1 = time.perf_counter()
				
				# with self.strategy.scope():
				if epoch <= self.GAN_pretrain_epochs or epoch <= self.AE_pretrain_epochs:
					if epoch <= self.GAN_pretrain_epochs:
						self.pretrain_step_GAN(image_batch)
					if epoch <= self.AE_pretrain_epochs:
						self.pretrain_step_AE(image_batch)
				else:
					self.train_step(image_batch)
					self.eval_metrics()
				
				train_time = time.perf_counter()-start_1

				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():4.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.postfix[3] = f'{self.AE_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())

				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f}; AE_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy(),self.AE_loss.numpy()))

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

			tf.print ('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
			self.save_epoch_h5models()

	def print_batch_outputs(self,epoch):		
		if (self.total_count.numpy() <= 2) or ((self.total_count.numpy() % self.save_step.numpy()) == 0) or ((self.total_count.numpy() % 250) ==0):
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 1000) == 0:
			if (self.total_count.numpy() % 20000) == 0:
				with tf.device("/CPU"):
					self.test_full()
			else:
				with tf.device("/CPU"):
					self.test()
			# assert tf.distribute.get_replica_context() is None
			# self.strategy.run(self.test, args=())
			
	def test(self):

		###### Random Samples
		for i in range(10):
			path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			noise = self.get_noise(self.batch_size)
			images = self.Decoder(noise)

			images = (images + 1.0)/2.0
			# sharpness = self.find_sharpness(images)
			# print(sharpness)
			# exit(0)
			# try:
			# 	sharpness_vec.append(sharpness)
			# except:
			# 	shaprpness_vec = [sharpness]

			size_figure_grid = self.num_to_print
			images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			# fig.text(0.5, 0.04, label, ha='center')
			plt.savefig(path)
			plt.close()

		###### Random Samples - Sharpness averaging measure
		# overall_sharpness = np.mean(np.array(shaprpness_vec))
		# if self.mode == 'test':
		# 	print("\n Random Sharpness - " + str(overall_sharpness))
		# if self.res_flag:
		# 	self.res_file.write("\n Random Sharpness - "+str(overall_sharpness))

		# i = 0
		# for image_batch in self.train_dataset:
		# 	i+=1
		# 	image_batch = (image_batch+1.0)/2.0
		# 	sharpness = self.find_sharpness(image_batch)
		# 	try:
		# 		sharpness_vec.append(sharpness)
		# 	except:
		# 		shaprpness_vec = [sharpness]
		# 	if i==100:
		# 		break

		# overall_sharpness = np.mean(np.array(shaprpness_vec))
		# if self.mode == 'test':
		# 	print("\n Dataset Sharpness 10k samples - " + str(overall_sharpness))
		# if self.res_flag:
		# 	self.res_file.write("\n Dataset Sharpness 10k samples - "+str(overall_sharpness))


		# ####### Recon - Output
		for image_batch in self.recon_dataset:				
			path = self.impath+'_TestingRecon_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			images = self.Decoder(self.Encoder(image_batch))

			# if self.topic in ['PolyGAN', 'WAEMMD']:
			# 	try:
			# 		recon_loss += tf.reduce_mean(tf.abs(image_batch - images))
			# 		recon_loss *= 0.5
			# 	except:
			# 		recon_loss = tf.reduce_mean(tf.abs(image_batch - images))
			# else:
			# 	try:
			# 		recon_loss = 0.5*(recon_loss) + 0.25*tf.reduce_mean(tf.abs(image_batch - images)) + 0.75*(mse(image_batch,images))
			# 	except:
			# 		recon_loss =0.5*tf.reduce_mean(tf.abs(image_batch - images))+1.5*(mse(image_batch,images))


			images = (images + 1.0)/2.0
			size_figure_grid = self.num_to_print
			images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()

		# ###### Recon - org
			path = self.impath+'_TestingReconOrg_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			images = image_batch
			images = (images + 1.0)/2.0
			size_figure_grid = self.num_to_print
			images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()
			break
			

		# if self.mode == 'test':
		# 	print("\n Reconstruction error - " + str(recon_loss))
		# if self.res_flag:
		# 	self.res_file.write("\n Reconstruction error - " + str(recon_loss))


		# ####### Interpolation
		# num_interps = self.num_to_print
		# if self.mode == 'test':
		# 	num_figs = int(400/(2*num_interps))
		# else:
		# 	num_figs = 9
		# # there are 400 samples in the batch. to make 10x10 images, 
		# for image_batch in self.interp_dataset:
		# 	for j in range(num_figs):
		# 		path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
		# 		current_batch = image_batch[2*num_interps*j:2*num_interps*(j+1)]
		# 		image_latents = self.Encoder(current_batch)
		# 		current_batch = (current_batch + 1.0)/2.0
		# 		for i in range(num_interps):
		# 			start = image_latents[i:1+i].numpy()
		# 			end = image_latents[num_interps+i:num_interps+1+i].numpy()
		# 			stack = np.vstack([start, end])

		# 			linfit = interp1d([1,num_interps+1], stack, axis=0)
		# 			interp_latents = linfit(list(range(1,num_interps+1)))
		# 			cur_interp_figs = self.Decoder(interp_latents)

		# 			cur_interp_figs = (cur_interp_figs + 1.0)/2.0

		# 			sharpness = self.find_sharpness(cur_interp_figs)

		# 			try:
		# 				sharpness_vec.append(sharpness)
		# 			except:
		# 				shaprpness_vec = [sharpness]
		# 			cur_interp_figs_with_ref = np.concatenate((current_batch[i:1+i],cur_interp_figs.numpy(),current_batch[num_interps+i:num_interps+1+i]), axis = 0)
		# 			# print(cur_interp_figs_with_ref.shape)
		# 			try:
		# 				batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs_with_ref),axis = 0)
		# 			except:
		# 				batch_interp_figs = cur_interp_figs_with_ref

		# 		images = batch_interp_figs#(batch_interp_figs + 1.0)/2.0
		# 		# print(images.shape)
		# 		size_figure_grid = num_interps
		# 		images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid+2),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 		fig1 = plt.figure(figsize=(num_interps,num_interps+2))
		# 		ax1 = fig1.add_subplot(111)
		# 		ax1.cla()
		# 		ax1.axis("off")
		# 		if images_on_grid.shape[2] == 3:
		# 			ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 		else:
		# 			ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 		label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
		# 		plt.title(label)
		# 		plt.tight_layout()
		# 		plt.savefig(path)
		# 		plt.close()
		# 		del batch_interp_figs

		# ###### Interpol samples - Sharpness
		# overall_sharpness = np.mean(np.array(shaprpness_vec))
		# if self.mode == 'test':
		# 	print("\n Interpolation Sharpness - " + str(overall_sharpness))
		# if self.res_flag:
		# 	self.res_file.write("\n Interpolation Sharpness - "+str(overall_sharpness))

		# ###### Kurtosis and Skewness:
		# for image_batch in self.interp_dataset:
		# 	encoded = self.Encoder(image_batch, training = False).numpy()
		# 	try:
		# 		encoded_all = np.concatenate((encoded_all,encoded), axis=0)
		# 	except:
		# 		encoded_all = encoded

		# self.eval_MardiaStats(encoded_all)
		# if self.mode =='test':
		# 	print("\n Final skewness score - "+str(self.skewness))
		# 	print("\n Final Kurtosis score - "+str(self.kurtosis))


	def test_full(self):

		###### Random Samples
		for i in range(10):
			path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			noise = self.get_noise(self.batch_size)
			images = self.Decoder(noise)

			images = (images + 1.0)/2.0
			sharpness = self.find_sharpness(images)
			# print(sharpness)
			# exit(0)
			try:
				sharpness_vec.append(sharpness)
			except:
				shaprpness_vec = [sharpness]

			size_figure_grid = self.num_to_print
			images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			# fig.text(0.5, 0.04, label, ha='center')
			plt.savefig(path)
			plt.close()

		###### Random Samples - Sharpness averaging measure
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("\n Random Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("\n Random Sharpness - "+str(overall_sharpness))

		i = 0
		for image_batch in self.train_dataset:
			i+=1
			image_batch = (image_batch+1.0)/2.0
			sharpness = self.find_sharpness(image_batch)
			try:
				sharpness_vec.append(sharpness)
			except:
				shaprpness_vec = [sharpness]
			if i==100:
				break

		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("\n Dataset Sharpness 10k samples - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("\n Dataset Sharpness 10k samples - "+str(overall_sharpness))


		# ####### Recon - Output
		for image_batch in self.recon_dataset:				
			path = self.impath+'_TestingRecon_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			images = self.Decoder(self.Encoder(image_batch))

			if self.topic in ['PolyGAN', 'WAEMMD']:
				try:
					recon_loss += tf.reduce_mean(tf.abs(image_batch - images))
					recon_loss *= 0.5
				except:
					recon_loss = tf.reduce_mean(tf.abs(image_batch - images))
			else:
				try:
					recon_loss = 0.5*(recon_loss) + 0.25*tf.reduce_mean(tf.abs(image_batch - images)) + 0.75*(mse(image_batch,images))
				except:
					recon_loss =0.5*tf.reduce_mean(tf.abs(image_batch - images))+1.5*(mse(image_batch,images))


			images = (images + 1.0)/2.0
			size_figure_grid = self.num_to_print
			images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()

		# ###### Recon - org
			path = self.impath+'_TestingReconOrg_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			images = image_batch
			images = (images + 1.0)/2.0
			size_figure_grid = self.num_to_print
			images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()
			break

		if self.mode == 'test':
			print("\n Reconstruction error - " + str(recon_loss))
		if self.res_flag:
			self.res_file.write("\n Reconstruction error - " + str(recon_loss))


		####### Interpolation
		num_interps = self.num_to_print
		if self.mode == 'test':
			num_figs = int(400/(2*num_interps))
		else:
			num_figs = 9
		# there are 400 samples in the batch. to make 10x10 images, 
		for image_batch in self.interp_dataset:
			for j in range(num_figs):
				path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
				current_batch = image_batch[2*num_interps*j:2*num_interps*(j+1)]
				image_latents = self.Encoder(current_batch)
				current_batch = (current_batch + 1.0)/2.0
				for i in range(num_interps):
					start = image_latents[i:1+i].numpy()
					end = image_latents[num_interps+i:num_interps+1+i].numpy()
					stack = np.vstack([start, end])

					linfit = interp1d([1,num_interps+1], stack, axis=0)
					interp_latents = linfit(list(range(1,num_interps+1)))
					cur_interp_figs = self.Decoder(interp_latents)

					cur_interp_figs = (cur_interp_figs + 1.0)/2.0

					sharpness = self.find_sharpness(cur_interp_figs)

					try:
						sharpness_vec.append(sharpness)
					except:
						shaprpness_vec = [sharpness]
					cur_interp_figs_with_ref = np.concatenate((current_batch[i:1+i],cur_interp_figs.numpy(),current_batch[num_interps+i:num_interps+1+i]), axis = 0)
					# print(cur_interp_figs_with_ref.shape)
					try:
						batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs_with_ref),axis = 0)
					except:
						batch_interp_figs = cur_interp_figs_with_ref

				images = batch_interp_figs#(batch_interp_figs + 1.0)/2.0
				# print(images.shape)
				size_figure_grid = num_interps
				images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid+2),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
				fig1 = plt.figure(figsize=(num_interps,num_interps+2))
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.axis("off")
				if images_on_grid.shape[2] == 3:
					ax1.imshow(np.clip(images_on_grid,0.,1.))
				else:
					ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

				label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
				plt.title(label)
				plt.tight_layout()
				plt.savefig(path)
				plt.close()
				del batch_interp_figs

		###### Interpol samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("\n Interpolation Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("\n Interpolation Sharpness - "+str(overall_sharpness))

		###### Kurtosis and Skewness:
		for image_batch in self.interp_dataset:
			encoded = self.Encoder(image_batch, training = False).numpy()
			try:
				encoded_all = np.concatenate((encoded_all,encoded), axis=0)
			except:
				encoded_all = encoded

		self.eval_MardiaStats(encoded_all)
		if self.mode =='test':
			print("\n Final skewness score - "+str(self.skewness))
			print("\n Final Kurtosis score - "+str(self.kurtosis))

'''***********************************************************************************
********** GAN Baseline setup ********************************************************
***********************************************************************************'''
class GAN_Flow(GAN_SRC, GAN_DATA_Base):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all fundamental ops and metric functions'''
		GAN_SRC.__init__(self,FLAGS_dict)
		''' Set up the GAN_DATA class'''
		GAN_DATA_Base.__init__(self)

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.arch+'_'+self.data+'()'
		if self.data not in ['g2','g1','gN']:
			self.score_model = 'self.score_model_'+self.arch+'_'+self.data+'()'
		else:
			#### Makes no difference. Dummy variable
			self.score_model = 'self.generator_model_'+self.arch+'_'+self.data+'()'

		self.disc_model = 'self.discriminator_model_'+self.arch+'_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		self.noise_setup()

	def get_data(self):
		with tf.device('/CPU'):
			self.train_data = eval(self.gen_func)

			self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
			''' Set PRINT and SAVE iters if 0'''
			self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
			self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

			self.train_dataset = eval(self.dataset_func)
			self.train_dataset_size = self.train_data.shape[0]

			# self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)

			print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size,  self.num_batches,self.print_step, self.save_step))


	# ###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	# def create_models(self):
	# 	# with self.strategy.scope():
	# 	self.total_count = tf.Variable(0,dtype='int64')
	# 	self.generator = eval(self.gen_model)
	# 	self.ScoreNet = eval(self.score_model)
	# 	self.discriminator = eval(self.disc_model)

	# 	if self.res_flag == 1:
	# 		with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
	# 			# Pass the file handle in as a lambda function to make it callable
	# 			fh.write("\n\n GENERATOR MODEL: \n\n")
	# 			self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
	# 			fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
	# 			self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
	# 			fh.write("\n\n SCORE MODEL: \n\n")
	# 			self.ScoreNet.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

	# 	print("Model Successfully made")

	# 	print(self.generator.summary())
	# 	print(self.discriminator.summary())
	# 	print(self.ScoreNet.summary())
	# 	return		


	# ###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	# def create_load_checkpoint(self):

	# 	self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
	# 							 Disc_optimizer = self.Disc_optimizer,
	# 							 S_optimizer = self.S_optimizer,
	# 							 generator = self.generator,
	# 							 ScoreNet = self.ScoreNet,
	# 							 discriminator = self.discriminator,
	# 							 total_count = self.total_count)
	# 	self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
	# 	self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

	# 	if self.resume:
	# 		try:
	# 			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
	# 		except:
	# 			print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
	# 			try:
	# 				self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
	# 				self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
	# 			except:
	# 				print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

	# 		print("Model restored...")
	# 		print("Starting at Iteration - "+str(self.total_count.numpy()))
	# 		print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
	# 	return

	def noise_setup(self):

		if self.noise_kind == 'non_para':
			self.non_para_dist_mat = sio.loadmat('data/non_param_pdf.mat')

		if self.noise_kind == 'gamma':
			self.gamma_shape = 0.5
			self.gamma_scale = 1.0

		if self.noise_kind == 'trip':
			self.num_latents_trip = 128
			self.num_components_trip = 10
			self.tt_int_trip = 40

		return

	def get_noise(self, shape):
		#shape = [self.batch_size, self.noise_dims]

		def random_gen(shape, pdf, points, epsilon):
			assert len(shape) == 2
			rn = np.random.choice(points, size = shape, p=pdf).astype(np.float32)
			for i in range(shape[0]):
				for j in range(shape[1]):
					rn[i,j] = np.random.uniform(rn[i,j], rn[i,j]+epsilon, 1).astype(np.float32)
			return rn

		def sample_spherical(npoints, ndim=3):
			vec = np.random.randn(ndim, npoints)
			vec /= np.linalg.norm(vec, axis=0)
			return vec

		# def TRIP()

		if self.noise_kind == 'non_para':
			pdf = np.reshape(self.non_para_dist_mat['X_final'], (1024))
			points = np.linspace(-2, 2, 1024)
			epsilon = 4./1024.
			noise = random_gen(shape, pdf, points, epsilon)

		elif self.noise_kind == 'gaussian':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = self.noise_stddev)

		elif self.noise_kind == 'gaussian075':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = 0.75)

		elif self.noise_kind == 'gamma':
			nu = sample_spherical(shape[0], shape[1]).transpose()
			r = np.random.gamma(self.gamma_shape, scale=self.gamma_scale, size=shape[0])
			root_r_mat = np.repeat(np.expand_dims(np.sqrt(r),axis=1), shape[1], axis = 1)

			noise = np.multiply(root_r_mat,nu)
			# print(root_r_mat.shape,nu.shape,noise.shape)
		elif self.noise_kind == 'cauchy':
			noise = np.random.standard_cauchy(size=shape)

		elif self.noise_kind == 'trip':
			prior = TRIP(self.num_latents_trip * (('c', self.num_components_trip),),tt_int=self.tt_int_trip, distr_init='uniform')

		return noise

	def train(self):

		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)   
			start = time.perf_counter()
			batch_count = tf.Variable(0,dtype='int64')
			start_time =0

			for image_batch in self.train_dataset:
				# print(image_batch.shape)
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_time = time.perf_counter()
				# with self.strategy.scope():
				self.train_step(image_batch.numpy())
				self.eval_metrics()
				self.D_loss = self.G_loss = tf.constant(0)
						

				train_time = time.perf_counter()-start_time

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

				if self.iters_flag:
					if self.num_iters == self.total_count.numpy():
						tf.print("\n Training for {} Iterations completed".format( self.total_count.numpy()))
						if self.pbar_flag:
							bar.close()
							del bar
						tf.print('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
						self.save_epoch_h5models()
						return

			if self.pbar_flag:
				bar.close()
				del bar
			tf.print('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
			self.save_epoch_h5models()


	def save_epoch_h5models(self):

		# self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)

		# if self.loss == 'FS':
		# 	self.discriminator_A.save(self.checkpoint_dir + '/model_discriminator_A.h5', overwrite = True)
		# 	self.discriminator_B.save(self.checkpoint_dir + '/model_discriminator_B.h5', overwrite = True)
		# elif self.loss == 'RBF':
		# 	self.discriminator_RBF.save(self.checkpoint_dir +'/model_discriminator_RBF.h5',overwrite=True)
		# elif self.topic not in ['SnakeGAN', 'ScoreGAN']:
		# 	self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return


	def print_batch_outputs(self,epoch):
		if ((self.total_count.numpy() % 5) == 0 and self.data in ['g1', 'g2']):### Was 10 - ICML22 plots
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() <= 5) and self.data in [ 'g1', 'g2', 'gmm2', 'gmm8']:
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 100) == 0 and self.data in ['gmm2', 'gmm8']):
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 10) == 0 and self.data in ['gmm2', 'gmm8'] and self.topic in ['ScoreGAN']):
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 100) == 0 and self.data in ['celeba']):
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def eval_sharpness(self):
		i = 0
		for train_batch in self.train_dataset:
			noise = self.get_noise([self.batch_size, self.noise_dims])
			preds = self.generator(noise, training = False)

			sharp = self.find_sharpness(preds)
			base_sharp = self.find_sharpness(train_batch)
			try:
				sharp_vec.append(sharp)
				base_sharp_vec.append(base_sharp)

			except:
				sharp_vec = [sharp]
				base_sharp_vec = [base_sharp]
			i += 1
			if i == 10:
				break
		###### Sharpness averaging measure
		sharpness = np.mean(np.array(sharp_vec))
		baseline_sharpness = np.mean(np.array(base_sharp_vec))

		return baseline_sharpness, sharpness



	def test(self):
		num_interps = 10
		if self.mode == 'test':
			num_figs = 20#int(400/(2*num_interps))
		else:
			num_figs = 9
		# there are 400 samples in the batch. to make 10x10 images, 
		for j in range(num_figs):
			# print("Interpolation Testing Image ",j)
			path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
			# noise = self.get_noise([20*num_figs, self.noise_dims])
			# current_batch = noise[2*num_interps*j:2*num_interps*(j+1)]
			# image_latents = self.generator(current_batch)
			for i in range(num_interps):
				# print("Pair ",i)
				# current_batch = self.get_noise([20*num_figs, self.noise_dims])
				start = self.get_noise([1, self.noise_dims])#current_batch[i:1+i].numpy()
				end = self.get_noise([1, self.noise_dims]) #current_batch[num_interps+i:num_interps+1+i].numpy()
				stack = np.vstack([start, end])
				# print(stack)



				linfit = interp1d([1,num_interps+1], stack, axis=0)
				interp_latents = linfit(list(range(1,num_interps+1)))

				# print(interp_latents)
				cur_interp_figs = self.generator(interp_latents)

				# print(cur_interp_figs)

				sharpness = self.find_sharpness(cur_interp_figs)

				try:
					sharpness_vec.append(sharpness)
				except:
					shaprpness_vec = [sharpness]
				# cur_interp_figs_with_ref = np.concatenate((current_batch[i:1+i],cur_interp_figs.numpy(),current_batch[num_interps+i:num_interps+1+i]), axis = 0)
				# print(cur_interp_figs_with_ref.shape)
				try:
					batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs),axis = 0)
				except:
					batch_interp_figs = cur_interp_figs

			images = (batch_interp_figs + 1.0)/2.0
			# print(images.shape)
			size_figure_grid = num_interps
			images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(num_interps,num_interps))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()
			del batch_interp_figs

		###### Interpol samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Interpolation Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("Interpolation Sharpness - "+str(overall_sharpness))


'''***********************************************************************************
********** WAE-GAN Setup *************************************************************
***********************************************************************************'''
class GAN_MMDGAN(GAN_SRC, GAN_DATA_WAE):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all GAN architectures'''

		GAN_SRC.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_WAE.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')

		self.noise_setup()

	def noise_setup(self):

		if self.noise_kind == 'non_para':
			self.non_para_dist_mat = sio.loadmat('data/non_param_pdf.mat')

		if self.noise_kind == 'gamma':
			self.gamma_shape = 0.5
			self.gamma_scale = 1.0

		if self.noise_kind == 'trip':
			self.num_latents_trip = 128
			self.num_components_trip = 10
			self.tt_int_trip = 40

		return

	def get_noise(self, shape):
		#shape = [self.batch_size, self.noise_dims]

		def random_gen(shape, pdf, points, epsilon):
			assert len(shape) == 2
			rn = np.random.choice(points, size = shape, p=pdf).astype(np.float32)
			for i in range(shape[0]):
				for j in range(shape[1]):
					rn[i,j] = np.random.uniform(rn[i,j], rn[i,j]+epsilon, 1).astype(np.float32)
			return rn

		def sample_spherical(npoints, ndim=3):
			vec = np.random.randn(ndim, npoints)
			vec /= np.linalg.norm(vec, axis=0)
			return vec

		# def TRIP()

		if self.noise_kind == 'non_para':
			pdf = np.reshape(self.non_para_dist_mat['X_final'], (1024))
			points = np.linspace(-2, 2, 1024)
			epsilon = 4./1024.
			noise = random_gen(shape, pdf, points, epsilon)

		elif self.noise_kind == 'gaussian':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = self.noise_stddev)

		elif self.noise_kind == 'gamma':
			nu = sample_spherical(shape[0], shape[1]).transpose()
			r = np.random.gamma(self.gamma_shape, scale=self.gamma_scale, size=shape[0])
			root_r_mat = np.repeat(np.expand_dims(np.sqrt(r),axis=1), shape[1], axis = 1)

			noise = np.multiply(root_r_mat,nu)
			# print(root_r_mat.shape,nu.shape,noise.shape)
		elif self.noise_kind == 'cauchy':
			noise = np.random.standard_cauchy(size=shape)

		elif self.noise_kind == 'trip':
			prior = TRIP(self.num_latents_trip * (('c', self.num_components_trip),),tt_int=self.tt_int_trip, distr_init='uniform')

		return noise

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		## In MMD-GANs, the Discriminator is autoencoding, while the generator is not. 
		self.gen_model = 'self.generator_model_'+self.arch+'_'+self.data+'()'
		self.encdec_model = 'self.encdec_model_'+self.arch+'_'+self.data+'()'

		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		# if self.loss == 'FS':
		# 	self.disc_model = 'self.discriminator_model_FS()' 
		# 	self.DEQ_func = 'self.discriminator_ODE()'


	def get_data(self):
		with tf.device('/CPU'):
			self.train_data = eval(self.gen_func)

			# self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
			self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
			''' Set PRINT and SAVE iters if 0'''
			self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
			self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

			self.train_dataset = eval(self.dataset_func)
			self.train_dataset_size = self.train_data.shape[0]

			print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size,  self.num_batches,self.print_step, self.save_step))


	def train(self):  

		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1 
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)
			start = time.perf_counter()
			batch_count = tf.Variable(0,dtype='int64')
			start_1 = 0

			for image_batch in self.train_dataset:
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_1 = time.perf_counter()
				
				# with self.strategy.scope():
				if epoch <= self.AE_pretrain_epochs:
					self.pretrain_step_AE(image_batch)
				else:
					self.train_step(image_batch)
					self.eval_metrics()
						
				
				train_time = time.perf_counter()-start_1

				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():4.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.postfix[3] = f'{self.AE_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())

				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f}; AE_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy(),self.AE_loss.numpy()))

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

			tf.print ('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
			self.save_epoch_h5models()

	def print_batch_outputs(self,epoch):		
		if (self.total_count.numpy() <= 2) or ((self.total_count.numpy() % self.save_step.numpy()) == 0) or ((self.total_count.numpy() % 250) ==0):
			self.generate_and_save_batch(epoch)
		# if (self.total_count.numpy() % self.test_steps) == 0:
		# 	self.test()


	def test(self):
		###### Random Samples
		for i in range(10):
			path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			noise = self.get_noise([self.batch_size, self.noise_dims])
			images = self.generator(noise)

			sharpness = self.find_sharpness(images)
			try:
				sharpness_vec.append(sharpness)
			except:
				shaprpness_vec = [sharpness]

			images = (images + 1.0)/2.0
			size_figure_grid = self.num_to_print
			images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			# fig.text(0.5, 0.04, label, ha='center')
			plt.savefig(path)
			plt.close()

		###### Random Samples - Sharpness averaging measure
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Random Sharpness - " + str(overall_sharpness))
			if self.res_flag:
				self.res_file.write("Random Sharpness - "+str(overall_sharpness))
		else:
			if self.res_flag:
				self.res_file.write("Random Sharpness - "+str(overall_sharpness))

		i = 0
		for image_batch in self.train_dataset:
			i+=1
			sharpness = self.find_sharpness(image_batch)
			try:
				sharpness_vec.append(sharpness)
			except:
				shaprpness_vec = [sharpness]
			if i==100:
				break

		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Dataset Sharpness 10k samples - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("Dataset Sharpness 10k samples - "+str(overall_sharpness))


		# # # ####### Recon - Output
		# for image_batch in self.recon_dataset:				
		# 	path = self.impath+'_TestingRecon_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
		# 	images = self.Decoder(self.Encoder(image_batch))
		# 	images = (images + 1.0)/2.0
		# 	size_figure_grid = self.num_to_print
		# 	images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 	fig1 = plt.figure(figsize=(7,7))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.axis("off")
		# 	if images_on_grid.shape[2] == 3:
		# 		ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 	else:
		# 		ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 	label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
		# 	plt.title(label)
		# 	plt.tight_layout()
		# 	plt.savefig(path)
		# 	plt.close()

		# # ###### Recon - org
		# 	path = self.impath+'_TestingReconOrg_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
		# 	images = image_batch
		# 	images = (images + 1.0)/2.0
		# 	size_figure_grid = self.num_to_print
		# 	images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 	fig1 = plt.figure(figsize=(7,7))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.axis("off")
		# 	if images_on_grid.shape[2] == 3:
		# 		ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 	else:
		# 		ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 	label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
		# 	plt.title(label)
		# 	plt.tight_layout()
		# 	plt.savefig(path)
		# 	plt.close()
		# 	break

		####### Interpolation
		num_interps = self.num_to_print
		if self.mode == 'test':
			num_figs = 20#int(400/(2*num_interps))
		else:
			num_figs = 9
		# there are 400 samples in the batch. to make 10x10 images, 
		for j in range(num_figs):
			# print("Interpolation Testing Image ",j)
			path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
			# noise = self.get_noise([20*num_figs, self.noise_dims])
			# current_batch = noise[2*num_interps*j:2*num_interps*(j+1)]
			# image_latents = self.generator(current_batch)
			for i in range(num_interps):
				start = self.get_noise([1, self.noise_dims])#current_batch[i:1+i].numpy()
				end = self.get_noise([1, self.noise_dims]) #current_batch[num_interps+i:num_interps+1+i].numpy()
				stack = np.vstack([start, end])

				linfit = interp1d([1,num_interps+1], stack, axis=0)
				interp_latents = linfit(list(range(1,num_interps+1)))

				cur_interp_figs = self.generator(interp_latents)

				sharpness = self.find_sharpness(cur_interp_figs)

				try:
					sharpness_vec.append(sharpness)
				except:
					shaprpness_vec = [sharpness]
				try:
					batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs),axis = 0)
				except:
					batch_interp_figs = cur_interp_figs

			images = (batch_interp_figs + 1.0)/2.0
			size_figure_grid = num_interps
			images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(num_interps,num_interps))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()
			del batch_interp_figs

		###### Interpol samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Interpolation Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("Interpolation Sharpness - "+str(overall_sharpness))


'''***********************************************************************************
********** GAN CycleGAN setup *******************************************************
***********************************************************************************'''
class GAN_CycleGAN(GAN_SRC, GAN_DATA_CycleGAN):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all fundamental ops and metric functions'''
		GAN_SRC.__init__(self,FLAGS_dict)
		''' Set up the GAN_DATA class'''
		GAN_DATA_CycleGAN.__init__(self)

	def initial_setup(self):

		''' Initial Setup function. define function names '''
		self.gen_func_A = 'self.gen_func_'+self.data_A+'()'
		self.gen_func_B = 'self.gen_func_'+self.data_B+'()'
		self.gen_model = 'self.generator_model_'+self.arch+'_'+self.data_A+'_'+self.data_B+'()'
		self.disc_model = 'self.discriminator_model_'+self.arch+'_'+self.data_A+'_'+self.data_B+'()'
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func_A = 'self.dataset_'+self.data_A+'(self.train_data_A, self.batch_size, reps_A)'
		self.dataset_func_B = 'self.dataset_'+self.data_B+'(self.train_data_B, self.batch_size, reps_B)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		# self.noise_setup()

	def get_data(self):

		with tf.device('/CPU'):

			#### From Spider to here, data -> B and noise -> A
			self.train_data_B = eval(self.gen_func_B)
			print(self.train_data_B.shape)
			self.train_data_A = eval(self.gen_func_A)
			print(self.train_data_A.shape)
			self.ratio = self.train_data_B.shape[0]/self.train_data_A.shape[0] # is the num of reps noise data needs, to match train data
			reps_B = np.ceil(1/float(self.ratio))
			reps_A = np.ceil(self.ratio)
			print("reps_dataset_A",reps_A)
			print("reps_dataset_B",reps_B)

			# self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
			self.max_data_size = max(self.train_data_B.shape[0],self.train_data_A.shape[0])
			self.num_batches = int(np.floor(self.max_data_size/self.batch_size))
			''' Set PRINT and SAVE iters if 0'''
			self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
			self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

			self.train_dataset_A = eval(self.dataset_func_A)
			self.train_dataset_B = eval(self.dataset_func_B)

			self.train_dataset_size = self.max_data_size

		print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size,  self.num_batches, self.print_step, self.save_step))


	def train(self):    
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data_B.shape[0])) + 1 
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)
			start = time.perf_counter()
			batch_count = tf.Variable(0,dtype='int64')
			start_1 = 0

			for image_batch_A, image_batch_B in zip(self.train_dataset_A, self.train_dataset_B):
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_1 = time.perf_counter()


				# with self.strategy.scope():
				self.train_step(image_batch_A, image_batch_B)
				self.eval_metrics()

				train_time = time.perf_counter()-start_1
					
				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():6.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())
				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; G_BA_loss - {:>2.4f}; D_A_loss - {:>2.4f}; G_AB_loss - {:>2.4f}; D_B_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.Gen_BA_loss.numpy(),self.Disc_A_loss.numpy(),self.Gen_AB_loss.numpy(),self.Disc_B_loss.numpy()))

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
			tf.print ('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
			self.save_epoch_h5models()


	def print_batch_outputs(self,epoch):		
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		# if (self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']:
		# 	self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)


	# def save_epoch_h5models(self):
	# 	if self.arch!='biggan':
	# 		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
	# 		self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
	# 	return


	def print_batch_outputs(self,epoch):
		if (self.total_count.numpy() <= 5) and self.data in ['g1', 'g2']:
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']):
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def eval_sharpness(self):
		i = 0
		for train_batch, noise_batch in zip(self.train_dataset, self.noise_dataset):
			preds = self.generator(noise_batch, training = False)

			sharp = self.find_sharpness(preds)
			base_sharp = self.find_sharpness(train_batch)
			try:
				sharp_vec.append(sharp)
				base_sharp_vec.append(base_sharp)

			except:
				sharp_vec = [sharp]
				base_sharp_vec = [base_sharp]
			i += 1
			if i == 10:
				break

		###### Sharpness averaging measure
		sharpness = np.mean(np.array(sharp_vec))
		baseline_sharpness = np.mean(np.array(base_sharp_vec))

		return baseline_sharpness, sharpness


	##### In SpiderGAN form. Needs rewriting
	def test(self):
		num_interps = 10
		if self.mode == 'test':
			num_figs = 20#int(400/(2*num_interps))
		else:
			num_figs = 9

		fig_count = 0
		# there are 400 samples in the batch. to make 10x10 images, 
		for noise_batch in self.noise_dataset:

			for j in range(5):
				fig_count += 1
				path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(fig_count)+'.png'
				noi_path = self.impath+'_TestingInterpolationV2_NOISE_'+str(self.total_count.numpy())+'_TestCase_'+str(fig_count)+'.png'
				OP1_path = self.impath+'_TestingInterpolationV2_OP1_'+str(self.total_count.numpy())+'_TestCase_'+str(fig_count)+'.png'

				current_batch = noise_batch[2*num_interps*j:2*num_interps*(j+1)]
				# image_latents = self.Encoder(current_batch)
				for i in range(num_interps):
					start = np.reshape(current_batch[i:1+i].numpy(),(self.input_size*self.input_size*self.input_dims))
					end = np.reshape(current_batch[num_interps+i:num_interps+1+i].numpy(),(self.input_size*self.input_size*self.input_dims))

					# print(start.shape, end.shape)
					stack = np.vstack([start, end])

					linfit = interp1d([1,num_interps+1], stack, axis=0)
					interp_latents = linfit(list(range(1,num_interps+1)))
					# print(interp_latents.shape)
					interp_noise_images = np.reshape(interp_latents, (interp_latents.shape[0],self.input_size,self.input_size,self.input_dims))
					# print(interp_noise_images.shape)

					if self.TanGAN_flag == 1:
						interp_OP1_images = self.TanGAN_generator(interp_noise_images)
						cur_interp_figs = self.generator(interp_OP1_images)
					elif self.BaseTanGAN_flag == 1:
						start = self.get_noise([1, self.noise_dims])#current_batch[i:1+i].numpy()
						end = self.get_noise([1, self.noise_dims]) 
						stack = np.vstack([start, end])

						linfit = interp1d([1,num_interps+1], stack, axis=0)
						interp_latents = linfit(list(range(1,num_interps+1)))
						interp_OP1_images = self.TanGAN_generator(interp_latents)
						cur_interp_figs = self.generator(interp_OP1_images)
					else:
						cur_interp_figs = self.generator(interp_noise_images)

					sharpness = self.find_sharpness(cur_interp_figs)
					try:
						sharpness_vec.append(sharpness)
					except:
						shaprpness_vec = [sharpness]

					try:
						batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs),axis = 0)
						batch_noise_figs = np.concatenate((batch_noise_figs,interp_noise_images), axis = 0)
						if self.TanGAN_flag or self.BaseTanGAN_flag:
							batch_OP1_figs = np.concatenate((batch_OP1_figs,interp_OP1_images), axis = 0)
					except:
						batch_interp_figs = cur_interp_figs
						batch_noise_figs = interp_noise_images
						if self.TanGAN_flag or self.BaseTanGAN_flag:
							batch_OP1_figs = interp_OP1_images

				images = (batch_interp_figs + 1.0)/2.0
				# print(images.shape)
				size_figure_grid = num_interps
				images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(images.shape[1],images.shape[2]),num_channels=images.shape[3])
				fig1 = plt.figure(figsize=(num_interps,num_interps))
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.axis("off")
				if images_on_grid.shape[2] == 3:
					ax1.imshow(np.clip(images_on_grid,0.,1.))
				else:
					ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

				label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
				plt.title(label)
				plt.tight_layout()
				plt.savefig(path)
				plt.close()

				noise_print_image = (batch_noise_figs + 1.0)/2.0
				# print(images.shape)
				size_figure_grid = num_interps
				images_on_grid = self.image_grid(input_tensor = noise_print_image, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(noise_print_image.shape[1],noise_print_image.shape[2]),num_channels=noise_print_image.shape[3])
				fig1 = plt.figure(figsize=(num_interps,num_interps))
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.axis("off")
				if images_on_grid.shape[2] == 3:
					ax1.imshow(np.clip(images_on_grid,0.,1.))
				else:
					ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

				label = 'INTERPOLATED NOISE IMAGES AT ITERATION '+str(self.total_count.numpy())
				plt.title(label)
				plt.tight_layout()
				plt.savefig(noi_path)
				plt.close()

				if self.TanGAN_flag or self.BaseTanGAN_flag:
					OP1_print_image = (batch_OP1_figs + 1.0)/2.0
					# print(images.shape)
					size_figure_grid = num_interps
					images_on_grid = self.image_grid(input_tensor = OP1_print_image, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(OP1_print_image.shape[1],OP1_print_image.shape[2]),num_channels=OP1_print_image.shape[3])
					fig1 = plt.figure(figsize=(num_interps,num_interps))
					ax1 = fig1.add_subplot(111)
					ax1.cla()
					ax1.axis("off")
					if images_on_grid.shape[2] == 3:
						ax1.imshow(np.clip(images_on_grid,0.,1.))
					else:
						ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

					label = 'INTERPOLATED OP1 IMAGES AT ITERATION '+str(self.total_count.numpy())
					plt.title(label)
					plt.tight_layout()
					plt.savefig(OP1_path)
					plt.close()		

				del batch_interp_figs
			if fig_count >= num_figs:
				break

		###### Interpol samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Interpolation Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("Interpolation Sharpness - "+str(overall_sharpness))


'''***********************************************************************************
********** The Fourier-series Solver *************************************************
***********************************************************************************'''
class FourierSolver():

	def __init__(self):
		from itertools import product as cart_prod

		self.M = self.terms #Number of terms in FS
		self.T = self.sigma
		# self.W = 1/self.T
		self.W = np.pi/self.T
		self.W0 = 1/self.T

		## For 1-D and 2-D Gaussians, no latent projections are needed. So latent dims are full dims itself.
		if self.data in ['g1', 'gmm2']:
			self.latent_dims = 1
		if self.data in ['g2', 'gmm8']:
			self.latent_dims = 2
		self.N = self.latent_dims

		''' If M is small, take all terms in FS expanse, else, a sample few of them '''
		if self.N <= 4:
			num_terms = list(np.arange(1,self.M+1))
			self.L = ((self.M)**self.N)
			print(num_terms) # nvec = Latent x Num_terms^latent
			self.n_vec = tf.cast(np.array([p for p in cart_prod(num_terms,repeat = self.N)]).transpose(), dtype = 'float32') # self.N x self.L lengthmatrix, each column is a desired N_vec to use
		else:
			# self.L = L#50000# + self.N + 1


			# with self.strategy.scope():


			'''need to do poisson disc sampling'''  #temp is self.M^self.N here
			temp = self.latent_dims

			#### This is what worked for WAEFR Images
			# vec1 = np.concatenate((np.ones([temp, 1]), \
			# 	np.concatenate(tuple([np.ones([temp,temp])+k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1)), axis = 1)

			### This is what is worked for WGAN Gaussians and Images
			# vec1 = np.concatenate((np.ones([temp, 1]), \
			# 	np.concatenate(tuple([np.zeros([temp,temp])+k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1), \
			# 	np.concatenate(tuple([np.ones([temp,temp])+k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1)), axis = 1)

			# ### This is what is worked for WGAN Gaussians and Images
			# vec1 = np.concatenate((np.ones([temp, 1]), \
			# 	np.concatenate(tuple([np.zeros([temp,temp])+k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1), \
			# 	np.concatenate(tuple([np.ones([temp,temp])+k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1), \
			# 	np.concatenate(tuple([2*np.ones([temp,temp])+k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1)), axis = 1)

			# vec1 = np.ones([temp, 1])

			#### This is what is worked for WGAN Gaussians and Images
			vec1 = np.concatenate((np.ones([temp, 1]), \
				np.concatenate(tuple([np.zeros([temp,temp])+k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1), \
				np.concatenate(tuple([np.ones([temp,temp])+k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1), \
				np.concatenate(tuple([2*np.ones([temp,temp])+k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1), \
				np.concatenate(tuple([3*np.ones([temp,temp])+k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1), \
				np.concatenate(tuple([4*np.ones([temp,temp])+k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1), \
				np.concatenate(tuple([5*np.ones([temp,temp])+k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1)), axis = 1)


			print("VEC1",vec1)
			vec2 = tf.cast(tf.random.uniform((temp,self.L),minval = 1, maxval = self.M, dtype = 'int32'),dtype='float32')
			# vec2_basis = np.random.choice(self.M-1,self.L) + 1
			# vec2 = np.concatenate(tuple([np.expand_dims(np.roll(vec2_basis,k),axis=0) for k in range(temp)]), axis = 0)
			print("VEC2",vec2)
			# self.n_vec = tf.cast(np.concatenate((vec1,vec2.numpy()), axis = 1),dtype='float32')
			self.n_vec = tf.cast(np.concatenate((vec1,vec2), axis = 1),dtype='float32')
			self.L += self.M*temp + 1
			self.L = self.n_vec.shape[1]
			print("NVEC",self.n_vec)


		# with self.strategy.scope():
		print(self.n_vec, self.W)
		self.Coeffs = tf.multiply(self.n_vec, self.W)
		print(self.Coeffs)
		self.n_norm = tf.expand_dims(tf.square(tf.linalg.norm(tf.transpose(self.n_vec), axis = 1)), axis=1)
		self.bias = np.array([0.])


		## Target data is for evaluateion of alphas (Check Main Manuscript)
		## Generator data is for evaluateion of betas (Check Main Manuscript)
		if self.gan == 'WGAN':
			self.target_data = 'self.reals_enc'
			self.generator_data = 'self.fakes_enc'
		elif self.gan == 'WAE':
			self.target_data = 'self.fakes_enc'
			self.generator_data = 'self.reals_enc'
		return

	def discriminator_model_FS_A(self):
		inputs = tf.keras.Input(shape=(self.latent_dims,)) #used to be self.N

		w0_nt_x = tf.keras.layers.Dense(self.L, activation=None, use_bias = False)(inputs)
		w0_nt_x2 = tf.math.scalar_mul(2., w0_nt_x)

		cos_terms = tf.keras.layers.Activation( activation = tf.math.cos)(w0_nt_x)
		sin_terms = tf.keras.layers.Activation( activation = tf.math.sin)(w0_nt_x)
		cos2_terms  = tf.keras.layers.Activation( activation = tf.math.cos)(w0_nt_x2)

		model = tf.keras.Model(inputs=inputs, outputs= [inputs, cos_terms, sin_terms, cos2_terms])
		return model

	def discriminator_model_FS_B(self):
		inputs = tf.keras.Input(shape=(self.latent_dims,))
		cos_terms = tf.keras.Input(shape=(self.L,)) #used to be self.N
		sin_terms = tf.keras.Input(shape=(self.L,))
		cos2_terms = tf.keras.Input(shape=(self.L,))

		cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos_terms)
		sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)

		cos2_c_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) #Tau_c weights
		cos2_s_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) #Tau_s weights

		lambda_x_term = tf.keras.layers.Subtract()([cos2_s_sum, cos2_c_sum]) #(tau_s  - tau_r)

		if self.latent_dims == 1:
			phi0_x = inputs
		else:
			phi0_x = 0.01*tf.divide(tf.reduce_sum(inputs,axis=-1,keepdims=True),self.latent_dims)

		if self.homo_flag:
			Out = tf.keras.layers.Add()([cos_sum, sin_sum, phi0_x])
		else:
			Out = tf.keras.layers.Add()([cos_sum, sin_sum])

		model = tf.keras.Model(inputs= [inputs, cos_terms, sin_terms, cos2_terms], outputs=[Out,lambda_x_term])
		return model

	def Fourier_Series_Comp(self,f):

		mu = tf.convert_to_tensor(np.expand_dims(np.mean(f,axis = 0),axis=1), dtype = 'float32')
		cov = tf.convert_to_tensor(np.cov(f,rowvar = False), dtype = 'float32')
		# print(self.reals.shape,self.fakes.shape)
		# self.T = tf.convert_to_tensor(2*max(np.mean(self.reals_enc), np.mean(self.fakes_enc)), dtype = 'float32')
		# # print("T",self.T)
		# self.W = 2*np.pi/self.T
		# self.freq = 1/self.T
		# self.Coeffs = tf.multiply(self.n_vec, self.W)
		# self.coefficients.set_weights([self.Coeffs, self.Coeffs])

		# with self.strategy.scope():

		if self.distribution == 'generic':
			_, ar, ai, _ = self.discriminator_A(f, training = False)
			ar = (1/self.T)*tf.expand_dims(tf.reduce_mean(ar, axis = 0), axis = 1)#Lx1 vector
			ai = (1/self.T)*tf.expand_dims(tf.reduce_mean(ai, axis = 0), axis = 1)#Lx1 vector
			# print(ai,ar)

			## Error calc between true and estimate values. Only for Sanity Check
			# if self.data != 'g1':
			# 	nt_mu = tf.linalg.matmul(tf.transpose(self.n_vec),mu)
			# 	nt_cov_n =  tf.expand_dims(tf.linalg.tensor_diag_part(tf.linalg.matmul(tf.transpose(self.n_vec),tf.linalg.matmul(cov,self.n_vec))), axis=1)
			# else:
			# 	nt_mu = mu*self.n_vec
			# 	nt_cov_n = cov * tf.expand_dims(tf.linalg.tensor_diag_part(tf.linalg.matmul(tf.transpose(self.n_vec),self.n_vec)), axis=1)
			# #### FIX POWER OF T
			# #tf.constant((1/(self.T))**1)
			# ar_true =  tf.constant((1/(self.T))**0) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2))*(tf.math.cos(tf.multiply(nt_mu, self.W)))
			# ai_true =  tf.constant((1/(self.T))**0) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2))*(tf.math.sin(tf.multiply(nt_mu, self.W)))

			# error = tf.reduce_mean(tf.abs(ar-ar_true)) + tf.reduce_mean(tf.abs(ai-ai_true))
			# # self.lambda_vec.append(np.log(error.numpy()))


		if self.distribution == 'gaussian':
			if self.data != 'g1':
				nt_mu = tf.linalg.matmul(tf.transpose(self.n_vec),mu)
				nt_cov_n =  tf.expand_dims(tf.linalg.tensor_diag_part(tf.linalg.matmul(tf.transpose(self.n_vec),tf.linalg.matmul(cov,self.n_vec))), axis=1)
				#### No Pow on T
				ar =  tf.constant((1/(self.T))**0) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2))*(tf.math.cos(tf.multiply(nt_mu, self.W)))
				ai =  tf.constant((1/(self.T))**0) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2 ))*(tf.math.sin(tf.multiply(nt_mu, self.W)))
			else:
				nt_mu = mu*tf.transpose(self.n_vec)
				nt_cov_n = cov * tf.expand_dims(tf.linalg.tensor_diag_part(tf.linalg.matmul(tf.transpose(self.n_vec,[1,0]),self.n_vec)), axis=1)
				ar =  tf.constant((1/(self.T))**1) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2))*(tf.math.cos(tf.multiply(nt_mu, self.W)))
				ai =  tf.constant((1/(self.T))**1) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2 ))*(tf.math.sin(tf.multiply(nt_mu, self.W)))

			
			# print(ar,ai)
		if self.distribution == 'uniform':
			#DEPRICATED - UNIFORM IS A BAD IDEA
			a_vec = tf.expand_dims(tf.reduce_min(f, axis = 0),0)
			b_vec = tf.expand_dims(tf.reduce_max(f, axis = 0),0)
			nt_a = tf.transpose(tf.linalg.matmul(a_vec, self.n_vec),[1,0])
			nt_b = tf.transpose(tf.linalg.matmul(b_vec, self.n_vec),[1,0])
			nt_bma = tf.transpose(tf.linalg.matmul(b_vec - a_vec, self.n_vec),[1,0])

			# tf.constant((1/(self.T))**1)
			ar =  1 * tf.divide(tf.math.sin(tf.multiply(nt_b ,self.W)) - tf.math.sin(tf.multiply(nt_a ,self.W)), tf.multiply(nt_bma,self.W))
			ai = - 1 * tf.divide(tf.math.cos(tf.multiply(nt_b ,self.W)) - tf.math.cos(tf.multiply(nt_a ,self.W)), tf.multiply(nt_bma,self.W))
		
		return  ar, ai

	def discriminator_ODE(self):
		self.alpha_c, self.alpha_s = self.Fourier_Series_Comp(eval(self.target_data))
		self.beta_c, self.beta_s = self.Fourier_Series_Comp(eval(self.generator_data))

		# with self.strategy.scope():
			# Vec of len Lx1 , wach entry is ||n||
		self.Gamma_s = tf.math.divide(tf.constant(1/(self.W**2))*tf.subtract(self.alpha_s, self.beta_s), self.n_norm)
		self.Gamma_c = tf.math.divide(tf.constant(1/(self.W**2))*tf.subtract(self.alpha_c, self.beta_c), self.n_norm)
		self.Tau_s = tf.math.divide(tf.constant(1/(2.*(self.W**2)))*tf.square(tf.subtract(self.alpha_s, self.beta_s)), self.n_norm)
		self.Tau_c = tf.math.divide(tf.constant(1/(2.*(self.W**2)))*tf.square(tf.subtract(self.alpha_c, self.beta_c)), self.n_norm)
		self.sum_Tau = 1.*tf.reduce_sum(tf.add(self.Tau_s,self.Tau_c))

	def find_lambda(self):
		self.lamb = tf.divide(tf.reduce_sum(self.lambda_x_terms_2) + tf.reduce_sum(self.lambda_x_terms_1),tf.cast(self.batch_size, dtype = 'float32')) + self.sum_Tau
		self.lamb = tf.cast((2*self.L+1), dtype = 'float32')*self.lamb
		self.lamb = tf.sqrt(self.lamb)
		

	def divide_by_lambda(self):
		self.real_output = tf.divide(self.real_output, self.lamb)
		self.fake_output = tf.divide(self.fake_output, self.lamb)



'''***********************************************************************************
********** The RBF-PHS Solver *************************************************
***********************************************************************************'''
class SnakeSolver():

	def __init__(self):
		from itertools import product as cart_prod


		## For 1-D and 2-D Gaussians, no latent projections are needed. So latent dims are full dims itself.
		if self.data in ['g1']:
			self.latent_dims = 1
		if self.data in ['g2', 'gmm8']:
			self.latent_dims = 2
		if self.data in ['gN', 'gmmN']:
			self.latent_dims = self.GaussN
		self.N = self.rbf_n = self.latent_dims

		# self.N_centers = self.batch_size

		self.c = 0

		self.G_loss_counter = 0
		self.first_iter_lambda = 0

		if self.rbf_n%2 == 1:
			if self.rbf_m < ((self.rbf_n+1)/2) :
				self.poly_case = 0
			else:
				self.poly_case = 1 ## odd_n, for all m
		else:
			if self.rbf_m <= ((self.rbf_n/2) - 1):
				self.poly_case = 2 ## even_n, negtive 2m-n
			else:
				self.poly_case = 3 ## even_n, positive 2m-n
			self.rbf_eta = self.rbf_n/2

			self.c_index = self.rbf_m - self.rbf_eta

			r1 = np.arange(1, self.c_index, 1)
			r2 = np.arange(self.rbf_eta, self.rbf_m + self.rbf_eta - 1, 1)

			if self.c_index != 0:
				for tau in r1:
					self.c += (1/(2*tau))
				for tau in r2:
					self.c += (1/(2*tau))




		######3 HACK CHECK##########
		# self.c = 0

		## Defining the Solution cases based on m and n


		## Target data is for evaluateion of alphas (Check Main Manuscript)
		## Generator data is for evaluateion of betas (Check Main Manuscript)
		if self.gan in ['WGAN', 'LSGAN']:
			self.target_data = 'self.reals'
			self.generator_data = 'self.fakes'
		elif self.gan in ['WAE']:
			self.target_data = 'self.fakes_enc'
			self.generator_data = 'self.reals_enc'
		elif self.gan == ['WGAN_AE','MMDGAN']:
			self.target_data = 'self.reals_enc'
			self.generator_data = 'self.fakes_enc'


		# self.snake_update_count = 0
		# self.kappa = 0.2
		# self.alpha = 0.005
		# self.beta  = 0.0000
		# self.gamma = 0.15
		# self.iterations = self.num_snake_iters

		if self.snake_kind == 'o':
			self.snake_update_count = 0
			self.kappa = 0.2
			self.alpha = 0.005
			self.beta  = 0.0000
			self.gamma = 0.75
			self.iterations = self.num_snake_iters

		if self.snake_kind == 'uo':
			self.snake_update_count = 0
			self.kappa = 0.2
			self.alpha = 0.005
			self.beta  = 0.0000
			self.gamma = 0.15
			self.iterations = self.num_snake_iters

		# self.snake_update_count = 0
		# self.kappa = 0.1
		# self.alpha = 0.005
		# self.beta  = 0.0000
		# self.gamma = 1.
		# self.iterations = 2

		scaled_circ = 0.05
		offset = 0
		locs = [[scaled_circ*1.+offset, 0.+offset], \
				[scaled_circ*1*0.7071+offset, scaled_circ*1*0.7071+offset],	 \
				[0.+offset, scaled_circ*1.+offset], \
				[scaled_circ*-1*0.7071+offset, scaled_circ*1*0.7071+offset], \
				[scaled_circ*-1.+offset,0.+offset], \
				[scaled_circ*-1*0.7071+offset, scaled_circ*-1*0.7071+offset], \
				[0.+offset,scaled_circ*-1.+offset], \
				[scaled_circ*1*0.7071+offset, scaled_circ*-1*0.7071+offset], ]

		self.num_snake_points = len(locs)
		print(locs)
		self.snake_gen_matrix = tf.cast(np.array(locs),dtype = 'float32')
		print(self.snake_gen_matrix.shape)
		self.snake_gen_matrix = tf.tile(self.snake_gen_matrix, [self.batch_size,1])
		print(self.snake_gen_matrix)
		print(self.snake_gen_matrix.shape)
		
		# test = tf.reshape(self.snake_gen_matrix, (self.batch_size,self.num_snake_points,2))
		# print(test[0,:,:])
		# exit(0)

		
		###
		### Repmatting the centers and adding this 


	def grad_kernel(self,fakes,real_centers,fake_centers):
		def calculate_squared_distances(a, b):
			'''returns the squared distances between all elements in a and in b as a matrix
			of shape #a * #b'''
			# a = np.array([[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.],])
			# b = np.array([[2.,2.],[5.,5.],[6.,7.],[10.,20.],[10.,100.],])
			na = tf.shape(a)[0]
			nb = tf.shape(b)[0]

			# print(tf.shape(a),tf.shape(b))
			# print(a,b)
			nas, nbs = list(a.shape), list(b.shape)
			a = tf.reshape(a, [na, 1, -1])
			b = tf.reshape(b, [1, nb, -1])
			# print(a,b)
			a.set_shape([nas[0], 1, np.prod(nas[1:])])
			b.set_shape([1, nbs[0], np.prod(nbs[1:])])
			a = tf.tile(a, [1, nb, 1]) #a_i repeated on axis 1
			b = tf.tile(b, [na, 1, 1]) #b_i repeated on axis 0
			d = a-b
			# print(d)
			# print(a,b)
			# print(d,d.shape)
			temp = tf.transpose(d,perm=[2,0,1])
			tempD = tf.linalg.diag_part(temp)
			tempD2 = tf.transpose(tempD,perm=[1,0])
			# print(tempD2,tempD2.shape)
			normD = tf.sqrt(tf.reduce_sum(tf.square(d), axis=2))
			normD = tf.expand_dims(tf.linalg.diag_part(normD),axis =1)
			# normD = tf.tile(normD, [na,-1]
			# print(normD)
			

			# grad_D = tf.multiply(tempD2,tf.math.log(normD))
			# grad_D = tf.multiply(tf.multiply(tempD2,normD),tf.math.log(normD))
			grad_D = tempD2/normD
			# print(grad_D,grad_D.shape)
			# exit(0)
			return grad_D


		reals_term = calculate_squared_distances(fakes,real_centers)
		fakes_term = calculate_squared_distances(fakes,fake_centers)

		return fakes_term,reals_term



	def compute_grad_disc(self,kappa,batch_size,fakes,real_centers,fake_centers):
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
			# print(a,b)
			# print(d,d.shape)
			temp = tf.transpose(d,perm=[2,0,1])
			tempD = tf.linalg.diag_part(temp)
			tempD2 = tf.transpose(tempD,perm=[1,0])
			# print(tempD2,tempD2.shape)
			normD = tf.sqrt(tf.reduce_sum(tf.square(d), axis=2))
			normD = tf.expand_dims(tf.linalg.diag_part(normD),axis =1)
			# normD = tf.tile(normD, [na,-1]
			# print(normD)
			grad_D = tempD2/normD
			# print(grad_D,grad_D.shape)
			return grad_D


		reals_term = calculate_squared_distances(fakes,real_centers)
		fakes_term = calculate_squared_distances(fakes,fake_centers)

		return fakes_term - reals_term


	# def compute_grad_disc(self,kappa,batch_size,generated_data_discriminator,true_data,generated_data_reference):
	# 	constant = kappa/tf.cast(batch_size,dtype=tf.float32)
	# 	grad_val_vector = []
	# 	for point_vector in generated_data_discriminator:
	# 		sum_1 = 0
	# 		sum_2 = 0
			
	# 		true_data_mod = tf.cast(true_data,tf.float32)
	# 		for j in generated_data_reference:
	# 				nr = point_vector - j
	# 				dr = tf.norm(point_vector - j)
	# 				sum_1 += nr/dr
	# 		for k in true_data_mod:
	# 				nr = tf.cast(point_vector,tf.float32)-k
	# 				dr = tf.norm(point_vector - k)
	# 				sum_2 += nr/dr
	# 		grad_val = constant * (sum_1 - sum_2)
	# 		grad_val_vector.append(grad_val)
	# 		print(grad_val_vector)

	# 	return tf.convert_to_tensor(grad_val_vector,dtype=tf.float32)

	# Hence the gradients at each of the points in the generated_data_reference have been computed
	# Next we apply the snake algorithm to move these points to the desired locations

	def create_A(self,a, b, N):

		row_1 = tf.constant([-2*a - 6*b,a + 4*b,-b],dtype=tf.float32)
		row_2 = tf.zeros(N-5,dtype=tf.float32)
		row_3 = tf.constant([-b,a + 4*b],dtype=tf.float32)
		row = tf.concat([row_1,row_2,row_3],axis=0)
		row_stack =[]
		for k in range(0,N):
			row_stack.append(tf.roll(row, k,axis=0))
		A = tf.stack(row_stack,axis=0)
		A = tf.reshape(A,(N,N))
		# print(A)
		# exit(0)
		return A

	def iterate_snake(self,points_vector,true_data,generated_data,a, b, kappa, gamma=0.1, n_iters=10, return_all=True):
		
		no_snake_points = points_vector.shape[0]
		A = self.create_A(a,b,no_snake_points)
		B = tf.linalg.inv(tf.eye(no_snake_points) - tf.constant(gamma,dtype=tf.float32)*A)
		# print(A)
		# print(points_vector.shape)
		

		def update(i,points_vector):
			Centres, Weights = self.find_rbf_centres_weights_given(true_data,generated_data)
			self.discriminator_RBF.set_weights([Centres,Weights])
			grads = self.compute_grad_disc(kappa,self.batch_size,points_vector,true_data,generated_data)
			# print(grads)
			# print(B.shape,points_vector.shape,grads.shape)
			points_vector_update = tf.tensordot(B, points_vector + tf.constant(gamma,dtype=tf.float32)*grads,1)
			i_next = i + 1
			return i_next,points_vector_update

		c = lambda i,points_vector: tf.less(i,n_iters)
		# init_points_vector = tf.zeros(tf.shape(points_vector))#(B, points_vector + tf.constant(gamma,dtype=tf.float32)*compute_grad_disc(kappa,batch_size,points_vector,true_data,generated_data),1) 
		# _,result = tf.while_loop(cond = c, body=update,loop_vars=(tf.constant(0,dtype=tf.int32),tf.zeros(tf.shape(points_vector))))
		_,result = tf.while_loop(cond = c, body=update,loop_vars=(tf.constant(0,dtype=tf.int32),points_vector))
		# print("Snake Iterations Computed for Iteration:" + str))  
		return result

	def snake_flow_uo(self,reals,fakes,reference_fakes):

		if self.gan not in ['WAE', 'MMDGAN', 'WGAN_AE'] and self.data not in ['g1', 'g2','gmm8', 'gN', 'gmmN']:
			reals_res = tf.reshape(reals, [reals.shape[0], reals.shape[1]*reals.shape[2]*reals.shape[3]])
			fakes_res = tf.reshape(fakes, [fakes.shape[0], fakes.shape[1]*fakes.shape[2]*fakes.shape[3]])
			reference_fakes_res = tf.reshape(reference_fakes, [reference_fakes.shape[0], reference_fakes.shape[1]*reference_fakes.shape[2]*reference_fakes.shape[3]])
		else:
			reals_res = reals
			fakes_res = fakes
			reference_fakes_res = reference_fakes

		snake = self.iterate_snake(
				points_vector = fakes_res,
				true_data = reals_res,
				generated_data = reference_fakes_res,    
				a = self.alpha,
				b = self.beta,
				kappa = self.kappa,
				gamma = self.gamma,
				n_iters = self.iterations,
				return_all = True)
		snakes = tf.convert_to_tensor(snake)
		# print(snakes.shape)
		if self.gan not in ['WAE', 'MMDGAN', 'WGAN_AE'] and self.data not in ['g1', 'g2','gmm8', 'gN', 'gmmN']:
			snakes = tf.reshape(snakes, [fakes.shape[0], fakes.shape[1],fakes.shape[2],fakes.shape[3]])
		return snakes

	def compute_grad_disc_o(self,kappa,batch_size,fakes,real_centers,fake_centers):
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
			# print(a,b)
			# print(d,d.shape)
			temp = tf.transpose(d,perm=[2,0,1])
			tempD = tf.linalg.diag_part(temp)
			tempD2 = tf.transpose(tempD,perm=[1,0])
			# print(tempD2,tempD2.shape)
			normD = tf.sqrt(tf.reduce_sum(tf.square(d), axis=2))
			normD = tf.expand_dims(tf.linalg.diag_part(normD),axis =1)
			# normD = tf.tile(normD, [na,-1]
			# print(normD)
			grad_D = tempD2/normD
			# print(grad_D,grad_D.shape)
			return grad_D


		reals_term = calculate_squared_distances(fakes,real_centers)
		fakes_term = calculate_squared_distances(fakes,fake_centers)

		return fakes_term - reals_term


	def create_A_o(self,a, b, N, big_N):

		row_1 = tf.constant([-2*a - 6*b,a + 4*b,-b],dtype=tf.float32)
		row_2 = tf.zeros(N-5,dtype=tf.float32)
		row_3 = tf.constant([-b,a + 4*b],dtype=tf.float32)
		row = tf.concat([row_1,row_2,row_3],axis=0)
		row_stack =[]
		for k in range(0,N):
			row_stack.append(tf.roll(row, k,axis=0))
		A = tf.stack(row_stack,axis=0)
		A = tf.reshape(A,(N,N))
		# print(A,A.shape)
		# A = tf.repeat(A, repeats= [big_N], axis = 0)
		# print(A,A.shape)
		# exit(0)
		return A

	def create_B_o(self,A,num_snake_points,num_repeat,gamma):

		each_inv = tf.linalg.inv(tf.eye(num_snake_points) - tf.constant(gamma,dtype=tf.float32)*A)
		# print(each_inv)

		zeros_once = tf.zeros((num_snake_points,num_snake_points),dtype=tf.float32)
		row_zeros = tf.tile(zeros_once,[1,num_repeat-1])

		# print(row_zeros.shape)
		OneBlock = tf.concat([each_inv,row_zeros],axis = 1)

		# print(OneBlock)
		row_stack =[]
		for k in range(0,num_repeat):
			row_stack.append(tf.roll(OneBlock, k*num_snake_points,axis=1))

		B = tf.stack(row_stack,axis=0)
		B = tf.reshape(B,(num_snake_points*num_repeat,num_snake_points*num_repeat))

		##3 Repeats the matrix for the number of times the point is there
		# B = tf.tile(each_inv, [num_repeat,1])
		# print(B)
		# exit(0)
		return B


	def iterate_snake_o(self,points_vector,true_data,generated_data,a, b, kappa, gamma=0.1, n_iters=10, return_all=True):
		
		
		num_snake_points = self.num_snake_points
		num_repeat = self.batch_size
		total_num_points = points_vector.shape[0]
		assert total_num_points == num_repeat * num_snake_points 


		A = self.create_A_o(a,b,num_snake_points,num_repeat)
		B = self.create_B_o(A,num_snake_points,num_repeat,gamma)

		
		# print(A)
		# print(points_vector.shape)
		

		# def update(i,points_vector):
		# 	grads = self.compute_grad_disc(kappa,self.batch_size,points_vector,true_data,generated_data)
		# 	print(grads)
		# 	points_vector_update = tf.tensordot(B, points_vector + tf.constant(gamma,dtype=tf.float32)*grads,1)

		# 	i_next = i + 1
		# 	return i_next,points_vector_update
		Centres, Weights = self.find_rbf_centres_weights_given(true_data,generated_data)
		self.discriminator_RBF.set_weights([Centres,Weights])

		def update(i,points_vector):
			
			with tf.GradientTape() as disc_tape:
				disc_tape.watch(points_vector)
				d_vals = self.discriminator_RBF(points_vector,training = False)
			grads = disc_tape.gradient(d_vals, [points_vector])[0]

			# print(B.shape, points_vector.shape,grads.shape)
			# grads = self.compute_grad_disc(kappa,self.batch_size,points_vector,true_data,generated_data)

			points_vector_update = tf.tensordot(B, points_vector + tf.constant(gamma,dtype=tf.float32)*grads,1)
			# points_vector_update = tf.matmul(B, points_vector - tf.constant(gamma,dtype=tf.float32)*grads,1)

			# print(points_vector_update.shape)

			i_next = i + 1
			return i_next,points_vector_update
			

		_,snakes1 = update(0,points_vector)
		_,snakes2 = update(1,snakes1)
		_,snakes3 = update(2,snakes2)
		_,snakes4 = update(3,snakes3)
		_,result = update(4,snakes4)



		result_pointwiseBatch1 = tf.reshape(snakes1, (num_snake_points,num_repeat,2))
		self.snakes1 = tf.reduce_mean(result_pointwiseBatch1, axis = 0)
		result_pointwiseBatch2 = tf.reshape(snakes2, (num_snake_points,num_repeat,2))
		self.snakes2 = tf.reduce_mean(result_pointwiseBatch2, axis = 0)
		result_pointwiseBatch3 = tf.reshape(snakes3, (num_snake_points,num_repeat,2))
		self.snakes3 = tf.reduce_mean(result_pointwiseBatch3, axis = 0)
		result_pointwiseBatch4 = tf.reshape(snakes4, (num_snake_points,num_repeat,2))
		self.snakes4 = tf.reduce_mean(result_pointwiseBatch4, axis = 0)

		# c = lambda i,points_vector: tf.less(i,n_iters)
		# # init_points_vector = tf.zeros(tf.shape(points_vector))#(B, points_vector + tf.constant(gamma,dtype=tf.float32)*compute_grad_disc(kappa,batch_size,points_vector,true_data,generated_data),1) 
		# # _,result = tf.while_loop(cond = c, body=update,loop_vars=(tf.constant(0,dtype=tf.int32),tf.zeros(tf.shape(points_vector))))
		# _,result = tf.while_loop(cond = c, body=update,loop_vars=(tf.constant(0,dtype=tf.int32),points_vector))

		### Need to average is chuncks of num_snake_points
		# print(result.shape)
		result_pointwiseBatch = tf.reshape(result, (num_snake_points,num_repeat,2))
		Centroids = tf.reduce_mean(result_pointwiseBatch, axis = 0)
		# print(Centroids)
		# exit(0)
		# print("Snake Iterations Computed for Iteration:" + str))  
		return Centroids


	def snake_flow_o(self,reals,fakes,reference_fakes):

		if self.gan not in ['WAE', 'MMDGAN', 'WGAN_AE'] and self.data not in ['g1', 'g2','gmm8', 'gN', 'gmmN']:
			reals_res = tf.reshape(reals, [reals.shape[0], reals.shape[1]*reals.shape[2]*reals.shape[3]])
			fakes_res = tf.reshape(fakes, [fakes.shape[0], fakes.shape[1]*fakes.shape[2]*fakes.shape[3]])
			reference_fakes_res = tf.reshape(reference_fakes, [reference_fakes.shape[0], reference_fakes.shape[1]*reference_fakes.shape[2]*reference_fakes.shape[3]])
		else:
			reals_res = reals
			fakes_res = fakes
			reference_fakes_res = reference_fakes



		fakes_res = tf.repeat(fakes_res, repeats=self.num_snake_points, axis=0)
		# print(fakes_res)
		fakes_res += self.snake_gen_matrix
		# print(fakes_res)


		snake = self.iterate_snake_o(
				points_vector = fakes_res,
				true_data = reals_res,
				generated_data = reference_fakes_res,    
				a = self.alpha,
				b = self.beta,
				kappa = self.kappa,
				gamma = self.gamma,
				n_iters = self.iterations,
				return_all = True)
		snakes = tf.convert_to_tensor(snake)


		# print(fakes_res[0:10],snakes[0:10])
		# exit(0)
		if self.gan not in ['WAE', 'MMDGAN'] and self.data not in ['g1', 'g2','gmm8', 'gN', 'gmmN']:
			snakes = tf.reshape(snakes, [fakes.shape[0], fakes.shape[1],fakes.shape[2],fakes.shape[3]])
		return snakes



	def discriminator_model_RBF(self):

		if self.gan not in ['WAE', 'MMDGAN', 'WGAN_AE'] and self.data not in ['g2','gmm8', 'gN', 'gmmN']:
			inputs = tf.keras.Input(shape=(self.output_size,self.output_size,self.output_dims))
			inputs_res = tf.keras.layers.Reshape(target_shape = [self.output_size*self.output_size*self.output_dims])(inputs)
		else:
			inputs = tf.keras.Input(shape=(self.latent_dims,))
			inputs_res = inputs

		### For Supp
		num_centers = 2*self.N_centers
		### For Main. Need to correct if need be
		# num_centers = 2*self.batch_size

		if self.gan in ['WGAN', 'WAE', 'MMDGAN', 'WGAN_AE']:
			D = RBFLayer(num_centres=num_centers, output_dim=1, order_m = self.rbf_m, batch_size = self.batch_size, const = self.c, rbf_pow = 1)(inputs_res)
			# lambda_term = RBFLayer(num_centres=num_centers, output_dim=1, order_m = self.rbf_m, batch_size = self.batch_size, const = self.c, rbf_pow = -2*self.rbf_n)(inputs_res)
			if self.latent_dims == 1:
				phi0_x = inputs
			else:
				phi0_x = tf.divide(tf.reduce_sum(inputs_res,axis=-1,keepdims=True),self.latent_dims)

			if self.homo_flag:
				Out = tf.keras.layers.Add()([D, phi0_x])
			else:
				Out = D 

			model = tf.keras.Model(inputs=inputs, outputs=Out)#[Out,lambda_term])

		if self.gan == 'LSGAN':
			[Out,A,B] = PHSLayer(num_centres=num_centers, output_dim=1,  dim_v = self.dim_v, rbf_k = self.rbf_m, batch_size = self.batch_size, multi_index = self.multi_index)(inputs_res)
			# Poly_of_x = tf.keras.layers.Dense(units = 1, use_bias = False)(XPoly)

			# Out = tf.keras.layers.Add()([D, Poly_of_x])

			model = tf.keras.Model(inputs=inputs, outputs= [Out,A,B])
		

		return model

	def find_rbf_centres_weights_given(self,reals,fakes):

		C_d = reals[0:self.N_centers] #SHould be NDxn
		C_g = fakes[0:self.N_centers]
		check = 1
		if self.gan not in ['WAE', 'MMDGAN', 'WGAN_AE'] and self.data not in ['g1', 'g2','gmm8', 'gN', 'gmmN']:
			C_d = tf.reshape(C_d, [C_d.shape[0], C_d.shape[1]*C_d.shape[2]*C_d.shape[3]])
			C_g = tf.reshape(C_g, [C_g.shape[0], C_g.shape[1]*C_g.shape[2]*C_g.shape[3]])

		Centres = np.concatenate((C_d,C_g), axis = 0)

		D_d = (-1/C_d.shape[0])*np.ones([C_d.shape[0]])
		D_g = (1/(C_g.shape[0]))*np.ones([C_g.shape[0]])
		Weights = np.concatenate((D_d,D_g), axis = 0)
		return Centres, Weights



	# def find_rbf_centres_weights(self):

	# 	C_d = eval(self.target_data)[0:self.N_centers] #SHould be NDxn
	# 	C_g = eval(self.generator_data)[0:self.N_centers]
	# 	check = 1
	# 	if self.gan not in ['WAE', 'MMDGAN'] and self.data not in ['g1', 'g2','gmm8', 'gN', 'gmmN']:
	# 		C_d = tf.reshape(C_d, [C_d.shape[0], C_d.shape[1]*C_d.shape[2]*C_d.shape[3]])
	# 		C_g = tf.reshape(C_g, [C_g.shape[0], C_g.shape[1]*C_g.shape[2]*C_g.shape[3]])

	# 	Centres = np.concatenate((C_d,C_g), axis = 0)

	# 	# D_d = check*(((-1)**(2*self.rbf_m - C_d.shape[1]+1))/self.reals.shape[0])*np.ones([self.batch_size])
	# 	# D_g = check*(((-1)**(2*self.rbf_m - C_g.shape[1]))/self.fakes.shape[0])*np.ones([self.batch_size])

	# 	if self.gan in ['WGAN', 'WAE', 'MMDGAN']:
	# 		if (2*self.rbf_m - self.rbf_n) <= 2 or self.rbf_n <= 350:
	# 			self.alpha = self.beta = 1
	# 			D_d = (-1/C_d.shape[0])*np.ones([C_d.shape[0]])
	# 			D_g = (1/(C_g.shape[0]))*np.ones([C_g.shape[0]])
	# 			W_lamb = 1*tf.ones_like(D_d)
	# 		else:
	# 			self.alpha = 1/(C_g.shape[0])**(2*self.rbf_m - self.rbf_n)
	# 			self.beta = 1#2 - (C_g.shape[0])**(2*self.rbf_m - self.rbf_n)
	# 			D_d = -1*np.ones([C_d.shape[0]])
	# 			D_g = 1*self.alpha*np.ones([C_g.shape[0]])
	# 			W_lamb = tf.ones_like(D_d)

	# 		Weights = np.concatenate((D_d,D_g), axis = 0)

	# 		Lamb_Weights = np.concatenate((W_lamb,W_lamb), axis = 0)

	# 		return Centres, Weights, Lamb_Weights

	# 	elif self.gan == 'LSGAN':
	# 		d_vals = self.label_b*tf.ones((C_d.shape[0],1))
	# 		g_vals = self.label_a*tf.ones((C_g.shape[0],1))
	# 		Values = np.concatenate((d_vals,g_vals), axis = 0)

	# 		Weights, PolyWeights = self.PHS_MatrixSolver(vals = Values)#centers = Centres, vals = Values, phs_deg = self.rbf_m, poly_deg = self.rbf_m-1)

	# 		# print(Weights)
	# 		# print(PolyWeights)

	# 		return Centres, Weights, PolyWeights


'''***********************************************************************************
********** The Grad RBF ScoreGAN Solver *************************************************
***********************************************************************************'''
class GradRBFSolver():

	def __init__(self):
		from itertools import product as cart_prod


		## For 1-D and 2-D Gaussians, no latent projections are needed. So latent dims are full dims itself.
		if self.data in ['g1']:
			self.latent_dims = 1
		if self.data in ['g2', 'gmm8']:
			self.latent_dims = 2
		if self.data in ['gN', 'gmmN']:
			self.latent_dims = self.GaussN
		self.N = self.rbf_n = self.latent_dims

		# self.N_centers = self.batch_size

		self.c = 0

		self.G_loss_counter = 0
		self.first_iter_lambda = 0

		if self.rbf_n%2 == 1:
			if self.rbf_m < ((self.rbf_n+1)/2) :
				self.poly_case = 0
			else:
				self.poly_case = 1 ## odd_n, for all m
		else:
			if self.rbf_m <= ((self.rbf_n/2) - 1):
				self.poly_case = 2 ## even_n, negtive 2m-n
			else:
				self.poly_case = 3 ## even_n, positive 2m-n
			self.rbf_eta = self.rbf_n/2

			self.c_index = self.rbf_m - self.rbf_eta

			r1 = np.arange(1, self.c_index, 1)
			r2 = np.arange(self.rbf_eta, self.rbf_m + self.rbf_eta - 1, 1)

			if self.c_index != 0:
				for tau in r1:
					self.c += (1/(2*tau))
				for tau in r2:
					self.c += (1/(2*tau))

		## Target data is for evaluateion of alphas (Check Main Manuscript)
		## Generator data is for evaluateion of betas (Check Main Manuscript)
		if self.gan in ['WGAN']:
			if self.topic in ['ScoreGAN']:
				self.target_data = 'self.reference_reals'
				self.generator_data = 'self.reference_fakes'
			else:
				self.target_data = 'self.reals'
				self.generator_data = 'self.fakes'
		elif self.gan in ['WAE']:
			self.target_data = 'self.fakes_enc'
			self.generator_data = 'self.reals_enc'
		elif self.gan in ['WGANFlow']:
			self.target_data = 'self.reference_reals'
			self.generator_data = 'self.reference_fakes'
		elif self.gan in ['MMDGAN', 'WGAN_AE']:
			self.target_data = 'self.reals_enc'
			self.generator_data = 'self.fakes_enc'
		return

	def discriminator_model_RBF(self):

		if self.gan not in ['WAE', 'MMDGAN', 'WGAN', 'WGANFlow'] and self.data not in ['g2','gmm8', 'gN', 'gmmN']:
			inputs = tf.keras.Input(shape=(self.output_size,self.output_size,self.output_dims))
			inputs_res = tf.keras.layers.Reshape(target_shape = [self.output_size*self.output_size*self.output_dims])(inputs)
		else:
			inputs = tf.keras.Input(shape=(self.latent_dims,))
			inputs_res = inputs

		### For Supp
		num_centers = self.N_centers
		# num_centers = 2*self.batch_size

		GradD_reals = GradRBFLayer(num_centres=num_centers, output_dim=self.latent_dims, order_m = self.rbf_m, batch_size = self.batch_size, const = self.c)(inputs_res)
		GradD_fakes = GradRBFLayer(num_centres=num_centers, output_dim=self.latent_dims, order_m = self.rbf_m, batch_size = self.batch_size, const = self.c)(inputs_res)
		model = tf.keras.Model(inputs=inputs, outputs=[GradD_reals,GradD_fakes])

		return model


	def find_rbf_centres_weights(self):

		C_d = eval(self.target_data)[0:self.N_centers] #SHould be NDxn
		C_g = eval(self.generator_data)[0:self.N_centers]
		check = 1
		if self.gan not in ['WAE', 'MMDGAN', 'WGAN_AE', 'WGANFlow'] and self.data not in ['g1', 'g2','gmm8', 'gN', 'gmmN']:
			C_d = tf.reshape(C_d, [C_d.shape[0], C_d.shape[1]*C_d.shape[2]*C_d.shape[3]])
			C_g = tf.reshape(C_g, [C_g.shape[0], C_g.shape[1]*C_g.shape[2]*C_g.shape[3]])

		# Centres = np.concatenate((C_d,C_g), axis = 0)


		# if (2*self.rbf_m - self.rbf_n) <= 2 or self.rbf_n <= 350:
			# self.alpha = self.beta = 1
		D_d = (1/C_d.shape[0])*np.ones([C_d.shape[0]])
		D_g = (1/(C_g.shape[0]))*np.ones([C_g.shape[0]])
			# W_lamb = 1*tf.ones_like(D_d)
		# else:
		# 	self.alpha = 1/(C_g.shape[0])**(2*self.rbf_m - self.rbf_n)
		# 	self.beta = 1#2 - (C_g.shape[0])**(2*self.rbf_m - self.rbf_n)
		# 	D_d = 1*np.ones([C_d.shape[0]])
		# 	D_g = 1*self.alpha*np.ones([C_g.shape[0]])
		# 	# W_lamb = tf.ones_like(D_d)

		# Weights = np.concatenate((D_d,D_g), axis = 0)


		return C_d, C_g, D_d, D_g 

	def find_lambda(self):

		# if (2*self.rbf_m - self.rbf_n) < 0 or (2*self.rbf_m - self.rbf_n) > 1 or self.data in ['mnist']:
		# 	if self.total_count.numpy() < 50 or self.total_count.numpy() % 500 == 0:
		# 		self.pow_val = np.ceil(np.log10(np.amax(np.abs(self.real_output.numpy()), axis = 0)))
		# 	self.lamb = tf.constant(1.0* (10.0 ** (self.pow_val)))
		# else:
		self.lamb = tf.constant(0.1)


	def divide_by_lambda(self):
		self.real_output = tf.divide(self.real_output, self.lamb)
		self.fake_output = tf.divide(self.fake_output, self.lamb)

class GradRBFLayer(tf.keras.layers.Layer):
	""" Layer of Gaussian RBF units.
	# Example
	```python
		model = Sequential()
		model.add(RBFLayer(10,
						   initializer=InitCentersRandom(X),
						   betas=1.0,
						   input_shape=(1,)))
		model.add(Dense(1))
	```
	# Arguments
		output_dim: number of hidden units (i.e. number of outputs of the
					layer)
		initializer: instance of initiliazer to initialize centers
		betas: float, initial value for betas
	"""

	def __init__(self, num_centres, output_dim, order_m, batch_size, const = 0, rbf_pow = None, initializer=None, **kwargs):

		self.m = order_m
		self.const = const
		self.output_dim = output_dim #1 for us
		self.num_hidden = num_centres #N for us 
		self.rbf_pow =rbf_pow
		# self.unif_weight = 1/batch_size
		if not initializer:
			self.initializer = tf.keras.initializers.RandomUniform(0.0, 1.0)
		else:
			self.initializer = initializer
		super(GradRBFLayer, self).__init__(**kwargs)


	def build(self, input_shape):
		print(input_shape) ## Should be NB x n
		self.n = input_shape[1]
		self.centers = self.add_weight(name='centers',
									   shape=(self.num_hidden, input_shape[1]), ## Nxn
									   initializer=self.initializer,
									   trainable=True)
		self.rbf_weights = self.add_weight(name='rbf_weights',
									 shape=(self.num_hidden,), ## N,1
									 # initializer=tf.keras.initializers.Constant(value=self.unif_weight),
									 initializer='ones',
									 trainable=True)

		super(GradRBFLayer, self).build(input_shape)

	def call(self, X):
		X = tf.expand_dims(X, axis = 2) ## X in Nonexnx1
		# print('Input X',X, X.shape)
		C = tf.expand_dims(self.centers, axis = 2) ## Nxnx1
		# print('Centers C', C, C.shape)
		C = tf.expand_dims(C, axis = 0)
		C_tiled = tf.tile(C, [tf.shape(X)[0],1,1,1])
		X = tf.expand_dims(X, axis = 1)
		X_tiled = tf.tile(X, [1,self.num_hidden,1,1])
		# print('C_tiled', C_tiled, C_tiled.shape)
		# print('X_tiled', X_tiled, X_tiled.shape)
		Tau =  X_tiled - C_tiled ## NonexNxnx1 = NonexNxnx1 - NonexNxnx1


		# print("diff_terms",diff_terms)
		# print('Tau', Tau)
		if self.rbf_pow == None:
			order = (2*self.m) - self.n
		else:
			order = self.rbf_pow
			# if order >= 25:
			# 	return tf.max(Tau, axis = 2)

		# sign = 1.
		if order < 0:
			sign = -1.
		else:
			sign = 1.

		if order < 2:
			epsilon = 1e-5
		else:
			epsilon = 0.
		
		if self.n%2 == 1 or (self.n%2 == 0 and (order)<0):
			norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
			ord_tensor = (order-2)*tf.ones_like(norm_tau)
			Phi = sign*order*tf.pow(norm_tau, ord_tensor) + epsilon ## Nx1
		else:
			norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
			ord_tensor = (order-2)*tf.ones_like(norm_tau)
			log_term = order*tf.math.log(norm_tau+10.0**(-100)) + 1
			Phi = sign*tf.multiply(tf.pow(norm_tau, ord_tensor) + epsilon,log_term)##NonexNx1

		RepPhi = tf.expand_dims(Phi, axis = 2) ##NonexNx1x1
		RepPhi = tf.tile(RepPhi, [1,1,self.n,1])

		if self.n%2 == 1 or (self.n%2 == 0 and (order)<0):
			GradD = tf.multiply(Tau,RepPhi)
		else:
			GradD = tf.multiply(Tau,RepPhi)

		W = tf.expand_dims(self.rbf_weights, axis = 1) ### N x 1

		WGradD =  tf.squeeze(tf.einsum('bNno,No->bno',GradD,W))

		# D += tf.squeeze(tf.linalg.matmul(W, Phi1, transpose_a=True, transpose_b=False),axis = 2)
		# print('WGradD',WGradD)
		# exit(0)
		return WGradD


	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)

	def get_config(self):
		# have to define get_config to be able to use model_from_json
		config = {
			'output_dim': self.output_dim
		}
		base_config = super(GradRBFLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class GradRBFLayerV1(tf.keras.layers.Layer):
	""" Layer of Gaussian RBF units.
	# Example
	```python
		model = Sequential()
		model.add(RBFLayer(10,
						   initializer=InitCentersRandom(X),
						   betas=1.0,
						   input_shape=(1,)))
		model.add(Dense(1))
	```
	# Arguments
		output_dim: number of hidden units (i.e. number of outputs of the
					layer)
		initializer: instance of initiliazer to initialize centers
		betas: float, initial value for betas
	"""

	def __init__(self, num_centres, output_dim, order_m, batch_size, const = 0, rbf_pow = None, initializer=None, **kwargs):

		self.m = order_m
		self.const = const
		self.output_dim = output_dim #1 for us
		self.num_hidden = num_centres #N for us 
		self.rbf_pow =rbf_pow
		# self.unif_weight = 1/batch_size
		if not initializer:
			self.initializer = tf.keras.initializers.RandomUniform(0.0, 1.0)
		else:
			self.initializer = initializer
		super(GradRBFLayer, self).__init__(**kwargs)


	def build(self, input_shape):
		print(input_shape) ## Should be NB x n
		self.n = input_shape[1]
		self.centers = self.add_weight(name='centers',
									   shape=(self.num_hidden, input_shape[1]), ## Nxn
									   initializer=self.initializer,
									   trainable=True)
		self.rbf_weights = self.add_weight(name='rbf_weights',
									 shape=(self.num_hidden,), ## N,1
									 # initializer=tf.keras.initializers.Constant(value=self.unif_weight),
									 initializer='ones',
									 trainable=True)

		super(GradRBFLayer, self).build(input_shape)

	def call(self, X):
		X = tf.expand_dims(X, axis = 2) ## X in Nonexnx1
		# print('Input X',X, X.shape)
		C = tf.expand_dims(self.centers, axis = 2) ## Nxnx1
		# print('Centers C', C, C.shape)
		C = tf.expand_dims(C, axis = 0)
		C_tiled = tf.tile(C, [tf.shape(X)[0],1,1,1])
		X = tf.expand_dims(X, axis = 1)
		X_tiled = tf.tile(X, [1,self.num_hidden,1,1])
		# print('C_tiled', C_tiled, C_tiled.shape)
		# print('X_tiled', X_tiled, X_tiled.shape)
		Tau = -1 * (C_tiled - X_tiled) ## NonexNxnx1 = NonexNxnx1 - NonexNxnx1


		# print("diff_terms",diff_terms)
		# print('Tau', Tau)
		if self.rbf_pow == None:
			order = (2*self.m) - self.n
		else:
			order = self.rbf_pow
			# if order >= 25:
			# 	return tf.max(Tau, axis = 2)

		sign = 1.
		# if order < -2:
		# 	sign = -1.
		# else:
		# 	sign = -1.
		
		if self.n%2 == 1 or (self.n%2 == 0 and (order)<0):
			norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
			ord_tensor = order*tf.ones_like(norm_tau)
			Phi = sign*tf.pow(norm_tau, ord_tensor) ## Nx1
		else:
			norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
			ord_tensor = order*tf.ones_like(norm_tau)
			Phi = 1*tf.multiply(tf.pow(norm_tau, ord_tensor),tf.math.log(norm_tau+10.0**(-100)))##NonexNx1

		RepPhi = tf.expand_dims(Phi, axis = 2) ##NonexNx1x1
		RepPhi = tf.tile(RepPhi, [1,1,self.n,1])

		Diff_times_norm = tf.multiply(Tau,RepPhi)

		W = tf.expand_dims(self.rbf_weights, axis = 1) ### N x 1
		# print('W', W)
		# D = tf.squeeze(tf.linalg.matmul(W, Phi, transpose_a=True, transpose_b=False),axis = 2)

		# print(D)

		GradD =  tf.squeeze(tf.einsum('bNno,No->bno',Diff_times_norm,W))

		# D += tf.squeeze(tf.linalg.matmul(W, Phi1, transpose_a=True, transpose_b=False),axis = 2)
		# print('D',D)
		# exit(0)
		return GradD


	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)

	def get_config(self):
		# have to define get_config to be able to use model_from_json
		config = {
			'output_dim': self.output_dim
		}
		base_config = super(RBFLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))



'''***********************************************************************************
********** The RBF-PHS Solver *************************************************
***********************************************************************************'''
class RBFSolver():

	def __init__(self):
		from itertools import product as cart_prod


		## For 1-D and 2-D Gaussians, no latent projections are needed. So latent dims are full dims itself.
		if self.data in ['g1']:
			self.latent_dims = 1
		if self.data in ['g2', 'gmm8']:
			self.latent_dims = 2
		if self.data in ['gN', 'gmmN']:
			self.latent_dims = self.GaussN
		self.N = self.rbf_n = self.latent_dims

		# self.N_centers = self.batch_size

		self.c = 0

		self.G_loss_counter = 0
		self.first_iter_lambda = 0

		if self.rbf_n%2 == 1:
			if self.rbf_m < ((self.rbf_n+1)/2) :
				self.poly_case = 0
			else:
				self.poly_case = 1 ## odd_n, for all m
		else:
			if self.rbf_m <= ((self.rbf_n/2) - 1):
				self.poly_case = 2 ## even_n, negtive 2m-n
			else:
				self.poly_case = 3 ## even_n, positive 2m-n
			self.rbf_eta = self.rbf_n/2

			self.c_index = self.rbf_m - self.rbf_eta

			r1 = np.arange(1, self.c_index, 1)
			r2 = np.arange(self.rbf_eta, self.rbf_m + self.rbf_eta - 1, 1)

			if self.c_index != 0:
				for tau in r1:
					self.c += (1/(2*tau))
				for tau in r2:
					self.c += (1/(2*tau))




		### Generating MultiIndices
		if self.gan in ['LSGAN']:
			self.multi_index = np.zeros((1,self.rbf_n))
			## m^th order penalty has an (m-1)^th order polynomial 
			for j in range(1,self.rbf_m): #### Used to use rbf_m+1
				x = np.arange(0,self.rbf_n,1)
				bins = np.arange(0,self.rbf_n+1,1)
				a = combinations_with_replacement(x,j)
				for elem in a:
					elem_arr = np.asarray(elem)
					elem_arr = np.expand_dims(elem_arr,axis = 0)
					# try:
					# 	elem_vec = np.concatenate((elem_vec,elem_arr),axis = 0)
					# except:
					# 	elem_vec = elem_arr
					h,bins = np.histogram(elem_arr,bins=list(bins))
					h = np.expand_dims(h,axis = 0)
					self.multi_index = np.concatenate((self.multi_index,h),axis = 0)
			# if self.rbf_m == 1:
			# 	self.multi_index = np.zeros((1,self.rbf_n))
			print(self.multi_index)
			self.indices = tf.cast(self.multi_index, dtype = 'float32')
			# print(multi_index.shape[0])
			print("~~~~~~~~~")


			self.dim_v = self.multi_index.shape[0]

			print("Dimensionality of PolyCoeff vector =",self.dim_v)

			self.LZeroMat = np.zeros((self.dim_v,self.dim_v))

			self.lambda_D = self.LSGANlambdaD

		######3 HACK CHECK##########
		# self.c = 0

		## Defining the Solution cases based on m and n


		## Target data is for evaluateion of alphas (Check Main Manuscript)
		## Generator data is for evaluateion of betas (Check Main Manuscript)
		if self.gan in ['WGAN', 'LSGAN']:
			self.target_data = 'self.reals'
			self.generator_data = 'self.fakes'
		elif self.gan in ['WAE']:
			self.target_data = 'self.fakes_enc'
			self.generator_data = 'self.reals_enc'
		elif self.gan in ['MMDGAN', 'WGAN_AE']:
			self.target_data = 'self.reals_enc'
			self.generator_data = 'self.fakes_enc'
		return

	def discriminator_model_RBF(self):

		if self.gan not in ['WAE', 'MMDGAN', 'WGAN_AE'] and self.data not in ['g1','g2','gmm8', 'gN', 'gmmN']:
			inputs = tf.keras.Input(shape=(self.output_size,self.output_size,self.output_dims))
			inputs_res = tf.keras.layers.Reshape(target_shape = [self.output_size*self.output_size*self.output_dims])(inputs)
		else:
			inputs = tf.keras.Input(shape=(self.latent_dims,))
			inputs_res = inputs

		### For Supp
		num_centers = 2*self.N_centers
		### For Main. Need to correct if need be
		# num_centers = 2*self.batch_size

		##, rbf_pow = 1

		if self.gan in ['WGAN', 'WAE', 'MMDGAN', 'WGAN_AE']:
			D = RBFLayer(num_centres=num_centers, output_dim=1, order_m = self.rbf_m, batch_size = self.batch_size, const = self.c)(inputs_res)
			lambda_term = RBFLayer(num_centres=num_centers, output_dim=1, order_m = self.rbf_m, batch_size = self.batch_size, const = self.c, rbf_pow = -2*self.rbf_n)(inputs_res)
			if self.latent_dims == 1:
				phi0_x = inputs
			else:
				phi0_x = tf.divide(tf.reduce_sum(inputs_res,axis=-1,keepdims=True),self.latent_dims)

			if self.homo_flag:
				Out = tf.keras.layers.Add()([D, phi0_x])
			else:
				Out = D 

			model = tf.keras.Model(inputs=inputs, outputs= [Out,lambda_term])

		if self.gan == 'LSGAN':
			[Out,A,B] = PHSLayer(num_centres=num_centers, output_dim=1,  dim_v = self.dim_v, rbf_k = self.rbf_m, batch_size = self.batch_size, multi_index = self.multi_index)(inputs_res)
			# Poly_of_x = tf.keras.layers.Dense(units = 1, use_bias = False)(XPoly)

			# Out = tf.keras.layers.Add()([D, Poly_of_x])

			model = tf.keras.Model(inputs=inputs, outputs= [Out,A,B])
		

		return model



	def PHS_MatrixSolver_Numpy(self, vals):#, centers, vals, phs_deg, poly_deg):
		N = self.A.shape[0]
		# print(self.B)
		Correction = -96*np.pi*self.lambda_D*np.eye(N)
		self.BT = tf.transpose(self.B)
		# print(self.A.shape, self.B.shape, self.LZeroMat.shape)
		M = np.concatenate( (np.concatenate((self.A+Correction,self.B), axis = 1),np.concatenate((self.BT,self.LZeroMat), axis = 1)), axis = 0)

		# print(M,M.shape)
		# print(np.linalg.matrix_rank(M))
		# print(vals, vals.shape)

		y = np.concatenate((vals,np.zeros((self.dim_v,1))), axis = 0)
		sols = np.linalg.solve(M,y)
		# print(sols)


		Weights = np.squeeze(sols[0:N])
		PolyWts = sols[N:]
		# PolyWts = np.zeros_like(PolyWts)
		
		return Weights, PolyWts

	def PHS_MatrixSolver(self, vals):#, centers, vals, phs_deg, poly_deg):
		N = self.A.shape[0]
		# print(self.B)
		Correction = -96*3.14159*self.lambda_D*tf.eye(N)

		# if self.data in ['cifar10','celeba']:
		# 	B_Correction = 0.01*tf.eye(self.B.shape[0])
		# 	self.B = self.B + B_Correction

		self.BT = tf.transpose(self.B)
		# print(self.A.shape, self.B.shape, self.LZeroMat.shape)
		M = tf.concat( (tf.concat((self.A+Correction,self.B), axis = 1),tf.concat((self.BT,self.LZeroMat), axis = 1)), axis = 0)

		# print(M,M.shape)
		# print(np.linalg.matrix_rank(M))
		# print(vals, vals.shape)

		y = tf.concat((vals,tf.zeros((self.dim_v,1))), axis = 0)
		sols = tf.linalg.solve(M,y)
		# print(sols)


		Weights = tf.squeeze(sols[0:N])
		PolyWts = sols[N:]
		# PolyWts = np.zeros_like(PolyWts)
		
		return Weights, PolyWts

	def find_rbf_centres_weights(self):

		C_d = eval(self.target_data)[0:self.N_centers] #SHould be NDxn
		C_g = eval(self.generator_data)[0:self.N_centers]
		check = 1
		if self.gan not in ['WAE', 'MMDGAN', 'WGAN_AE'] and self.data not in ['g1', 'g2','gmm8', 'gN', 'gmmN']:
			C_d = tf.reshape(C_d, [C_d.shape[0], C_d.shape[1]*C_d.shape[2]*C_d.shape[3]])
			C_g = tf.reshape(C_g, [C_g.shape[0], C_g.shape[1]*C_g.shape[2]*C_g.shape[3]])

		Centres = np.concatenate((C_d,C_g), axis = 0)

		# D_d = check*(((-1)**(2*self.rbf_m - C_d.shape[1]+1))/self.reals.shape[0])*np.ones([self.batch_size])
		# D_g = check*(((-1)**(2*self.rbf_m - C_g.shape[1]))/self.fakes.shape[0])*np.ones([self.batch_size])

		if self.gan in ['WGAN', 'WAE', 'MMDGAN', 'WGAN_AE']:
			if (2*self.rbf_m - self.rbf_n) <= 2 or self.rbf_n <= 350:
				self.alpha = self.beta = 1
				D_d = (-1/C_d.shape[0])*np.ones([C_d.shape[0]])
				D_g = (1/(C_g.shape[0]))*np.ones([C_g.shape[0]])
				W_lamb = 1*tf.ones_like(D_d)
			else:
				self.alpha = 1/(C_g.shape[0])**(2*self.rbf_m - self.rbf_n)
				self.beta = 1#2 - (C_g.shape[0])**(2*self.rbf_m - self.rbf_n)
				D_d = -1*np.ones([C_d.shape[0]])
				D_g = 1*self.alpha*np.ones([C_g.shape[0]])
				W_lamb = tf.ones_like(D_d)

			Weights = np.concatenate((D_d,D_g), axis = 0)

			Lamb_Weights = np.concatenate((W_lamb,W_lamb), axis = 0)

			return Centres, Weights, Lamb_Weights

		elif self.gan == 'LSGAN':
			d_vals = self.label_b*tf.ones((C_d.shape[0],1))
			g_vals = self.label_a*tf.ones((C_g.shape[0],1))
			Values = np.concatenate((d_vals,g_vals), axis = 0)

			Weights, PolyWeights = self.PHS_MatrixSolver(vals = Values)#centers = Centres, vals = Values, phs_deg = self.rbf_m, poly_deg = self.rbf_m-1)

			# print(Weights)
			# print(PolyWeights)

			return Centres, Weights, PolyWeights


	def find_lambda(self):

		if (2*self.rbf_m - self.rbf_n) < 0 or (2*self.rbf_m - self.rbf_n) > 1 or self.data in ['mnist']:
			if self.total_count.numpy() < 50 or self.total_count.numpy() % 500 == 0:
				self.pow_val = np.ceil(np.log10(np.amax(np.abs(self.real_output.numpy()), axis = 0)))
			self.lamb = tf.constant(1.0* (10.0 ** (self.pow_val)))
		else:
			self.lamb = tf.constant(1.0)


	def divide_by_lambda(self):
		self.real_output = tf.divide(self.real_output, self.lamb)
		self.fake_output = tf.divide(self.fake_output, self.lamb)


class RBFLayer(tf.keras.layers.Layer):
	""" Layer of Gaussian RBF units.
	# Example
	```python
		model = Sequential()
		model.add(RBFLayer(10,
						   initializer=InitCentersRandom(X),
						   betas=1.0,
						   input_shape=(1,)))
		model.add(Dense(1))
	```
	# Arguments
		output_dim: number of hidden units (i.e. number of outputs of the
					layer)
		initializer: instance of initiliazer to initialize centers
		betas: float, initial value for betas
	"""

	def __init__(self, num_centres, output_dim, order_m, batch_size, const = 0, rbf_pow = None, initializer=None, **kwargs):

		self.m = order_m
		self.const = const
		self.output_dim = output_dim #1 for us
		self.num_hidden = num_centres #N for us 
		self.rbf_pow =rbf_pow
		# self.fake_centers = batch_size
		# self.unif_weight = 1/batch_size
		if not initializer:
			self.initializer = tf.keras.initializers.RandomUniform(0.0, 1.0)
		else:
			self.initializer = initializer
		super(RBFLayer, self).__init__(**kwargs)


	def build(self, input_shape):
		print(input_shape) ## Should be NB x n
		self.n = input_shape[1]
		self.centers = self.add_weight(name='centers',
									   shape=(self.num_hidden, input_shape[1]), ## Nxn
									   initializer=self.initializer,
									   trainable=True)
		self.rbf_weights = self.add_weight(name='rbf_weights',
									 shape=(self.num_hidden,), ## N,1
									 # initializer=tf.keras.initializers.Constant(value=self.unif_weight),
									 initializer='ones',
									 trainable=True)

		super(RBFLayer, self).build(input_shape)

	def call(self, X):
		X = tf.expand_dims(X, axis = 2) ## X in Nonexnx1
		# print('Input X',X, X.shape)
		C = tf.expand_dims(self.centers, axis = 2) ## Nxnx1
		# print('Centers C', C, C.shape)
		C = tf.expand_dims(C, axis = 0)
		C_tiled = tf.tile(C, [tf.shape(X)[0],1,1,1])
		X = tf.expand_dims(X, axis = 1)
		X_tiled = tf.tile(X, [1,self.num_hidden,1,1])
		# print('C_tiled', C_tiled, C_tiled.shape)
		# print('X_tiled', X_tiled, X_tiled.shape)
		Tau = C_tiled - X_tiled ## NonexNxnx1 = NonexNxnx1 - NonexNxnx1
		# print('Tau', Tau)
		if self.rbf_pow == None:
			order = (2*self.m) - self.n
		else:
			order = self.rbf_pow
			# if order >= 25:
			# 	return tf.max(Tau, axis = 2)

		if order < 0:
			sign = -1.
		else:
			sign = 1.
		
		if self.n%2 == 1 or (self.n%2 == 0 and (2*self.m-self.n)<0):
			# order = 3
			norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
			ord_tensor = order*tf.ones_like(norm_tau)
			Phi = sign*tf.pow(norm_tau, ord_tensor) ## Nx1
		else:
			norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
			ord_tensor = order*tf.ones_like(norm_tau)
			# ord_ones = 0.0*tf.ones_like(norm_tau)

			# if (2*self.m - self.n) <= 25:
			# Phi = 1*tf.multiply(tf.pow(norm_tau, ord_tensor),(self.const-tf.math.log(norm_tau+10.0**(-100))))##Nx1
			Phi = 1*tf.multiply(tf.pow(norm_tau, ord_tensor),tf.math.log(norm_tau+10.0**(-100)))##Nx1

			# else:
				# Phi = tf.math.exp((order-1)*tf.math.log(norm_tau) + tf.math.log((self.const*norm_tau-tf.math.xlogy(norm_tau,norm_tau))))
			# Phi1 = 1*sign*tf.multiply(tf.pow(norm_tau, ord_ones),(0-tf.math.log(norm_tau)))

		# print('Phi', Phi)
		W = tf.expand_dims(self.rbf_weights, axis = 1)
		# print('W', W)
		D = tf.squeeze(tf.linalg.matmul(W, Phi, transpose_a=True, transpose_b=False),axis = 2)

		# print(D)

		# D += tf.squeeze(tf.linalg.matmul(W, Phi1, transpose_a=True, transpose_b=False),axis = 2)
		# print('D',D)
		# exit(0)
		return D


	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)

	def get_config(self):
		# have to define get_config to be able to use model_from_json
		config = {
			'output_dim': self.output_dim
		}
		base_config = super(RBFLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class PHSLayer(tf.keras.layers.Layer):
	""" Layer of Gaussian RBF units.
	# Example
	```python
		model = Sequential()
		model.add(RBFLayer(10,
						   initializer=InitCentersRandom(X),
						   betas=1.0,
						   input_shape=(1,)))
		model.add(Dense(1))
	```
	# Arguments
		output_dim: number of hidden units (i.e. number of outputs of the
					layer)
		initializer: instance of initiliazer to initialize centers
		betas: float, initial value for betas
	"""

	def __init__(self, num_centres, output_dim, dim_v, rbf_k, multi_index, batch_size, initializer=None, **kwargs):

		# self.m = order_m
		self.dim_v = dim_v
		self.output_dim = output_dim #1 for us
		self.num_centres = num_centres #N for us 
		self.rbf_k =rbf_k ## Shoudl be m?
		self.multi_index = tf.cast(multi_index, dtype = 'float32')
		print(self.multi_index.shape)
		# self.unif_weight = 1/batch_size
		if not initializer:
			self.initializer = tf.keras.initializers.RandomUniform(0.0, 1.0)
		else:
			self.initializer = initializer
		super(PHSLayer, self).__init__(**kwargs)


	def build(self, input_shape):
		# print(input_shape) ## Should be NB x n
		self.n = input_shape[1]
		self.centers = self.add_weight(name='centers',
									   shape=(self.num_centres, input_shape[1]), ## Nxn
									   initializer=self.initializer,
									   trainable=True)
		self.rbf_weights = self.add_weight(name='rbf_weights',
									 shape=(self.num_centres,), ## N,1
									 # initializer=tf.keras.initializers.Constant(value=self.unif_weight),
									 initializer='ones',
									 trainable=True)
		self.poly_weights = self.add_weight(name='poly_weights',
									 shape=(self.dim_v,1), ## L,1
									 # initializer=tf.keras.initializers.Constant(value=self.unif_weight),
									 initializer='ones',
									 trainable=True)

		super(PHSLayer, self).build(input_shape)

	def call(self, X):

		def odd_PHS(f,order):
			norm_f = tf.norm(f, ord = 'euclidean', axis = 2)
			ord_tensor = order*tf.ones_like(norm_f)
			Phi = 1*tf.pow(norm_f, ord_tensor) ## Nx1
			return Phi

		def even_PHS(f,order):
			norm_f = tf.norm(f, ord = 'euclidean', axis = 2)
			ord_tensor = order*tf.ones_like(norm_f)
			Phi = 1*tf.multiply(tf.pow(norm_f, ord_tensor),tf.math.log(norm_f+10.0**(-100)))##Nx1
			return Phi

		X = tf.expand_dims(X, axis = 2) ## X in Nonexnx1
		# print('Input X',X, X.shape)
		Cp = C = tf.expand_dims(self.centers, axis = 2) ## Nxnx1
		# print('Centers C', C, C.shape)
		C = tf.expand_dims(C, axis = 0) ## 1xNxnx1
		C_tiled = tf.tile(C, [tf.shape(X)[0],1,1,1]) ## NonexNxnx1
		X = tf.expand_dims(X, axis = 1) ## Nonex1xnx1
		X_tiled = tf.tile(X, [1,self.num_centres,1,1]) ## NonexNxnx1
		# print('C_tiled', C_tiled, C_tiled.shape)
		# print('X_tiled', X_tiled, X_tiled.shape)
		Tau = C_tiled - X_tiled ## NonexNxnx1 = NonexNxnx1 - NonexNxnx1
		# print('Tau', Tau)

		#### 1) We compute the Polyharmonic part of PHS D(x)
		self.m_given_k = tf.math.ceil((self.rbf_k + self.n)/2.)
		if self.rbf_k%2 == 1:
			Phi = odd_PHS(Tau,self.rbf_k) ## NonexNx1
		else:
			Phi = even_PHS(Tau,self.rbf_k) ## NonexNx1

		# print('Phi', Phi)
		W = tf.expand_dims(self.rbf_weights, axis = 1) ## Nx1
		# print('W', W)
		D_PHS = tf.squeeze(tf.linalg.matmul(W, Phi, transpose_a=True, transpose_b=False),axis = 2) ## Nonex1



		#### 2) We compute the Poly terms for X

		PolyPow = tf.expand_dims(self.multi_index, axis = 2) ## Lxnx1
		PolyPow = tf.expand_dims(PolyPow, axis = 0) ## 1xLxnx1
		# print('PolyPow',PolyPow)
		PolyPow_PowTiled = tf.tile(PolyPow, [tf.shape(X)[0],1,1,1]) ## NonexLxnx1
		# print('PolyPow_PowTiled',PolyPow_PowTiled)
		X_PowTiled = tf.tile(X, [1,self.dim_v,1,1]) ## NonexLxnx1
		# print('X_PowTiled',X_PowTiled)
		X_Pow = tf.pow(X_PowTiled, PolyPow_PowTiled) ## NonexLxnx1
		# print('X_Pow',X_Pow)
		X_PowPord = tf.reduce_prod(X_Pow, axis = 2) ## NonexLx1
		# print('X_PowPord',X_PowPord)
		# V = tf.expand_dims(self.poly_weights, axis = 1) ## Lx1
		V = self.poly_weights
		# print('V',V)
		D_Poly = tf.squeeze(tf.linalg.matmul(V, X_PowPord, transpose_a=True, transpose_b=False),axis = 2) ## Nonex1
		
		D = D_PHS + 0.0*D_Poly

		#### 3) We compute the matrix A for PHS weight computation

		C_NxN = tf.tile(C, [self.num_centres,1,1,1]) ## 1xNxnx1 -> ## NxNxnx1
		Tau_C = C_NxN - tf.transpose(C_NxN, perm=[1,0,2,3]) ## NxNxnx1

		corr = 8*tf.acos(-1.)*10e-1*tf.eye(self.num_centres)
		corr = tf.expand_dims(corr,axis = 2)
		corr = tf.expand_dims(corr,axis = 3)
		corr = tf.tile(corr, [1,1,self.n,1])

		Tau_C = Tau_C + corr

		# print('Tau_C',Tau_C)
		if self.rbf_k%2 == 1:
			A = odd_PHS(Tau_C,self.rbf_k) ## NxNx1
		else:
			A = even_PHS(Tau_C,self.rbf_k) ## NxNx1
		# print('A', A)
		A = tf.squeeze(A) ## NxN
		# print('A squeezed', A)

		#### 4) We compute the matrix B for PHS weight computation
		
		PolyPow_CTiled = tf.tile(PolyPow, [self.num_centres,1,1,1]) ## NxLxnx1
		# print('PolyPow_CTiled',PolyPow_CTiled)
		Cp = tf.expand_dims(Cp, axis = 1) ## Nx1xnx1
		# print('Cp',Cp)
		Cp_Tiled = tf.tile(Cp, [1, self.dim_v,1,1]) ## NxLxnx1
		# print('Cp_Tiled',Cp_Tiled)
		C_Pow = tf.pow(Cp_Tiled, PolyPow_CTiled) ## NxLxnx1
		# print('C_Pow',C_Pow)
		C_PowPord = tf.reduce_prod(C_Pow, axis = 2) ## NxLx1
		# print('C_PowPord',C_PowPord)
		B = tf.squeeze(C_PowPord) ## NxL
		if self.rbf_k == 1: ###  1 if the monomials code does not include rbf_m + 1
			B = tf.expand_dims(B,axis = 1)
		# print('B squeezed', B)

		return [D,A,B]


		# if self.rbf_pow == None:
		# 	order = (2*self.m) - self.n
		# else:
		# 	order = self.rbf_pow
			# if order >= 25:
			# 	return tf.max(Tau, axis = 2)

		# if self.n%2 == 1 or (self.n%2 == 0 and (2*self.m-self.n)<0) :
		# 	norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
		# 	ord_tensor = order*tf.ones_like(norm_tau)
		# 	Phi = sign*tf.pow(norm_tau, ord_tensor) ## Nx1
		# else:
		# 	norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
		# 	ord_tensor = order*tf.ones_like(norm_tau)
		# 	Phi = 1*tf.multiply(tf.pow(norm_tau, ord_tensor),tf.math.log(norm_tau+10.0**(-100)))##Nx1


	def compute_output_shape(self, input_shape):
		return [(input_shape[0], self.output_dim), \
				(self.num_centres, self.num_centres), \
				(self.dim_v, self.num_centres)]

	def get_config(self):
		# have to define get_config to be able to use model_from_json
		config = {
			'output_dim': self.output_dim
		}
		base_config = super(PHSLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


'''***********************************************************************************
********** GAN SpiderGAN setup *******************************************************
***********************************************************************************'''
class GAN_SpiderGAN(GAN_SRC, GAN_DATA_SpiderGAN):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all fundamental ops and metric functions'''
		GAN_SRC.__init__(self,FLAGS_dict)
		''' Set up the GAN_DATA class'''
		GAN_DATA_SpiderGAN.__init__(self)

	def initial_setup(self):

		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_func_noise = 'self.gen_func_'+self.noise_data+'()'
		self.gen_model = 'self.generator_model_'+self.arch+'_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.arch+'_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size, reps_train)'
		self.dataset_func_noise = 'self.dataset_'+self.noise_data+'(self.train_data_noise, self.batch_size, reps_noise)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		self.noise_setup()

		''' Define dataset and tf.data function. batch sizing done'''
		# self.get_data()
		# print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.num_batches,self.print_step, self.save_step))


	def get_data(self):

		# with tf.device('/CPU'):
		# with self.strategy.scope():
		self.train_data = eval(self.gen_func)
		print(self.train_data.shape)
		self.train_data_noise = eval(self.gen_func_noise)
		print(self.train_data_noise.shape)
		self.ratio = self.train_data.shape[0]/self.train_data_noise.shape[0] # is the num of reps noise data needs, to match train data
		reps_train = np.ceil(1/float(self.ratio))
		reps_noise = np.ceil(self.ratio)
		print("reps_train",reps_train)
		print("reps_noise",reps_noise)

		# self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
		self.max_data_size = max(self.train_data.shape[0],self.train_data_noise.shape[0])
		self.num_batches = int(np.floor(self.max_data_size/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)
		self.noise_dataset = eval(self.dataset_func_noise)

		self.train_dataset_size = self.max_data_size

		# self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
		# self.noise_dataset = self.strategy.experimental_distribute_dataset(self.noise_dataset)

		## with was till here
		print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size,  self.num_batches, self.print_step, self.save_step))


	def noise_setup(self):

		if self.noise_kind == 'non_para':
			self.non_para_dist_mat = sio.loadmat('data/non_param_pdf.mat')

		if self.noise_kind == 'gamma':
			self.gamma_shape = 0.5
			self.gamma_scale = 1.0

		return

	def get_noise(self, shape):

		def random_gen(shape, pdf, points, epsilon):
			assert len(shape) == 2
			rn = np.random.choice(points, size = shape, p=pdf).astype(np.float32)
			for i in range(shape[0]):
				for j in range(shape[1]):
					rn[i,j] = np.random.uniform(rn[i,j], rn[i,j]+epsilon, 1).astype(np.float32)
			return rn


		def sample_spherical(npoints, ndim=3):
			vec = np.random.randn(ndim, npoints)
			vec /= np.linalg.norm(vec, axis=0)
			return vec


		if self.noise_kind == 'non_para':
			pdf = np.reshape(self.non_para_dist_mat['X_final'], (1024))
			points = np.linspace(-2, 2, 1024)
			epsilon = 4./1024.
			noise = random_gen(shape, pdf, points, epsilon)

		elif self.noise_kind == 'gaussian':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = self.noise_stddev)


		elif self.noise_kind == 'gaussian025':
			noise = tf.random.normal(shape, mean = 0.0, stddev = 0.25, dtype=tf.float32)

		elif self.noise_kind == 'gaussian01':
			noise = tf.random.normal(shape, mean = 0.0, stddev = 0.1, dtype=tf.float32)

		elif self.noise_kind == 'zeros':
			noise = tf.zeros(shape, dtype=tf.float32)


		elif self.noise_kind == 'gamma':
			nu = sample_spherical(shape[0], shape[1]).transpose()
			r = np.random.gamma(self.gamma_shape, scale=self.gamma_scale, size=shape[0])
			root_r_mat = np.repeat(np.expand_dims(np.sqrt(r),axis=1), shape[1], axis = 1)
			noise = np.multiply(root_r_mat,nu)
			# print(root_r_mat.shape, nu.shape, noise.shape)
			noise = noise.astype(dtype = 'float32')

		elif self.noise_kind == 'cauchy':
			noise = np.random.standard_cauchy(size=shape)
			noise = noise.astype(dtype = 'float32')

		return noise

	###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	def create_models(self):
		# with self.strategy.scope():
		self.total_count = tf.Variable(0,dtype='int64')
		self.generator = eval(self.gen_model)
		self.discriminator = eval(self.disc_model)

		if self.res_flag == 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n GENERATOR MODEL: \n\n")
				self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
				self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

		print("Model Successfully made")

		print(self.generator.summary())
		print(self.discriminator.summary())

		if self.TanGAN_flag == 1 or self.BaseTanGAN_flag == 1:
			self.TanGAN_generator = tf.keras.models.load_model(self.latent_gen_folder+self.latent_gen_run_id+'/checkpoints/model_generator.h5')

			print(self.TanGAN_generator.summary())
			if self.res_flag == 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n TanGAN GENERATOR MODEL: \n\n")
					self.TanGAN_generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return		


	###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
								 generator = self.generator,
								 discriminator = self.discriminator,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

				###### UGLY HACK - Gamma Model C10 -> SVHN FAILED
				assert 0 == 1
				print("Checkpoint restored...")
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
					print("H5 restored...")
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")


			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return


	def train(self):    
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1 
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)
			start = time.perf_counter()
			batch_count = tf.Variable(0,dtype='int64')
			start_1 = 0

			for image_batch, noise_batch1, noise_batch2 in zip(self.train_dataset, self.noise_dataset, self.noise_dataset):
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_1 = time.perf_counter()


				#### ---- Sanity check to see if 2 pulls give 2 images ---- ####
				####
				# print(noise_batch1.shape)
				# print(noise_batch2.shape)
				# self.save_image_batch(images = noise_batch1,label = 'Noise BAtch 1',path = 'batch2.png')
				# self.save_image_batch(images = noise_batch2,label = 'Noise Batch 2',path = 'batch1.png')
				# exit(0)
				####
				####

				# with self.strategy.scope():
				self.train_step(image_batch, noise_batch1, noise_batch2)
				self.eval_metrics()

				train_time = time.perf_counter()-start_1
					
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

				if self.iters_flag:
					if self.num_iters == self.total_count.numpy():
						tf.print("\n Training for {} Iterations completed".format( self.total_count.numpy()))
						if self.pbar_flag:
							bar.close()
							del bar
						self.save_epoch_h5models()
						return

			if self.pbar_flag:
				bar.close()
				del bar
			tf.print ('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
			self.save_epoch_h5models()


	def print_batch_outputs(self,epoch):		
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		# if (self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']:
		# 	self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)


	def save_epoch_h5models(self):
		if self.arch!='biggan':
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return


	def print_batch_outputs(self,epoch):
		if (self.total_count.numpy() <= 5) and self.data in ['g1', 'g2']:
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']):
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def eval_sharpness(self):
		i = 0
		for train_batch, noise_batch in zip(self.train_dataset, self.noise_dataset):
			preds = self.generator(noise_batch, training = False)

			sharp = self.find_sharpness(preds)
			base_sharp = self.find_sharpness(train_batch)
			try:
				sharp_vec.append(sharp)
				base_sharp_vec.append(base_sharp)

			except:
				sharp_vec = [sharp]
				base_sharp_vec = [base_sharp]
			i += 1
			if i == 10:
				break

		###### Sharpness averaging measure
		sharpness = np.mean(np.array(sharp_vec))
		baseline_sharpness = np.mean(np.array(base_sharp_vec))

		return baseline_sharpness, sharpness

	def test(self):
		num_interps = 10
		if self.mode == 'test':
			num_figs = 20#int(400/(2*num_interps))
		else:
			num_figs = 9

		fig_count = 0
		# there are 400 samples in the batch. to make 10x10 images, 
		for noise_batch in self.noise_dataset:

			for j in range(5):
				fig_count += 1
				path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(fig_count)+'.png'
				noi_path = self.impath+'_TestingInterpolationV2_NOISE_'+str(self.total_count.numpy())+'_TestCase_'+str(fig_count)+'.png'
				OP1_path = self.impath+'_TestingInterpolationV2_OP1_'+str(self.total_count.numpy())+'_TestCase_'+str(fig_count)+'.png'

				current_batch = noise_batch[2*num_interps*j:2*num_interps*(j+1)]
				# image_latents = self.Encoder(current_batch)
				for i in range(num_interps):
					start = np.reshape(current_batch[i:1+i].numpy(),(self.input_size*self.input_size*self.input_dims))
					end = np.reshape(current_batch[num_interps+i:num_interps+1+i].numpy(),(self.input_size*self.input_size*self.input_dims))

					# print(start.shape, end.shape)
					stack = np.vstack([start, end])

					linfit = interp1d([1,num_interps+1], stack, axis=0)
					interp_latents = linfit(list(range(1,num_interps+1)))
					# print(interp_latents.shape)
					interp_noise_images = np.reshape(interp_latents, (interp_latents.shape[0],self.input_size,self.input_size,self.input_dims))
					# print(interp_noise_images.shape)

					if self.TanGAN_flag == 1:
						interp_OP1_images = self.TanGAN_generator(interp_noise_images)
						cur_interp_figs = self.generator(interp_OP1_images)
					elif self.BaseTanGAN_flag == 1:
						start = self.get_noise([1, self.noise_dims])#current_batch[i:1+i].numpy()
						end = self.get_noise([1, self.noise_dims]) 
						stack = np.vstack([start, end])

						linfit = interp1d([1,num_interps+1], stack, axis=0)
						interp_latents = linfit(list(range(1,num_interps+1)))
						interp_OP1_images = self.TanGAN_generator(interp_latents)
						cur_interp_figs = self.generator(interp_OP1_images)
					else:
						cur_interp_figs = self.generator(interp_noise_images)

					sharpness = self.find_sharpness(cur_interp_figs)
					try:
						sharpness_vec.append(sharpness)
					except:
						shaprpness_vec = [sharpness]

					try:
						batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs),axis = 0)
						batch_noise_figs = np.concatenate((batch_noise_figs,interp_noise_images), axis = 0)
						if self.TanGAN_flag or self.BaseTanGAN_flag:
							batch_OP1_figs = np.concatenate((batch_OP1_figs,interp_OP1_images), axis = 0)
					except:
						batch_interp_figs = cur_interp_figs
						batch_noise_figs = interp_noise_images
						if self.TanGAN_flag or self.BaseTanGAN_flag:
							batch_OP1_figs = interp_OP1_images

				images = (batch_interp_figs + 1.0)/2.0
				# print(images.shape)
				size_figure_grid = num_interps
				images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(images.shape[1],images.shape[2]),num_channels=images.shape[3])
				fig1 = plt.figure(figsize=(num_interps,num_interps))
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.axis("off")
				if images_on_grid.shape[2] == 3:
					ax1.imshow(np.clip(images_on_grid,0.,1.))
				else:
					ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

				label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
				plt.title(label)
				plt.tight_layout()
				plt.savefig(path)
				plt.close()

				noise_print_image = (batch_noise_figs + 1.0)/2.0
				# print(images.shape)
				size_figure_grid = num_interps
				images_on_grid = self.image_grid(input_tensor = noise_print_image, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(noise_print_image.shape[1],noise_print_image.shape[2]),num_channels=noise_print_image.shape[3])
				fig1 = plt.figure(figsize=(num_interps,num_interps))
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.axis("off")
				if images_on_grid.shape[2] == 3:
					ax1.imshow(np.clip(images_on_grid,0.,1.))
				else:
					ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

				label = 'INTERPOLATED NOISE IMAGES AT ITERATION '+str(self.total_count.numpy())
				plt.title(label)
				plt.tight_layout()
				plt.savefig(noi_path)
				plt.close()

				if self.TanGAN_flag or self.BaseTanGAN_flag:
					OP1_print_image = (batch_OP1_figs + 1.0)/2.0
					# print(images.shape)
					size_figure_grid = num_interps
					images_on_grid = self.image_grid(input_tensor = OP1_print_image, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(OP1_print_image.shape[1],OP1_print_image.shape[2]),num_channels=OP1_print_image.shape[3])
					fig1 = plt.figure(figsize=(num_interps,num_interps))
					ax1 = fig1.add_subplot(111)
					ax1.cla()
					ax1.axis("off")
					if images_on_grid.shape[2] == 3:
						ax1.imshow(np.clip(images_on_grid,0.,1.))
					else:
						ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

					label = 'INTERPOLATED OP1 IMAGES AT ITERATION '+str(self.total_count.numpy())
					plt.title(label)
					plt.tight_layout()
					plt.savefig(OP1_path)
					plt.close()		

				del batch_interp_figs
			if fig_count >= num_figs:
				break

		###### Interpol samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Interpolation Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("Interpolation Sharpness - "+str(overall_sharpness))




			# path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			# label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())

			# size_figure_grid = self.num_to_print
			# test_batch_size = size_figure_grid*size_figure_grid
			# noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)

			# images = self.generator(noise, training=False)
			# if self.data != 'celeba':
			# 	images = (images + 1.0)/2.0

			# self.save_image_batch(images = images,label = label, path = path)


'''***********************************************************************************
********** GAN SpiderGAN setup *******************************************************
***********************************************************************************'''
class GAN_CondSpiderGAN(GAN_SRC, GAN_DATA_CondSpiderGAN):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all fundamental ops and metric functions'''
		GAN_SRC.__init__(self,FLAGS_dict)
		''' Set up the GAN_DATA class'''
		GAN_DATA_CondSpiderGAN.__init__(self)

	def initial_setup(self):

		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_func_noise = 'self.gen_func_'+self.noise_data+'()'
		self.gen_model = 'self.generator_model_'+self.arch+'_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.arch+'_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.train_labels, self.batch_size, reps_train)'
		self.dataset_func_noise = 'self.dataset_'+self.noise_data+'(self.train_data_noise, self.noise_labels, self.batch_size, reps_noise)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		self.noise_setup()

		''' Define dataset and tf.data function. batch sizing done'''
		# self.get_data()
		# print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.num_batches,self.print_step, self.save_step))


	def get_data(self):

		# with tf.device('/CPU'):
		# with self.strategy.scope():
		self.train_data, self.train_labels = eval(self.gen_func)
		self.train_data_noise, self.noise_labels = eval(self.gen_func_noise)
		self.ratio = self.train_data.shape[0]/self.train_data_noise.shape[0] # is the num of reps noise data needs, to match train data
		reps_train = np.ceil(1/float(self.ratio))
		reps_noise = np.ceil(self.ratio)
		print("reps_train",reps_train)
		print("reps_noise",reps_noise)

		# self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
		self.max_data_size = max(self.train_data.shape[0],self.train_data_noise.shape[0])
		self.num_batches = int(np.floor(self.max_data_size/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)
		self.noise_dataset = eval(self.dataset_func_noise)

		self.train_dataset_size = self.max_data_size
			#
		## with was till here

		print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size,  self.num_batches, self.print_step, self.save_step))


	def noise_setup(self):

		if self.noise_kind == 'non_para':
			self.non_para_dist_mat = sio.loadmat('data/non_param_pdf.mat')

		if self.noise_kind == 'gamma':
			self.gamma_shape = 0.5
			self.gamma_scale = 1.0

		return

	def get_noise(self, shape):

		def random_gen(shape, pdf, points, epsilon):
			assert len(shape) == 2
			rn = np.random.choice(points, size = shape, p=pdf).astype(np.float32)
			for i in range(shape[0]):
				for j in range(shape[1]):
					rn[i,j] = np.random.uniform(rn[i,j], rn[i,j]+epsilon, 1).astype(np.float32)
			return rn


		def sample_spherical(npoints, ndim=3):
			vec = np.random.randn(ndim, npoints)
			vec /= np.linalg.norm(vec, axis=0)
			return vec


		if self.noise_kind == 'non_para':
			pdf = np.reshape(self.non_para_dist_mat['X_final'], (1024))
			points = np.linspace(-2, 2, 1024)
			epsilon = 4./1024.
			noise = random_gen(shape, pdf, points, epsilon)

		elif self.noise_kind == 'gaussian':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = self.noise_stddev)


		elif self.noise_kind == 'gaussian025':
			noise = tf.random.normal(shape, mean = 0.0, stddev = 0.25, dtype=tf.float32)

		elif self.noise_kind == 'zeros':
			noise = tf.zeros(shape, dtype=tf.float32)


		elif self.noise_kind == 'gamma':
			nu = sample_spherical(shape[0], shape[1]).transpose()
			r = np.random.gamma(self.gamma_shape, scale=self.gamma_scale, size=shape[0])
			root_r_mat = np.repeat(np.expand_dims(np.sqrt(r),axis=1), shape[1], axis = 1)
			noise = np.multiply(root_r_mat,nu)
			# print(root_r_mat.shape, nu.shape, noise.shape)
			noise = noise.astype(dtype = 'float32')

		elif self.noise_kind == 'cauchy':
			noise = np.random.standard_cauchy(size=shape)
			noise = noise.astype(dtype = 'float32')

		return noise

	###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	def create_models(self):
		# with self.strategy.scope():
		self.total_count = tf.Variable(0,dtype='int64')
		self.generator = eval(self.gen_model)
		self.discriminator = eval(self.disc_model)

		if self.res_flag == 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n GENERATOR MODEL: \n\n")
				self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
				self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

		print("Model Successfully made")

		print(self.generator.summary())
		print(self.discriminator.summary())
		return		


	###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
								 generator = self.generator,
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
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return


	def train(self):    
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1 
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)
			start = time.perf_counter()
			batch_count = tf.Variable(0,dtype='int64')
			start_1 = 0

			for A,B,C in zip(self.train_dataset, self.noise_dataset, self.noise_dataset):
				image_batch, image_labels = A
				noise_batch1, noise_labels1 = B 
				noise_batch2, noise_labels2 = C
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_1 = time.perf_counter()

				# with self.strategy.scope():
				self.train_step(image_batch, image_labels, noise_batch1, noise_labels1, noise_batch2, noise_labels2)
				self.eval_metrics()

				train_time = time.perf_counter()-start_1
					
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
			tf.print ('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
			self.save_epoch_h5models()


	def print_batch_outputs(self,epoch):		
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		# if (self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']:
		# 	self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)


	def save_epoch_h5models(self):
		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
		self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return


	# def print_batch_outputs(self,epoch):
	# 	if (self.total_count.numpy() <= 5) and self.data in ['g1', 'g2']:
	# 		self.generate_and_save_batch(epoch)
	# 	if ((self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']):
	# 		self.generate_and_save_batch(epoch)
	# 	if (self.total_count.numpy() % self.save_step.numpy()) == 0:
	# 		self.generate_and_save_batch(epoch)

	def eval_sharpness(self):
		i = 0
		for train_batch, noise_batch in zip(self.train_dataset, self.noise_dataset):
			preds = self.generator(noise_batch, training = False)

			sharp = self.find_sharpness(preds)
			base_sharp = self.find_sharpness(train_batch)
			try:
				sharp_vec.append(sharp)
				base_sharp_vec.append(base_sharp)

			except:
				sharp_vec = [sharp]
				base_sharp_vec = [base_sharp]
			i += 1
			if i == 10:
				break

		###### Sharpness averaging measure
		sharpness = np.mean(np.array(sharp_vec))
		baseline_sharpness = np.mean(np.array(base_sharp_vec))

		return baseline_sharpness, sharpness

	def test(self):
		num_interps = 10
		if self.mode == 'test':
			num_figs = 5#int(400/(2*num_interps))
		else:
			num_figs = 9
		# there are 400 samples in the batch. to make 10x10 images, 
		for noise_batch in self.noise_dataset:
			for j in range(num_figs):
				path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
				noi_path = self.impath+'_TestingInterpolationV2_NOISE_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
				current_batch = noise_batch[2*num_interps*j:2*num_interps*(j+1)]
				# image_latents = self.Encoder(current_batch)
				for i in range(num_interps):
					start = np.reshape(current_batch[i:1+i].numpy(),(self.input_size*self.input_size*self.input_dims))
					end = np.reshape(current_batch[num_interps+i:num_interps+1+i].numpy(),(self.input_size*self.input_size*self.input_dims))

					# print(start.shape, end.shape)
					stack = np.vstack([start, end])

					linfit = interp1d([1,num_interps+1], stack, axis=0)
					interp_latents = linfit(list(range(1,num_interps+1)))
					# print(interp_latents.shape)
					interp_noise_images = np.reshape(interp_latents, (interp_latents.shape[0],self.input_size,self.input_size,self.input_dims))
					# print(interp_noise_images.shape)
					cur_interp_figs = self.generator(interp_noise_images)
					# print(cur_interp_figs.shape)

					sharpness = self.find_sharpness(cur_interp_figs)

					try:
						sharpness_vec.append(sharpness)
					except:
						shaprpness_vec = [sharpness]
					# cur_interp_figs_with_ref = np.concatenate((current_batch[i:1+i],cur_interp_figs.numpy(),current_batch[num_interps+i:num_interps+1+i]), axis = 0)
					# print(cur_interp_figs_with_ref.shape)
					try:
						batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs),axis = 0)
						batch_noise_figs = np.concatenate((batch_noise_figs,interp_noise_images), axis = 0)
					except:
						batch_interp_figs = cur_interp_figs
						batch_noise_figs = interp_noise_images

				images = (batch_interp_figs + 1.0)/2.0
				# print(images.shape)
				size_figure_grid = num_interps
				images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(images.shape[1],images.shape[2]),num_channels=images.shape[3])
				fig1 = plt.figure(figsize=(num_interps,num_interps))
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.axis("off")
				if images_on_grid.shape[2] == 3:
					ax1.imshow(np.clip(images_on_grid,0.,1.))
				else:
					ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

				label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
				plt.title(label)
				plt.tight_layout()
				plt.savefig(path)
				plt.close()

				noise_print_image = (batch_noise_figs + 1.0)/2.0
				# print(images.shape)
				size_figure_grid = num_interps
				images_on_grid = self.image_grid(input_tensor = noise_print_image, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(noise_print_image.shape[1],noise_print_image.shape[2]),num_channels=noise_print_image.shape[3])
				fig1 = plt.figure(figsize=(num_interps,num_interps))
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.axis("off")
				if images_on_grid.shape[2] == 3:
					ax1.imshow(np.clip(images_on_grid,0.,1.))
				else:
					ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

				label = 'INTERPOLATED NOISE IMAGES AT ITERATION '+str(self.total_count.numpy())
				plt.title(label)
				plt.tight_layout()
				plt.savefig(noi_path)
				plt.close()

				del batch_interp_figs
			break

		###### Interpol samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Interpolation Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("Interpolation Sharpness - "+str(overall_sharpness))




			# path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			# label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())

			# size_figure_grid = self.num_to_print
			# test_batch_size = size_figure_grid*size_figure_grid
			# noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)

			# images = self.generator(noise, training=False)
			# if self.data != 'celeba':
			# 	images = (images + 1.0)/2.0

			# self.save_image_batch(images = images,label = label, path = path)






################## RBF Garbage Dump #################


# C = K.expand_dims(self.centers)
# H = K.transpose(C-K.transpose(x))
# return K.exp(-self.betas * K.sum(H**2, axis=1))

# # C = self.centers[np.newaxis, :, :]
# # X = x[:, np.newaxis, :]

# # diffnorm = K.sum((C-X)**2, axis=-1)
# # ret = K.exp( - self.betas * diffnorm)
# # return ret

	# def PHS_MatrixSolver(self, centers, vals, phs_deg, poly_deg):
		
	# 	### itertool prod gives dupes. Need to Rewrite with multiindex

	# 	# import cardinality
	# 	N = centers.shape[0]
	# 	Atemp = np.zeros((N,N))
	# 	B_T = tf.ones((1,N)) # Topmost row of B_Transpose
	# 	for i in range(N):
	# 		Ci = centers[i]

	# 		for k in range(N):
	# 			Ck = centers[k]
	# 			Atemp[i,k] = tf.norm(Ci - Ck, ord = 'euclidean')

	# 		for j in range(1,poly_deg+1):
	# 			if j == 1:
	# 				# print(Ci)
	# 				Ci_pow_j_vec = np.transpose(Ci)
	# 			else:
	# 				temp = combinations_with_replacement(Ci, j)
	# 				# print(cardinality.count(temp))
	# 				# temp=product(Ci, repeat=j)
	# 				for elem in temp:
	# 					# print(elem)
	# 					elem_prod = np.product(elem)
	# 					elem_vec = np.empty((1,1))
	# 					elem_vec[:,0] = elem_prod
	# 					# print(elem_vec)
	# 					try:
	# 						Ci_pow_j_vec = np.concatenate((Ci_pow_j_vec,elem_vec), axis = 0) # A vec of jth order monomials, appended for all 
	# 					except:
	# 						Ci_pow_j_vec = elem_vec #A vec of j^th order monomial, init
	# 					# print(Ci_pow_j_vec, Ci_pow_j_vec.shape)
	# 			try:
	# 				B_T_jthPow = np.concatenate( (B_T_jthPow,Ci_pow_j_vec), axis = 0) #A taaaal vec of jth pow monomials for all j, appended
	# 				# print("Hi,B_T_jthPow",B_T_jthPow)
	# 				del Ci_pow_j_vec
	# 			except:
	# 				B_T_jthPow = Ci_pow_j_vec #A taaaal vec of jth pow monomials for all j, init 
	# 				# print("Hi,B_T_jthPow",B_T_jthPow)
	# 				del Ci_pow_j_vec
	# 		# print(B_T_jthPow, B_T_jthPow.shape)
	# 		try:
	# 			# B_T_all_i = np.concatenate((B_T_all_i,np.expand_dims(B_T_jthPow, axis = 1)),axis = 1)
	# 			B_T_all_i = np.concatenate((B_T_all_i,B_T_jthPow),axis = 1)
	# 			del B_T_jthPow
	# 		except:
	# 			B_T_all_i = B_T_jthPow#np.expand_dims(B_T_jthPow, axis = 1)
	# 			del B_T_jthPow
	# 		# print(B_T_all_i, B_T_all_i.shape)

	# 	B_T = np.concatenate((B_T,B_T_all_i), axis = 0)
	# 	B_T = B_T_all_i
	# 	B = tf.transpose(B_T)
	# 	L = B_T.shape[0]
	# 	Z = np.zeros((L,L))

	# 	ord_tensor = phs_deg*tf.ones_like(Atemp)
	# 	if self.rbf_n%2 == 1:
	# 		A = tf.pow(Atemp, ord_tensor) ## Nx1
	# 	else:
	# 		A = 1*tf.multiply(tf.pow(Atemp, ord_tensor),tf.math.log(Atemp+10.0**(-100)))##Nx1

	# 	M = np.concatenate( (np.concatenate((A,B), axis = 1),np.concatenate((B_T,Z), axis = 1)), axis = 0)

	# 	# print(M,M.shape)
	# 	# print(np.linalg.matrix_rank(M))
	# 	# print(vals, vals.shape)

	# 	y = np.concatenate((vals,np.zeros((L,1))), axis = 0)

	# 	sols = np.linalg.solve(M,y)

	# 	Weights = np.squeeze(sols[0:N])
	# 	PolyWts = np.expand_dims(sols[N:], axis = 1)
		
	# 	return Weights, PolyWts
