from __future__ import print_function
import os, sys, time, argparse
from datetime import date
import numpy as np
# from tensorflow.python import eager,pywrap_tfe
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
from absl import app
from absl import flags
import json
import glob
from tqdm.autonotebook import tqdm
import shutil
from cleanfid import fid
import ot

import tensorflow_probability as tfp
tfd = tfp.distributions


##FOR FID
from numpy import cov
from numpy import trace
from scipy.linalg import sqrtm
import scipy as sp
from numpy import iscomplexobj
from numpy.linalg import norm as norml2
import torch

import os.path
import tarfile
from six.moves import urllib
import scipy.misc




import multiprocessing

from ops import *

class GAN_Metrics():

	def __init__(self):

		self.PR_flag = 0
		self.class_prob_flag = 0
		self.loss_flag = 0
		self.IS_flag = 0
		self.KLD_flag = 0
		self.SinD_flag = 0
		self.SinID_flag = 0
		self.W22_flag = 0
		self.FID_flag = 0
		self.ReconFID_flag = 0
		self.KID_flag = 0
		self.SID_flag = 0
		self.DID_flag = 0
		self.DatasetCID_flag = 0
		self.PR_flag = 0
		self.RIP_flag = 0
		self.sharp_flag = 0
		self.lambda_flag = 0
		self.recon_flag = 0
		self.interpol_figs_flag = 0
		self.MardiaStats_flag = 0
		self.GradGrid_flag = 0
		self.class_prob_flag = 0
		self.LapD_flag = 0
		self.metric_counter_vec = []


		if 'lamb' in self.metrics and self.loss in ['FS'] and self.mode != 'metrics':
			self.lambda_flag = 1
			self.lambda_vec = []

		if 'interpol_figs' in self.metrics :
			self.interpol_figs_flag = 1

			if self.data in ['mnist', 'fmnist']:
				if self.mode in ['metrics']:
					self.FID_num_samples = 10000
				else:
					self.FID_num_samples = 5000 #was 5k
			elif self.data in ['cifar10', 'svhn']:
				if self.mode in ['metrics'] and self.testcase != 'single':
					self.FID_num_samples = 10000
				elif self.testcase != 'single':
					self.FID_num_samples = 5000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['tinyimgnet']:
				if self.mode in ['metrics']:
					self.FID_num_samples = 20000
				else:
					self.FID_num_samples = 20000
			elif self.data in ['celeba', 'church', 'bedroom']:
				self.FID_steps = 5000 #2500 for Rumi
				if self.mode in ['metrics']:
					self.FID_num_samples = 5000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['ukiyoe']:
				if self.mode in ['metrics']:
					self.FID_num_samples = 7000
				else:
					self.FID_num_samples = 5000
			else:
				self.FID_flag = 0
				print('Interpol FID cannot be evaluated on this dataset')

		if 'loss' in self.metrics:
			self.loss_flag = 1
			if self.data in ['g1','g2','gmm8', 'gN', 'mnist']:
				self.loss_steps = 10
			else:
				self.loss_steps = 100
			self.Dloss_vec = []
			self.Gloss_vec = []
			if self.gan in ['WAE'] or self.topic in ['MMDGAN']:
				self.AEloss_vec = []


		if 'KLD' in self.metrics:				
			self.KLD_flag = 1
			self.KLD_vec = []

			if self.data in ['g1', 'g2', 'gmm8', 'gN', 'gmmN']:
				self.KLD_steps = 10
				if self.data in ['gmm8', 'gmmN']:
					self.KLD_func = self.KLD_sample_estimate
				else:
					self.KLD_func = self.KLD_Gaussian
			else:
				self.KLD_flag = 1
				self.KLD_steps = 50
				if self.loss in ['FS','RBF'] and self.gan != 'WAE':
					if self.distribution == 'gaussian' or self.data in ['g1','g2']:
						self.KLD_func = self.KLD_Gaussian
					else:
						self.KLD_func = self.KLD_sample_estimate
				if self.gan == 'WAE':
					if 'gaussian' in self.noise_kind:
						self.KLD_func = self.KLD_Gaussian
					else:
						self.KLD_func = self.KLD_sample_estimate
				print('KLD is not an accurate metric on this datatype')

		if 'W22' in self.metrics:				
			self.W22_flag = 1
			self.W22_vec = []

			if self.data in ['g1', 'g2']:
				self.W22_steps = 10
				# self.W22_func = self.W22_Gaussian
			elif self.data in ['gN']:
				self.W22_steps = 50
			else:
				self.W22_flag = 1
				self.W22_steps = 50
				# if self.loss in ['FS','RBF'] and self.gan != 'WAE':
				# 	if self.distribution == 'gaussian' or self.data in ['g1','g2']:
				# 		self.W22_func = self.W22_Gaussian
				# 	else:
				# 		self.W22_func = self.W22_sample_estimate
				# if self.gan == 'WAE':
				# 	if 'gaussian' in self.noise_kind:
				# 		self.W22_func = self.W22_Gaussian
				# 	else:
				# 		self.W22_func = self.W22_sample_estimate
				print('W22 is not an accurate metric on this datatype')

		if self.models_for_metrics == 1:

			if self.data in ['mnist','cifar10', 'svhn']:
				self.model_steps = 500 #was 500, make 2500 to run on Colab 
			elif self.data in ['tinyimgnet']:
				self.model_steps = 1000
			elif self.data in ['celeba', 'church', 'bedroom', 'ukiyoe']:
				if self.gan not in ['WGAN_AE']:
					self.model_steps = 2500
				else:
					self.model_steps = 500
			else:
				self.model_steps = 5000

		if 'ReconFID' in self.metrics:
			self.ReconFID_flag = 1
			self.ReconFID_load_flag = 0
			self.ReconFID_vec = []
			self.ReconFID_vec_new = []

			if self.data in ['mnist', 'fmnist']:
				self.ReconFID_steps = 500
				if self.mode in ['metrics']:
					self.ReconFID_num_samples = 10000
				else:
					self.ReconFID_num_samples = 5000 #was 5k
			elif self.data in ['cifar10', 'svhn']:
				self.ReconFID_steps = 500
				if self.mode in ['metrics'] and self.testcase != 'single':
					self.ReconFID_num_samples = 10000
				elif self.testcase != 'single':
					self.ReconFID_num_samples = 5000
				else:
					self.ReconFID_num_samples = 5000
			elif self.data in ['tinyimgnet']:
				self.ReconFID_steps = 1000 
				if self.mode in ['metrics']:
					self.ReconFID_num_samples = 20000
				else:
					self.ReconFID_num_samples = 20000
			elif self.data in ['celeba', 'church', 'bedroom']:
				self.ReconFID_steps = 5000 #2500 for Rumi
				if self.mode in ['metrics']:
					self.ReconFID_num_samples = 5000
				else:
					self.ReconFID_num_samples = 5000
			elif self.data in ['ukiyoe']:
				self.ReconFID_steps = 5000 #2500 for Rumi
				if self.mode in ['metrics']:
					self.ReconFID_num_samples = 7000
				else:
					self.ReconFID_num_samples = 5000
			elif self.data in ['gN']:
				self.ReconFID_steps = 200#50
			else:
				self.ReconFID_flag = 0
				print('FID cannot be evaluated on this dataset')



		if 'FID' in self.metrics:
			self.FID_flag = 1
			self.FID_load_flag = 0
			self.FID_vec = []
			self.FID_vec_new = []

			if self.data in ['mnist', 'fmnist']:
				self.FID_steps = 250 #250 for MNIST SpiderGAN, was 500, make 2500 to run on Colab 
				if self.gan == 'WAE':
					self.FID_steps = 500
				if self.mode in ['metrics']:
					self.FID_num_samples = 10000
				else:
					self.FID_num_samples = 5000 #was 5k
			elif self.data in ['cifar10', 'svhn']:
				self.FID_steps = 500
				if self.mode in ['metrics'] and self.testcase != 'single':
					self.FID_num_samples = 10000
				elif self.testcase != 'single':
					self.FID_num_samples = 5000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['tinyimgnet']:
				self.FID_steps = 1000 
				if self.mode in ['metrics']:
					self.FID_num_samples = 20000
				else:
					self.FID_num_samples = 20000
			elif self.data in ['celeba', 'church', 'bedroom']:
				self.FID_steps = 5000 #2500 for Rumi
				if self.gan == 'WGAN_AE':
					self.FID_steps = 500
				if self.mode in ['metrics']:
					self.FID_num_samples = 5000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['ukiyoe']:
				self.FID_steps = 5000 #2500 for Rumi
				if self.mode in ['metrics']:
					self.FID_num_samples = 7000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['gN']:
				self.FID_steps = 200#50
			else:
				self.FID_flag = 0
				print('FID cannot be evaluated on this dataset')

		if 'IS' in self.metrics:
			self.IS_flag = 1
			self.IS_vec = []
			self.IS_num_samples = 1000
			if self.data not in ['g1', 'g2', 'gN', 'gmm8', 'gmm2', 'gmmN']:
				self.IS_steps = 100
			else:
				self.IS_flag = 0
				print('IS cannot be evaluated on this dataset')


		if 'SID' in self.metrics:
			self.SID_flag = 1
			self.SID_load_flag = 0 ### FID functions load data
			self.SID_vec = []
			self.SID_vec_new = []
			self.SID_batch_size = 200
			self.SID_num_samples = 5000
			if self.data in ['mnist', 'fmnist']:
				self.SID_steps = 250
			elif self.data in ['cifar10', 'svhn']:
				self.SID_steps = 500
			else:
				self.SID_steps = 5000

			# if self.data in ['mnist']:
			# 	self.SID_steps = 500 #was 500, make 2500 to run on Colab 
			# 	if self.gan == 'WAE':
			# 		self.SID_steps = 500
			# 	if self.mode in ['metrics']:
			# 		self.SID_num_samples = 50000 ### FID functions load data
			# 	else:
			# 		self.SID_num_samples = 1000 #was 5k
			# elif self.data in ['cifar10', 'svhn']:
			# 	self.SID_steps = 500
			# 	if self.mode in ['metrics'] and self.testcase != 'single':
			# 		self.SID_num_samples = 50000
			# 	elif self.testcase != 'single':
			# 		self.SID_num_samples = 1000
			# 	else:
			# 		self.SID_num_samples = 5000
			# elif self.data in ['celeba', 'church', 'bedroom']:
			# 	self.SID_steps = 5000 #2500 for Rumi
			# 	if self.mode in ['metrics']:
			# 		self.SID_num_samples = 5000
			# 	else:
			# 		self.SID_num_samples = 5000
			# elif self.data in ['ukiyoe']:
			# 	self.SID_steps = 5000 #2500 for Rumi
			# 	if self.mode in ['metrics']:
			# 		self.SID_num_samples = 7000
			# 	else:
			# 		self.SID_num_samples = 5000
			# elif self.data in ['g1','g2','gmm2','gmm8','gN']:
			# 	self.SID_steps = 100
			# else:
			# 	self.SID_flag = 0
			# 	print('SID cannot be evaluated on this dataset')


		if 'KID' in self.metrics:
			self.KID_flag = 1
			self.FID_load_flag = 0 ### FID functions load data
			self.KID_vec = []
			self.KID_vec_new = []

			if self.data in ['mnist', 'fmnist']:
				self.KID_steps = 250 #250 for Spider, was 500, make 2500 to run on Colab 
				if self.gan == 'WAE':
					self.KID_steps = 500
				if self.mode in ['metrics']:
					self.FID_num_samples = 10000 ### FID functions load data
				else:
					self.FID_num_samples = 5000 #was 5k
			elif self.data in ['cifar10', 'svhn']:
				self.KID_steps = 500
				if self.mode in ['metrics'] and self.testcase != 'single':
					self.FID_num_samples = 10000
				elif self.testcase != 'single':
					self.FID_num_samples = 5000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['tinyimgnet']:
				self.KID_steps = 1000 
				if self.mode in ['metrics']:
					self.FID_num_samples = 20000
				else:
					self.FID_num_samples = 20000
			elif self.data in ['celeba', 'church', 'bedroom']:
				self.KID_steps = 5000 #2500 for Rumi
				if self.mode in ['metrics']:
					self.FID_num_samples = 5000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['ukiyoe']:
				self.KID_steps = 5000 #2500 for Rumi
				if self.mode in ['metrics']:
					self.FID_num_samples = 7000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['gN']:
				self.KID_steps = 200#50
			else:
				self.KID_flag = 0
				print('KID cannot be evaluated on this dataset')



		if 'recon' in self.metrics:
			self.recon_flag = 1
			self.recon_vec = []
			self.FID_vec_new = []

			if self.data in ['mnist', 'svhn']:
				self.recon_steps = 500
			elif self.data in ['cifar10']:
				self.recon_steps = 500
			elif self.data in ['celeba', 'church', 'bedroom']:
				self.recon_steps = 5000 
			elif self.data in ['ukiyoe']:
				self.recon_steps = 1500# 1500 of 
			elif self.data in ['gN']:
				self.recon_steps = 100
			else:
				self.recon_flag = 0
				print('Reconstruction cannot be evaluated on this dataset')

		if 'GradGrid' in self.metrics:
			if self.data in ['g2', 'gmm8']:
				self.GradGrid_flag = 1
				self.GradGrid_steps = 10 ## 5 for Video ### PolyGAN Paper has 50
			elif self.gan in ['Langevin'] and self.data == 'mnist' and self.latent_dims == 2:
				self.GradGrid_flag = 1
				self.GradGrid_steps = 25 ## 5 for Video ### PolyGAN Paper has 50
				self.MIN = -6.0
				self.MAX = 6.0
			else:
				print("Cannot plot Gradient grid. Not a 2D dataset")


		if 'PR' in self.metrics:
			### NEed to DeisGN
			self.PR_flag = 1
			self.PR_vec = []
			self.PR_steps = self.FID_steps


		if 'sharpness' in self.metrics:
			self.sharp_flag = 1
			self.sharp_vec = []
			self.sharp_steps = 1000

		if 'LapD' in self.metrics:
			self.LapD_flag = 1
			self.LapD_vec = []
			self.LapD_steps = 1


		if 'MardiaStats' in self.metrics:
			self.MardiaStats_flag = 1
			self.skew_vec = []
			self.kurt_vec = []
			self.MardiaStats_steps = 1000

		if 'DatasetFID' in self.metrics:
			self.DID_flag = 1
			self.DFID_vec = []
			self.DKID_vec = []
			self.DSID_vec = []
			self.DID_load_flag = 0
			self.DID_steps = 1000
			self.DID_batch_size = 100

			if self.data == 'ukiyoe' or self.noise_data == 'ukiyoe':
				self.DID_num_samples = 5000
			else:
				self.DID_num_samples = 10000

		if 'DatasetSinD' in self.metrics:
			self.SinD_flag = 1
			self.SinD_vec = []
			self.DID_load_flag = 0 ### Could be SinD... same for simplicity in the gan_data file
			self.SinD_steps = 1000

		if 'DatasetSinID' in self.metrics:
			self.SinID_flag = 1
			self.SinID_vec = []
			self.DID_load_flag = 0 ### Could be SinD... same for simplicity in the gan_data file
			self.SinID_steps = 1000
			if self.data == 'ukiyoe' or self.noise_data == 'ukiyoe':
				self.DID_num_samples = 5000
			else:
				self.DID_num_samples = 10000


		if 'DatasetCID' in self.metrics:
			self.DatasetCID_flag = 1
			self.CID_vec = []
			self.DatasetCID_load_flag = 0
			self.DatasetCID_steps = 1000
			self.DatasetCID_batch_size = 100
			self.DatasetCID_num_samples = 500


		if 'RIP' in self.metrics:
			self.RIP_flag = 1
			self.RIP_vec = []
			self.RIP_sigmaMax_vec = []
			self.RIP_sigmaMin_vec = []
			self.RIP_steps = 25

		if 'ClassProbs' in self.metrics:
			self.class_prob_vec = []

			if self.data in ['mnist', 'pfmnist', 'fmnist']:
				self.class_prob_flag = 1
				self.class_prob_steps = 2500 
				self.classifier_load_flag = 1
			else:
				print("Cannot find class-wise probabilites for this dataset")

	def eval_metrics(self):
		update_flag = 0

		if self.interpol_figs_flag:
			self.save_interpol_figs()



		if self.FID_flag and (self.total_count.numpy()%self.FID_steps == 0 or self.mode in ['metrics', 'model_metrics'] or self.total_count.numpy() < 2):
			update_flag = 1
			self.update_FID()
			if self.mode != 'metrics':
				np.save(self.metricpath+'FID.npy',np.array(self.FID_vec))
				if self.topic == 'RumiGAN' and self.data == 'mnist':
					self.print_FID_Rumi()
				elif self.topic in['cGAN', 'ACGAN'] and self.data == 'mnist':
					self.print_FID_ACGAN()
				else:
					self.print_FID()

		if self.ReconFID_flag and (self.total_count.numpy()%self.ReconFID_steps == 0 or self.mode in ['metrics', 'model_metrics'] or self.total_count.numpy() < 2):
			update_flag = 1
			self.update_ReconFID()
			if self.mode != 'metrics':
				np.save(self.metricpath+'FID.npy',np.array(self.ReconFID_vec))
				self.print_ReconFID()


		if self.KID_flag and (self.total_count.numpy()%self.KID_steps == 0 or self.mode in ['metrics', 'model_metrics'] or self.total_count.numpy() < 2):
			update_flag = 1
			self.update_KID()
			if self.mode != 'metrics':
				np.save(self.metricpath+'KID.npy',np.array(self.KID_vec))
				self.print_KID()

		if self.IS_flag and (self.total_count.numpy()%self.IS_steps == 0 or self.mode in ['metrics', 'model_metrics'] or self.total_count.numpy() < 2):
			update_flag = 1
			self.update_IS()
			if self.mode != 'metrics':
				np.save(self.metricpath+'IS.npy',np.array(self.IS_vec))
				# self.print_KID()

		if self.SID_flag and (self.total_count.numpy()%self.SID_steps == 0 or self.mode in ['metrics', 'model_metrics'] or self.total_count.numpy() < 2):
			update_flag = 1
			self.update_SID()
			if self.mode != 'metrics':
				np.save(self.metricpath+'SID_all.npy',np.array(self.SID_vec))
				self.print_SID()
			else:
				np.save(self.metricpath+'SID_MetricsEval.npy',np.array(self.SID_vec))
				self.print_SID()

		if self.DID_flag and (self.total_count.numpy()%self.DID_steps == 0 or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_DID()
			if self.mode != 'metrics':
				np.save(self.metricpath+'DFID.npy',np.array(self.DFID_vec))
				np.save(self.metricpath+'DKID.npy',np.array(self.DKID_vec))

		if self.SinD_flag and (self.total_count.numpy()%self.SinD_steps == 0 or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_DatasetSinD()
			if self.mode != 'metrics':
				np.save(self.metricpath+'SinD.npy',np.array(self.SinD_vec))

		if self.SinID_flag and (self.total_count.numpy()%self.SinID_steps == 0 or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_DatasetSinID()
			if self.mode != 'metrics':
				np.save(self.metricpath+'SinID.npy',np.array(self.SinID_vec))

		if self.models_for_metrics and (self.mode not in ['metrics', 'model_metrics']) and (self.total_count.numpy()%self.model_steps == 0):
			update_flag = 1
			self.h5_for_metrics()

		if self.loss_flag and (self.total_count.numpy()%self.loss_steps == 0 or self.total_count.numpy() <= 10 or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_loss()
			self.print_loss()
			if self.mode != 'metrics':
				np.save(self.metricpath+'Gloss.npy',np.array(self.Gloss_vec))
				np.save(self.metricpath+'Dloss.npy',np.array(self.Dloss_vec))
				if self.gan in ['WAE'] or self.topic in ['MMDGAN']:
					np.save(self.metricpath+'AEloss.npy',np.array(self.AEloss_vec))

		if self.PR_flag and (self.total_count.numpy()%self.PR_steps == 0 or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_PR()
			if self.mode != 'metrics':
				np.save(self.metricpath+'PR_all.npy',np.array(self.PR_vec))
				self.print_PR()
			else:
				np.save(self.metricpath+'PR_MetricsEval.npy',np.array(self.PR_vec))
				self.print_PR()

		if self.DatasetCID_flag and (self.total_count.numpy()%self.DatasetCID_steps == 0 or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_DatasetCID()
			if self.mode != 'metrics':
				np.save(self.metricpath+'DatasetCID_all.npy',np.array(self.CID_vec))
				# self.print_CID()
			else:
				np.save(self.metricpath+'DatasetCID_MetricsEval.npy',np.array(self.CID_vec))
				# self.print_CID()

		if self.sharp_flag and (self.total_count.numpy()%self.sharp_steps == 0 or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_sharpness()
			if self.mode != 'metrics':
				np.save(self.metricpath+'Sharpness_all.npy',np.array(self.sharp_vec))
			else:
				np.save(self.metricpath+'Sarpness_MetricsEval.npy',np.array(self.sharp_vec))

		if self.LapD_flag and (self.total_count.numpy()%self.LapD_steps == 0 or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_LapD()
			if self.mode != 'metrics':
				np.save(self.metricpath+'LapD_all.npy',np.array(self.LapD_vec))
			else:
				np.save(self.metricpath+'LapD_MetricsEval.npy',np.array(self.LapD_vec))

		if self.MardiaStats_flag and (self.total_count.numpy()%self.MardiaStats_steps == 0 or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_MardiaStats()
			if self.mode != 'metrics':
				np.save(self.metricpath+'MadriaSkew.npy',np.array(self.skew_vec))
				np.save(self.metricpath+'MadriaKurt.npy',np.array(self.kurt_vec))
			else:
				np.save(self.metricpath+'MadriaSkew_MetricsEval.npy',np.array(self.skew_vec))
				np.save(self.metricpath+'MadriaKurt_MetricsEval.npy',np.array(self.kurt_vec))

		if self.RIP_flag and (self.total_count.numpy()%self.RIP_steps == 0 or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_RIP()
			if self.mode != 'metrics':
				np.save(self.metricpath+'RIP_all.npy',np.array(self.RIP_vec))
				np.save(self.metricpath+'RIP_sigmaMax_all.npy',np.array(self.RIP_sigmaMax_vec))
				np.save(self.metricpath+'RIP_sigmaMin_all.npy',np.array(self.RIP_sigmaMin_vec))
				self.print_RIP()
			else:
				np.save(self.metricpath+'RIP_MetricsEval.npy',np.array(self.RIP_vec))
				np.save(self.metricpath+'RIP_sigmaMax_all.npy',np.array(self.RIP_sigmaMax_vec))
				np.save(self.metricpath+'RIP_sigmaMin_all.npy',np.array(self.RIP_sigmaMin_vec))
				self.print_RIP()

		if self.class_prob_flag and (self.total_count.numpy()%self.class_prob_steps == 0 or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			# self.class_prob_metric()
			if self.mode != 'metrics':
				np.save(self.metricpath+'ClassProbs.npy',np.array(self.class_prob_vec))
				self.print_ClassProbs()


		if self.KLD_flag and ((self.total_count.numpy()%self.KLD_steps == 0 or self.total_count.numpy() == 1)  or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_KLD()
			if self.mode != 'metrics':
				np.save(self.metricpath+'KLD.npy',np.array(self.KLD_vec))
				self.print_KLD()

		if self.W22_flag and ((self.total_count.numpy()%self.W22_steps == 0 or self.total_count.numpy() <= 10 or self.total_count.numpy() == 1)  or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_W22()
			if self.mode != 'metrics':
				np.save(self.metricpath+'W22.npy',np.array(self.W22_vec))
				self.print_W22()

		if self.recon_flag and ((self.total_count.numpy()%self.recon_steps == 0 or self.total_count.numpy() == 1)  or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.eval_recon()
			if self.mode != 'metrics':
				np.save(self.metricpath+'recon.npy',np.array(self.recon_vec))
				self.print_recon()

		if self.lambda_flag and (self.loss in ['RBF','FS'] or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_Lambda()
			if self.mode != 'metrics':
				np.save(self.metricpath+'Lambda.npy',np.array(self.lambda_vec))
				self.print_Lambda()

		if self.GradGrid_flag and ((self.total_count.numpy()%self.GradGrid_steps == 0 or self.total_count.numpy() == 1) or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.print_GradGrid()


		if self.res_flag and update_flag:
			self.res_file.write("Metrics avaluated at Iteration " + str(self.total_count.numpy()) + '\n')

	def model_metrics(self):


		def load_run_WAE(i,models_Enc,models_Dec):
			cur_Enc = models_Enc[i][1]
			# print(cur_Enc)
			self.Encoder = tf.keras.models.load_model(cur_Enc)
			cur_Dec = models_Dec[i][1]
			self.Decoder = tf.keras.models.load_model(cur_Dec)
			self.eval_metrics()

		def load_run_GAN(i,models_Gen):
			cur_Gen = models_Gen[i][1]
			self.generator = tf.keras.models.load_model(cur_Gen)
			self.eval_metrics()

		def sorted_list(names):
			l = []
			for n in names:
				num = int(n.split('Iter')[1].split('_')[0])
				l.append([num,n])
			return sorted(l)

		if self.gan == 'WAE':
			models_Enc = sorted_list(sorted(glob.glob(self.modelspath+'*_Encoder.h5')))
			models_Dec = sorted_list(sorted(glob.glob(self.modelspath+'*_Decoder.h5')))
			iter_vals = np.array(models_Enc)[:,0]
		elif self.gan not in ['MMDGAN']:
			models_Gen = sorted_list(sorted(glob.glob(self.modelspath+'*_generator.h5')))
			iter_vals = np.array(models_Gen)[:,0]

		if self.loss == 'FS':
			### WGAN-FS's discriminator is 2 part: Disc-A and Disc-B
			models_DiscA = sorted_list(sorted(glob.glob(self.modelspath+'*_discriminator_A.h5')))
			models_DiscB = sorted_list(sorted(glob.glob(self.modelspath+'*_discriminator_B.h5')))
		elif self.loss == 'RBF':
			models_DiscRBF = sorted_list(sorted(glob.glob(self.modelspath+'*_discriminator_RBF.h5')))
		elif self.gan not in ['MMDGAN']:
			models_Disc = sorted_list(sorted(glob.glob(self.modelspath+'*_discriminator.h5')))


		if self.gan == 'MMDGAN':
			models_Gen = sorted_list(sorted(glob.glob(self.modelspath+'*_generator.h5')))
			models_Enc = sorted_list(sorted(glob.glob(self.modelspath+'*_Encoder.h5')))
			models_Dec = sorted_list(sorted(glob.glob(self.modelspath+'*_Decoder.h5')))
			if self.loss == 'RBF':
				models_DiscRBF = sorted_list(sorted(glob.glob(self.modelspath+'*_discriminator_RBF.h5')))
			iter_vals = np.array(models_Gen)[:,0]


		for i, cur in enumerate(iter_vals):
			# context_handle = tfpy.eager.context.context()._context_handle
			# if context_handle is not None:
			# 	tfpy.pywrap_tfe.TFE_ContextClearCaches(context_handle)

			if int(cur) > self.stop_metric_iters:
				break

			if int(cur) < self.start_metric_iters:
				continue

			self.total_count.assign(cur)
			print(i, cur)

			# p = multiprocessing.Process(target=run_inference_or_training,args=(i,models_Enc,models_Dec))
			
			if self.gan == 'WAE':
				cur_Enc = models_Enc[i][1]
				# print(cur_Enc)
				# self.Encoder = tf.keras.models.load_model(cur_Enc)
				self.Encoder.load_weights(cur_Enc)
				cur_Dec = models_Dec[i][1]
				# self.Decoder = tf.keras.models.load_model(cur_Dec)
				self.Decoder.load_weights(cur_Dec)
				# p=multiprocessing.Process(target=load_run_WAE,args=(i,models_Enc,models_Dec))
				# print(models_Enc)
			else:
				cur_Gen = models_Gen[i][1]
				# self.generator = tf.keras.models.load_model(cur_Gen)
				self.generator.load_weights(cur_Gen)
				# p=multiprocessing.Process(target=load_run_GAN,args=(i,models_Gen))

				

			# if self.loss == 'FS':
			# 	### WGAN-FS's discriminator is 2 part: Disc-A and Disc-B
			# 	cur_DiscA = models_DiscA[i][1]
			# 	self.discriminator_A = tf.keras.models.load_model(cur_DiscA)
			# 	cur_DiscB = models_DiscB[i][1]
			# 	self.discriminator_B = tf.keras.models.load_model(cur_DiscB)
			# elif self.loss == 'RBF':
			# 	cur_DiscRBF = models_DiscRBF[i][1]
			# 	self.discriminator_RBF = tf.keras.models.load_model(cur_DiscRBF)
			# else:
			# 	cur_Disc = models_Disc[i][1]
			# 	self.discriminator = tf.keras.models.load_model(cur_Disc)

			# p.start()
			# p.join()
			if 'NeurIPS22_bias' in self.metrics:
				self.generate_NeurIPS22_bias()
				# return
			self.eval_metrics()

		

	def update_RIP(self):
		## Get weight matrices in order:
		W_list = [] 
		for m in self.generator.get_weights():
			if m.ndim == 2:
				# print(m.shape)
				W_list.append(m)
		W_prod = W_list[0]
		# print(W_prod.shape)
		for i in range(1,len(W_list)):
			W_prod = np.matmul(W_prod, W_list[i])
			# print(W_prod.shape)

		Sigma = tf.linalg.svd(W_prod, compute_uv=False)
		num_nonzero = len(np.nonzero(Sigma)[0])
		S = Sigma[Sigma != 0]
		# print(num_nonzero, S)
		self.RIP_sigmaMax_vec.append([max(S),self.total_count.numpy()])
		self.RIP_sigmaMin_vec.append([min(S),self.total_count.numpy()])
		K_number = np.log(max(S)/min(S))
		self.RIP_vec.append([K_number,self.total_count.numpy()])
		return


	def print_RIP(self):
		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
		if self.colab==1 or self.latex_plot_flag==0:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages

		vals = list(np.array(self.RIP_vec)[:,0])
		locs = list(np.array(self.RIP_vec)[:,1])

		vals_max = list(np.array(self.RIP_sigmaMax_vec)[:,0])
		locs_max = list(np.array(self.RIP_sigmaMax_vec)[:,1])

		vals_min = list(np.array(self.RIP_sigmaMin_vec)[:,0])
		locs_min = list(np.array(self.RIP_sigmaMin_vec)[:,1])

		with PdfPages(path+'RIP_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs_max,vals_max, c='r',label = 'Max Sigma vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs_min,vals_min, c='r',label = 'Min Sigma vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)


			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'K_number vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

	def update_IS(self):
		#### Need to fix for Non WAE
		preds = self.Decoder(self.get_noise(tf.constant(1000)), training=False)
		preds = tf.image.resize(preds, [299,299])
		if preds.shape[3] == 1:
			preds = tf.image.grayscale_to_rgb(preds)
		preds = preds.numpy()*255
		self.IS = self.get_inception_score(preds)
		self.IS_vec.append([IS,self.total_count.numpy()])

		if self.mode in ['metrics', 'model_metrics']:
			print("Final IS score - "+str(self.IS))
			self.res_file.write("Final IS score - "+str(self.IS))
			# if self.res_flag:
			# 	self.res_file.write("Final FID score - "+str(self.IS))

		if self.res_flag:
			self.res_file.write("IS score - "+str(self.IS))

		return

	# Call this function with list of images. Each of elements should be a 
	# numpy array with values ranging from 0 to 255.
	def get_inception_score(self,images, splits=10):
		MODEL_DIR = '/tmp/imagenet'
		DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
		## The init func's code

		if not os.path.exists(MODEL_DIR):
			os.makedirs(MODEL_DIR)
		filename = DATA_URL.split('/')[-1]
		filepath = os.path.join(MODEL_DIR, filename)
		if not os.path.exists(filepath):
			def _progress(count, block_size, total_size):
				sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
				sys.stdout.flush()
			filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
			print()
			statinfo = os.stat(filepath)
			print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
		tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
		with tf.compat.v1.gfile.FastGFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
			graph_def = tf.compat.v1.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.compat.v1.import_graph_def(graph_def, name='')
		# Works with an arbitrary minibatch size.
		with tf.compat.v1.Session() as sess:
			pool3 = sess.graph.get_tensor_by_name('pool_3:0')
			ops = pool3.graph.get_operations()
			for op_idx, op in enumerate(ops):
				for o in op.outputs:
					shape = o.get_shape()
					shape = [s.value for s in shape]
					new_shape = []
					for j, s in enumerate(shape):
						if s == 1 and j == 0:
							new_shape.append(None)
						else:
							new_shape.append(s)
					o.set_shape(tf.TensorShape(new_shape))
			w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
			logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
			softmax = tf.nn.softmax(logits)


		assert(type(images) == list)
		assert(type(images[0]) == np.ndarray)
		assert(len(images[0].shape) == 3)
		assert(np.max(images[0]) > 10)
		assert(np.min(images[0]) >= 0.0)
		inps = []
		for img in images:
			img = img.astype(np.float32)
			inps.append(np.expand_dims(img, 0))
		bs = 1
		with tf.compat.v1.Session() as sess:
			preds = []
			n_batches = int(math.ceil(float(len(inps)) / float(bs)))
			for i in range(n_batches):
				sys.stdout.write(".")
				sys.stdout.flush()
				inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
				inp = np.concatenate(inp, 0)
				pred = sess.run(softmax, {'ExpandDims:0': inp})
				preds.append(pred)
			preds = np.concatenate(preds, 0)
			scores = []
			for i in range(splits):
				part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
				kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
				kl = np.mean(np.sum(kl, 1))
				scores.append(np.exp(kl))
			return np.mean(scores), np.std(scores)




	def update_PR(self):
		min_size = min(self.act1.shape[0], self.act2.shape[0])
		self.PR = compute_prd_from_embedding(self.act2[0:min_size], self.act1[0:min_size])
		# self.PR = compute_prd_from_embedding(self.act1, self.act2) #Wong
		np.save(self.metricpath+'latest_PR.npy',np.array(self.PR))
		# if self.mode != 'metrics':
		self.PR_vec.append([self.PR,self.total_count.numpy()])

	def print_PR(self):
		path = self.metricpath
		
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages


		with PdfPages(path+'PR_plot.pdf') as pdf:
			for PR in self.PR_vec:
				fig1 = plt.figure(figsize=(3.5, 3.5), dpi=400)
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.set_xlim([0, 1])
				ax1.set_ylim([0, 1])
				ax1.get_xaxis().set_visible(True)
				ax1.get_yaxis().set_visible(True)
				precision, recall = PR[0]
				ax1.plot(recall, precision, color = 'g', linestyle = 'solid', alpha=0.5, linewidth=3)
				ax1.set_xlabel('RECALL')
				ax1.set_ylabel('PRECISION')
				title = 'PR at Iteration '+str(PR[1])
				plt.title(title, fontsize=8)
				pdf.savefig(fig1, bbox_inches='tight', dpi=400)
				plt.close(fig1)


	def update_DatasetCID(self):

		def ukiyoe_reader(filename):
			# with tf.device('/CPU'):
			with self.strategy.scope():
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([1024,1024,3])
				image = tf.image.resize(image,[75,75])
				# This will convert to float values in [0, 1]
				image = tf.subtract(image,127.5)
				image = tf.divide(image,127.5)
			return image

		def tinyimgnet_reader(filename):
			# with tf.device('/CPU'):
			with self.strategy.scope():
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([64,64,3])
				# image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[75,75])

				# This will convert to float values in [-1, 1]
				image = tf.subtract(image,127.0)
				image = tf.divide(image,127.0)
			return image

		def celeba_reader(filename):
			# with tf.device('/CPU'):
			with self.strategy.scope():
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[75,75])

				# This will convert to float values in [-1, 1]
				image = tf.subtract(image,127.5)
				image = tf.divide(image,127.5)
			return image

		def data_preprocess(image):
			# with tf.device('/CPU'):
			with self.strategy.scope():
				image = tf.image.resize(image,[75,75])
				if image.shape[2] == 1:
					image = tf.image.grayscale_to_rgb(image)
			return image


		with self.strategy.scope():
			if self.DatasetCID_load_flag == 0:
				### First time FID call setup
				self.DatasetCID_load_flag = 1

				### Noise is SRC, Data is TAR

				random_points_src = tf.keras.backend.random_uniform([self.DatasetCID_num_samples], minval=0, maxval=int(self.dfid_noise_images.shape[0]), dtype='int32', seed=None)
				self.src_images = self.dfid_noise_images[random_points_src]

				random_points_tar = tf.keras.backend.random_uniform([self.DatasetCID_num_samples], minval=0, maxval=int(self.dfid_data_images.shape[0]), dtype='int32', seed=None)
				self.tar_images = self.dfid_data_images[random_points_tar]


				self.src_images_dataset = tf.data.Dataset.from_tensor_slices((self.src_images))
				if self.noise_data == 'celeba': 
					self.src_images_dataset = self.src_images_dataset.map(celeba_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.noise_data == 'ukiyoe': 
					self.src_images_dataset = self.src_images_dataset.map(ukiyoe_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.noise_data == 'tinyimgnet': 
					self.src_images_dataset = self.src_images_dataset.map(tinyimgnet_reader,num_parallel_calls=int(self.num_parallel_calls))
				else:
					self.src_images_dataset = self.src_images_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))

				self.src_images_dataset = self.src_images_dataset.batch(self.DatasetCID_batch_size)

				self.tar_images_dataset = tf.data.Dataset.from_tensor_slices((self.tar_images))
				if self.data == 'celeba': 
					self.tar_images_dataset = self.tar_images_dataset.map(celeba_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.data == 'ukiyoe': 
					self.tar_images_dataset = self.tar_images_dataset.map(ukiyoe_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.data == 'tinyimgnet': 
					self.tar_images_dataset = self.tar_images_dataset.map(tinyimgnet_reader,num_parallel_calls=int(self.num_parallel_calls))
				else:
					self.tar_images_dataset = self.tar_images_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))

				self.tar_images_dataset = self.tar_images_dataset.batch(self.DatasetCID_batch_size)


				self.CID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(75,75,3), classes=1000)

		with self.strategy.scope():
			for src_batch,tar_batch in zip(self.src_images_dataset, self.tar_images_dataset):
				act1 = self.CID_model.predict(src_batch)
				act2 = self.CID_model.predict(tar_batch)
				try:
					self.act1 = np.concatenate((self.act1,act1), axis = 0)
					self.act2 = np.concatenate((self.act2,act2), axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
			self.eval_DatasetCID()
		return

	def eval_DatasetCID(self):
		self.act1_dataset = tf.data.Dataset.from_tensor_slices((self.act1))
		self.act1_dataset = self.act1_dataset.batch(self.DatasetCID_batch_size)

		self.act2_dataset = tf.data.Dataset.from_tensor_slices((self.act2))
		self.act2_dataset = self.act2_dataset.batch(self.DatasetCID_batch_size)


		def D(X,C):
			#####
			## X :          Data of dimension N x n - batch size N and dimensionality n
			## c_d, c_g :   Centers from data/generator of dimension M x n - M is batch size
			## D(x) :       Computes the "Columb Potential" of each x, given all c_d, c_g of dimension N x 1
			##
			#####
			### Data vector
			N = X.shape[0] #### Batch size of data
			M = C.shape[0] #### Batch size of each set of centers

			W = (1/N)*np.ones([C.shape[0]])
			W = tf.expand_dims(W, axis = 1)
			# print('W', W)

			X = tf.expand_dims(X, axis = 2) ## Nxnx1
			X = tf.expand_dims(X, axis = 1) ## Nx1xnx1
			# print('Input X',X, X.shape)

			C = tf.expand_dims(C, axis = 2) ## Mxnx1
			C = tf.expand_dims(C, axis = 0) ## 1xMxnx1
			# print('Centers C', C, C.shape)

			C_tiled = tf.tile(C, [N,1,1,1])  ## NxMxnx1 ### was tf.shape(X)[0]
			X_tiled = tf.tile(X, [1,M,1,1])  ## NxMxnx1 ### was self.num_hidden
			# print('C_tiled', C_tiled, C_tiled.shape)
			# print('X_tiled', X_tiled, X_tiled.shape)

			Tau = C_tiled - X_tiled ## Nx2Mxnx1 = Nx2Mxnx1 - Nx2Mxnx1
			# print('Tau', Tau)

			#### Columb power is ||x||^{-3}? --- check

			order = -1
			sign = 1.
			if order < 0:
				sign *= -1

			norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
			ord_tensor = order*tf.ones_like(norm_tau)
			Phi = sign*tf.pow(norm_tau, ord_tensor) ## Nx1

			# print(Phi)
			# exit(0)


			W = tf.cast(W, dtype = 'float32')
			Phi = tf.cast(Phi, dtype = 'float32')
			D = tf.squeeze(tf.linalg.matmul(W, Phi, transpose_a=True, transpose_b=False),axis = 2)

			return D


		#### Centered about the target dataset
		mu_d = np.mean(self.act2,axis = 0)
		cov_d = np.cov(self.act2,rowvar = False)
		step_size = tf.math.sqrt(tf.linalg.diag_part(cov_d))
		if self.mode in ['metrics', 'model_metrics']:
			print("Step Size - ",step_size)
		# 	self.res_file.write("\n Step Size - "+step_size,"\n")
		# if self.res_flag:
		# 	self.res_file.write("\n Step Size - ",step_size,"\n")

		self.CID_cur_vec = []
		with tf.device(':/CPU'):
			for r in range(1,100):
				for act1,act2 in zip(self.act1_dataset, self.act2_dataset):

					Unif = tfp.distributions.Uniform(low=mu_d - (r/2)*step_size, high=mu_d + (r/2)*step_size)
					X = tf.cast(Unif.sample([self.DatasetCID_batch_size]),dtype = 'float32')
					# print(X.shape)

					##### CID Between source act1 and target act 2:
					# CID_val = CID(D,X,self.act1,self.act2)
					#### functions for CID and D computations
					Dxd_cur = D(X,act2)
					Dxg_cur = D(X,act1)
					try:
						Dxg = np.concatenate((Dxg,Dxg), axis = 0)
						Dxd = np.concatenate((Dxd,Dxd), axis = 0)
					except:
						Dxg = Dxg_cur
						Dxd = Dxd_cur

				CID_val = tf.reduce_mean(Dxg - Dxd)

				print('\nCID(pz|pd) :',CID_val,' at r = ',r)
				self.CID_cur_vec.append([CID_val,r])
				del Dxg,Dxd
			
		self.CID_vec.append([np.array(self.CID_cur_vec), self.total_count.numpy()])
		np.save(self.metricpath+'Dataset_CID_Iter'+str(self.total_count.numpy()).zfill(6)+'.npy',np.array(self.CID_cur_vec))
		return



	def eval_DatasetCID_unbatched(self):

		#### functions for CID and D computations
		def CID(D,X,cen1,cen2):
			Dxd = D(X,cen1)
			Dxg = D(X,cen2)
			CID_val = tf.reduce_mean(Dxg - Dxd)
			return CID_val

		def D(X,C):
			#####
			## X :          Data of dimension N x n - batch size N and dimensionality n
			## c_d, c_g :   Centers from data/generator of dimension M x n - M is batch size
			## D(x) :       Computes the "Columb Potential" of each x, given all c_d, c_g of dimension N x 1
			##
			#####
			### Data vector
			N = X.shape[0] #### Batch size of data
			M = C.shape[0] #### Batch size of each set of centers

			W = (1/N)*np.ones([C.shape[0]])
			W = tf.expand_dims(W, axis = 1)
			# print('W', W)

			X = tf.expand_dims(X, axis = 2) ## Nxnx1
			X = tf.expand_dims(X, axis = 1) ## Nx1xnx1
			# print('Input X',X, X.shape)

			C = tf.expand_dims(C, axis = 2) ## Mxnx1
			C = tf.expand_dims(C, axis = 0) ## 1xMxnx1
			# print('Centers C', C, C.shape)

			C_tiled = tf.tile(C, [N,1,1,1])  ## NxMxnx1 ### was tf.shape(X)[0]
			X_tiled = tf.tile(X, [1,M,1,1])  ## NxMxnx1 ### was self.num_hidden
			# print('C_tiled', C_tiled, C_tiled.shape)
			# print('X_tiled', X_tiled, X_tiled.shape)

			Tau = C_tiled - X_tiled ## Nx2Mxnx1 = Nx2Mxnx1 - Nx2Mxnx1
			# print('Tau', Tau)

			#### Columb power is ||x||^{-3}? --- check

			order = -1
			sign = 1.
			if order < 0:
				sign *= -1

			norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
			ord_tensor = order*tf.ones_like(norm_tau)
			Phi = sign*tf.pow(norm_tau, ord_tensor) ## Nx1

			# print(Phi)
			# exit(0)


			W = tf.cast(W, dtype = 'float32')
			Phi = tf.cast(Phi, dtype = 'float32')
			D = tf.squeeze(tf.linalg.matmul(W, Phi, transpose_a=True, transpose_b=False),axis = 2)

			return D


		#### Centered about the target dataset
		mu_d = np.mean(self.act1,axis = 0)
		cov_d = np.cov(self.act1,rowvar = False)
		step_size = tf.math.sqrt(tf.linalg.diag_part(cov_d))
		if self.mode in ['metrics', 'model_metrics']:
			print("Step Size - ",step_size)
		# 	self.res_file.write("\n Step Size - "+step_size,"\n")
		# if self.res_flag:
		# 	self.res_file.write("\n Step Size - ",step_size,"\n")

		self.CID_cur_vec = []
		with tf.device(':/CPU'):
			for r in range(1,100):
				d = self.act1.shape[1] ## Dimensionality of data

				# Unif = tfp.distributions.Uniform(low=mu_d - (r/2)*tf.ones_like(mu_d), high=mu_d + (r/2)*tf.ones_like(mu_d))
				Unif = tfp.distributions.Uniform(low=mu_d - (r/2)*step_size, high=mu_d + (r/2)*step_size)
				X = tf.cast(Unif.sample([self.DatasetCID_batch_size]),dtype = 'float32')
				# print(X.shape)

				##### CID Between source act1 and target act 2:
				CID_val = CID(D,X,self.act1,self.act2)
				print('\nCID(pz|pd) :',CID_val,' at r = ',r)
				self.CID_cur_vec.append([CID_val,r])
			
		self.CID_vec.append([np.array(self.CID_cur_vec), self.total_count.numpy()])
		np.save(self.metricpath+'Dataset_CID_Iter'+str(self.total_count.numpy()).zfill(6)+'.npy',np.array(self.CID_cur_vec))
		return

	def print_CID(self):
		path = self.metricpath
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages


		with PdfPages(path+'CID_Data_'+self.data+'_Noise_'+self.noise_data+'_plot.pdf') as pdf:
			for CID_cur,count in self.CID_vec:
				fig1 = plt.figure(figsize=(3.5, 3.5), dpi=400)
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.set_xlim([0, 1])
				ax1.set_ylim([0, 1])
				ax1.get_xaxis().set_visible(True)
				ax1.get_yaxis().set_visible(True)
				CID_vals,locs = CID_cur
				ax1.plot(locs, CID_vals, color = 'g', linestyle = 'solid', alpha=0.5, linewidth=3)
				ax1.set_xlabel(r'\mathrm{DISTANCE }r')
				ax1.set_ylabel(r'CID(p_z|p_d)')
				title = 'CID at Iteration '+str(count)
				plt.title(title, fontsize=8)
				pdf.savefig(fig1, bbox_inches='tight', dpi=400)
				plt.close(fig1)


	def update_SID(self):
		from ops.gan_sid import my_sid as sid

		### Fix === remove flags and pass only order or whatever is needed/ 
		self.SID_cur_arr = sid.compute_sid(fdir1 = self.FIDfakes_dir,  fdir2 = self.FIDreals_dir, order = self.SID_order)

		self.SID_vec.append([self.SID_cur_arr, self.total_count.numpy()])
		np.save(self.metricpath+'_SID_Iter'+str(self.total_count.numpy()).zfill(6)+'.npy',np.array(self.SID_cur_arr))


	def update_SID_local(self):
		def ukiyoe_reader(filename):
			# with tf.device('/CPU'):
			with self.strategy.scope():
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([1024,1024,3])
				image = tf.image.resize(image,[75,75])
				# This will convert to float values in [0, 1]
				image = tf.subtract(image,127.5)
				image = tf.divide(image,127.5)
			return image

		def tinyimgnet_reader(filename):
			# with tf.device('/CPU'):
			with self.strategy.scope():
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([64,64,3])
				# image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[75,75])

				# This will convert to float values in [-1, 1]
				image = tf.subtract(image,127.0)
				image = tf.divide(image,127.0)
			return image

		def celeba_reader(filename):
			# with tf.device('/CPU'):
			with self.strategy.scope():
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[75,75])

				# This will convert to float values in [-1, 1]
				image = tf.subtract(image,127.5)
				image = tf.divide(image,127.5)
			return image

		def data_preprocess(image):
			# with tf.device('/CPU'):
			# with self.strategy.scope():
			image = tf.image.resize(image,[75,75])
			if image.shape[2] == 1:
				image = tf.image.grayscale_to_rgb(image)
			return image

		# def SID(D,X,cen1,cen2):
		# 	Dxd = D(X,cen1)
		# 	Dxg = D(X,cen2)
		# 	SID_val = tf.reduce_mean(Dxg - Dxd)
		# 	return SID_val

		@tf.function
		def D(X,C):
			#####
			## X :          Data of dimension N x n - batch size N and dimensionality n
			## c_d, c_g :   Centers from data/generator of dimension M x n - M is batch size
			## D(x) :       Computes the "Columb Potential" of each x, given all c_d, c_g of dimension Nx1
			##
			#####
			### Data vector
			N = X.shape[0] #### Batch size of data
			M = C.shape[0] #### Batch size of each set of centers

			W = (1/N)*tf.ones([C.shape[0]])
			W = tf.expand_dims(W, axis = 1)
			# print('W', W)

			X = tf.expand_dims(X, axis = 2) ## Nxnx1
			X = tf.expand_dims(X, axis = 1) ## Nx1xnx1
			# print('Input X',X, X.shape)

			C = tf.expand_dims(C, axis = 2) ## Mxnx1
			C = tf.expand_dims(C, axis = 0) ## 1xMxnx1
			# print('Centers C', C, C.shape)

			C_tiled = tf.tile(C, [N,1,1,1])  ## NxMxnx1 ### was tf.shape(X)[0]
			X_tiled = tf.tile(X, [1,M,1,1])  ## NxMxnx1 ### was self.num_hidden
			# print('C_tiled', C_tiled, C_tiled.shape)
			# print('X_tiled', X_tiled, X_tiled.shape)

			Tau = C_tiled - X_tiled ## Nx2Mxnx1 = Nx2Mxnx1 - Nx2Mxnx1
			# print('Tau', Tau)

			#### Columb power is ||x||^{-3}? --- check

			order = -1
			sign = 1.
			if order < 0:
				sign *= -1

			norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
			ord_tensor = order*tf.ones_like(norm_tau)
			Phi = sign*tf.pow(norm_tau, ord_tensor) ## Nx1

			# print(Phi)
			# exit(0)


			W = tf.cast(W, dtype = 'float32')
			Phi = tf.cast(Phi, dtype = 'float32')
			D = tf.squeeze(tf.linalg.matmul(W, Phi, transpose_a=True, transpose_b=False),axis = 2)

			return D

		# def eval_SID(act1,act2):
		# 	mu_d = np.mean(act2,axis = 0)
		# 	cov_d = np.cov(act2,rowvar = False)
		# 	step_size = tf.math.sqrt(tf.linalg.diag_part(cov_d))
		# 	# if self.mode in ['metrics', 'model_metrics']:
		# 		# print("Step Size - ",step_size)


		# 	self.SID_cur_arr = np.zeros([len(range(1,100),2)])
		# 	with tf.device(':/CPU'):
		# 		for r in range(1,100):
		# 			for act1,act2 in zip(self.act1_dataset, self.act2_dataset):

		# 				cur_step = (r/2)*step_size

		# 				Unif = tfp.distributions.Uniform(low=mu_d - cur_step, high=mu_d + cur_step)
		# 				X = tf.cast(Unif.sample([self.SID_batch_size]),dtype = 'float32')
		# 				# print(X.shape)

		# 				##### SID Between source act1 and target act 2:
		# 				# SID_val = SID(D,X,self.act1,self.act2)
		# 				#### functions for SID and D computations
		# 				Dxd_cur = D(X,act2)
		# 				Dxg_cur = D(X,act1)
		# 				try:
		# 					Dxg = np.concatenate((Dxg,Dxg), axis = 0)
		# 					Dxd = np.concatenate((Dxd,Dxd), axis = 0)
		# 				except:
		# 					Dxg = Dxg_cur
		# 					Dxd = Dxd_cur

		# 			SID_val = tf.reduce_mean(Dxg - Dxd)

		# 			print('\nSID(pz|pd) :',SID_val,' at r = ',r)
		# 			self.SID_cur_vec.append([SID_val,r])
		# 			del Dxg,Dxd

		# 	return sid


		# with self.strategy.scope():
		if self.SID_load_flag == 0:
			### First time FID call setup
			self.SID_load_flag = 1

			### Noise is SRC, Data is TAR

			# random_points_src = tf.keras.backend.random_uniform([self.SID_num_samples], minval=0, maxval=int(self.dfid_noise_images.shape[0]), dtype='int32', seed=None)
			# self.src_images = self.dfid_noise_images[random_points_src]

			random_points_tar = tf.keras.backend.random_uniform([self.SID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			self.tar_images = self.fid_train_images[random_points_tar]


			# self.src_images_dataset = tf.data.Dataset.from_tensor_slices((self.src_images))
			# if self.data == 'celeba': 
			# 	self.src_images_dataset = self.src_images_dataset.map(celeba_reader,num_parallel_calls=int(self.num_parallel_calls))
			# elif self.data == 'ukiyoe': 
			# 	self.src_images_dataset = self.src_images_dataset.map(ukiyoe_reader,num_parallel_calls=int(self.num_parallel_calls))
			# elif self.data == 'tinyimgnet': 
			# 	self.src_images_dataset = self.src_images_dataset.map(tinyimgnet_reader,num_parallel_calls=int(self.num_parallel_calls))
			# else:
			# 	self.src_images_dataset = self.src_images_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))

			# self.src_images_dataset = self.src_images_dataset.batch(self.SID_batch_size)

			self.tar_images_dataset = tf.data.Dataset.from_tensor_slices((self.tar_images))
			if self.data == 'celeba': 
				self.tar_images_dataset = self.tar_images_dataset.map(celeba_reader,num_parallel_calls=int(self.num_parallel_calls))
			elif self.data == 'ukiyoe': 
				self.tar_images_dataset = self.tar_images_dataset.map(ukiyoe_reader,num_parallel_calls=int(self.num_parallel_calls))
			elif self.data == 'tinyimgnet': 
				self.tar_images_dataset = self.tar_images_dataset.map(tinyimgnet_reader,num_parallel_calls=int(self.num_parallel_calls))
			else:
				self.tar_images_dataset = self.tar_images_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))

			self.tar_images_dataset = self.tar_images_dataset.batch(self.SID_batch_size)

			self.SID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(75,75,3), classes=1000)

		with self.strategy.scope():
			range_var = range(1,50)
			self.SID_cur_arr = np.zeros([len(range_var),2])
			for r in range_var:
				if self.topic not in ['SpiderGAN']:
					for batch_idx,tar_batch in enumerate(self.tar_images_dataset):
						
						noise = self.get_noise([self.batch_size, self.noise_dims])

						if self.gan in ['WAE']:
							fakes = self.decoder(noise, training=False)
						else:
							fakes = self.generator(noise, training=False)

						fakes = tf.image.resize(fakes, [75,75])
						if fakes.shape[3] == 1:
							fakes = tf.image.grayscale_to_rgb(fakes)
						fakes = fakes.numpy()


						act1 = self.SID_model.predict(fakes)
						act2 = self.SID_model.predict(tar_batch)

						if batch_idx <= 1:
							mu_d = np.mean(act2,axis = 0)
							cov_d = np.cov(act2,rowvar = False)
							step_size = tf.math.sqrt(tf.linalg.diag_part(cov_d))
							cur_step = tf.reduce_mean((r/2)*step_size)

						self.SID_cur_arr[r-1,1] = cur_step

						Unif = tfp.distributions.Uniform(low=mu_d - cur_step, high=mu_d + cur_step)
						X = tf.cast(Unif.sample([self.SID_batch_size]),dtype = 'float32')

						Dxd = D(X,act2)
						Dxg = D(X,act1)

						SID_val = tf.reduce_mean(Dxg - Dxd)
						# del Dxg,Dxd

						if self.SID_cur_arr[r-1,0] == 0:
							self.SID_cur_arr[r-1,0] = SID_val
						else:
							self.SID_cur_arr[r-1,0] = 0.5*self.SID_cur_arr[r-1,0] + 0.5*SID_val

				else:
					for batch_idx,(tar_batch,src_batch) in enumerate(zip(self.tar_images_dataset,self.noise_dataset)):

						fakes = self.generator(src_batch, training=False)

						fakes = tf.image.resize(fakes, [75,75])
						if fakes.shape[3] == 1:
							fakes = tf.image.grayscale_to_rgb(fakes)
						fakes = fakes.numpy()


						act1 = self.SID_model.predict(fakes)
						act2 = self.SID_model.predict(tar_batch)

						if batch_idx <= 1:
							mu_d = np.mean(act2,axis = 0)
							cov_d = np.cov(act2,rowvar = False)
							step_size = tf.math.sqrt(tf.linalg.diag_part(cov_d))
							cur_step = tf.reduce_mean((r/2)*step_size)

						self.SID_cur_arr[r-1,1] = cur_step

						Unif = tfp.distributions.Uniform(low=mu_d - cur_step, high=mu_d + cur_step)
						X = tf.cast(Unif.sample([self.SID_batch_size]),dtype = 'float32')

						Dxd = D(X,act2)
						Dxg = D(X,act1)

						SID_val = tf.reduce_mean(Dxg - Dxd)
						# del Dxg,Dxd

						if self.SID_cur_arr[r-1,0] == 0:
							self.SID_cur_arr[r-1,0] = SID_val
						else:
							self.SID_cur_arr[r-1,0] = 0.5*self.SID_cur_arr[r-1,0] + 0.5*SID_val



			self.SID_vec.append([self.SID_cur_arr, self.total_count.numpy()])
			np.save(self.metricpath+'_SID_Iter'+str(self.total_count.numpy()).zfill(6)+'.npy',np.array(self.SID_cur_arr))
			# print('\nSID(pz|pd) :',SID_val,' at r = ',r)
			# cur_SID = eval_SID(act1,act2)
			
		return

	def eval_SID(self):
		self.act1_dataset = tf.data.Dataset.from_tensor_slices((self.act1))
		self.act1_dataset = self.act1_dataset.batch(self.SID_batch_size)

		self.act2_dataset = tf.data.Dataset.from_tensor_slices((self.act2))
		self.act2_dataset = self.act2_dataset.batch(self.SID_batch_size)





		#### Centered about the target dataset
		mu_d = np.mean(self.act2,axis = 0)
		cov_d = np.cov(self.act2,rowvar = False)
		step_size = tf.math.sqrt(tf.linalg.diag_part(cov_d))
		if self.mode in ['metrics', 'model_metrics']:
			print("Step Size - ",step_size)
		# 	self.res_file.write("\n Step Size - "+step_size,"\n")
		# if self.res_flag:
		# 	self.res_file.write("\n Step Size - ",step_size,"\n")

		self.SID_cur_vec = []
		with tf.device(':/CPU'):
			for r in range(1,100):
				for act1,act2 in zip(self.act1_dataset, self.act2_dataset):

					Unif = tfp.distributions.Uniform(low=mu_d - (r/2)*step_size, high=mu_d + (r/2)*step_size)
					X = tf.cast(Unif.sample([self.SID_batch_size]),dtype = 'float32')
					# print(X.shape)

					##### SID Between source act1 and target act 2:
					# SID_val = SID(D,X,self.act1,self.act2)
					#### functions for SID and D computations
					Dxd_cur = D(X,act2)
					Dxg_cur = D(X,act1)
					try:
						Dxg = np.concatenate((Dxg,Dxg), axis = 0)
						Dxd = np.concatenate((Dxd,Dxd), axis = 0)
					except:
						Dxg = Dxg_cur
						Dxd = Dxd_cur

				SID_val = tf.reduce_mean(Dxg - Dxd)

				print('\nSID(pg|pd) :',SID_val,' at r = ',r)
				self.SID_cur_vec.append([SID_val,r])
				del Dxg,Dxd
			
		self.SID_vec.append([np.array(self.SID_cur_vec), self.total_count.numpy()])
		np.save(self.metricpath+'_SID_Iter'+str(self.total_count.numpy()).zfill(6)+'.npy',np.array(self.SID_cur_vec))
		return

	def print_SID(self):
		path = self.metricpath
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages

		max_list = []
		min_list = []
		with PdfPages(path+'SID_Data_'+self.data+'_plot.pdf') as pdf:
			for SID_cur,count in self.SID_vec:
				fig1 = plt.figure(figsize=(3.5, 3.5), dpi=400)
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.get_xaxis().set_visible(True)
				ax1.get_yaxis().set_visible(True)
				SID_vals = SID_cur[:,0]
				max_list.append(max(SID_vals))
				min_list.append(min(SID_vals))
				locs = SID_cur[:,1]
				ax1.plot(locs, SID_vals, color = 'g', linestyle = 'solid', alpha=0.5, linewidth=3)
				ax1.set_xlabel(r'$\mathrm{DISTANCE }r$')
				ax1.set_ylabel(r'$SID(p_g|p_d)$')
				title = 'SID at Iteration '+str(count)
				plt.title(title, fontsize=8)
				pdf.savefig(fig1, bbox_inches='tight', dpi=400)
				plt.close(fig1)

		ymin = min(min_list)
		ymax = max(max_list)
		with PdfPages(path+'SID_Data_'+self.data+'_sameY_plot.pdf') as pdf:
			for SID_cur,count in self.SID_vec:
				fig1 = plt.figure(figsize=(3.5, 3.5), dpi=400)
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.set_ylim([ymin, ymax])
				ax1.get_xaxis().set_visible(True)
				ax1.get_yaxis().set_visible(True)
				SID_vals = SID_cur[:,0]
				locs = SID_cur[:,1]
				ax1.plot(locs, SID_vals, color = 'g', linestyle = 'solid', alpha=0.5, linewidth=3)
				ax1.set_xlabel(r'$\mathrm{DISTANCE }r$')
				ax1.set_ylabel(r'$SID(p_g|p_d)$')
				title = 'SID at Iteration '+str(count)
				plt.title(title, fontsize=8)
				pdf.savefig(fig1, bbox_inches='tight', dpi=400)
				plt.close(fig1)

	def update_DatasetSinD(self):

		def ukiyoe_reader(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([1024,1024,3])
				# image = tf.image.resize(image,[299,299])
				image = tf.image.resize(image,[self.SinD_size,self.SinD_size])
				image = tf.reshape(image,[self.SinD_size*self.SinD_size*3])
				# This will convert to float values in [0, 1]
				image = tf.subtract(image,127.5)
				image = tf.divide(image,127.5)
			return image

		def tinyimgnet_reader(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([64,64,3])
				# image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				# image = tf.image.resize(image,[299,299])
				image = tf.image.resize(image,[self.SinD_size,self.SinD_size])
				image = tf.reshape(image,[self.SinD_size*self.SinD_size*3])
				# This will convert to float values in [-1, 1]
				image = tf.subtract(image,127.0)
				image = tf.divide(image,127.0)
			return image

		def church_reader(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image  = tf.image.crop_to_bounding_box(image, 0, 0, 256,256)
				# image = tf.image.resize(image,[299,299])
				image = tf.image.resize(image,[self.SinD_size,self.SinD_size])
				image = tf.reshape(image,[self.SinD_size*self.SinD_size*3])				
				# This will convert to float values in [0, 1]
				# image = tf.divide(image,255.0)
				image = tf.subtract(image,127.0)
				image = tf.divide(image,127.0)
				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image

		def celeba_reader(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				# image = tf.image.resize(image,[299,299])
				image = tf.image.resize(image,[self.SinD_size,self.SinD_size])
				image = tf.reshape(image,[self.SinD_size*self.SinD_size*3])
				# This will convert to float values in [-1, 1]
				image = tf.subtract(image,127.5)
				image = tf.divide(image,127.5)
			return image

		def data_preprocess(image):
			with tf.device('/CPU'):
				if image.shape[2] == 1:
					image = tf.image.grayscale_to_rgb(image)
				image = tf.image.resize(image,[self.SinD_size,self.SinD_size])
				image = tf.reshape(image,[self.SinD_size*self.SinD_size*3])			
				return image


		with tf.device('/CPU'):
			if self.DID_load_flag == 0:
				### First time FID call setup
				self.DID_load_flag = 1

				### Noise is SRC, Data is TAR

				# random_points_src = tf.keras.backend.random_uniform([self.DID_num_samples], minval=0, maxval=int(self.dfid_noise_images.shape[0]), dtype='int32', seed=None)
				self.src_images = self.dfid_noise_images
				self.src_batch_size = self.dfid_noise_images.shape[0]

				# random_points_tar = tf.keras.backend.random_uniform([self.DID_num_samples], minval=0, maxval=int(self.dfid_data_images.shape[0]), dtype='int32', seed=None)
				self.tar_images = self.dfid_data_images
				self.tar_batch_size = self.dfid_data_images.shape[0]


				self.src_images_dataset = tf.data.Dataset.from_tensor_slices((self.src_images))
				if self.noise_data == 'celeba': 
					self.src_images_dataset = self.src_images_dataset.map(celeba_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.noise_data == 'ukiyoe': 
					self.src_images_dataset = self.src_images_dataset.map(ukiyoe_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.noise_data == 'tinyimgnet': 
					self.src_images_dataset = self.src_images_dataset.map(tinyimgnet_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.noise_data == 'church': 
					self.src_images_dataset = self.src_images_dataset.map(church_reader,num_parallel_calls=int(self.num_parallel_calls))
				else:
					self.src_images_dataset = self.src_images_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))

				self.src_images_dataset = self.src_images_dataset.batch(self.src_batch_size)


				self.tar_images_dataset = tf.data.Dataset.from_tensor_slices((self.tar_images))
				if self.data == 'celeba': 
					self.tar_images_dataset = self.tar_images_dataset.map(celeba_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.data == 'ukiyoe': 
					self.tar_images_dataset = self.tar_images_dataset.map(ukiyoe_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.data == 'tinyimgnet': 
					self.tar_images_dataset = self.tar_images_dataset.map(tinyimgnet_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.data == 'church': 
					self.tar_images_dataset = self.tar_images_dataset.map(church_reader,num_parallel_calls=int(self.num_parallel_calls))
				else:
					self.tar_images_dataset = self.tar_images_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))

				self.tar_images_dataset = self.tar_images_dataset.batch(self.tar_batch_size)


				self.src_images_dataset = self.strategy.experimental_distribute_dataset(self.src_images_dataset)
				self.tar_images_dataset = self.strategy.experimental_distribute_dataset(self.tar_images_dataset)


				# self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(299,299,3), classes=1000)
		self.eval_DatasetSinD()
		self.print_DatasetSinD()
		### Convert to clea
		# for src_batch,tar_batch in zip(self.src_images_dataset, self.tar_images_dataset):
		# 	i += 1
		# 	for j in range(self.DID_batch_size):
		# 		src = src_batch[j]
		# 		tar = tar_batch[j]
		# 		total_num_figs += 1
		# 		if total_num_figs <= self.DID_num_samples:
		# 			tf.keras.preprocessing.image.save_img(self.DIDsrcpath+str(i*self.DID_batch_size + j)+'.png', src, scale=True)
		# 			tf.keras.preprocessing.image.save_img(self.DIDtarpath+str(i*self.DID_batch_size + j)+'.png', tar, scale=True)

		return

	def eval_DatasetSinD(self):

		feats_dim = self.SinD_size * self.SinD_size * 3			
		for src_batch,tar_batch in zip(self.src_images_dataset, self.tar_images_dataset):
			### Just get the one batch out of it. This is redundant, but consistent with other code
			assert src_batch.shape == (self.src_batch_size, self.SinD_size*self.SinD_size*3)
			assert tar_batch.shape == (self.tar_batch_size, self.SinD_size*self.SinD_size*3)
			break
		##

		if self.src_batch_size > 100000 or self.tar_batch_size > 100000:
			device = '/CPU'
		else:
			device = self.device
		with tf.device(device):
			src_cov = tfp.stats.covariance(src_batch)
			tar_cov = tfp.stats.covariance(tar_batch)
			src_sigma, src_U, src_V = tf.linalg.svd(src_cov)
			tar_sigma, tar_U, tar_V = tf.linalg.svd(tar_cov)
			diff_sigma, siff_U, diff_V = tf.linalg.svd(src_cov - tar_cov)

		## Cov Diff Op Norm is l_inf of diff of Eigenvalues
		cov_op_norm = tf.linalg.norm(diff_sigma, ord = np.inf)
		# print('cov_op_norm',cov_op_norm)
		## Cov Diff Op Norm is l_inf of diff of Eigenvalues
		cov_F_norm = tf.linalg.norm(src_cov - tar_cov, ord = 'fro', axis = [-2,-1])
		# print('cov_F_norm',cov_F_norm)

		low = 3
		high = int(0.1*feats_dim)
		mid = int((low+high)/2)
		iter_range = range(low,high)
		num_sinD = len(iter_range)
		self.SinD_mat = np.zeros((num_sinD,2))
		## Finding Covariance matrices
		for s in iter_range:
			## r=0, r-1 = -\infty; s = loop_Var, s+1 inferred
			d = s+1 ## d=s-r+1 = s+1 
			num = 2*tf.minimum((d**(0.5))*cov_op_norm, cov_F_norm).numpy()
			# den = src_sigma[s] - src_sigma[s+1] ## technically min(\infty, current_den)
			den = tar_sigma[s] - tar_sigma[s+1] ## technically min(\infty, current_den)
			self.SinD_mat[s-low,0] = num/den
			self.SinD_mat[s-low,1] = s
			if self.mode in ['metrics', 'model_metrics']:
				print("\nDataset Sin Theta Distance for "+str(s)+ " st/nd/rd/th Eigenvalue - "+str(self.SinD_mat[s-low,0]))
				if self.res_flag:
					self.res_file.write("\nDataset Sin Theta Distance for "+str(s)+" st/nd/rd/th Eigenvalues - "+str(self.SinD_mat[s-low,0]))
				if self.res_flag:
					self.res_file.write("\nDataset Sin Theta Distance for "+str(s)+" st/nd/rd/th Eigenvalues - "+str(self.SinD_mat[s-low,0]))

				

		self.SinD_vec.append([self.SinD_mat, self.total_count.numpy()])
		np.save(self.metricpath+'_SinD_Dataset_'+str(self.total_count.numpy()).zfill(6)+'.npy',np.array(self.SinD_vec))

		# if self.mode in ['metrics', 'model_metrics']:
		# 	print("Dataset Sin Theta Distance for "+str(low)+ " Eigenvalues - "+str(self.SinD_mat[low-low,0]))
		# 	if self.res_flag:
		# 		self.res_file.write("Dataset Sin Theta Distance for "+str(low)+" Eigenvalues - "+str(self.SinD_mat[low-low,0]))
		# if self.res_flag:
		# 	self.res_file.write("Dataset Sin Theta Distance for "+str(low)+" Eigenvalues - "+str(self.SinD_mat[low-low,0]))



		# if self.mode in ['metrics', 'model_metrics']:
		# 	print("Dataset Sin Theta Distance for "+str(mid)+ " Eigenvalues - "+str(self.SinD_mat[mid-low,0]))
		# 	if self.res_flag:
		# 		self.res_file.write("Dataset Sin Theta Distance for "+str(mid)+" Eigenvalues - "+str(self.SinD_mat[mid-low,0]))
		# if self.res_flag:
		# 	self.res_file.write("Dataset Sin Theta Distance for "+str(mid)+" Eigenvalues - "+str(self.SinD_mat[mid-low,0]))


		# if self.mode in ['metrics', 'model_metrics']:
		# 	print("Dataset Sin Theta Distance for "+str(high)+" Eigenvalues - "+str(self.SinD_mat[-1,0]))
		# 	if self.res_flag:
		# 		self.res_file.write("Dataset Sin Theta Distance for "+str(high)+" Eigenvalues - "+str(self.SinD_mat[-1,0]))
		# if self.res_flag:
		# 	self.res_file.write("Dataset Sin Theta Distance for "+str(high)+" Eigenvalues - "+str(self.SinD_mat[-1,0]))

		return

	def print_DatasetSinD(self):

		path = self.metricpath
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages

		with PdfPages(path+'SinD_Noise_'+self.noise_data+'_Data_'+self.data+'_plot.pdf') as pdf:
			for SinD_cur,count in self.SinD_vec:
				fig1 = plt.figure(figsize=(3.5, 3.5), dpi=400)
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.get_xaxis().set_visible(True)
				ax1.get_yaxis().set_visible(True)
				SID_vals = SinD_cur[:,0]
				locs = SinD_cur[:,1]
				ax1.plot(locs, SID_vals, color = 'g', linestyle = 'solid', alpha=0.5, linewidth=3)
				ax1.set_xlabel(r'MAX-EIGENVALUE INDEX $s: \{\lambda_1,\lambda_s\}$')
				ax1.set_ylabel(r'$\left\|\sin\Theta(\mathcal{S}_{g}|\mathcal{S}_{d})\right\|_{F}$')
				title = 'Iteration '+str(count)
				plt.title(title, fontsize=8)
				pdf.savefig(fig1, bbox_inches='tight', dpi=400)
				plt.close(fig1)


		with PdfPages(path+'SinD_Noise_'+self.noise_data+'_Data_'+self.data+'_LogPlot.pdf') as pdf:
			for SinD_cur,count in self.SinD_vec:
				fig1 = plt.figure(figsize=(3.5, 3.5), dpi=400)
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.get_xaxis().set_visible(True)
				ax1.get_yaxis().set_visible(True)
				SID_vals = SinD_cur[:,0]
				locs = SinD_cur[:,1]
				ax1.plot(locs, np.abs(np.log10(SID_vals+1e-10)), color = 'g', linestyle = 'solid', alpha=0.5, linewidth=3)
				ax1.set_xlabel(r'MAX-EIGENVALUE INDEX $s: \{\lambda_1,\lambda_s\}$')
				ax1.set_ylabel(r'$\left\|\sin\Theta(\mathcal{S}_{g}|\mathcal{S}_{d})\right\|_{F}$')
				title = 'Iteration '+str(count)
				plt.title(title, fontsize=8)
				pdf.savefig(fig1, bbox_inches='tight', dpi=400)
				plt.close(fig1)

	def update_DatasetSinID(self):

		def ukiyoe_reader(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([1024,1024,3])
				# image = tf.image.resize(image,[299,299])
				image = tf.image.resize(image,[192,192])
				
				# This will convert to float values in [0, 1]
				image = tf.subtract(image,127.5)
				image = tf.divide(image,127.5)
			return image

		def tinyimgnet_reader(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([64,64,3])
				# image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				# image = tf.image.resize(image,[299,299])
				image = tf.image.resize(image,[192,192])
				
				# This will convert to float values in [-1, 1]
				image = tf.subtract(image,127.0)
				image = tf.divide(image,127.0)
			return image

		def church_reader(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image  = tf.image.crop_to_bounding_box(image, 0, 0, 256,256)
				# image = tf.image.resize(image,[299,299])
				image = tf.image.resize(image,[192,192])
								
				# This will convert to float values in [0, 1]
				# image = tf.divide(image,255.0)
				image = tf.subtract(image,127.0)
				image = tf.divide(image,127.0)
				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image

		def celeba_reader(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				# image = tf.image.resize(image,[299,299])
				image = tf.image.resize(image,[192,192])
				
				# This will convert to float values in [-1, 1]
				image = tf.subtract(image,127.5)
				image = tf.divide(image,127.5)
			return image

		def data_preprocess(image):
			with tf.device('/CPU'):
				if image.shape[2] == 1:
					image = tf.image.grayscale_to_rgb(image)
				image = tf.image.resize(image,[192,192])
							
				return image


		with tf.device('/CPU'):
			if self.DID_load_flag == 0:
				### First time FID call setup
				self.DID_load_flag = 1

				### Noise is SRC, Data is TAR

				# random_points_src = tf.keras.backend.random_uniform([self.DID_num_samples], minval=0, maxval=int(self.dfid_noise_images.shape[0]), dtype='int32', seed=None)
				# self.src_images = self.dfid_noise_images
				# self.src_batch_size = self.dfid_noise_images.shape[0]

				# # random_points_tar = tf.keras.backend.random_uniform([self.DID_num_samples], minval=0, maxval=int(self.dfid_data_images.shape[0]), dtype='int32', seed=None)
				# self.tar_images = self.dfid_data_images
				# self.tar_batch_size = self.dfid_data_images.shape[0]

				random_points_src = tf.keras.backend.random_uniform([self.DID_num_samples], minval=0, maxval=int(self.dfid_noise_images.shape[0]), dtype='int32', seed=None)
				self.src_images = self.dfid_noise_images[random_points_src]

				random_points_tar = tf.keras.backend.random_uniform([self.DID_num_samples], minval=0, maxval=int(self.dfid_data_images.shape[0]), dtype='int32', seed=None)
				self.tar_images = self.dfid_data_images[random_points_tar]


				self.src_images_dataset = tf.data.Dataset.from_tensor_slices((self.src_images))
				if self.noise_data == 'celeba': 
					self.src_images_dataset = self.src_images_dataset.map(celeba_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.noise_data == 'ukiyoe': 
					self.src_images_dataset = self.src_images_dataset.map(ukiyoe_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.noise_data == 'tinyimgnet': 
					self.src_images_dataset = self.src_images_dataset.map(tinyimgnet_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.noise_data == 'church': 
					self.src_images_dataset = self.src_images_dataset.map(church_reader,num_parallel_calls=int(self.num_parallel_calls))
				else:
					self.src_images_dataset = self.src_images_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))

				self.src_images_dataset = self.src_images_dataset.batch(100)#self.src_batch_size)


				self.tar_images_dataset = tf.data.Dataset.from_tensor_slices((self.tar_images))
				if self.data == 'celeba': 
					self.tar_images_dataset = self.tar_images_dataset.map(celeba_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.data == 'ukiyoe': 
					self.tar_images_dataset = self.tar_images_dataset.map(ukiyoe_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.data == 'tinyimgnet': 
					self.tar_images_dataset = self.tar_images_dataset.map(tinyimgnet_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.data == 'church': 
					self.tar_images_dataset = self.tar_images_dataset.map(church_reader,num_parallel_calls=int(self.num_parallel_calls))
				else:
					self.tar_images_dataset = self.tar_images_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))

				self.tar_images_dataset = self.tar_images_dataset.batch(100)#self.tar_batch_size)


				self.src_images_dataset = self.strategy.experimental_distribute_dataset(self.src_images_dataset)
				self.tar_images_dataset = self.strategy.experimental_distribute_dataset(self.tar_images_dataset)


				# self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(299,299,3), classes=1000)
		### Convert to clea

				self.SinD_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(75,75,3), classes=1000)

		with self.strategy.scope():
			for src_batch in self.src_images_dataset:
				act1 = self.SinD_model.predict(src_batch)
				try:
					self.act1 = np.concatenate((self.act1,act1), axis = 0)
				except:
					self.act1 = act1

			self.src_feats = self.act1

			for tar_batch in self.tar_images_dataset:
				act2 = self.SinD_model.predict(tar_batch)
				try:
					self.act2 = np.concatenate((self.act2,act2), axis = 0)
				except:
					self.act2 = act2

			self.tar_feats = self.act2

		self.eval_DatasetSinID()
		self.print_DatasetSinID()

		return

	def eval_DatasetSinID(self):

		# feats_dim = self.SinD_size * self.SinD_size * 3			
		# for src_batch,tar_batch in zip(self.src_images_dataset, self.tar_images_dataset):
		# 	### Just get the one batch out of it. This is redundant, but consistent with other code
		# 	assert src_batch.shape == (self.src_batch_size, self.SinD_size*self.SinD_size*3)
		# 	assert tar_batch.shape == (self.tar_batch_size, self.SinD_size*self.SinD_size*3)
		# 	break
		##
		src_batch = self.src_feats
		tar_batch = self.tar_feats
		feats_dim = src_batch.shape[1]

		if src_batch.shape[0] > 100000 or tar_batch.shape[0] > 100000:
			device = '/CPU'
		else:
			device = self.device
		with tf.device(device):
			src_cov = tfp.stats.covariance(src_batch)
			tar_cov = tfp.stats.covariance(tar_batch)
			src_sigma, src_U, src_V = tf.linalg.svd(src_cov)
			tar_sigma, tar_U, tar_V = tf.linalg.svd(tar_cov)

		## Cov Diff Op Norm is l_inf of diff of Eigenvalues
		cov_op_norm = tf.linalg.norm(src_sigma - tar_sigma, ord = np.inf)
		# print('cov_op_norm',cov_op_norm)
		## Cov Diff Op Norm is l_inf of diff of Eigenvalues
		cov_F_norm = tf.linalg.norm(src_cov - tar_cov, ord = 'fro', axis = [-2,-1])
		# print('cov_F_norm',cov_F_norm)

		low = 3
		high = int(0.1*feats_dim)
		mid = int((low+high)/2)
		iter_range = range(low,high)
		num_sinD = len(iter_range)
		self.SinD_mat = np.zeros((num_sinD,2))
		## Finding Covariance matrices
		for s in iter_range:
			## r=0, r-1 = -\infty; s = loop_Var, s+1 inferred
			d = s+1 ## d=s-r+1 = s+1 
			num = 2*tf.minimum((d**(0.5))*cov_op_norm, cov_F_norm).numpy()
			# den = src_sigma[s] - src_sigma[s+1] ## technically min(\infty, current_den)
			den = tar_sigma[s] - tar_sigma[s+1] ## technically min(\infty, current_den)
			self.SinD_mat[s-low,0] = num/den
			self.SinD_mat[s-low,1] = s
			if self.mode in ['metrics', 'model_metrics']:
				print("\nDataset Sin Theta Inception Distance for "+str(s)+ " st/nd/rd/th Eigenvalue - "+str(self.SinD_mat[s-low,0]))
				if self.res_flag:
					self.res_file.write("\nDataset Sin Theta Inception Distance for "+str(s)+" st/nd/rd/th Eigenvalues - "+str(self.SinD_mat[s-low,0]))
				if self.res_flag:
					self.res_file.write("\nDataset Sin Theta Inception Distance for "+str(s)+" st/nd/rd/th Eigenvalues - "+str(self.SinD_mat[s-low,0]))


		self.SinID_vec.append([self.SinD_mat, self.total_count.numpy()])
		np.save(self.metricpath+'_SinID_Dataset_'+str(self.total_count.numpy()).zfill(6)+'.npy',np.array(self.SinID_vec))

		# if self.mode in ['metrics', 'model_metrics']:
		# 	print("Dataset Sin Theta Distance for "+str(mid)+ " Eigenvalues - "+str(self.SinD_mat[mid-low,0]))
		# 	if self.res_flag:
		# 		self.res_file.write("Dataset Sin Theta Distance for "+str(mid)+" Eigenvalues - "+str(self.SinD_mat[mid-low,0]))
		# if self.res_flag:
		# 	self.res_file.write("Dataset Sin Theta Distance for "+str(mid)+" Eigenvalues - "+str(self.SinD_mat[mid-low,0]))


		# if self.mode in ['metrics', 'model_metrics']:
		# 	print("Dataset Sin Theta Distance for "+str(high)+" Eigenvalues - "+str(self.SinD_mat[-1,0]))
		# 	if self.res_flag:
		# 		self.res_file.write("Dataset Sin Theta Distance for "+str(high)+" Eigenvalues - "+str(self.SinD_mat[-1,0]))
		# if self.res_flag:
		# 	self.res_file.write("Dataset Sin Theta Distance for "+str(high)+" Eigenvalues - "+str(self.SinD_mat[-1,0]))
		return

	def print_DatasetSinID(self):

		path = self.metricpath
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages

		with PdfPages(path+'SinD_Noise_'+self.noise_data+'_Data_'+self.data+'_plot.pdf') as pdf:
			for SinD_cur,count in self.SinID_vec:
				fig1 = plt.figure(figsize=(3.5, 3.5), dpi=400)
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.get_xaxis().set_visible(True)
				ax1.get_yaxis().set_visible(True)
				SID_vals = SinD_cur[:,0]
				locs = SinD_cur[:,1]
				ax1.plot(locs, SID_vals, color = 'g', linestyle = 'solid', alpha=0.5, linewidth=3)
				ax1.set_xlabel(r'MAX-EIGENVALUE INDEX $s: \{\lambda_1,\lambda_s\}$')
				ax1.set_ylabel(r'$\left\|\sin\Theta(\mathcal{S}_{g}|\mathcal{S}_{d})\right\|_{F}$')
				title = 'Iteration '+str(count)
				plt.title(title, fontsize=8)
				pdf.savefig(fig1, bbox_inches='tight', dpi=400)
				plt.close(fig1)


		with PdfPages(path+'SinD_Noise_'+self.noise_data+'_Data_'+self.data+'_LogPlot.pdf') as pdf:
			for SinD_cur,count in self.SinID_vec:
				fig1 = plt.figure(figsize=(3.5, 3.5), dpi=400)
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.get_xaxis().set_visible(True)
				ax1.get_yaxis().set_visible(True)
				SID_vals = SinD_cur[:,0]
				locs = SinD_cur[:,1]
				ax1.plot(locs, np.abs(np.log10(SID_vals+1e-10)), color = 'g', linestyle = 'solid', alpha=0.5, linewidth=3)
				ax1.set_xlabel(r'MAX-EIGENVALUE INDEX $s: \{\lambda_1,\lambda_s\}$')
				ax1.set_ylabel(r'$\left\|\sin\Theta(\mathcal{S}_{g}|\mathcal{S}_{d})\right\|_{F}$')
				title = 'Iteration '+str(count)
				plt.title(title, fontsize=8)
				pdf.savefig(fig1, bbox_inches='tight', dpi=400)
				plt.close(fig1)

	def update_DID(self):

		def ukiyoe_reader(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([1024,1024,3])
				# image = tf.image.resize(image,[299,299])
				image = tf.image.resize(image,[self.ukiyoe_size,self.ukiyoe_size])
				# This will convert to float values in [0, 1]
				image = tf.subtract(image,127.5)
				image = tf.divide(image,127.5)
			return image

		def tinyimgnet_reader(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([64,64,3])
				# image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				# image = tf.image.resize(image,[299,299])
				image = tf.image.resize(image,[self.tinyimgnet_size,self.tinyimgnet_size])

				# This will convert to float values in [-1, 1]
				image = tf.subtract(image,127.0)
				image = tf.divide(image,127.0)
			return image

		def church_reader(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image  = tf.image.crop_to_bounding_box(image, 0, 0, 256,256)
				# image = tf.image.resize(image,[299,299])
				image = tf.image.resize(image,[self.church_size,self.church_size])
				# This will convert to float values in [0, 1]
				# image = tf.divide(image,255.0)
				image = tf.subtract(image,127.0)
				image = tf.divide(image,127.0)
				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image

		def celeba_reader(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				# image = tf.image.resize(image,[299,299])
				image = tf.image.resize(image,[self.celeba_size,self.celeba_size])

				# This will convert to float values in [-1, 1]
				image = tf.subtract(image,127.5)
				image = tf.divide(image,127.5)
			return image

		def data_preprocess(image):
			with tf.device('/CPU'):
				# image = tf.image.resize(image,[299,299])
				if image.shape[2] == 1:
					image = tf.image.grayscale_to_rgb(image)
			return image


		with tf.device('/CPU'):
			if self.DID_load_flag == 0:
				### First time FID call setup
				self.DID_load_flag = 1

				### Noise is SRC, Data is TAR

				random_points_src = tf.keras.backend.random_uniform([self.DID_num_samples], minval=0, maxval=int(self.dfid_noise_images.shape[0]), dtype='int32', seed=None)
				self.src_images = self.dfid_noise_images[random_points_src]

				random_points_tar = tf.keras.backend.random_uniform([self.DID_num_samples], minval=0, maxval=int(self.dfid_data_images.shape[0]), dtype='int32', seed=None)
				self.tar_images = self.dfid_data_images[random_points_tar]


				self.src_images_dataset = tf.data.Dataset.from_tensor_slices((self.src_images))
				if self.noise_data == 'celeba': 
					self.src_images_dataset = self.src_images_dataset.map(celeba_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.noise_data == 'ukiyoe': 
					self.src_images_dataset = self.src_images_dataset.map(ukiyoe_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.noise_data == 'tinyimgnet': 
					self.src_images_dataset = self.src_images_dataset.map(tinyimgnet_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.noise_data == 'church': 
					self.src_images_dataset = self.src_images_dataset.map(church_reader,num_parallel_calls=int(self.num_parallel_calls))
				else:
					self.src_images_dataset = self.src_images_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))

				self.src_images_dataset = self.src_images_dataset.batch(self.DID_batch_size)


				self.tar_images_dataset = tf.data.Dataset.from_tensor_slices((self.tar_images))
				if self.data == 'celeba': 
					self.tar_images_dataset = self.tar_images_dataset.map(celeba_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.data == 'ukiyoe': 
					self.tar_images_dataset = self.tar_images_dataset.map(ukiyoe_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.data == 'tinyimgnet': 
					self.tar_images_dataset = self.tar_images_dataset.map(tinyimgnet_reader,num_parallel_calls=int(self.num_parallel_calls))
				elif self.data == 'church': 
					self.tar_images_dataset = self.tar_images_dataset.map(church_reader,num_parallel_calls=int(self.num_parallel_calls))
				else:
					self.tar_images_dataset = self.tar_images_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))

				self.tar_images_dataset = self.tar_images_dataset.batch(self.DID_batch_size)


				self.src_images_dataset = self.strategy.experimental_distribute_dataset(self.src_images_dataset)
				self.tar_images_dataset = self.strategy.experimental_distribute_dataset(self.tar_images_dataset)


				# self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(299,299,3), classes=1000)

		### Convert to clean
		total_num_figs = 0 
		i = -1
		for src_batch,tar_batch in zip(self.src_images_dataset, self.tar_images_dataset):
			i += 1
			for j in range(self.DID_batch_size):
				src = src_batch[j]
				tar = tar_batch[j]
				total_num_figs += 1
				if total_num_figs <= self.DID_num_samples:
					tf.keras.preprocessing.image.save_img(self.DIDsrcpath+str(i*self.DID_batch_size + j)+'.png', src, scale=True)
					tf.keras.preprocessing.image.save_img(self.DIDtarpath+str(i*self.DID_batch_size + j)+'.png', tar, scale=True)

		self.eval_CleanDID()
		# with self.strategy.scope():
		# for src_batch,tar_batch in zip(self.src_images_dataset, self.tar_images_dataset):
		# 	act1 = self.FID_model.predict(src_batch)
		# 	act2 = self.FID_model.predict(tar_batch)
		# 	try:
		# 		self.act1 = np.concatenate((self.act1,act1), axis = 0)
		# 		self.act2 = np.concatenate((self.act2,act2), axis = 0)
		# 	except:
		# 		self.act1 = act1
		# 		self.act2 = act2
		# self.eval_DID()
		return

	def eval_CleanDID(self):
		from ops.gan_sid import my_sid as sid

		self.fid = fid.compute_fid(self.DIDsrc_dir, self.DIDtar_dir, mode = "clean")
		self.kid = fid.compute_kid(self.DIDsrc_dir, self.DIDtar_dir, mode = "clean")
		### Fix === remove flags and pass only order or whatever is needed/ 
		self.sid = sid.compute_sid(fdir1 = self.DIDsrc_dir,  fdir2 = self.DIDtar_dir, order = self.SID_order)

		sid.print_and_save_sid(path = self.metricpath, SID_vec = self.sid)
		# np.save(self.metricpath+'_DSID_all'+str(self.total_count.numpy()).zfill(6)+'.npy',np.array(self.SID_vec))

		self.DSID_vec.append([self.sid, self.total_count.numpy()])
		self.DFID_vec.append([self.fid, self.total_count.numpy()])
		self.DKID_vec.append([self.kid, self.total_count.numpy()])

		if self.mode in ['metrics', 'model_metrics']:
			print("Final DFID score - "+str(self.fid))
			if self.res_flag:
				self.res_file.write("Final DFID score - "+str(self.fid))
		if self.res_flag:
			self.res_file.write("DFID score - "+str(self.fid))



		if self.mode in ['metrics', 'model_metrics']:
			print("Final DKID score - "+str(self.kid))
			if self.res_flag:
				self.res_file.write("Final DKID score - "+str(self.kid))
		if self.res_flag:
			self.res_file.write("DKID score - "+str(self.kid))



		if self.mode in ['metrics', 'model_metrics']:
			print("Final DSID sum score - "+str(np.sum(self.sid[:,0])))
			if self.res_flag:
				self.res_file.write("Final DSID sum score - "+str(np.sum(self.sid[:,0])))
		if self.res_flag:
			self.res_file.write("DSID sum score - "+str(np.sum(self.sid[:,0])))


		return

	def eval_DID(self):
		mu1, sigma1 = self.act1.mean(axis=0), cov(self.act1, rowvar=False)
		mu2, sigma2 = self.act2.mean(axis=0), cov(self.act2, rowvar=False)
		# calculate sum squared difference between means
		ssdiff = np.sum((mu1 - mu2)**2.0)
		# calculate sqrt of product between cov
		covmean = sqrtm(sigma1.dot(sigma2))
		# check and correct imaginary numbers from sqrt
		if iscomplexobj(covmean):
			covmean = covmean.real
		# calculate score
		self.fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
		self.DID_vec.append([self.fid, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final DID score - "+str(self.fid))
			if self.res_flag:
				self.res_file.write("Final DID score - "+str(self.fid))

		if self.res_flag:
			self.res_file.write("DID score - "+str(self.fid))

		return


	def update_FID(self):
		#FID Funcs vary per dataset. We therefore call the corresponding child func foundin the arch_*.py files
		## Also save the model files for FID computation in the fututre

		if self.FID_kind == 'clean':
			self.same_images_FID()
			self.eval_CleanFID()
		else:
			if self.FID_kind == 'torch':
				torch.cuda.set_per_process_memory_fraction(0.5)
			eval(self.FID_func)


	def eval_CleanFID(self):
		mode = 'legacy_tensorflow'
		fid_name = self.data+str(self.output_size)+mode
		try:
			fid.test_stats_exists(fid_name, mode = mode)
		except:
			fid.make_custom_stats(fid_name, self.FIDreals_dir, mode=mode)

		if self.data == 'cifar10' and self.testcase != 'single':
			self.fid = fid.compute_fid(self.FIDfakes_dir, dataset_name="cifar10", dataset_res=32, dataset_split="train", mode="clean")
		else:
			# self.fid = 
			self.fid = fid.compute_fid(self.FIDfakes_dir, self.FIDreals_dir, mode = "clean")
			# self.fid = fid.compute_fid(self.FIDfakes_dir, self.FIDreals_dir, mode = "legacy_pytorch")
			# self.fid = fid.compute_fid(self.FIDfakes_dir, self.FIDreals_dir, mode = "legacy_tensorflow")


		self.FID_vec.append([self.fid, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final FID score - "+str(self.fid))
			self.res_file.write("Final FID score - "+str(self.fid))
			# if self.res_flag:
			# 	self.res_file.write("Final FID score - "+str(self.fid))

		if self.res_flag:
			self.res_file.write("FID score - "+str(self.fid))
		return



	def update_ReconFID(self):
		#FID Funcs vary per dataset. We therefore call the corresponding child func foundin the arch_*.py files
		## Also save the model files for FID computation in the fututre

		self.save_images_ReconFID()
		self.eval_ReconFID()


	def eval_ReconFID(self):
		mode = 'legacy_tensorflow'
		# fid_name = self.data+str(self.output_size)+mode
		# try:
		# 	fid.test_stats_exists(fid_name, mode = mode)
		# except:
		# 	fid.make_custom_stats(fid_name, self.FIDreals_dir, mode=mode)

		if self.data == 'cifar10' and self.testcase != 'single':
			self.recon_fid = fid.compute_fid(self.ReconFIDfakes_dir, dataset_name="cifar10", dataset_res=32, dataset_split="train", mode="clean")
		else:
			# self.recon_fid = 
			self.recon_fid = fid.compute_fid(self.ReconFIDfakes_dir, self.ReconFIDreals_dir, mode = "clean")
			# self.recon_fid = fid.compute_fid(self.FIDfakes_dir, self.FIDreals_dir, mode = "legacy_pytorch")
			# self.recon_fid = fid.compute_fid(self.FIDfakes_dir, self.FIDreals_dir, mode = "legacy_tensorflow")


		self.ReconFID_vec.append([self.recon_fid, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final Recon FID score - "+str(self.recon_fid))
			self.res_file.write("Final Recon FID score - "+str(self.recon_fid))
			# if self.res_flag:
			# 	self.res_file.write("Final FID score - "+str(self.recon_fid))

		if self.res_flag:
			self.res_file.write("Recon FID score - "+str(self.recon_fid))
		return


	def eval_FID(self):
		from scipy import linalg
		mu1, sigma1 = self.act1.mean(axis=0), cov(self.act1, rowvar=False)
		mu2, sigma2 = self.act2.mean(axis=0), cov(self.act2, rowvar=False)
		mu1 = np.atleast_1d(mu1)
		mu2 = np.atleast_1d(mu2)
		sigma1 = np.atleast_2d(sigma1)
		sigma2 = np.atleast_2d(sigma2)

		assert mu1.shape == mu2.shape, \
			'Training and test mean vectors have different lengths'
		assert sigma1.shape == sigma2.shape, \
			'Training and test covariances have different dimensions'

		diff = mu1 - mu2

		# Product might be almost singular
		covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
		if not np.isfinite(covmean).all():
			msg = ('fid calculation produces singular product; '
				   'adding %s to diagonal of cov estimates') % eps
			# print(msg)
			offset = np.eye(sigma1.shape[0]) * eps
			covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

		# Numerical error might give slight imaginary component
		if np.iscomplexobj(covmean):
			if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
				m = np.max(np.abs(covmean.imag))
				raise ValueError('Imaginary component {}'.format(m))
			covmean = covmean.real

		tr_covmean = np.trace(covmean)

		self.fid = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
		self.FID_vec.append([self.fid, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final FID score - "+str(self.fid))
			self.res_file.write("Final FID score - "+str(self.fid))
			# if self.res_flag:
			# 	self.res_file.write("Final FID score - "+str(self.fid))

		if self.res_flag:
			self.res_file.write("FID score - "+str(self.fid))

		# import gc
		# del variables
		# gc.collect()
		return

	def eval_OtherFID(self):
		mu1, sigma1 = self.act1.mean(axis=0), cov(self.act1, rowvar=False)
		mu2, sigma2 = self.act2.mean(axis=0), cov(self.act2, rowvar=False)
		# calculate sum squared difference between means
		ssdiff = np.sum((mu1 - mu2)**2.0)
		# calculate sqrt of product between cov
		covmean = sqrtm(np.matmul(sigma1,sigma2))
		# check and correct imaginary numbers from sqrt
		if iscomplexobj(covmean):
			covmean = covmean.real
		# calculate score
		self.fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
		self.FID_vec.append([self.fid, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final FID score - "+str(self.fid))
			self.res_file.write("Final FID score - "+str(self.fid))
			# if self.res_flag:
			# 	self.res_file.write("Final FID score - "+str(self.fid))

		if self.res_flag:
			self.res_file.write("FID score - "+str(self.fid))
		return

	def eval_OOOOOOLD_FID(self):
		mu1, sigma1 = self.act1.mean(axis=0), cov(self.act1, rowvar=False)
		mu2, sigma2 = self.act2.mean(axis=0), cov(self.act2, rowvar=False)
		# calculate sum squared difference between means
		ssdiff = np.sum((mu1 - mu2)**2.0)
		# calculate sqrt of product between cov
		covmean = sqrtm(sigma1.dot(sigma2))
		# check and correct imaginary numbers from sqrt
		if iscomplexobj(covmean):
			covmean = covmean.real
		# calculate score
		self.fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
		self.FID_vec.append([self.fid, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final FID score - "+str(self.fid))
			self.res_file.write("Final FID score - "+str(self.fid))
			# if self.res_flag:
			# 	self.res_file.write("Final FID score - "+str(self.fid))

		if self.res_flag:
			self.res_file.write("FID score - "+str(self.fid))
		return


	def print_FID_ACGAN(self):

		np.save(self.metricpath+'FID_even.npy',np.array(self.FID_vec_even))
		np.save(self.metricpath+'FID_odd.npy',np.array(self.FID_vec_odd))
		np.save(self.metricpath+'FID_sharp.npy',np.array(self.FID_vec_sharp))
		np.save(self.metricpath+'FID_single.npy',np.array(self.FID_vec_single))

		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
			
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages

		vals = list(np.array(self.FID_vec_even)[:,0])
		locs = list(np.array(self.FID_vec_even)[:,1])

		with PdfPages(path+'FID_plot_even.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

		vals = list(np.array(self.FID_vec_odd)[:,0])
		locs = list(np.array(self.FID_vec_odd)[:,1])

		with PdfPages(path+'FID_plot_odd.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

		vals = list(np.array(self.FID_vec_sharp)[:,0])
		locs = list(np.array(self.FID_vec_sharp)[:,1])

		with PdfPages(path+'FID_plot_sharp.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

		vals = list(np.array(self.FID_vec_single)[:,0])
		locs = list(np.array(self.FID_vec_single)[:,1])

		with PdfPages(path+'FID_plot_single.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

	def print_FID_Rumi(self):

		np.save(self.metricpath+'FID_pos.npy',np.array(self.FID_vec_pos))
		np.save(self.metricpath+'FID_neg.npy',np.array(self.FID_vec_neg))

		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages

		vals = list(np.array(self.FID_vec_pos)[:,0])
		locs = list(np.array(self.FID_vec_pos)[:,1])

		with PdfPages(path+'FID_plot_pos.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

		vals = list(np.array(self.FID_vec_neg)[:,0])
		locs = list(np.array(self.FID_vec_neg)[:,1])

		with PdfPages(path+'FID_plot_neg.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

	def print_FID(self):
		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages

		vals = list(np.array(self.FID_vec)[:,0])
		locs = list(np.array(self.FID_vec)[:,1])

		with PdfPages(path+'FID_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

	def update_KID(self):
		#FID Funcs vary per dataset. We therefore call the corresponding child func foundin the arch_*.py files
		## Also save the model files for FID computation in the fututre

		if self.KID_kind == 'clean':
			if 'FID' not in self.metrics:
				self.same_images_FID()
			self.eval_CleanKID()
		# else:
		# 	if self.FID_kind == 'torch':
		# 		torch.cuda.set_per_process_memory_fraction(0.5)
		# 	eval(self.KID_func)


	def eval_CleanKID(self):
		# mode = 'legacy_tensorflow'
		# fid_name = self.data+str(self.output_size)+mode
		# try:
		# 	fid.test_stats_exists(fid_name, mode = mode)
		# except:
		# 	fid.make_custom_stats(fid_name, self.FIDreals_dir, mode=mode)

		# if self.data == 'cifar10' and self.testcase != 'single':
		# 	self.fid = fid.compute_fid(self.FIDfakes_dir, dataset_name="cifar10", dataset_res=32, dataset_split="train", mode="clean")
		# else:
		# 	# self.fid = 
		# 	# self.fid = fid.compute_fid(self.FIDfakes_dir, self.FIDreals_dir, mode = "clean")
		# 	# self.fid = fid.compute_fid(self.FIDfakes_dir, self.FIDreals_dir, mode = "legacy_pytorch")
		self.kid = fid.compute_kid(self.FIDfakes_dir, self.FIDreals_dir)

		self.KID_vec.append([self.kid, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final KID score - "+str(self.kid))
			self.res_file.write("Final KID score - "+str(self.kid))
			# if self.res_flag:
			# 	self.res_file.write("Final FID score - "+str(self.fid))

		if self.res_flag:
			self.res_file.write("KID score - "+str(self.kid))
		return

	def print_KID(self):
		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages

		vals = list(np.array(self.KID_vec)[:,0])
		locs = list(np.array(self.KID_vec)[:,1])

		with PdfPages(path+'KID_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'KID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)


	def eval_recon(self):
		# print('Evaluating Recon Loss\n')
		mse = tf.keras.losses.MeanSquaredError()
		for image_batch in self.recon_dataset:
			# print("batch 1\n")
			recon_images = self.Decoder(self.Encoder(image_batch, training= False) , training = False)
			if self.topic in ['PolyGAN', 'WAEMMD']:
				try:
					recon_loss += tf.reduce_mean(tf.abs(image_batch - recon_images))
					recon_loss *= 0.5
				except:
					recon_loss = tf.reduce_mean(tf.abs(image_batch - recon_images))
			else:
				try:
					recon_loss += 0.75*tf.reduce_mean(tf.abs(image_batch - recon_images)) + 0.25*mse(image_batch,recon_images)
					recon_loss *= 0.5
				except:
					recon_loss = 0.75*tf.reduce_mean(tf.abs(image_batch - recon_images)) + 0.25*mse(image_batch,recon_images)
				# try:
				# 	recon_loss = 0.5*(recon_loss) + 0.25*tf.reduce_mean(tf.abs(image_batch - recon_images)) + 0.75*(mse(image_batch,recon_images))
				# except:
				# 	recon_loss = 0.5*tf.reduce_mean(tf.abs(image_batch - recon_images)) + 1.5*(mse(image_batch,recon_images))

		print(recon_loss)
		self.recon_vec.append([recon_loss, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final Reconstruction error - "+str(recon_loss))
			if self.res_flag:
				self.res_file.write("Final Reconstruction error - "+str(recon_loss))

		if self.res_flag:
			self.res_file.write("Reconstruction error - "+str(recon_loss))

	def print_recon(self):
		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
		if self.colab==1 or self.latex_plot_flag==0:
			from matplotlib.backends.backend_pdf import PdfPages
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		vals = list(np.array(self.recon_vec)[:,0])
		locs = list(np.array(self.recon_vec)[:,1])

		with PdfPages(path+'recon_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'Reconstruction Error')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)


	def KLD_sample_estimate(self,P,Q):
		def skl_estimator(s1, s2, k=1):
			from sklearn.neighbors import NearestNeighbors
			### Code Courtesy nheartland 
			### URL : https://github.com/nhartland/KL-divergence-estimators/blob/master/src/knn_divergence.py
			""" KL-Divergence estimator using scikit-learn's NearestNeighbours
			s1: (N_1,D) Sample drawn from distribution P
			s2: (N_2,D) Sample drawn from distribution Q
			k: Number of neighbours considered (default 1)
			return: estimated D(P|Q)
			"""
			n, m = len(s1), len(s2)
			d = float(s1.shape[1])
			D = np.log(m / (n - 1))

			s1_neighbourhood = NearestNeighbors(k+1, 10).fit(s1)
			s2_neighbourhood = NearestNeighbors(k, 10).fit(s2)

			for p1 in s1:
				s1_distances, indices = s1_neighbourhood.kneighbors([p1], k+1)
				s2_distances, indices = s2_neighbourhood.kneighbors([p1], k)
				rho = s1_distances[0][-1]
				nu = s2_distances[0][-1]
				D += (d/n)*np.log(nu/rho)
			return D
		KLD = skl_estimator(P,Q)
		self.KLD_vec.append([KLD, self.total_count.numpy()])
		return

	def KLD_Gaussian(self,P,Q):

		def get_mean(f):
			return np.mean(f,axis = 0).astype(np.float64)
		def get_cov(f):
			return np.cov(f,rowvar = False).astype(np.float64)
		def get_std(f):
			return np.std(f).astype(np.float64)
		try:
			if self.data == 'g1':
				Distribution = tfd.Normal
				P_dist = Distribution(loc=get_mean(P), scale=get_std(P))
				Q_dist = Distribution(loc=get_mean(Q), scale=get_std(Q))
			else:
				Distribution = tfd.MultivariateNormalFullCovariance
				P_dist = Distribution(loc=get_mean(P), covariance_matrix=get_cov(P))
				Q_dist = Distribution(loc=get_mean(Q), covariance_matrix=get_cov(Q))
	
			self.KLD_vec.append([P_dist.kl_divergence(Q_dist).numpy(), self.total_count.numpy()])
		except:
			print("KLD error - Falling back to prev value")
			try:
				self.KLD_vec.append([self.KLD_vec[-1]*0.9, self.total_count.numpy()])
			except:
				self.KLD_vec.append([0, self.total_count.numpy()])
			# print('KLD: ',self.KLD_vec[-1])
		return


	def update_KLD(self):
		if self.topic == 'ELeGANt' and self.gan == 'WGAN':
			self.KLD_func(self.reals,self.fakes)
		elif self.gan == 'WAE':
			self.KLD_func(self.fakes_enc,self.reals_enc)
		else:
			self.KLD_func(self.reals,self.fakes)

	def print_KLD(self):
		path = self.metricpath
		if self.colab==1 or self.latex_plot_flag==0:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		vals = list(np.array(self.KLD_vec)[:,0])
		locs = list(np.array(self.KLD_vec)[:,1])
		# if self.topic == 'ELeGANt':
		# 	if self.loss == 'FS' and self.latent_kind == 'AE':
		# 		locs = list(np.array(self.KLD_vec)[:,1] - self.AE_steps)
		

		with PdfPages(path+'KLD_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'KL Divergence Vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

	def update_W22(self):
		if self.topic == 'ELeGANt' and self.gan == 'WGAN':
			if self.data in ['g1','g2']:
				self.eval_W22(self.reals,self.fakes)
			else:
				self.estimate_W22(self.reals,self.fakes)
		elif self.gan == 'WAE':
			self.W22_from_model_latents()
		else:
			if self.data in ['g1','g2', 'gN']:
				self.eval_W22(self.reals,self.fakes)
			else:
				self.estimate_W22(self.reals,self.fakes)

	def W22_from_model_latents(self):
		fakes_enc = self.get_noise(self.batch_size)
		for image_batch in self.train_dataset:
			reals_enc = self.Encoder(image_batch, training = True)
			break
		self.eval_W22(fakes_enc,reals_enc)
		return

	def eval_W22(self,act1,act2):
		if tf.is_tensor(act1):
			act1 = act1.numpy()
		if tf.is_tensor(act2):
			act2 = act2.numpy()

		mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
		mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
		# print(act1.numpy(),act2.numpy())
		# print(mu1, sigma1)
		# print(mu2, sigma2)
		# calculate sum squared difference between means
		ssdiff = np.sum((mu1 - mu2)**2.0)
		# calculate sqrt of product between cov
		if self.data not in ['g1', 'gmm2']:
			covmean = sqrtm(sigma1.dot(sigma2))
		else:
			covmean = np.sqrt(sigma1*sigma2)
		# check and correct imaginary numbers from sqrt
		if iscomplexobj(covmean):
			covmean = covmean.real
		# calculate score
		if self.data not in ['g1', 'gmm2']:
			self.W22_val = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
		else:
			self.W22_val = ssdiff + sigma1 + sigma2 - 2.0 * covmean
		self.W22_vec.append([self.W22_val, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("\n Final W22 score - "+str(self.W22_val))
			if self.gan == 'WAE':
				self.res_file.write("\n Final W22 score at Iteration "+str(self.total_count.numpy())+" - "+str(self.W22_val))
			else:
				self.res_file.write("\n Final W22 score - "+str(self.W22_val))
			# if self.res_flag:
			# 	self.res_file.write("Final FID score - "+str(self.fid))

		if self.res_flag:
			if self.gan == 'WAE':
				self.res_file.write("\n W22 score at Iteration "+str(self.total_count.numpy())+" - "+str(self.W22_val))
			else:
				self.res_file.write("\n W22 score - "+str(self.W22_val))
		return

	def estimate_W22(self,target_sample, gen_sample, q=2, p=2):
		target_sample = tf.cast(target_sample, dtype = 'float32').numpy()
		gen_sample = tf.cast(gen_sample, dtype = 'float32').numpy()
		target_weights = np.ones(target_sample.shape[0]) / target_sample.shape[0]
		gen_weights = np.ones(gen_sample.shape[0]) / gen_sample.shape[0]

		x = target_sample[:, None, :] - gen_sample[None, :, :]

		# print(x.shape)
		# M = torch.linalg.norm(target_sample[:, None, :] - gen_sample[None, :, :], dim=2, ord=q)**p / p
		# M = tf.cast(tf.norm(target_sample[:, None, :] - gen_sample[None, :, :], ord=q)**p / p, dtype = 'float')
		M = tf.norm(x, ord=q, axis = 2)**p / p
		# print(target_sample.shape, gen_sample.shape, M.shape)
		T = ot.emd2(target_weights, gen_weights, M.numpy())
		self.W22_val = W = ((M.numpy() * T).sum())**(1. / p)

		self.W22_vec.append([self.W22_val, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final W22 score - "+str(self.W22_val))
			self.res_file.write("Final W22 score - "+str(self.W22_val))
			# if self.res_flag:
			# 	self.res_file.write("Final FID score - "+str(self.fid))

		if self.res_flag:
			self.res_file.write("W22 score - "+str(self.W22_val))
		return


	def print_W22(self):
		path = self.metricpath
		if self.colab==1 or self.latex_plot_flag==0:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		vals = list(np.array(self.W22_vec)[:,0])
		locs = list(np.array(self.W22_vec)[:,1])
		# if self.topic == 'ELeGANt':
		# 	if self.loss == 'FS' and self.latent_kind == 'AE':
		# 		locs = list(np.array(self.KLD_vec)[:,1] - self.AE_steps)
		

		with PdfPages(path+'W22_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = r'$\mathcal{W}^{2,2}(p_d,p_g)$ Vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

	def update_Lambda(self):
		# if self.loss == 'FS':
		# 	lamb_val = self.lamb.numpy()
		# else:
		# 	lamb_val = 
		self.lambda_vec.append([self.lamb.numpy(),self.total_count.numpy()])
		return

	def print_Lambda(self):
		path = self.metricpath
		if self.colab==1 or self.latex_plot_flag==0:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size": 12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		# lbasis  = np.expand_dims(np.array(np.arange(0,len(self.lambda_vec))),axis=1)
		vals = list(np.array(self.lambda_vec)[:,0])
		locs = list(np.array(self.lambda_vec)[:,1])

		with PdfPages(path+'Lambda_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'Lambda Vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

	def update_loss(self):
		self.Dloss_vec.append([self.D_loss.numpy(),self.total_count.numpy()])
		self.Gloss_vec.append([self.G_loss.numpy(),self.total_count.numpy()])
		if self.gan in ['WAE'] or self.topic in ['MMDGAN']:
			self.AEloss_vec.append([self.AE_loss.numpy(),self.total_count.numpy()])
		return

	def print_loss(self):
		path = self.metricpath
		if self.colab==1 or self.latex_plot_flag==0:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size": 12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		# lbasis  = np.expand_dims(np.array(np.arange(0,len(self.lambda_vec))),axis=1)
		vals = list(np.array(self.Dloss_vec)[:,0])
		locs = list(np.array(self.Dloss_vec)[:,1])

		with PdfPages(path+'Dloss_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'D_loss Vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

		vals = list(np.array(self.Gloss_vec)[:,0])
		locs = list(np.array(self.Gloss_vec)[:,1])

		with PdfPages(path+'Gloss_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'G_loss Vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

		if self.gan in ['WAE'] or self.topic in ['MMDGAN']:
			vals = list(np.array(self.AEloss_vec)[:,0])
			locs = list(np.array(self.AEloss_vec)[:,1])

			with PdfPages(path+'Dloss_plot.pdf') as pdf:

				fig1 = plt.figure(figsize=(3.5, 3.5))
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.get_xaxis().set_visible(True)
				ax1.get_yaxis().set_visible(True)
				ax1.plot(locs,vals, c='r',label = 'AE_loss Vs. Iterations')
				ax1.legend(loc = 'upper right')
				pdf.savefig(fig1)
				plt.close(fig1)
	# def update_LapD(self):
	# 	# if self.loss == 'FS':
	# 	# 	lamb_val = self.lamb.numpy()
	# 	# else:
	# 	# 	lamb_val = 
	# 	self.lapD_vec.append([self.lamb.numpy(),self.total_count.numpy()])
	# 	return

	# def print_LapD(self):
	# 	path = self.metricpath
	# 	if self.colab==1 or self.latex_plot_flag==0:
	# 		from matplotlib.backends.backend_pdf import PdfPages
	# 		plt.rc('text', usetex=False)
	# 	else:
	# 		from matplotlib.backends.backend_pgf import PdfPages
	# 		plt.rcParams.update({
	# 			"pgf.texsystem": "pdflatex",
	# 			"font.family": "helvetica",  # use serif/main font for text elements
	# 			"font.size": 12,
	# 			"text.usetex": True,     # use inline math for ticks
	# 			"pgf.rcfonts": False,    # don't setup fonts from rc parameters
	# 		})

	# 	# lbasis  = np.expand_dims(np.array(np.arange(0,len(self.lambda_vec))),axis=1)
	# 	vals = list(np.array(self.lambda_vec)[:,0])
	# 	locs = list(np.array(self.lambda_vec)[:,1])

	# 	with PdfPages(path+'Lambda_plot.pdf') as pdf:

	# 		fig1 = plt.figure(figsize=(3.5, 3.5))
	# 		ax1 = fig1.add_subplot(111)
	# 		ax1.cla()
	# 		ax1.get_xaxis().set_visible(True)
	# 		ax1.get_yaxis().set_visible(True)
	# 		ax1.plot(locs,vals, c='r',label = 'Lambda Vs. Iterations')
	# 		ax1.legend(loc = 'upper right')
	# 		pdf.savefig(fig1)
	# 		plt.close(fig1)

	def print_GradGrid(self):

		path = self.metricpath + str(self.total_count.numpy()).zfill(6) + '_'

		if self.colab==1 or self.latex_plot_flag==0:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size": 12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		
		from itertools import product as cart_prod

		x = np.arange(self.MIN,self.MAX+0.1,0.1)
		y = np.arange(self.MIN,self.MAX+0.1,0.1)

		# X, Y = np.meshgrid(x, y)
		prod = np.array([p for p in cart_prod(x,repeat = 2)])
		# print(x,prod)

		X = prod[:,0]
		Y = prod[:,1]

		# print(prod,X,Y)
		# print(XXX)

		with tf.GradientTape() as disc_tape:
			prod = tf.cast(prod, dtype = 'float32')
			disc_tape.watch(prod)
			if self.loss == 'FS':
				d_vals =self.discriminator_B(self.discriminator_A(prod,training = False),training = False)
			elif (self.loss == 'RBF' and self.topic != 'CoulombGAN') or (self.loss == 'score' and self.topic == 'ScoreGAN'):
				d_vals = self.discriminator_RBF(prod,training = False)
			elif self.gan == 'Langevin':
				_,_,d_vals,_ = self.discriminator(prod,training = False)
			else:
				d_vals = self.discriminator(prod,training = False)
		grad_vals = disc_tape.gradient(d_vals, [prod])[0]

		# print(np.array(d_vals).shape)
		# exit(0)

		#Flag to control normalization of D(x) values for printing on the contour plot
		# if self.loss == 'FS' and self.total_count.numpy()>1500:
		# 	Normalize_Flag = True
		# else:
		Normalize_Flag = True
		# if self.gan not in ['Langevin']:
		try:
			if Normalize_Flag and ((min(d_vals[0]) <= -2) or (max(d_vals[0]) >= 2)):
				### IF NORMALIZATION IS NEEDED

				d_vals_sub = d_vals[0] - min(d_vals[0])
				d_vals_norm = d_vals_sub/max(d_vals_sub)
				d_vals_norm -= 0.5
				d_vals_norm *= 3

				
				# d_vals_new = np.expand_dims(np.array(d_vals_norm),axis = 1)
				d_vals_new = np.reshape(d_vals_norm,(x.shape[0],y.shape[0])).transpose()


				# d_vals_norm = np.expand_dims(np.array(d_vals_sub/max(d_vals_sub)),axis = 1)
				# d_vals_new = np.subtract(d_vals_norm,0.5)
				# d_vals_new = np.multiply(d_vals_new,3.)
				# print(d_vals_new)
			else:
				### IF NORMALIZATION IS NOT NEEDED
				d_vals_norm = d_vals[0]
				d_vals_new = np.reshape(d_vals_norm,(x.shape[0],y.shape[0])).transpose()
		except:
			d_vals_new = np.reshape(d_vals,(x.shape[0],y.shape[0])).transpose()
		# else:
		# 	d_vals_new = np.reshape(np.array(d_vals)[:,0:1],(x.shape[0],y.shape[0])).transpose()
		# print(d_vals_new)
		dx = grad_vals[:,1]
		dy = grad_vals[:,0]
		# print(XXX)
		n = -1
		color_array = np.sqrt(((dx-n)/2)**2 + ((dy-n)/2)**2)

		# with PdfPages(path+'GradGrid_plot.pdf') as pdf:

			# fig1 = plt.figure(figsize=(3.5, 3.5))
			# ax1 = fig1.add_subplot(111)
			# ax1.cla()
			# ax1.get_xaxis().set_visible(True)
			# ax1.get_yaxis().set_visible(True)
			# ax1.set_xlim([self.MIN,self.MAX])
			# ax1.set_ylim(bottom=self.MIN,top=self.MAX)
			# ax1.quiver(X,Y,dx,dy,color_array)
			# ax1.scatter(self.reals[:1000,0], self.reals[:1000,1], c='r', linewidth = 1, label='Real Data', marker = '.', alpha = 0.1)
			# ax1.scatter(self.fakes[:1000,0], self.fakes[:1000,1], c='g', linewidth = 1, label='Fake Data', marker = '.', alpha = 0.1)
			# pdf.savefig(fig1)
			# plt.close(fig1)

		# with PdfPages(path+'Contourf_plot.pdf') as pdf:

			# fig1 = plt.figure(figsize=(3.5, 3.5))
			# ax1 = fig1.add_subplot(111)
			# ax1.cla()
			# ax1.get_xaxis().set_visible(False)
			# ax1.get_yaxis().set_visible(False)
			# ax1.set_xlim([self.MIN,self.MAX])
			# ax1.set_ylim([self.MIN,self.MAX])
			# cs = ax1.contourf(x,y,d_vals_new,alpha = 0.5, levels = list(np.arange(-1.5,1.5,0.1)), extend = 'both' )
			# ax1.scatter(self.reals[:,0], self.reals[:,1], c='r', linewidth = 1, marker = '.', alpha = 0.75)
			# ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.75)
			# # cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
			# # Can be used with figure size (2,10) to generate a colorbar with diff colors as plotted
			# # Good for a table wit \multicol{5}
			# # cbar = fig1.colorbar(cs, aspect = 40, shrink=1., ticks = [0, 1.0], orientation = 'horizontal')
			# # cbar.ax.set_xticklabels(['Min', 'Max'])
			# # # cbar.set_ticks_position(['bottom', 'top'])
			# pdf.savefig(fig1)
			# plt.close(fig1)

		# with PdfPages(path+'Contourf_plot_cBar.pdf') as pdf:

			# fig1 = plt.figure(figsize=(8, 8))
			# ax1 = fig1.add_subplot(111)
			# ax1.cla()
			# ax1.get_xaxis().set_visible(False)
			# ax1.get_yaxis().set_visible(False)
			# ax1.set_xlim([self.MIN,self.MAX])
			# ax1.set_ylim([self.MIN,self.MAX])
			# cs = ax1.contourf(x,y,d_vals_new,alpha = 0.5, levels = list(np.arange(-1.5,1.6,0.1)), extend = 'both' )
			# # cs = ax1.contourf(x,y,d_vals_new,alpha = 0.5, extend = 'both' )
			# ax1.scatter(self.reals[:,0], self.reals[:,1], c='r', linewidth = 1, marker = '.', alpha = 0.75)
			# ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.75)
			# # cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
			# # Can be used with figure size (10,2) to generate a colorbar with diff colors as plotted
			# # Good for a table wit \multicol{5}
			# cbar = fig1.colorbar(cs, aspect = 40, shrink=1., ticks = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5], orientation = 'horizontal')
			# cbar.ax.set_xticklabels(['$-1.5$', '$-1$', '$-0.5$', '$0$', '$0.5$', '$1$', '$1.5$'])
			# # # cbar.set_ticks_position(['bottom', 'top'])
			# pdf.savefig(fig1)
			# plt.close(fig1)

		# with PdfPages(path+'Contour_plot.pdf') as pdf:

		# 	fig1 = plt.figure(figsize=(3.5, 3.5))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.get_xaxis().set_visible(False)
		# 	ax1.get_yaxis().set_visible(False)
		# 	ax1.set_xlim([self.MIN,self.MAX])
		# 	ax1.set_ylim([self.MIN,self.MAX])
		# 	ax1.contour(x,y,d_vals_new,12,linewidths = 0.8, alpha = 0.5 )
		# 	ax1.scatter(self.reals[:,0], self.reals[:,1], c='r', linewidth = 1, marker = '.', alpha = 0.7)
		# 	ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.7)
		# 	# cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
		# 	plt.tight_layout()
		# 	pdf.savefig(fig1)
		# 	plt.close(fig1)

		dnumpy = d_vals_norm.numpy()
		D_pos = np.maximum(0, dnumpy)
		D_neg = np.minimum(0, dnumpy)
		print(D_pos,D_neg)
		# d_vals_new_inv = (d_vals_new+0.1)**(-1)

		d_vals_inv  = np.sqrt(D_pos) - np.sqrt(-D_neg)
		# print(d_vals_inv)
		# exit(0)
		d_vals_inv_sub = d_vals_inv - min(d_vals_inv)
		d_vals_inv_norm = d_vals_inv_sub/max(d_vals_inv_sub)
		d_vals_inv_norm -= 0.5
		d_vals_inv_norm *= 3

		d_vals_inv_new = np.reshape(d_vals_inv_norm,(x.shape[0],y.shape[0])).transpose()

		###### png contours for Qualcomm Video
		# fig1 = plt.figure(figsize=(5.5, 5.5))
		### Thesis
		fig1 = plt.figure(figsize=(7.5, 7.5))
		ax1 = fig1.add_subplot(111)
		ax1.cla()
		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)
		if self.noise_kind == 'ThesisMoon':
			ax1.set_xlim([self.MIN_x,self.MAX_x])
			ax1.set_ylim([self.MIN_y,self.MAX_y])
		else:
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim([self.MIN,self.MAX])
		if self.loss == 'FS' and self.total_count.numpy() >=12500:
			# ax1.contour(x,y,d_vals_new,linewidths = 0.8,levels = list(np.arange(-15,15,0.5)),alpha = 0.5 )
			ax1.contour(x,y,d_vals_new,5,linewidths = 1.2, alpha = 0.5 )
			cs = ax1.contourf(x,y,d_vals_new,alpha = 0.5, levels = 5, extend = 'both')
		else:
			# ax1.contour(x,y,d_vals_new,15,linewidths = 1.2, alpha = 0.5, cmap='PRGn' )
			# cs = ax1.contourf(x,y,d_vals_new,alpha = 0.5, levels = 15, extend = 'both', cmap='PRGn')
			#### For un-sqrt'd
			# ax1.contour(x,y,d_vals_new,50,linewidths = 0.75, alpha = 0.35, cmap='gray' )
			# ax1.contour(x,y,d_vals_new,50,linewidths = 1.2, alpha = 0.75, cmap='PRGn' )
			# cs = ax1.contourf(x,y,d_vals_new, alpha = 0.75, levels = 250, extend = 'both', cmap='PRGn')
			#### For sqrt'd
			ax1.contour(x,y,d_vals_inv_new,10,linewidths = 1.75, alpha = 0.35, cmap='ocean' )
			ax1.contour(x,y,d_vals_inv_new,10,linewidths = 4.25, alpha = 0.5, cmap='PRGn' )
			cs = ax1.contourf(x,y,d_vals_inv_new, alpha = 0.5, levels = 50, extend = 'both', cmap='PRGn')

		if self.gan in ['Langevin']:
			ax1.scatter(self.reals_enc[:,0], self.reals_enc[:,1], c='purple', linewidth = 1, marker = '.', alpha = 0.7)
			ax1.scatter(self.fakes_enc[:,0], self.fakes_enc[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.7)
		else:
			ax1.scatter(self.reals[:,0], self.reals[:,1], c='purple', linewidth = 0.5, marker = '.', alpha = 0.7)
			ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 0.5, marker = '.', alpha = 0.7)
		# cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
		plt.tight_layout()
		# pdf.savefig(fig1)
		fig1.savefig(path+'Contour_plot.png', dpi = 600, bbox_inches="tight")
		plt.close(fig1)


		with PdfPages(path+'_Thesis_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim([self.MIN,self.MAX])
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim([self.MIN,self.MAX])
			ax1.contour(x,y,d_vals_inv_new,25,linewidths = 1.25, alpha = 1., cmap='PRGn' )
			# ax1.contour(x,y,d_vals_new,25,linewidths = 0.5, alpha = 0.75, cmap='PRGn' )
			cs = ax1.contourf(x,y,d_vals_new, alpha = 0.75, levels = 250, extend = 'both', cmap='PRGn')
			
			ax1.scatter(self.reals[:,0], self.reals[:,1], c='purple', linewidth = 1, marker = '.', alpha = 0.9)
			ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.9)
			# cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
			plt.tight_layout()
			pdf.savefig(fig1)
			plt.close(fig1)

		# if self.total_count.numpy() >2:
		# 	exit(0)


		###### Surface png for Qualcomm Video
		# fig1 = plt.figure(figsize=(3.5, 3.5))
		# ax1 = plt.axes(projection='3d')
		# ax1.cla()
		# # ax1.set_axis_off()
		# # ax1.set_xlim([self.MIN,self.MAX])
		# # ax1.set_ylim([self.MIN,self.MAX])
		# # ax1.contour3D(x,y,d_vals_new, 250, cmap='viridis')
		# ax1.plot_surface(x, y, d_vals_new, cmap='viridis', edgecolor='none',rstride=10,cstride=10,alpha=0.5,antialiased=False)
		# # ax1.contour(x,y,d_vals_new,12,linewidths = 0.8, alpha = 0.5 )
		# ax1.view_init(0, 290)
		# # cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
		# for line in ax1.xaxis.get_ticklines():
		# 	line.set_visible(False)
		# for line in ax1.yaxis.get_ticklines():
		# 	line.set_visible(False)
		# for line in ax1.zaxis.get_ticklines():
		# 	line.set_visible(False)
		# # plt.tight_layout()
		# # pdf.savefig(fig1)
		# fig1.savefig(path+'Surf_plot.png', dpi = 300, bbox_inches="tight")
		# plt.close(fig1)

	def update_sharpness(self):

		self.baseline_sharpness, self.sharpness = self.eval_sharpness()
		if self.mode in ['metrics', 'model_metrics']:
			print("\n Final Sharpness score - "+str(self.sharpness))
			print(" \n Baseline Sharpness score - "+str(self.baseline_sharpness))
			if self.res_flag:
				self.res_file.write("\n Final Sharpness score - "+str(self.sharpness))
				self.res_file.write("\n Baseline Sharpness score - "+str(self.baseline_sharpness))

		if self.res_flag:
			self.res_file.write("\n Sharpness score - "+str(self.sharpness))
			self.res_file.write("\n Baseline Sharpness score - "+str(self.baseline_sharpness))

	def find_sharpness(self,images):
		sample_size = tf.shape(images)[0]
		# First convert to greyscale
		if tf.shape(images)[-1] > 1:
			# We have RGB
			images = tf.image.rgb_to_grayscale(images)
		# Next convolve with the Laplace filter
		lap_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
		lap_filter = lap_filter.reshape([3, 3, 1, 1])
		conv = tf.nn.conv2d(images, lap_filter,
							strides=[1, 1, 1, 1], padding='VALID')
		_, lapvar = tf.nn.moments(conv, axes=[1, 2, 3])
		return lapvar


	def find_sharpness_old(self,input_ims):
		def laplacian(input, ksize, mode=None, constant_values=None, name=None):
			"""
			Apply Laplacian filter to image.
			Args:
			  input: A 4-D (`[N, H, W, C]`) Tensor.
			  ksize: A scalar Tensor. Kernel size.
			  mode: A `string`. One of "CONSTANT", "REFLECT", or "SYMMETRIC"
				(case-insensitive). Default "CONSTANT".
			  constant_values: A `scalar`, the pad value to use in "CONSTANT"
				padding mode. Must be same type as input. Default 0.
			  name: A name for the operation (optional).
			Returns:
			  A 4-D (`[N, H, W, C]`) Tensor.
			"""

			input = tf.convert_to_tensor(input)
			ksize = tf.convert_to_tensor(ksize)

			tf.debugging.assert_none_equal(tf.math.mod(ksize, 2), 0)

			ksize = tf.broadcast_to(ksize, [2])

			total = ksize[0] * ksize[1]
			index = tf.reshape(tf.range(total), ksize)
			g = tf.where(
				tf.math.equal(index, tf.math.floordiv(total - 1, 2)),
				tf.cast(1 - total, input.dtype),
				tf.cast(1, input.dtype),
			)

			# print(g)

			# input = pad(input, ksize, mode, constant_values)

			channel = tf.shape(input)[-1]
			shape = tf.concat([ksize, tf.constant([1, 1], ksize.dtype)], axis=0)
			g = tf.reshape(g, shape)
			shape = tf.concat([ksize, [channel], tf.constant([1], ksize.dtype)], axis=0)
			g = tf.broadcast_to(g, shape)
			return tf.nn.depthwise_conv2d(input, g, [1, 1, 1, 1], padding="VALID")

		# import tensorflow_io as tfio
		lap_img = laplacian(input_ims,3)
		if input_ims.shape[3] == 3:
			reduction_axis = [1,2,3]
		else:
			reduction_axis = [1,2]
		var = tf.square(tf.math.reduce_std(lap_img, axis = reduction_axis))
		var_out = np.mean(var)
		# print(var_out)
		return var_out


	# def print_ClassProbs(self):
	# 	path = self.metricpath
	# 	if self.latex_plot_flag:
	# 		from matplotlib.backends.backend_pgf import PdfPages
	# 		plt.rcParams.update({
	# 			"pgf.texsystem": "pdflatex",
	# 			"font.family": "helvetica",  # use serif/main font for text elements
	# 			"font.size":12,
	# 			"text.usetex": True,     # use inline math for ticks
	# 			"pgf.rcfonts": False,    # don't setup fonts from rc parameters
	# 		})
	# 	else:
	# 		from matplotlib.backends.backend_pdf import PdfPages
	# 		plt.rc('text', usetex = False)

	# 	# vals = list(np.array(self.class_prob_vec)[:,0][-1])
	# 	# locs = list(np.arange(10))

	# 	if self.classifier_load_flag:
	# 		self.classifier_load_flag = 0
	# 		try:
	# 			self.classifier = tf.keras.models.load_model('PFMNIST_Classifer.h5')
	# 		except:
	# 			os.system("python PartialFMNIST_Classifer.py") ### Might not work if env is bad
	# 			self.classifier = tf.keras.models.load_model('PFMNIST_Classifer.h5')

	# 	num_samples = 90000
	# 	batch_size = self.batch_size
	# 	num_batches = num_samples//batch_size
	# 	cur_batch = 0
	# 	num_ones = 0

	# 	if self.topic == 'SpiderGAN':
	# 		while cur_batch < num_batches:
	# 			for noise_batch in self.noise_dataset:
	# 				cur_batch += 1
	# 				images = self.generator(noise_batch, training = False)
	# 				predictions = self.classifier.predict(images)
	# 				# print(predictions)
	# 				label = np.argmax(predictions, axis = 1)
	# 				# print(label)
	# 				try:
	# 					try:
	# 						# ones = np.concatentate((ones,images[np.where(label == 10)[0]]), axis = 0 )
	# 						ones_prob = np.concatentate((ones_prob,predictions[np.where(label == 10)[0]]), axis = 0)
	# 						ones = np.concatentate((ones,label[np.where(label == 10)[0]]), axis = 0 )
	# 					except: 
	# 						# ones = images[np.where(label == 10)[0]]
	# 						ones_prob = predictions[np.where(label == 10)[0]]
	# 						ones = label[np.where(label == 10)[0]]
	# 					num_ones += ones.shape[0]
	# 					# print(ones)
	# 				except:
	# 					continue

	# 	if self.topic == 'Base':
	# 		while cur_batch < num_batches:
	# 			noise = self.get_noise([batch_size, self.noise_dims])
	# 			cur_batch += 1
	# 			images = self.generator(noise, training = False)
	# 			predictions = self.classifier.predict(images)
	# 			label = np.argmax(predictions, axis = 1)
	# 			try:
	# 				try:
	# 					# ones = np.concatentate((ones,images[np.where(label == 10)[0]]), axis = 0 )
	# 					ones_prob = np.concatentate((ones_prob,predictions[np.where(label == 10)[0]]), axis = 0)
	# 					ones = np.concatentate((ones,label[np.where(label == 10)[0]]), axis = 0 )
	# 				except: 
	# 					# ones = images[np.where(label == 10)[0]]
	# 					ones_prob = predictions[np.where(label == 10)[0]]
	# 					ones = label[np.where(label == 10)[0]]
	# 				num_ones += ones.shape[0]
	# 			except:
	# 				continue
	# 	# try:
	# 	# 	avg_prb = tf.reduce_mean(ones_prob, axis = 0)
	# 	# except:
	# 	# 	ones_prob = 0
	# 	# 	avg_prob = 0
	# 	# self.W22_vec.append([self.W22_val, self.total_count.numpy()])
	# 	if self.mode in ['metrics', 'model_metrics']:
	# 		print("Final Number of 1 in 9e5 - "+str(num_ones))
	# 		self.res_file.write("Final Number of 1 in 9e5 - "+str(num_ones))
	# 		# print("Final AbgProb in 9e5 - "+str(avg_prob))
	# 		# self.res_file.write("Final AbgProb in 9e5 - "+str(avg_prob))
	# 		# if self.res_flag:
	# 		# 	self.res_file.write("Final FID score - "+str(self.fid))

	# 	if self.res_flag:
	# 		print("Final Number of 1 in 9e5 - "+str(num_ones))
	# 		self.res_file.write("Final Number of 1 in 9e5 - "+str(num_ones))
	# 		# print("Final AbgProb in 9e5 - "+str(avg_prob))
	# 		# self.res_file.write("Final AbgProb in 9e5 - "+str(avg_prob))
	# 	return


	# 	# with PdfPages(path+'ClassProbs_stem_'+str(self.total_count.numpy())+'.pdf') as pdf:

	# 	# 	fig1 = plt.figure(figsize=(3.5, 3.5))
	# 	# 	ax1 = fig1.add_subplot(111)
	# 	# 	ax1.cla()
	# 	# 	ax1.get_xaxis().set_visible(True)
	# 	# 	ax1.get_yaxis().set_visible(True)
	# 	# 	ax1.set_ylim([0,0.5])
	# 	# 	ax1.stem(vals,label = 'alpha_p='+str(self.alphap))
	# 	# 	ax1.legend(loc = 'upper right')
	# 	# 	pdf.savefig(fig1)
	# 	# 	plt.close(fig1)

	# 	# with PdfPages(path+'ClassProbs_plot_'+str(self.total_count.numpy())+'.pdf') as pdf:

	# 	# 	fig1 = plt.figure(figsize=(3.5, 3.5))
	# 	# 	ax1 = fig1.add_subplot(111)
	# 	# 	ax1.cla()
	# 	# 	ax1.get_xaxis().set_visible(True)
	# 	# 	ax1.get_yaxis().set_visible(True)
	# 	# 	ax1.plot(locs,vals, c='r',label = 'alpha_p='+str(self.alphap))
	# 	# 	ax1.legend(loc = 'upper right')
	# 	# 	pdf.savefig(fig1)
	# 	# 	plt.close(fig1)



	def print_ClassProbs(self):
		path = self.metricpath
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex = False)

		if self.classifier_load_flag:
			self.classifier_load_flag = 0
			try:
				self.classifier = tf.keras.models.load_model('PFMNIST_Classifer.h5')
			except:
				os.system("python PartialFMNIST_Classifer.py") ### Might not work if env is bad
				self.classifier = tf.keras.models.load_model('PFMNIST_Classifer.h5')

		num_samples = 90000
		batch_size = self.batch_size
		num_batches = num_samples//batch_size
		cur_batch = 0
		num_ones = 0
		good_ones = np.zeros((400,28,28,1))
		good_ones_counter = 0 
		avg_prob = 0 
		
		while cur_batch < num_batches:
			for noise_batch in self.noise_dataset:
				cur_batch += 1
				images = self.generator(noise_batch, training = False)
				predictions = self.classifier.predict(images)
				for j in range(batch_size):
					label = np.argmax(predictions[j,:])
					if label == 10:
						print(good_ones_counter)
						if avg_prob == 0:
							avg_prob = predictions[j,10]
						else:
							avg_prob = 0.5*avg_prob + 0.5*predictions[j,10]
						good_ones[good_ones_counter,:,:,:] = images[j,:,:,:]
						good_ones_counter+= 1
					if good_ones_counter == 400:
						break
				if good_ones_counter == 400:
						break
			if good_ones_counter == 400:
						break

		good_ones = (good_ones + 1.0)/2.0
		path_noise = self.metricpath +'GAN_Gen1_'+str(self.total_count.numpy())+'.png'
		self.save_image_batch(images = good_ones, label ='Generated 1',  path = path_noise, size = 14)



		# print(predictions)
		# label = np.argmax(predictions, axis = 1)
		# print(label)
		# try:
		# 	try:
		# 		# ones = np.concatentate((ones,images[np.where(label == 10)[0]]), axis = 0 )
		# 		ones_prob = np.concatentate((ones_prob,predictions[np.where(label == 10)[0]]), axis = 0)
		# 		ones = np.concatentate((ones,label[np.where(label == 10)[0]]), axis = 0 )
		# 	except: 
		# 		# ones = images[np.where(label == 10)[0]]
		# 		ones_prob = predictions[np.where(label == 10)[0]]
		# 		ones = label[np.where(label == 10)[0]]
		# 	num_ones += ones.shape[0]
			# print(ones)
		# except:
		# 	continue

		if self.mode in ['metrics', 'model_metrics']:
			print("Final Number of 1 in 9e5 - "+str(num_ones))
			self.res_file.write("Final Number of 1 in 9e5 - "+str(num_ones))
			print("Final AbgProb in 9e5 - "+str(avg_prob))
			self.res_file.write("Final AbgProb in 9e5 - "+str(avg_prob))


		if self.res_flag:
			# print("Final Number of 1 in 9e5 - "+str(num_ones))
			# self.res_file.write("Final Number of 1 in 9e5 - "+str(num_ones))
			print("Final AbgProb in 9e5 - "+str(avg_prob))
			self.res_file.write("Final AbgProb in 9e5 - "+str(avg_prob))
		return




	def eval_MardiaStats(self,endcoded):
		def b_1_d(X, chunks: int = None):
			N = np.shape(X)[0]
			if chunks is None:
				return (1/N**2)*np.sum(np.matmul(X, X.T)**3)
			else:
				sum = 0
				Cs = np.array_split(X, chunks)
				for c2 in Cs:
					c2t = c2.T
					for c1 in Cs:
						sum += np.sum(np.matmul(c1, c2t)**3)

				return (1/N**2)*sum


		def b_2_d(X):
			return np.mean(norml2(X, axis=1)**4)


		def skewness_test(X, chunks: int = None):
			return b_1_d(X, chunks)

		def kurtosis_test(X):
			D = np.shape(X)[1]
			return b_2_d(X) - D*(D+2)

		self.skewness = skewness_test(endcoded, chunks = 2)
		self.kurtosis = kurtosis_test(endcoded)

		# self.skew_vec.append([self.skewness, self.total_count.numpy()])
		# self.kurt_vec.append([self.kurtosis, self.total_count.numpy()])

		if self.mode in ['metrics', 'model_metrics']:
			print("\n Final skewness score - "+str(self.skewness))
			print("\n Final Kurtosis score - "+str(self.kurtosis))
			if self.res_flag:
				self.res_file.write("\n Final skewness score - "+str(self.skewness))
				self.res_file.write("\n Final Kurtosis score - "+str(self.kurtosis))

		if self.res_flag:
			self.res_file.write("\n Skewness score - "+str(self.skewness))
			self.res_file.write("\n Kurtosis score - "+str(self.kurtosis))
		return



		

