from __future__ import print_function
import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from absl import app
from absl import flags

from gan_topics import *

'''***********************************************************************************
********** Standard GAN  **************************************************************
***********************************************************************************'''
class SGAN_Base(GAN_Base):

	def __init__(self,FLAGS_dict):
		GAN_Base.__init__(self,FLAGS_dict)

	def create_optimizer(self):
		with tf.device(self.device):
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)
		print("Optimizers Successfully made")
		return


	def train_step(self,reals_all):
		for i in tf.range(self.Dloop):
			# noise = tf.random.normal([self.batch_size, self.noise_dims], self.noise_mean, self.noise_stddev)
			noise = self.get_noise([self.batch_size, self.noise_dims])
			self.reals = reals_all
			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
				self.fakes = self.generator(noise, training=True)

				self.real_output = self.discriminator(self.reals, training=True)
				self.fake_output = self.discriminator(self.fakes, training=True)

				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.Disc_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))
			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	def loss_base(self):
		cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

		D_real_loss = cross_entropy(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = cross_entropy(tf.zeros_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss

		G_fake_loss = cross_entropy(tf.ones_like(self.fake_output), self.fake_output)

		self.G_loss = G_fake_loss

