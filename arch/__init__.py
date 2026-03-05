from absl import flags
import os, sys, time, argparse

FLAGS = flags.FLAGS
FLAGS(sys.argv)

if FLAGS.gan not in ['WAE']:
	if FLAGS.loss == 'RBF' and FLAGS.topic == 'PolyGAN':
		from .arch_RBF import *
	else:
		from .arch_base import *
	
if FLAGS.gan == 'WAE':
	from .arch_WAE import *