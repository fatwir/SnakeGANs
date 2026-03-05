from __future__ import print_function
import os, sys, time, argparse, signal, json, struct
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from tensorflow.python import eager
import tensorflow as tf
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
# import tensorflow_addons as tfa
# from tensorflow.python import debug as tf_debug
import traceback

print(tf.__version__)
from absl import app
from absl import flags


# from mnist_cnn_icp_eval import *
tf.keras.backend.set_floatx('float32')

def signal_handler(sig, frame):
	print('\n\n\nYou pressed Ctrl+C! \n\n\n')
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

'''Generic set of FLAGS. learning_rate and batch_size are redefined in GAN_ARCH if g1/g2'''
FLAGS = flags.FLAGS
flags.DEFINE_float('lr_G', 0.0001, """learning rate for generator""")
flags.DEFINE_float('lr_D', 0.0001, """learning rate for discriminator""")
flags.DEFINE_float('beta1', 0.5, """beta1 for Adam""")
flags.DEFINE_float('beta2', 0.9, """beta2 for Adam""")
flags.DEFINE_float('decay_rate', 1.0, """decay rate for lr""")
flags.DEFINE_integer('decay_steps', 5000000, """ decay steps for lr""")
flags.DEFINE_integer('colab', 0, """ set 1 to run code in a colab friendy way """)
flags.DEFINE_integer('homo_flag', 0, """ set 1 to read data in a colab friendy way """)
flags.DEFINE_integer('batch_size', 100, """Batch size.""")
flags.DEFINE_integer('paper', 1, """1 for saving images for a paper""")
flags.DEFINE_integer('resume', 1, """1 vs 0 for Yes Vs. No""")
flags.DEFINE_integer('saver', 1, """1-Save events for Tensorboard. 0 O.W.""")
flags.DEFINE_integer('models_for_metrics', 0, """1-Save H5 Models at FID iters. 0 O.W.""")
flags.DEFINE_integer('res_flag', 1, """1-Write results to a file. 0 O.W.""")
flags.DEFINE_integer('update_fig', 1, """1-Write results to a file. 0 O.W.""")
flags.DEFINE_integer('pbar_flag', 1, """1-Display Progress Bar, 0 O.W.""")
flags.DEFINE_integer('latex_plot_flag', 1, """1-Plot figs with latex, 0 O.W.""")
flags.DEFINE_integer('gaussian_stats_flag', 0, """1-Plot figs with latex, 0 O.W.""")
flags.DEFINE_integer('out_size', 32, """CelebA output reshape size""")
flags.DEFINE_list('metrics', '', 'CSV for the metrics to evaluate. KLD, FID, PR')
flags.DEFINE_integer('video_flag', 0, 'Set 1 for GradGrid Video')
flags.DEFINE_integer('save_all', 0, """1-Save all the models. 0 for latest 10""") #currently functions as save_all internally
flags.DEFINE_integer('seed', 42, """Initialize the random seed of the run (for reproducibility).""")
flags.DEFINE_integer('num_epochs', 200, """Number of epochs to train for.""")
flags.DEFINE_integer('num_iters', 20000, """Number of epochs to train for.""")
flags.DEFINE_integer('iters_flag', 0, """Flag to stop at number of iters, not epochs""")
flags.DEFINE_integer('Dloop', 1, """Number of loops to run for D.""")
flags.DEFINE_integer('Gloop', 1, """Number of loops to run for G.""")
flags.DEFINE_integer('ODE_step', 1, """Number of loops to run for G. DEPRICATED UNTIL FIX""")
flags.DEFINE_integer('num_parallel_calls', 5, """Number of parallel calls for dataset map function""")
flags.DEFINE_string('run_id', 'default', """ID of the run, used in saving.""")
flags.DEFINE_string('log_folder', 'default', """ID of the run, used in saving.""")
flags.DEFINE_string('mode', 'train', """Operation mode: train, test, fid """)
flags.DEFINE_string('topic', 'ELeGANt', """ELeGANt or RumiGAN""")
flags.DEFINE_string('data', 'mnist', """Type of Data to run for""")
flags.DEFINE_string('gan', 'sgan', """Type of GAN for""")
flags.DEFINE_string('loss', 'base', """Type of Loss function to use""")
flags.DEFINE_string('loss_norm', '1', """Type of norm in losses that use it: 1,2,mix""")
flags.DEFINE_string('GPU', '0,1', """GPU's made visible '0', '1', or '0,1' """)
flags.DEFINE_string('device', '0', """Which GPU device to run on: 0,1 or -1(CPU)""")
flags.DEFINE_string('noise_kind', 'gaussian', """Type of Noise for WAE latent prior or for SpiderGAN""")
flags.DEFINE_string('noise_data', 'mnist', """Type of Data to feed as noise""")
flags.DEFINE_string('arch', 'dcgan', """resnet vs dcgan""")
flags.DEFINE_string('data_1', 'celeba', """Type of Data to run for""")
flags.DEFINE_string('data_2', 'ukiyoe', """Type of Data to run for""")
# flags.DEFINE_string('data_3', 'imagenet', """Type of Data to run for""")
flags.DEFINE_integer('tinyimgnet_size', 64, """ Output size for Tiny-ImageNet data""")
flags.DEFINE_integer('ukiyoe_size', 128, """ Output size for Ukiyo-E data""")
flags.DEFINE_integer('ffhq_size', 128, """ Output size for FFHQ data""")
flags.DEFINE_integer('celeba_size', 64, """ Output size for CelebA data""")
flags.DEFINE_integer('cifar10_size', 32, """ Output size for CIFAR-10 data""")
flags.DEFINE_integer('svhn_size', 32, """ Output size for CIFAR-10 data""")
flags.DEFINE_integer('church_size', 128, """ Output size for LSUN Churches data""")
flags.DEFINE_integer('bedroom_size', 128, """ Output size for LSUN Bedroom data""")
flags.DEFINE_integer('zero_size', 32, """ Output size for zeros-noise data""")
flags.DEFINE_string('latent_gen_run_id', 'default', """ Output size for CelebA data""")
flags.DEFINE_string('latent_gen_folder', 'default', """ Output size for CelebA data""")
flags.DEFINE_integer('TanGAN_flag', 0, """ set 1 to read data in a colab friendy way """)
flags.DEFINE_integer('BaseTanGAN_flag', 0, """ set 1 to read data in a colab friendy way """)
flags.DEFINE_integer('sn_flag', 1, """ set 1 to use spectral normalization """)

# '''Flags just for metric computations'''
flags.DEFINE_integer('stop_metric_iters', 1000000, """1-Display Progress Bar, 0 O.W.""")
flags.DEFINE_integer('start_metric_iters', 20, """1-Display Progress Bar, 0 O.W.""")
flags.DEFINE_integer('append', 1, """1-Display Progress Bar, 0 O.W.""")

# '''Flags just for RumiGAN'''
# flags.DEFINE_integer('number', 3, """ Class selector in Multi-class data""")
# flags.DEFINE_integer('num_few', 200, """ 200 for MNIST, 1k for C10 and 10k for CelebA""")
# flags.DEFINE_integer('GaussN', 3, """ N for Gaussian""")
# flags.DEFINE_string('testcase', 'female', """Test cases for RumiGAN""")
# flags.DEFINE_string('label_style', 'base', """base vs. embed for how labels are fed to the net""")
# flags.DEFINE_float('label_a', -0.5, """Class label - a """)
# flags.DEFINE_float('label_bp', 2.0, """Class label - bp for +ve data """)
# flags.DEFINE_float('label_bn', -2.0, """Class label - bn for -ve data """)
# flags.DEFINE_float('label_c', 2.0, """Class label - c for generator """)
# flags.DEFINE_float('alphap', 0.9, """alpha weight for +ve class """)
# flags.DEFINE_float('alphan', 0.1, """alpha weight for -ve class""")
'''
Defined Testcases:
1. even - learn only the even numbers 
2. odd - learn only the odd mnist numbers
3. male - learn males in CelebA
4. female - learn females in CelebA
5. single - learn a single digit in MNIST - uses number flag to deice number
'''
'''Flags just for CycleGAN Paper analysis'''
flags.DEFINE_string('data_A', 'mnist', """Type of Data to run for""")
flags.DEFINE_string('data_B', 'svhn', """Type of Data to run for""")

'''Flags just for RumiGAN Paper analysis'''
flags.DEFINE_integer('number', 3, """ Class selector in Multi-class data on mnist/fmnist/cifar10""")
flags.DEFINE_integer('num_few', 200, """Num of images for minority 200((F)MNIST), 1k(Cifar10), 5k(CelebA)""")
flags.DEFINE_string('label_style', 'base', """Label input style to cGAN/ACGANs :base/embed/multiply""")

flags.DEFINE_float('label_a', -0.5, """Class label - a """)
flags.DEFINE_float('label_b', 0.5, """Class label - b """)
flags.DEFINE_float('label_bp', 2.0, """Class label - bp for +ve data """)
flags.DEFINE_float('label_bn', -2.0, """Class label - bn for -ve data """)
flags.DEFINE_float('label_c', 2.0, """Class label - c for generator """)
flags.DEFINE_float('LSGANlambdaD', 1, """beta1 for Adam""")

flags.DEFINE_float('alphap', 2.5, """alpha_plus/beta_plus weight for +ve class loss term """)
flags.DEFINE_float('alphan', 0.5, """alpha_minus/beta_minus weight for -ve class loss term""")

flags.DEFINE_string('testcase', 'none', """Test cases for RumiGAN""")
flags.DEFINE_string('mnist_variant', 'none', """Set to 'fashion' for Fashion-MNIST dataset""")
'''
Defined Testcases:
MNIST/FMNIST:
1. even - even numbers as positive class
2. odd - odd numbers as positive class
3. overlap - "Not true random - determinitic to the set selected in the paper" 
4. rand - 6 random classes as positive, 6 as negative
5. single - learn a single digit in MNIST - uses "number" flag to deice which number
6. few - learn a single digit (as minority positive) in MNIST - uses "number" flag to deice which number, "num_few" to decide how many samples to pick for minority class 
CelebA:
1. male - learn males in CelebA as positive
2. female - learn females in CelebA as positive
3. fewmale - learn males as minority positive class in CelebA - "num_few" used as in MNIST.6
4. fewfemale - learn females as minority positive class in CelebA - "num_few" used as in MNIST.6
5. hat - learn hat in CelebA as positive
6. bald - learn bald in CelebA as positive
7. cifar10 - learn all of CelebA, with CIFAR-10 as negative class (R3 Rebuttal response)
CIFAR-10:
1. single - as in MNIST
2. few - as in MNIST
3. animals - learn animals as positive class, vehicles as negative
'''

# ''' Flags for PolyGAN RBF'''

flags.DEFINE_integer('rbf_m', 2, """Gradient order for RBF. The m in k=2m-n""") #
flags.DEFINE_integer('GaussN', 3, """ N for Gaussian""")
flags.DEFINE_integer('N_centers', 100, """ N for number of centres in PolyRBF""")
flags.DEFINE_integer('num_snake_iters', 2, """ Number of iterations of snake""")
flags.DEFINE_string('snake_kind', 'o', """ordered(o)/unordered(uo)""")

# '''Flags just for WGAN-FS forms'''
flags.DEFINE_float('data_mean', 10.0, """Mean of taget Gaussian data""")
flags.DEFINE_float('data_var', 1.0, """Variance of taget Gaussian data""")

'''Flags just for WGAN-FS forms'''
flags.DEFINE_integer('terms', 50, """N for 0-M for FS.""") #Matters only if g
flags.DEFINE_float('sigma',75, """approximation sigma of data distribution""") 
flags.DEFINE_integer('lambda_d', 20000, """Period as a multiple of sigmul*sigma""") ##NeedToKill
flags.DEFINE_string('latent_kind', 'base', """AE/DCT/W/AE2/AE3/Cycle - need to make W""") ##NeedToKill
flags.DEFINE_string('distribution', 'generic', """generic/gaussian""")
flags.DEFINE_integer('latent_dims', 10, """Dimension of latent representation""") #20 on GMM 8 worked #Matters only if not g  ;AE3 takes lxl 14 or 7; DCt lxl
flags.DEFINE_integer('L', 10000, """Number of terms in summation""")
# GAN pretraining for WAEFR, AE pretraining for PolyMMDGAN
flags.DEFINE_integer('GAN_pretrain_epochs', 0, """Num of GAN pre-training Epochs""")
flags.DEFINE_integer('AE_pretrain_epochs', 0, """Num of AE pre-training Epochs""")
flags.DEFINE_integer('train_D', 0, """Set 1 to backprop and update Disc FS weights with backprop""")
flags.DEFINE_integer('input_noise', 0, """Set 1 to add Noise to input images""")
flags.DEFINE_string('FID_kind', 'clean', """ latent: FID on WAE latent; clean: Clean-FID lirary """)
flags.DEFINE_string('KID_kind', 'clean', """ latent: KID on WAE latent; clean: Clean-FID lirary """)
flags.DEFINE_integer('SID_order', -1, """The choice of order for computing SID (default: 1)""")
flags.DEFINE_integer('SinD_size', 32, """The choice of order for computing SID (default: 1)""")

flags.DEFINE_float('lr_AE_Enc', 0.01, """learning rate""")
flags.DEFINE_float('lr_AE_Dec', 0.01, """learning rate""")

# '''Flags just for WGAN-FS forms'''
# flags.DEFINE_integer('terms', 15, """N for 0-M for FS.""") #Matters only if g
# flags.DEFINE_float('sigma',10, """approximation sigma of data distribution""") 
# flags.DEFINE_integer('lambda_d', 20000, """Period as a multiple of sigmul*sigma""")
# flags.DEFINE_integer('sigmul', 1, """Period as a multiple of sigmul*sigma""") #10e5 is mnist, 10e3 if g ### NEW 100 works for MNIST, 10^5 is celeba ### Set to 1. make 100 for save for paper in 1D
# flags.DEFINE_string('latent_kind', 'AE', """AE/DCT/W/AE2/AE3/Cycle - need to make W""")
# flags.DEFINE_string('distribution', 'generic', """generic/gaussian""")
# flags.DEFINE_integer('latent_dims', 10, """Dimension of latent representation""") #20 on GMM 8 worked #Matters only if not g  ;AE3 takes lxl 14 or 7; DCt lxl
# flags.DEFINE_integer('L', 25000, """Number of terms in summation""")
# flags.DEFINE_integer('AE_steps', 20000, """Dimension of latent representation""") #1000 for GMM8
# flags.DEFINE_integer('AE_count', 5, """Num of AE pre-training Epochs""")
# flags.DEFINE_integer('train_D', 0, """Dimension of latent representation""")
# flags.DEFINE_string('FID_kind', 'none', """if FID is latent, calculates on WAE latent space""")

# flags.DEFINE_float('lr_AE_Enc', 0.01, """learning rate""")
# flags.DEFINE_float('lr_AE_Dec', 0.01, """learning rate""")
# flags.DEFINE_float('lr_GenEnc', 0.0001, """learning rate""") #0.1 here works on base g
# flags.DEFINE_float('lr_GenDec', 0.01, """learning rate""")
# flags.DEFINE_float('lr_Disc', 0.0, """learning rate""")
# flags.DEFINE_float('lr2_AE_Enc', 0.0000001, """learning rate""")
# flags.DEFINE_float('lr2_AE_Dec', 0.0000001, """learning rate""")
# flags.DEFINE_float('lr2_GenEnc', 0.000001, """learning rate""") #0.001 is good? 0.00001 too high for g2
# flags.DEFINE_float('lr2_GenDec', 0, """learning rate""")
# flags.DEFINE_float('lr2_Disc', 0.0, """learning rate""")

# '''Flags just for SGRLD-GAN forms'''
flags.DEFINE_integer('hl_nodes', 8, """Hidden layer nodes (8 best for mnist. ~20 for lr30d?)""")
flags.DEFINE_float('epsilon', 80, """Decay param Epsilon""")
flags.DEFINE_float('tau0', 100, """Decay param tau""")
flags.DEFINE_float('kappa', 0.9, """Decay param kappa""")
flags.DEFINE_integer('mnist_size', 28, """ Output size for MNIST data""")
flags.DEFINE_integer('lr30d_size', 30, """ Output size for lr30d data""")
flags.DEFINE_integer('iris_size', 4, """ Output size for iris dataset""")
flags.DEFINE_integer('gen_size', 2, """ Output size for generated data""")
# flags.DEFINE_integer('autodiff', 1, """1 vs 0 for Yes Vs. No""")
flags.DEFINE_bool('autodiff', True, """--autodiff vs --noautodiff for Yes Vs. No""")
# flags.DEFINE_integer('bias', 1, """1 vs 0 for Yes Vs. No""")
flags.DEFINE_bool('use_bias', True, """--use_bias vs --nouse_bias for Yes Vs. No""")
flags.DEFINE_string('approach', 'Ours', """ Ours or Euclidean or Patterson or NGD""")
flags.DEFINE_string('fim_type', 'approx', """ 'our' (our derivations) or 'approx' (Karakida and Osawa, 2020, NeurIPS) FIM """)

# '''Flags just for ScoreGAN forms'''
flags.DEFINE_float('lr_S', 0.001, """learning rate for Score Net""")
flags.DEFINE_integer('Score_pretrain_epochs', 0, """Num of GAN pre-training Epochs""")




def email_success():
	import smtplib, ssl

	port = 587  # For starttls
	smtp_server = "smtp.gmail.com"
	sender_email = "darthsidcodes@gmail.com"
	receiver_email = "darthsidcodes@gmail.com"
	password = "kamehamehaX100#"
	SUBJECT = "Execution Completed"
	TEXT = "Execution of code: "+str(gan.run_id)+" completed Successfully."
	message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT) 
	context = ssl.create_default_context()
	with smtplib.SMTP(smtp_server, port) as server:
		server.ehlo()  # Can be omitted
		server.starttls(context=context)
		server.ehlo()  # Can be omitted
		server.login(sender_email, password)
		server.sendmail(sender_email, receiver_email, message)


def email_error(error):
	traceback.print_exc()
	error = traceback.format_exc()
	import smtplib, ssl

	port = 587  # For starttls
	smtp_server = "smtp.gmail.com"
	sender_email = "darthsidcodes@gmail.com"
	receiver_email = "darthsidcodes@gmail.com"
	password = "kamehamehaX100#"
	SUBJECT = "An Error Occured"
	TEXT = "Error: "+error
	message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT) 
	context = ssl.create_default_context()
	with smtplib.SMTP(smtp_server, port) as server:
		server.ehlo()  # Can be omitted
		server.starttls(context=context)
		server.ehlo()  # Can be omitted
		server.login(sender_email, password)
		server.sendmail(sender_email, receiver_email, message)



FLAGS(sys.argv)
from models import *


if __name__ == '__main__':
	'''Enable Flags and various tf declarables on GPU processing '''
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU #'0' or '0,1', or '0,1,2' or '1,2,3'
	# physical_devices = tf.test.gpu_device_name()

	
	physical_devices = list(tf.config.experimental.list_physical_devices('GPU'))
	for gpu in physical_devices:
		print(gpu)
		tf.config.experimental.set_memory_growth(gpu, True)

	print('Visible Physical Devices: ',physical_devices)
	tf.config.threading.set_inter_op_parallelism_threads(12)
	tf.config.threading.set_intra_op_parallelism_threads(12)
	

	
	# Level | Level for Humans | Level Description                  
	# ------|------------------|------------------------------------ 
	# 0     | DEBUG            | [Default] Print all messages       
	# 1     | INFO             | Filter out INFO messages           
	# 2     | WARNING          | Filter out INFO & WARNING messages 
	# 3     | ERROR            | Filter out all messages
	tf.get_logger().setLevel('ERROR')
	# tf.get_logger().setLevel('DEBUG')
	# tf.debugging.set_log_device_placement(True)
	# if FLAGS.colab and FLAGS.data == 'celeba':
	os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "500G"
	if FLAGS.colab:
		import warnings
		warnings.filterwarnings("ignore")



	''' Set random seed '''
	np.random.seed(FLAGS.seed)
	tf.random.set_seed(FLAGS.seed)

	FLAGS_dict = FLAGS.flag_values_dict()

	###	EXISTING Variants:
	##
	##	(1) SGAN - 
	##		(A) Base
	##		(B) RumiGAN
	##		(C) ACGAN
	##
	##	(2) LSGAN - 
	##		(A) Base
	##		(B) RumiGAN
	##
	##	(3) WGAN - 
	##		(A) Base
	##		(B) ELeGANt
	##		(C) Rumi
	##
	##	(4) WAE - 
	##		(A) Base
	##		(B) ELeGANt
	### -----------------
	### Have to add CycleGAN for future work. Potentially a cGAN to separate out ACGAN style stuff from cGAN. Currently, no plans.

	gan_call = FLAGS.gan + '_' + FLAGS.topic + '(FLAGS_dict)'

	if FLAGS.gan == 'Langevin':
		lang = Langevin_SGRLD(FLAGS_dict)
		lang.initial_setup()
		lang.get_data()
		# gan.create_load_checkpoint()
		# lang.pretrain_AE()
		# lang.train_sgrld()
		lang.train()
		exit(0)

	# with tf.device('/GPU:'+FLAGS.device):
	# try:
	print('trying')
	gan = eval(gan_call)
	# with gan.strategy.scope():
	with tf.device(gan.device):
		gan.initial_setup()
		gan.get_data()
		gan.create_models()
		gan.create_optimizer()
		gan.create_load_checkpoint()
		print('Worked')

		if gan.mode == 'train':
			print(gan.mode)
			# with tf.device(gan.device):
			gan.train()
			# email_success()
			if gan.data not in ['g1', 'g2', 'gN', 'gmm8', 'gmmN']:
				gan.test()
		if gan.mode == 'h5_from_checkpoint':
			gan.h5_from_checkpoint()
		if gan.mode == 'test':
			# with tf.device(gan.device):
			gan.test()
		if gan.mode == 'metrics':
			# with tf.device(gan.device):
			gan.eval_metrics()
		if gan.mode == 'model_metrics':
			# with tf.device(gan.device):
			gan.model_metrics()

	# except Exception as e:
	# 	email_error(str(e))
	# 	print("\nExiting Execution due to error\n")
	# 	print(e)
	# 	exit(0)

###############################################################################  
	
	
	print('Completed.')
