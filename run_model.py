import sys
import time
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple

# from data import Vocab
# from batcher import Batcher
from model import QAModel
# from decode import BeamSearchDecoder
import utils

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', 'data', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Where to save output
tf.app.flags.DEFINE_string('log_root', 'log', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', 'base', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_size', 80, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('decoder_hidden_size', 128, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('embed_size', 100, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')

tf.app.flags.DEFINE_integer('max_allowed_inputs', 130, 'max_allowed_inputs')
tf.app.flags.DEFINE_integer('num_train', 9000, 'num_train')

tf.app.flags.DEFINE_integer('babi_id', 1, 'bAbi train task id')
tf.app.flags.DEFINE_integer('babi_test_id', None, 'bAbi test task id')

tf.app.flags.DEFINE_integer('max_epochs', 3, 'maximum number of epochs')
tf.app.flags.DEFINE_integer('early_stopping', 20, 'early stopping')

tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')

tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('dropout', 0.9, 'dropout')
tf.app.flags.DEFINE_float('l2', 0.001, 'L2 weight decay')

tf.app.flags.DEFINE_boolean('cap_grads', True, 'Gradient clipping')
tf.app.flags.DEFINE_float('max_grad_norm', 10.0, 'Gradient clipping threshold')
tf.app.flags.DEFINE_boolean('noisy_grads', True, 'Add gradient noise')
tf.app.flags.DEFINE_float('grad_noise_std', 0.001, 'Stddev of gaussian noise to add to the gradients')
tf.app.flags.DEFINE_boolean('word2vec_init', True, 'Use word embedding')
tf.app.flags.DEFINE_float('embedding_init', np.sqrt(3), 'Stddev of gaussian noise to add to the gradients')

tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')

# Pointer Memory Network or baseline model
tf.app.flags.DEFINE_boolean('pointer_memnet', False, 'If True, use pointer memory network model. If False, use baseline model.')
tf.app.flags.DEFINE_integer('num_generators', 3, 'Total number of generator sources: Copy from question, copy from context and word softmax')

class Config(object):
	"""Holds model hyperparams and data information."""
	
	batch_size = 32
	embed_size = 100
	hidden_size = 80
	decoder_hidden_size = 128
	
	max_epochs = 256
	early_stopping = 20
	
	dropout = 0.9
	lr = 0.001
	l2 = 0.001
	
	cap_grads = True
	max_grad_norm = 10
	noisy_grads = True
	grad_noise_std = 0.001
	
	word2vec_init = True
	embedding_init = np.sqrt(3)
	
	# set to zero with strong supervision to only train gates
	strong_supervision = False
	beta = 1
	
	# NOTE not currently used hence non-sensical anneal_threshold
	anneal_threshold = 1000
	anneal_by = 1.5
	
	num_hops = 3
	num_attention_features = 4
	
	max_allowed_inputs = 130
	num_train = 9000
	
	floatX = np.float32
	
	babi_id = "1"
	babi_test_id = ""
	
	mode = "train"
	num_generators = 3 # Copy from question, copy from context, word softmax
	pointer_memnet = False

def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
	"""Calculate the running average loss via exponential decay.
	This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.
	Args:
		loss: loss on the most recent eval step
		running_avg_loss: running_avg_loss so far
		summary_writer: FileWriter object to write for tensorboard
		step: training iteration step
		decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.
	Returns:
		running_avg_loss: new running average loss
	"""
	if running_avg_loss == 0:  # on the first iteration just take the loss
		running_avg_loss = loss
	else:
		running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
	# running_avg_loss = min(running_avg_loss, 12)  # clip
	loss_sum = tf.Summary()
	tag_name = 'running_avg_loss/decay=%f' % (decay)
	loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
	summary_writer.add_summary(loss_sum, step)
	tf.logging.info('running_avg_loss: %f', running_avg_loss)
	return running_avg_loss

# def setup_training(model, batcher):
def setup_training(model):
	"""Does setup before starting training (run_training)"""
	train_dir = os.path.join(FLAGS.log_root, "train")
	if not os.path.exists(train_dir): os.makedirs(train_dir)
	
	default_device = tf.device('/gpu:0')
	with default_device:
		model.build_graph() # build the graph
		saver = tf.train.Saver(max_to_keep=2) # only keep 5 checkpoint at a time
	
	sv = tf.train.Supervisor(logdir=train_dir,
	                   is_chief=True,
	                   saver=saver,
	                   summary_op=None,
	                   save_summaries_secs=60, # save summaries for tensorboard every 60 secs
	                   save_model_secs=60, # checkpoint every 60 secs
	                   global_step=model.global_step)
	train_writer = sv.summary_writer
	tf.logging.info("Preparing or waiting for session...")
	sess_context_manager = sv.prepare_or_wait_for_session(config=utils.get_config())
	tf.logging.info("Created session.")
	try:
		# run_training(model, batcher, sess_context_manager, sv, summary_writer) # this is an infinite loop until interrupted
		run_training(model, sess_context_manager, sv, train_writer) # this is an infinite loop until interrupted
	except KeyboardInterrupt:
		tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
		sv.stop()

def setup_testing(model):
	"""Does setup before starting training (run_training)"""
	test_dir = os.path.join(FLAGS.log_root, "train")
	if not os.path.exists(test_dir): os.makedirs(test_dir)
	
	default_device = tf.device('/gpu:0')
	with default_device:
		model.build_graph() # build the graph
		saver = tf.train.Saver(max_to_keep=2) # only keep 5 checkpoint at a time
	
	sv = tf.train.Supervisor(logdir=test_dir,
	                   is_chief=True,
	                   saver=saver,
	                   summary_op=None,
	                   save_summaries_secs=60, # save summaries for tensorboard every 60 secs
	                   save_model_secs=60, # checkpoint every 60 secs
	                   global_step=model.global_step)
	test_writer = sv.summary_writer
	tf.logging.info("Preparing or waiting for session...")
	sess_context_manager = sv.prepare_or_wait_for_session(config=utils.get_config())
	tf.logging.info("Created session.")
	try:
		model.run_inference_step(sess_context_manager)
	except KeyboardInterrupt:
		tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
		sv.stop()

# def run_training(model, batcher, sess_context_manager, sv, summary_writer):
def run_training(model, sess_context_manager, sv, train_writer):
	"""Repeatedly runs training iterations, logging loss to screen and writing summaries"""

	eval_dir = os.path.join(FLAGS.log_root, "eval")
	if not os.path.exists(eval_dir): os.makedirs(eval_dir)
	eval_writer = tf.summary.FileWriter(eval_dir)

	test_dir = os.path.join(FLAGS.log_root, "test")
	if not os.path.exists(test_dir): os.makedirs(test_dir)
	test_writer = tf.summary.FileWriter(test_dir)

	tf.logging.info("starting run_training")
	with sess_context_manager as sess:
		for epoch_id in range(FLAGS.max_epochs): # repeats until interrupted
			# batch = batcher.next_batch()
			
			# tf.logging.info('running training step...')
			# t0=time.time()
			model.run_train_epoch(sess, epoch_id, train_writer)
			# model.run_eval_epoch(sess, eval_writer)
			model.run_test_epoch(sess, test_writer)
			# t1=time.time()
			# tf.logging.info('seconds for training step: %.3f', t1-t0)
			
			# loss = results['loss']
			# tf.logging.info('loss: %f', loss) # print the loss to screen
			# 
			# # get the summaries and iteration number so we can write summaries to tensorboard
			# summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
			# train_step = results['global_step'] # we need this to update our running average loss
			# 
			# summary_writer.add_summary(summaries, train_step) # write the summaries
			# if train_step % 100 == 0: # flush the summary writer every so often
			# 	summary_writer.flush()


def main(unused_argv):
	if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
		raise Exception("Problem with flags: %s" % unused_argv)
	
	tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
	tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))
	
	# Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
	FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
	if not os.path.exists(FLAGS.log_root):
		if FLAGS.mode in ["train", "eval", "test"] :
			os.makedirs(FLAGS.log_root)
		else:
			raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))
	
	# vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size) # create a vocabulary
	# 
	# # If in decode mode, set batch_size = beam_size
	# # Reason: in decode mode, we decode one example at a time.
	# # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
	# if FLAGS.mode == 'decode':
	# 	FLAGS.batch_size = FLAGS.beam_size
	# 
	# # If single_pass=True, check we're in decode mode
	# if FLAGS.single_pass and FLAGS.mode!='decode':
	# 	raise Exception("The single_pass flag should only be True in decode mode")
	# 
	# # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
	# hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen']
	# hps_dict = {}
	# for key,val in FLAGS.__flags.iteritems(): # for each flag
	# 	if key in hparam_list: # if it's in the list
	# 		hps_dict[key] = val # add it to the dict
	# hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
	# 
	# # Create a batcher object that will create minibatches of data
	# batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)
	
	tf.set_random_seed(111) # a seed value for randomness
	
	# print "creating model..."
	# model = QAModel()
	# setup_training(model)

	if FLAGS.mode == 'train':
		print "creating model..."
		model = QAModel()
		setup_training(model)
	elif FLAGS.mode == 'test':
		print "creating model..."
		model = QAModel()
		setup_testing(model)
	# elif hps.mode == 'decode':
	# 	decode_model_hps = hps  # This will be the hyperparameters for the decoder model
	# 	decode_model_hps = hps._replace(max_dec_steps=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
	# 	model = SummarizationModel(decode_model_hps, vocab)
	# 	decoder = BeamSearchDecoder(model, batcher, vocab)
	# 	decoder.decode() # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
	# else:
	# 	raise ValueError("The 'mode' flag must be one of train/eval/decode")

if __name__ == '__main__':
	tf.app.run()
