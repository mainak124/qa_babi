import os
import sys
import time
import numpy as np
import tensorflow as tf
from decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector

import babi_input
from nn import weight, bias, batch_norm

from tensorflow.python.ops import rnn_cell_impl

# pylint: disable=protected-access
_linear = rnn_cell_impl._linear
# pylint: enable=protected-access

FLAGS = tf.app.flags.FLAGS


def _position_encoding(sentence_size, embedding_size):
	"""Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)"""
	encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
	ls = sentence_size+1
	le = embedding_size+1
	for i in range(1, le):
		for j in range(1, ls):
			encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
	encoding = 1 + 4 * encoding / embedding_size / sentence_size
	return np.transpose(encoding)

def _add_gradient_noise(t, stddev=1e-3, name=None):
	"""Adds gradient noise as described in http://arxiv.org/abs/1511.06807
	The input Tensor `t` should be a gradient.
	The output will be `t` + gaussian noise.
	0.001 was said to be a good fixed value for memory networks."""
	with tf.name_scope(values=[t, stddev], name="add_gradient_noise") as name:
		t = tf.convert_to_tensor(t, name="t")
		gn = tf.random_normal(tf.shape(t), stddev=stddev)
		return tf.add(t, gn, name=name)

class QAModel(object):
	"""A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""
	
	def __init__(self):
		# self.config = config
		# self.vocab = vocab
		
		data1, data2, data3, word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab = \
		babi_input.load_babi(split_sentences=True)
		self.ivocab = {v: k for k, v in self.vocab.iteritems()} 
		self.vocab_size = len(self.vocab)
		print("Vocabulary: \n", self.vocab)
		
		if FLAGS.mode in ['train', 'test']:
			self.train, self.valid, self.test = data1, data2, data3
			self.total_train_steps = len(self.train[0]) // FLAGS.batch_size
			self.total_eval_steps = len(self.valid[0]) // FLAGS.batch_size
			self.total_test_steps = len(self.test[0]) // FLAGS.batch_size
		else:
			self.train, self.valid, self.test = None, None, data1
			self.total_train_steps = None
			self.total_eval_steps = None
			self.total_test_steps = len(self.test[0]) // FLAGS.batch_size
		
		self.encoding = _position_encoding(self.max_sen_len, FLAGS.embed_size)
		self.embeddings = tf.Variable(word_embedding.astype(np.float32), name="Embedding")
	
	def add_placeholders(self):
	
		self.question_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, self.max_q_len))
		self.input_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, self.max_input_len, self.max_sen_len))
		
		self.question_len_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size,))
		self.input_len_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size,))
		self.answer_placeholder = tf.placeholder(tf.int64, shape=(FLAGS.batch_size, 1))
		self.rel_label_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, self.num_supporting_facts))
		self.dropout_placeholder = tf.placeholder(tf.float32)
	
	
	def make_feed_dict(self, data):
	
		"""Make a feed dictionary mapping parts of the batch to the appropriate placeholders.
		Args:
			batch: Batch object
			just_enc: Boolean. If True, only feed the parts needed for the encoder.
		"""
		
		dp = FLAGS.dropout
		# if self.train_op is None:
		# 	self.train_op = tf.no_op()
		# 	dp = 1
		
		# shuffle data
		p = np.random.permutation(len(data[0]))
		qp, ip, ql, il, im, a, r = data
		qp, ip, ql, il, im, a, r = qp[p], ip[p], ql[p], il[p], im[p], a[p], r[p]
		
		step=0
		
		index = range(step*FLAGS.batch_size, (step+1)*FLAGS.batch_size)
		
		feed_dict = {
						self.question_placeholder: qp[index],
						self.input_placeholder: ip[index],
						self.question_len_placeholder: ql[index],
						self.input_len_placeholder: il[index],
						self.answer_placeholder: a[index],
						self.rel_label_placeholder: r[index],
						self.dropout_placeholder: dp
					}
		
		return feed_dict
	
	def encode_question(self):
		
		with tf.variable_scope("question_encoder") as scope:
			questions = tf.nn.embedding_lookup(self.embeddings, self.question_placeholder)
			
			gru_cell = tf.contrib.rnn.GRUCell(FLAGS.decoder_hidden_size)
			self.q_outs, self.q_vec = tf.nn.dynamic_rnn(gru_cell,
							questions,
							dtype=np.float32,
							sequence_length=self.question_len_placeholder)
	
	
	def encode_context(self):
	
		with tf.variable_scope("context_encoder") as scope:
			inputs = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
			
			# use encoding to get sentence representation
			inputs = tf.reduce_sum(inputs * self.encoding, 2)
			
			forward_gru_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)
			backward_gru_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)
			context_outs, self.context_vec = tf.nn.bidirectional_dynamic_rnn(
							forward_gru_cell,
							backward_gru_cell,
							inputs,
							dtype=np.float32,
							sequence_length=self.input_len_placeholder)
			
			# Our encoder is bidirectional and our decoder is unidirectional
			# so we need to reduce the final encoder hidden state
			# to the right size to be the initial decoder hidden state
			
			with tf.variable_scope("brnn_combine") as inner_scope:
				outs_cat = tf.concat([context_outs[0], context_outs[1]], 2)
				_, n_words, in_dim = outs_cat.get_shape().as_list()
				out_dim = FLAGS.decoder_hidden_size
				outs_flat = tf.reshape(outs_cat, [-1, in_dim])
				w = weight('W', [in_dim, out_dim])
				b = bias('b', out_dim)
				res = tf.nn.bias_add(tf.matmul(outs_flat, w), b)
				outs = tf.reshape(res, [-1, n_words, out_dim])
				self.context_outs = outs
	
	def get_question_attention(self):
	
		# Take the cosine distance begtween every word in context and every word in question
		l = tf.matmul(self.context_outs, tf.transpose(self.q_outs, perm=[0,2,1])) 
		a_q = tf.nn.softmax(l, 2)
		a_p = tf.nn.softmax(l, 1)
		
		self.p_question = tf.reduce_mean(a_q, 1)
	
	def add_decoder(self):
		"""Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.
		Args:
			inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)
		Returns:
			outputs: List of tensors; the outputs of the decoder
			out_state: The final state of the decoder
			attn_dists: A list of tensors; the attention distributions
			p_gens: A list of scalar tensors; the generation probabilities
		"""
		# list length max_dec_steps containing shape (batch_size, emb_size)
		decoder_inputs = [tf.nn.embedding_lookup(self.embeddings, tf.zeros_like(tf.unstack(self.answer_placeholder, axis=1)[0]))] #+ [tf.nn.embedding_lookup(self.embeddings, x) for x in tf.unstack(self.answer_placeholder, axis=1)]
		encoder_states = self.context_outs
		
		cell = tf.nn.rnn_cell.LSTMCell(FLAGS.decoder_hidden_size, state_is_tuple=True, initializer=tf.random_uniform_initializer(-0.02, 0.02, seed=123))
		decoder_init_state = tf.nn.rnn_cell.LSTMStateTuple(self.q_vec, self.q_vec)
		
		outputs, out_state, attn_dists, p_gens = attention_decoder(decoder_inputs, decoder_init_state, encoder_states, cell) #, initial_state_attention=(FLAGS.mode=="decode"))
		
		return outputs, out_state, attn_dists, p_gens
	
	def greedy_decoder(self, vocab_lists):
		return tf.argmax(vocab_lists[0], 1)
	
	def add_seq2seq(self):
		"""Add the whole sequence-to-sequence model to the graph."""
		
		with tf.variable_scope('seq2seq'):
			# # Some initializers
			# self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
			# self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
			
			# Add the encoder
			with tf.variable_scope('encoder'):
				self.encode_question()
				self.encode_context()
			
			# Add the decoder.
			with tf.variable_scope('decoder'):
				self.get_question_attention()
				decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens = self.add_decoder()
			
			# Add the output projection to obtain the vocabulary distribution
			with tf.variable_scope('output_projection'):
				w = tf.get_variable('w', [FLAGS.decoder_hidden_size, self.vocab_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-4))
				w_t = tf.transpose(w)
				b = tf.get_variable('b', [self.vocab_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-4))
				vocab_scores = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
				for i, output in enumerate(decoder_outputs):
					if i > 0:
						tf.get_variable_scope().reuse_variables()
					vocab_scores.append(tf.nn.xw_plus_b(output, w, b)) # apply the linear layer
				
				# The vocabulary distributions. List length max_dec_steps of (batch_size, vocab_size) arrays. The words are in the order they appear in the vocabulary file.
				vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]
				
				self.predictions = self.greedy_decoder(vocab_dists)
				# tf.summary.scalar('predictions', self.predictions)
				
				#final_dists = calc_final_dist(vocab_dists, p_contexts, p_question, p_gens)
				## Take log of final distribution
				#log_dists = [tf.log(dist) for dist in final_dists]	
			
			
			# For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution, then take log
			if FLAGS.pointer_memnet:
				final_dists = self.calc_final_dist(vocab_dists)
				# Take log of final distribution
				log_dists = [tf.log(dist) for dist in final_dists]
			else: # just take log of vocab_dists
				log_dists = [tf.log(dist) for dist in vocab_dists]
			
			
			if FLAGS.mode in ['train', 'eval', 'test']:
				# Calculate the loss
				with tf.variable_scope('loss'):
					if FLAGS.pointer_memnet: # calculate loss from log_dists
						# Calculate the loss per step
						# This is fiddly; we use tf.gather_nd to pick out the log probabilities of the target words
						loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
						batch_nums = tf.range(0, limit=FLAGS.batch_size) # shape (batch_size)
						for dec_step, log_dist in enumerate(log_dists):
							targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
							indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
							losses = tf.gather_nd(-log_dist, indices) # shape (batch_size). loss on this step for each batch
							loss_per_step.append(losses)
						
						# Apply padding_mask mask and get loss
						self.loss = self.mask_and_avg(loss_per_step, self._padding_mask)
					
					else: # baseline model
						self.loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self.answer_placeholder, tf.ones([FLAGS.batch_size, 1]))
					
					tf.summary.scalar('loss', self.loss)
			
			with tf.name_scope('acc'):
				correct_vec = tf.equal(tf.argmax(vocab_scores[0], 1), tf.squeeze(self.answer_placeholder))
				num_corrects = tf.reduce_sum(tf.cast(correct_vec, 'float'), name='num_corrects')
				self.acc = tf.reduce_mean(tf.cast(correct_vec, 'float'), name='acc')
				tf.summary.scalar('acc', self.acc)
					
		# if hps.mode == "decode":
		# 	# We run decode beam search mode one decoder step at a time
		# 	assert len(log_dists)==1 # log_dists is a singleton list containing shape (batch_size, extended_vsize)
		# 	log_dists = log_dists[0]
		# 	self._topk_log_probs, self._topk_ids = tf.nn.top_k(log_dists, hps.batch_size*2) # note batch_size=beam_size in decode mode
	
	
	
	def add_train_op(self):
		"""Sets self.train_op, the op to run for training."""
		
		# Take gradients of the trainable variables w.r.t. the loss function to minimize
		tvars = tf.trainable_variables()
		grads = tf.gradients(self.loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
		for v in tvars: print(v)
		
		# optionally cap and noise gradients to regularize
		if FLAGS.cap_grads:
			grads, global_norm = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
		else:
			global_norm = tf.sqrt(sum([tf.square(tf.norm(t)) for t in grads]))
		if FLAGS.noisy_grads:
			grads = [_add_gradient_noise(grad, stddev=FLAGS.grad_noise_std) for grad in grads]
		
		# Add a summary
		tf.summary.scalar('global_norm', global_norm)
		
		# Apply adagrad optimizer
		# opt = tf.train.AdagradOptimizer(FLAGS.lr, initial_accumulator_value=0.1)
		opt = tf.train.AdamOptimizer(FLAGS.lr)
		with tf.device("/gpu:0"):
			self.train_op = opt.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')
	
	
	def build_graph(self):
		"""Add the placeholders, model, global step, train_op and summaries to the graph"""
		
		tf.logging.info('Building graph...')
		t0 = time.time()
		self.add_placeholders()
		with tf.device("/gpu:0"):
			self.add_seq2seq()
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		if FLAGS.mode == 'train':
			self.add_train_op()
		self.summaries = tf.summary.merge_all()
		t1 = time.time()
		tf.logging.info('Time to build graph: %i seconds', t1 - t0)
	
	def run_train_step(self, sess):
		"""Runs one training iteration. Returns a dictionary containing train op, summaries, loss and global_step."""
		feed_dict = self.make_feed_dict(self.train)
		to_return = {
			'train_op': self.train_op,
			'summaries': self.summaries,
			'loss': self.loss,
			# 'predictions': self.predictions,
			# 'answers': self.answer_placeholder,
			'global_step': self.global_step,
			'acc': self.acc,
		}
		return sess.run(to_return, feed_dict)
	
	def run_train_epoch(self, sess, epoch_id, summary_writer=None):
		"""Runs one training epoch. Returns a dictionary containing train op, summaries, loss and global_step."""
		total_loss = []
		accuracy = []
		for i in range(self.total_train_steps):
			results = self.run_train_step(sess)
			
			loss = results['loss']
			# preds = results['predictions']
			# ans = results['answers']
			accuracy.append(results['acc'])
			
			# get the summaries and iteration number so we can write summaries to tensorboard
			summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
			train_step = results['global_step'] # we need this to update our running average loss
			
			total_loss.append(loss)
			
			if summary_writer is not None:
				summary_writer.add_summary(summaries, train_step) # write the summaries
			
			if train_step % 100 == 0: # flush the summary writer every so often
				summary_writer.flush()
			
		# 	if verbose and step % verbose == 0:
		# 		sys.stdout.write('\r{} / {} : loss = {}'.format(
		# 			step, total_steps, np.mean(total_loss)))
		# 		sys.stdout.flush()
		# 
		# 
		# if verbose:
		# 	sys.stdout.write('\r')
		
		avg_loss = np.mean(total_loss)
		avg_accuracy = np.mean(accuracy)
		tf.logging.info('Task ID: %d,    Epoch Idx: %d,    Train loss: %f,    accuracy: %f', FLAGS.babi_id, epoch_id, avg_loss, avg_accuracy) # print the loss and accuracy to screen
	
	def run_eval_step(self, sess):
		"""Runs one evaluation iteration. Returns a dictionary containing summaries, loss and global_step."""
		feed_dict = self.make_feed_dict(self.valid)
		to_return = {
			'summaries': self.summaries,
			'loss': self.loss,
			'global_step': self.global_step,
			'acc': self.acc,
		}
		return sess.run(to_return, feed_dict)

	def run_eval_epoch(self, sess, summary_writer=None):
		"""Runs one training epoch. Returns a dictionary containing train op, summaries, loss and global_step."""
		eval_loss = []
		eval_accuracy = []
		for i in range(self.total_eval_steps):
			results = self.run_eval_step(sess)
			
			loss = results['loss']
			# preds = results['predictions']
			# ans = results['answers']
			eval_accuracy.append(results['acc'])
			
			# get the summaries and iteration number so we can write summaries to tensorboard
			summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
			train_step = results['global_step'] # we need this to update our running average loss
			
			eval_loss.append(loss)
			
			if summary_writer is not None:
				summary_writer.add_summary(summaries, train_step) # write the summaries
			
			if train_step % 100 == 0: # flush the summary writer every so often
				summary_writer.flush()
			
		# 	if verbose and step % verbose == 0:
		# 		sys.stdout.write('\r{} / {} : loss = {}'.format(
		# 			step, total_steps, np.mean(total_loss)))
		# 		sys.stdout.flush()
		# 
		# 
		# if verbose:
		# 	sys.stdout.write('\r')
		
		avg_loss = np.mean(eval_loss)
		avg_accuracy = np.mean(eval_accuracy)
		tf.logging.info('batch validation loss %f,\taccuracy: %f', avg_loss, avg_accuracy) # print the loss and accuracy to screen

	def run_test_step(self, sess):
		"""Runs one evaluation iteration. Returns a dictionary containing summaries, loss and global_step."""
		feed_dict = self.make_feed_dict(self.test)
		to_return = {
			'summaries': self.summaries,
			'loss': self.loss,
			'global_step': self.global_step,
			'acc': self.acc,
			'predictions': self.predictions,
			'answers': self.answer_placeholder,
			'questions': self.question_placeholder,
			'contexts': self.input_placeholder,
			'attentions': self.attn_dists
		}
		return sess.run(to_return, feed_dict)

	def run_inference_step(self, sess):
		results = self.run_test_step(sess)
		loss = results['loss']
		accuracy = results['acc']
		predictions = results['predictions']
		answers = results['answers']
		questions = results['questions']
		contexts = results['contexts']
		attentions = results['attentions']

		print("Answers, question, context, attention shape: ", answers.shape, questions.shape, contexts.shape, attentions[0].shape)
		print("Accuracy: ", accuracy)
		# ans = [self.ivocab[x] for x in answers.squeeze().tolist()]
		# preds = [self.ivocab[x] for x in predictions.squeeze().tolist()]
		for context, question, ans, pred, attention in zip(contexts.tolist(), questions.tolist(), answers.squeeze().tolist(), predictions.squeeze().tolist(), attentions[0]):
			for sentence, weight in zip(context, attention):
				sys.stdout.write(' '.join([self.ivocab[word] for word in sentence if word > 0])+ '. ' + str(weight) + '\n')
			#sys.stdout.write('\n')
			print(' '.join([self.ivocab[x] for x in question if x > 0]))
			print(self.ivocab[ans], self.ivocab[pred])
			print('\n\n')

	def run_test_epoch(self, sess, summary_writer=None):
		"""Runs one training epoch. Returns a dictionary containing train op, summaries, loss and global_step."""
		test_loss = []
		test_accuracy = []
		for i in range(self.total_test_steps):
			results = self.run_test_step(sess)
			
			loss = results['loss']
			# preds = results['predictions']
			# ans = results['answers']
			test_accuracy.append(results['acc'])
			
			# get the summaries and iteration number so we can write summaries to tensorboard
			summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
			train_step = results['global_step'] # we need this to update our running average loss
			
			test_loss.append(loss)
			
			if summary_writer is not None:
				summary_writer.add_summary(summaries, train_step) # write the summaries
			
			if train_step % 100 == 0: # flush the summary writer every so often
				summary_writer.flush()
			
		# 	if verbose and step % verbose == 0:
		# 		sys.stdout.write('\r{} / {} : loss = {}'.format(
		# 			step, total_steps, np.mean(total_loss)))
		# 		sys.stdout.flush()
		# 
		# 
		# if verbose:
		# 	sys.stdout.write('\r')
		
		avg_loss = np.mean(test_loss)
		avg_accuracy = np.mean(test_accuracy)
		tf.logging.info('batch test loss %f,\taccuracy: %f', avg_loss, avg_accuracy) # print the loss and accuracy to screen

def _mask_and_avg(values, padding_mask):
	"""Applies mask to values then returns overall average (a scalar)
	Args:
		values: a list length max_dec_steps containing arrays shape (batch_size).
		padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.
	Returns:
		a scalar
	"""
	
	dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
	values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
	values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
	return tf.reduce_mean(values_per_ex) # overall average
