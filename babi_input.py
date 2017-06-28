from __future__ import division
from __future__ import print_function

import sys

import os as os
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# can be sentence or word
input_mask_mode = "sentence"

# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
def init_babi(fname):
	
	print("==> Loading test from %s" % fname)
	tasks = []
	task = None
	for i, line in enumerate(open(fname)):
		id = int(line[0:line.find(' ')])
		if id == 1:
			task = {"C": "", "Q": "", "A": "", "S": ""} 
			counter = 0
			id_map = {}
		    
		line = line.strip()
		line = line.replace('.', ' . ')
		line = line[line.find(' ')+1:]
		# if not a question
		if line.find('?') == -1:
			task["C"] += line
			id_map[id] = counter
			counter += 1
		    
		else:
			idx = line.find('?')
			tmp = line[idx+1:].split('\t')
			task["Q"] = line[:idx]
			task["A"] = tmp[1].strip()
			task["S"] = []
			if task["A"].find(',') != -1: # Must be path finding task (task-19)
				assert FLAGS.babi_id == 19
				d = {'s': 'south', 'n': 'north', 'e': 'east', 'w': 'west'}
				tmp_ = [d[p_] for p_ in task["A"].split(',')]
				task["A"] = ' '.join(tmp_)
			for num in tmp[2].split():
				task["S"].append(id_map[int(num.strip())])
			tasks.append(task.copy())
	
	return tasks


def get_babi_raw(id, test_id):
    babi_map = {
        "1": "qa1_single-supporting-fact",
        "2": "qa2_two-supporting-facts",
        "3": "qa3_three-supporting-facts",
        "4": "qa4_two-arg-relations",
        "5": "qa5_three-arg-relations",
        "6": "qa6_yes-no-questions",
        "7": "qa7_counting",
        "8": "qa8_lists-sets",
        "9": "qa9_simple-negation",
        "10": "qa10_indefinite-knowledge",
        "11": "qa11_basic-coreference",
        "12": "qa12_conjunction",
        "13": "qa13_compound-coreference",
        "14": "qa14_time-reasoning",
        "15": "qa15_basic-deduction",
        "16": "qa16_basic-induction",
        "17": "qa17_positional-reasoning",
        "18": "qa18_size-reasoning",
        "19": "qa19_path-finding",
        "20": "qa20_agents-motivations",
        "MCTest": "MCTest",
        "19changed": "19changed",
        "joint": "all_shuffled", 
        "sh1": "../shuffled/qa1_single-supporting-fact",
        "sh2": "../shuffled/qa2_two-supporting-facts",
        "sh3": "../shuffled/qa3_three-supporting-facts",
        "sh4": "../shuffled/qa4_two-arg-relations",
        "sh5": "../shuffled/qa5_three-arg-relations",
        "sh6": "../shuffled/qa6_yes-no-questions",
        "sh7": "../shuffled/qa7_counting",
        "sh8": "../shuffled/qa8_lists-sets",
        "sh9": "../shuffled/qa9_simple-negation",
        "sh10": "../shuffled/qa10_indefinite-knowledge",
        "sh11": "../shuffled/qa11_basic-coreference",
        "sh12": "../shuffled/qa12_conjunction",
        "sh13": "../shuffled/qa13_compound-coreference",
        "sh14": "../shuffled/qa14_time-reasoning",
        "sh15": "../shuffled/qa15_basic-deduction",
        "sh16": "../shuffled/qa16_basic-induction",
        "sh17": "../shuffled/qa17_positional-reasoning",
        "sh18": "../shuffled/qa18_size-reasoning",
        "sh19": "../shuffled/qa19_path-finding",
        "sh20": "../shuffled/qa20_agents-motivations",
    }
    if (test_id == "" or test_id is None):
        test_id = str(id) 
    babi_name = babi_map[str(id)]
    babi_test_name = babi_map[str(test_id)]
    babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-10k/%s_train.txt' % babi_name))
    babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-10k/%s_test.txt' % babi_test_name))
    return babi_train_raw, babi_test_raw

            
def load_glove(dim):
	word2vec = {}
	
	print("==> loading glove")
	with open(("./data/glove/glove.6B/glove.6B." + str(dim) + "d.txt")) as f:
		for line in f:    
			l = line.split()
			word2vec[l[0]] = map(float, l[1:])
	        
	print("==> glove is loaded")
	
	return word2vec


def create_vector(word, word2vec, word_vector_size, silent=True):
	# if the word is missing from Glove, create some fake vector and store in glove!
	vector = np.random.uniform(0.0,1.0,(word_vector_size,))
	word2vec[word] = vector
	if (not silent):
		print("utils.py::create_vector => %s is missing" % word)
	return vector

def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=True):
	if not word in word2vec:
		create_vector(word, word2vec, word_vector_size, silent)
	if not word in vocab: 
		next_index = len(vocab)
		vocab[word] = next_index
		ivocab[next_index] = word
	
	if to_return == "word2vec":
		return word2vec[word]
	elif to_return == "index":
		return vocab[word]
	elif to_return == "onehot":
		raise Exception("to_return = 'onehot' is not implemented yet")

def process_input(data_raw, floatX, word2vec, vocab, ivocab, embed_size, split_sentences=False):
	questions = []
	inputs = []
	answers = []
	input_masks = []
	relevant_labels = []
	for x in data_raw:
		if split_sentences:
			inp = x["C"].lower().split(' . ') 
			inp = [w for w in inp if len(w) > 0]
			inp = [i.split() for i in inp]
		else:
			inp = x["C"].lower().split(' ') 
			inp = [w for w in inp if len(w) > 0]
		
		q = x["Q"].lower().split(' ')
		q = [w for w in q if len(w) > 0]
		
		if split_sentences: 
			inp_vector = [[process_word(word = w, 
										word2vec = word2vec, 
										vocab = vocab, 
										ivocab = ivocab, 
										word_vector_size = embed_size, 
										to_return = "index") for w in s] for s in inp]
		else:
			inp_vector = [process_word(word = w, 
										word2vec = word2vec, 
										vocab = vocab, 
										ivocab = ivocab, 
										word_vector_size = embed_size, 
										to_return = "index") for w in inp]
		                            
		q_vector = [process_word(word = w, 
									word2vec = word2vec, 
									vocab = vocab, 
									ivocab = ivocab, 
									word_vector_size = embed_size, 
									to_return = "index") for w in q]
		
		if split_sentences:
			inputs.append(inp_vector)
		else:
			inputs.append(np.vstack(inp_vector).astype(floatX))
		questions.append(np.vstack(q_vector).astype(floatX))
		if x["A"].find(' ') != -1:
			assert FLAGS.babi_id == 19
			a_vector = [process_word(word = w_, 
									word2vec = word2vec, 
									vocab = vocab, 
									ivocab = ivocab, 
									word_vector_size = embed_size, 
									to_return = "index") for w_ in x["A"].split(' ')]
			answers.append(np.vstack(a_vector).astype(floatX))
		else:
			answers.append(process_word(word = x["A"], 
			                            word2vec = word2vec, 
			                            vocab = vocab, 
			                            ivocab = ivocab, 
			                            word_vector_size = embed_size, 
			                            to_return = "index"))
		# NOTE: here we assume the answer is one word! 
		
		if not split_sentences:
			if input_mask_mode == 'word':
				input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32)) 
			elif input_mask_mode == 'sentence': 
				input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32)) 
			else:
				raise Exception("invalid input_mask_mode")
		
		relevant_labels.append(x["S"])
	
	return inputs, questions, answers, input_masks, relevant_labels 

def get_lens(inputs, split_sentences=False):
    lens = np.zeros((len(inputs)), dtype=int)
    for i, t in enumerate(inputs):
        lens[i] = t.shape[0]
    return lens

def get_sentence_lens(inputs):
    lens = np.zeros((len(inputs)), dtype=int)
    sen_lens = []
    max_sen_lens = []
    for i, t in enumerate(inputs):
        sentence_lens = np.zeros((len(t)), dtype=int)
        for j, s in enumerate(t):
            sentence_lens[j] = len(s)
        lens[i] = len(t)
        sen_lens.append(sentence_lens)
        max_sen_lens.append(np.max(sentence_lens))
    return lens, sen_lens, max(max_sen_lens)
    

def pad_inputs(inputs, lens, max_len, mode="", sen_lens=None, max_sen_len=None):
    if mode == "mask":
        padded = [np.pad(inp, (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
        return np.vstack(padded)

    elif mode == "split_sentences":
        padded = np.zeros((len(inputs), max_len, max_sen_len))
        for i, inp in enumerate(inputs):
            padded_sentences = [np.pad(s, (0, max_sen_len - sen_lens[i][j]), 'constant', constant_values=0) for j, s in enumerate(inp)]
            # trim array according to max allowed inputs
            if len(padded_sentences) > max_len:
                padded_sentences = padded_sentences[(len(padded_sentences)-max_len):]
                lens[i] = max_len
            padded_sentences = np.vstack(padded_sentences)
            padded_sentences = np.pad(padded_sentences, ((0, max_len - lens[i]),(0,0)), 'constant', constant_values=0)
            padded[i] = padded_sentences
        return padded

    padded = [np.pad(np.squeeze(inp, axis=1), (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
    return np.vstack(padded)

def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = word2vec[word]
    return embedding

def load_babi(split_sentences=False):
	vocab = {}
	ivocab = {}
	
	babi_train_raw, babi_test_raw = get_babi_raw(FLAGS.babi_id, FLAGS.babi_test_id)
	
	if FLAGS.word2vec_init:
		word2vec = load_glove(FLAGS.embed_size)
	else:
		word2vec = {}
	
	# set word at index zero to be end of sentence token so padding with zeros is consistent
	process_word(word = "<eos>", 
				word2vec = word2vec, 
				vocab = vocab, 
				ivocab = ivocab, 
				word_vector_size = FLAGS.embed_size, 
				to_return = "index")

	process_word(word = "<go>", 
				word2vec = word2vec, 
				vocab = vocab, 
				ivocab = ivocab, 
				word_vector_size = FLAGS.embed_size, 
				to_return = "index")
	
	print('==> get train inputs')
	train_data = process_input(babi_train_raw, np.float32, word2vec, vocab, ivocab, FLAGS.embed_size, split_sentences)
	print('vocab length after train inputs:', len(vocab))
	print('==> get test inputs')
	test_data = process_input(babi_test_raw, np.float32, word2vec, vocab, ivocab, FLAGS.embed_size, split_sentences)
	print('vocab length after test inputs:', len(vocab))
	
	if FLAGS.word2vec_init:
		# assert FLAGS.embed_size == 100
		word_embedding = create_embedding(word2vec, ivocab, FLAGS.embed_size)
	else:
		word_embedding = np.random.uniform(-FLAGS.embedding_init, FLAGS.embedding_init, (len(ivocab), FLAGS.embed_size))
	
	max_q_len, max_input_len, max_mask_len = 0, 0, 0
	
	loopcount = 2 # if FLAGS.mode == 'train' else 1
	for iter_ in range(loopcount):
		inputs, questions, answers, input_masks, rel_labels = train_data if iter_==0 else test_data
		
		if split_sentences:
			input_lens, sen_lens, max_sen_len = get_sentence_lens(inputs)
			max_mask_len = max(max_sen_len, max_mask_len)
		else:
			input_lens = get_lens(inputs)
			mask_lens = get_lens(input_masks)
			max_mask_len = max(np.max(mask_lens), max_mask_len)
		
		q_lens = get_lens(questions)
		
		max_q_len = max(np.max(q_lens), max_q_len)
		max_input_len = max(min(np.max(input_lens), FLAGS.max_allowed_inputs), max_input_len)
		
		#pad out arrays to max
		if split_sentences:
			inputs = pad_inputs(inputs, input_lens, max_input_len, "split_sentences", sen_lens, max_sen_len)
			input_masks = np.zeros(len(inputs))
			print("inputs shape: ", inputs.shape)
		else:
			inputs = pad_inputs(inputs, input_lens, max_input_len)
			input_masks = pad_inputs(input_masks, mask_lens, max_mask_len, "mask")
		
		questions = pad_inputs(questions, q_lens, max_q_len)
		
		answers = np.squeeze(np.stack(answers)) if FLAGS.babi_id == 19 else np.expand_dims(np.stack(answers), 1)
		print("max input length: ", max_input_len)
		print("Answer shape: ", answers.shape)
		
		rel_labels = np.zeros((len(rel_labels), len(rel_labels[0])))
		
		for i, tt in enumerate(rel_labels):
			rel_labels[i] = np.array(tt, dtype=int)
		
		if iter_==0:
			train = questions[:FLAGS.num_train], inputs[:FLAGS.num_train], q_lens[:FLAGS.num_train], input_lens[:FLAGS.num_train], input_masks[:FLAGS.num_train], answers[:FLAGS.num_train], rel_labels[:FLAGS.num_train] 
			
			valid = questions[FLAGS.num_train:], inputs[FLAGS.num_train:], q_lens[FLAGS.num_train:], input_lens[FLAGS.num_train:], input_masks[FLAGS.num_train:], answers[FLAGS.num_train:], rel_labels[FLAGS.num_train:] 
		else:
			test = questions, inputs, q_lens, input_lens, input_masks, answers, rel_labels

	data = [train, valid, test] if (FLAGS.mode == 'train') else [train, valid, test]
	
	return data[0], data[1], data[2], word_embedding, max_q_len, max_input_len, max_mask_len, rel_labels.shape[1], vocab


    
