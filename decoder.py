import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl

FLAGS = tf.app.flags.FLAGS

def attention_decoder(decoder_inputs, initial_state, encoder_states, cell, initial_state_attention=True):
	"""
	Args:
		decoder_inputs: A list of 2D Tensors [batch_size x input_size].
		initial_state: 2D Tensor [batch_size x cell.state_size].
		encoder_states: 3D Tensor [batch_size x attn_length x attn_size].
		cell: rnn_cell.RNNCell defining the cell function and size.
		initial_state_attention:
			Note that this attention decoder passes each decoder input through a linear layer with the previous step's context vector to get a modified version of the input. If initial_state_attention is False, on the first decoder step the "previous context vector" is just a zero vector. If initial_state_attention is True, we use initial_state to (re)calculate the previous step's context vector. We set this to False for train/eval mode (because we call attention_decoder once for all decoder steps) and True for decode mode (because we call attention_decoder once for each decoder step).
		pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
	Returns:
		outputs: A list of the same length as decoder_inputs of 2D Tensors of
			shape [batch_size x cell.output_size]. The output vectors.
		state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
		attn_dists: A list containing tensors of shape (batch_size,attn_length).
			The attention distributions for each decoder step.
		p_gens: List of scalars. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
	"""
	with tf.variable_scope("attention_decoder") as scope:
		attn_size = encoder_states.get_shape()[2].value # if this line fails, it's because the attention length isn't defined
		
		# Reshape encoder_states (need to insert a dim)
		encoder_states = tf.expand_dims(encoder_states, axis=2) # now is shape (batch_size, attn_len, 1, attn_size)
		
		# To calculate attention, we calculate
		#   v^T tanh(W_h h_i + W_s s_t + b_attn)
		# where h_i is an encoder state, and s_t a decoder state.
		# attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
		# We set it to be equal to the size of the encoder states.
		attention_vec_size = attn_size
		
		# Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
		# To calculate W_h * h_i we use a 1-by-1 convolution, need to reshape before.
		W_h = tf.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
		encoder_features = tf.nn.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME") # shape (batch_size,attn_length,1,attention_vec_size)
		
		# Get the weight vectors v and w_c (w_c is for coverage)
		v = tf.get_variable("v", [attention_vec_size])
	
	def attention(query):
	
	    """Point on hidden using hidden_features and query."""
	    with tf.variable_scope("Attention"):
	    
	        y = rnn_cell_impl._linear(query, attention_vec_size, True)
	        y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
	        
	        # Attention mask is a softmax of v^T * tanh(...).
	        s = tf.reduce_sum(v * tf.nn.tanh(encoder_features + y), [2, 3])
	        attn = tf.nn.softmax(s)
	    
	        p_context = attn
	    
	        # Calculate the context vector from attn_dist and encoder_states
	        context_vector = tf.reduce_sum(tf.reshape(attn, [FLAGS.batch_size, -1, 1, 1]) * encoder_states, [1, 2]) # shape (batch_size, attn_size).
	        context_vector = tf.reshape(context_vector, [-1, attn_size])
	
	        return p_context, context_vector
	
	outputs = []
	p_contexts = []
	p_gens = []
	state = initial_state
	
	context_vector = tf.zeros([FLAGS.batch_size, attn_size])
	context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.
	
	if initial_state_attention: # true in decode mode
		# Re-calculate the context vector from the previous step so that we can pass it through
		# a linear layer with this step's input to get a modified version of the input
		_, context_vector = attention(initial_state)
	    
	    
	#Note that this attention decoder passes each decoder input through a linear layer
	#with the previous step's context vector to get a modified version of the input.
	#If initial_state_attention is False, on the first decoder step the 
	#"previous context vector" is just a zero vector. If initial_state_attention is True,
	#we use initial_state to (re)calculate the previous step's context vector.
	#We set this to False for train/eval mode (because we call attention_decoder once for all decoder steps)
	#and True for decode mode (because we call attention_decoder once for each decoder step).
	
	# tf.logging.info("decoder input shape", decoder_inputs[0].get_shape())
	# tf.logging.info("state shap", state.get_shape())
	
	for i, inp in enumerate(decoder_inputs):
		tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(decoder_inputs))
		if i > 0:
			tf.get_variable_scope().reuse_variables()
		
		# Merge input and previous attentions into one vector x of the same size as inp
		input_size = inp.get_shape().with_rank(2)[1]
		if input_size.value is None:
			raise ValueError("Could not infer input size from input: %s" % inp.name)
		x = rnn_cell_impl._linear([inp] + [context_vector], input_size, True)
		
		# Run the decoder RNN cell. cell_output = decoder state
		cell_output, state = cell(x, state)
		
		# Run the attention mechanism.
		if i == 0 and initial_state_attention:  # always true in decode mode
			with tf.variable_scope(tf.get_variable_scope(), reuse=True): # you need this because you've already run the initial attention(...) call
				p_context, context_vector = attention(state)
		else:
			p_context, context_vector = attention(state)
		p_contexts.append(p_context)
		
		if FLAGS.pointer_memnet:
			# Calculate p_gen
			with tf.variable_scope('calculate_pgen'):
				p_gen = rnn_cell_impl._linear([context_vector, state.c, state.h, x], FLAGS.num_generators, True) # a scalar
				p_gen = tf.nn.softmax(p_gen)
				p_gens.append(p_gen)
		
		# Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
		# This is V[s_t, h*_t] + b in the paper
		with tf.variable_scope("AttnOutputProjection"):
			output = rnn_cell_impl._linear([cell_output] + [context_vector], cell.output_size, True)
		outputs.append(output)
		
		return outputs, state, p_contexts, p_gens
