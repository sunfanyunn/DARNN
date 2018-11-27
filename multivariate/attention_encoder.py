from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl   #omit when tf = 1.3
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

# TODO(ebrevdo): Remove once _linear is fully deprecated.
# linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access #omit when tf = 1.3
linear = rnn_cell_impl._linear #add when tf = 1.3

def attention_encoder(encoder_inputs, attention_states, cell,
                      output_size=None, num_heads=1,
                      dtype=dtypes.float32, scope=None):

    """RNN encoder with attention.
  In this context "attention" means that, during encoding, the RNN can look up
  information in the additional tensor "attention_states", which is constructed by transpose the dimensions of time steps and input features of the inputs,
  and it does this to focus on a few features of the input.

  Args:
    encoder_inputs: A list of 2D Tensors [batch_size x n_input_encoder].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".

  Returns:
    A tuple of the form (outputs, state, attn_weights), where:
      outputs: A list of the encoder hidden states. Each element is a 2D Tensor of shape [batch_size x output_size]. 
      state: The state of encoder cell at the final time-step. It is a 2D Tensor of shape [batch_size x cell.state_size].
      attn_weights: A list of the input attention weights. Each element is a 2D Tensor of shape [batch_size x attn_length]
  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
    if not encoder_inputs:
        raise ValueError("Must provide at least 1 input to attention encoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention encoder.")
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(scope or "attention_encoder"):
        # get the batch_size of the encoder_input
        batch_size = array_ops.shape(encoder_inputs[0])[0]  # Needed for reshaping.
        # attention_state.shape (batch_size, n_input_encoder, n_steps_encoder)
        attn_length = attention_states.get_shape()[1].value #  n_input_encoder
        attn_size = attention_states.get_shape()[2].value  # n_steps_encoder

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        # hidden_features shape: (batch_size, attn_length, 1, attn_size)
        hidden = array_ops.reshape(
            attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = variable_scope.get_variable("Attn_EncoderW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(variable_scope.get_variable("AttnEncoderV_%d" % a,
                                           [attention_vec_size]))
        # how to get the initial_state
        initial_state_size = array_ops.stack([batch_size, output_size])
        initial_state = [array_ops.zeros(initial_state_size,dtype=dtype) for _ in xrange(2)]
        state = initial_state

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list,1)
            for a in xrange(num_heads):
                with variable_scope.variable_scope("AttentionEncoder_%d" % a):
                    # y with the shape (batch_size, attention_vec_size)
                    y = linear(query, attention_vec_size, True)
                    # y with the shape (batch_size, 1, 1, attention_vec_size)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    # hidden_features with the shape (batch_size, attn_length, 1, attn_size)
                    s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                    # a with shape (batch_size, attn_length)
                    # a is the attention weight                    
                    a = nn_ops.softmax(s)
                    ds.append(a)
            return ds

        outputs = []
        attn_weights = []
        batch_attn_size = array_ops.stack([batch_size, attn_length])
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
                 for _ in xrange(num_heads)]

        # i is the index of the which time step
        # inp is numpy.array and the shape of inp is (batch_size, n_feature)
        for i, inp in enumerate(encoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            # multiply attention weights with the original input
            # get the newly input
            x = attns[0]*inp
            # Run the BasicLSTM with the newly input
            cell_output, state = cell(x, state)

            # Run the attention mechanism.
            attns = attention(state)

            with variable_scope.variable_scope("AttnEncoderOutputProjection"):
                output = cell_output

            outputs.append(output)
            attn_weights.append(attns)
            
    return outputs, state, attn_weights

