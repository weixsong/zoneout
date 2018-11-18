# This is a minimal gist of what you'd have to
# add to TensorFlow code to implement zoneout.

# To see this in action, see zoneout_seq2seq.py
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops


# Wrapper for the TF RNN cell
# For an LSTM, the 'cell' is a tuple containing state and cell
# We use TF's dropout to implement zoneout
class ZoneoutWrapper(tf.nn.rnn_cell.RNNCell):
    """Operator adding zoneout to all states (states+cells) of the given cell.
    # This is official implementation, optimized by weso(watkinsong@163.com)
    # Concern:
    # 1. why add (1 - state_part_zoneout_prob) to dropout layer, dropout on difference of new_state and old state
    #    and then add old state, is enough to represent the zoneout equation.
    #    ** This aim to make the expected sum is unchanged.
    #    ** dropout ops actually scaled the output so that the expected sum is unchanged.
    #    ** So we need to scale the output back to it true value
    # 2. return output, new_state is zoneout, but output still not zoneout,
    #    as actual output will be used by next layer
    #    ** output is actually not zoneout, I think this is a bug.
    # 3. paper not mentioned in inference if zoneout still used or as dropout not used.
    #    in inference, why not use the state directly?
    #    ** Paper said, like dropout, "we use the expectation of the random noise at test time."
    # 4. The implementation is different with the paper equation.
    #    Is that by design or just this implementation is OK?
    #    ** Yes, by design, as in training process, using equation to keep expected sum is unchanged.

    """

    def __init__(self, cell, zoneout_prob, is_training=True, seed=None):
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not an RNNCell.")
        if not isinstance(zoneout_prob, tuple):
            raise TypeError("Paramter zoneout_prob must be a tuple, such as (0.1, 0.2)")

        for prob in zoneout_prob:
            if not 0.0 <= prob <= 1:
                raise ValueError("Parameter zoneout_prob must be between 0 and 1: %s" % str(prob))

        self._cell = cell
        self._zoneout_prob = zoneout_prob
        self._seed = seed
        self.is_training = is_training

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        if isinstance(self.state_size, tuple) != isinstance(self._zoneout_prob, tuple):
            raise TypeError("Subdivided states need subdivided zoneouts.")
        if isinstance(self.state_size, tuple) and len(tuple(self.state_size)) != len(tuple(self._zoneout_prob)):
            raise ValueError("State and zoneout need equally many parts.")

        output, new_state = self._cell(inputs, state, scope)
        if isinstance(self.state_size, tuple):
            if self.is_training:
                # dropout ops actually scaled the output so that the expected sum is unchanged.
                # So we need to scale the output back to it true value
                new_state = tuple((1 - state_part_zoneout_prob) * tf.python.nn_ops.dropout(
                    new_state_part - state_part, (1 - state_part_zoneout_prob), seed=self._seed)
                                  + state_part
                                  for new_state_part, state_part, state_part_zoneout_prob
                                  in zip(new_state, state, self._zoneout_prob))
            else:
                new_state = tuple(state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
                                  for new_state_part, state_part, state_part_zoneout_prob
                                  in zip(new_state, state, self._zoneout_prob))
        else:
            # here only consider the non-proj LSTM cell, if you want to use
            # proj LSTM, remember to update hte slice ops to proj dimension.
            state_size = self._cell.state_size
            c_prev = tf.slice(state, [0, 0], [-1, state_size / 2])
            h_prev = tf.slice(state, [0, state_size / 2], [-1, state_size / 2])
            c = tf.slice(new_state, [0, 0], [[-1, state_size / 2]])
            h = tf.slice(new_state, [0, state_size / 2], [-1, state_size / 2])
            c_zoneout_prob, h_zoneout_prob = self._zoneout_prob[0], self._zoneout_prob[1]
            if self.is_training:
                c_new = (1 - c_zoneout_prob) * tf.python.nn_ops.dropout(c - c_prev,
                                                                        1 - c_zoneout_prob,
                                                                        seed=self._seed) + c_prev
                h_new = (1 - h_zoneout_prob) * tf.python.nn_ops.dropout(h - h_prev,
                                                                        1 - h_zoneout_prob,
                                                                        seed=self._seed) + h_prev
                new_state = tf.concat([c_new, h_new], axis=1)
            else:
                c_new = (1 - c_zoneout_prob) * c + c_zoneout_prob * c_prev
                h_new = (1 - h_zoneout_prob) * h + h_zoneout_prob * h_prev
                new_state = tf.concat([c_new, h_new], axis=1)

        return output, new_state


# Wrap your cells like this
cell = ZoneoutWrapper(tf.nn.rnn_cell.LSTMCell(128), zoneout_prob=(0.1, 0.2))


class ZoneoutLSTMCell(RNNCell):
  """ZONEOUT: REGULARIZING RNNS BY RANDOMLY PRESERVING HIDDEN ACTIVATIONS
  This is the same equation as Paper written.
  """

  def __init__(self,
               num_units,
               is_training,
               use_peepholes=False,
               cell_clip=None,
               initializer=None,
               num_proj=None,
               proj_clip=None,
               forget_bias=1.0,
               zoneout_prob_cell=0.0,
               zoneout_prob_output=0.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None):
    """Initialize the parameters for an ZoneoutLSTMCell cell.
    Args:
      num_units: int, The number of units in the ZoneoutLSTMCell cell.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training. Must set it manually to `0.0` when restoring from
        CudnnLSTM trained checkpoints.
      zoneout_prob_cell: zoneout probability for cell state
      zoneout_prob_output: zoneout probability for output state
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      When restoring from CudnnLSTM-trained checkpoints, use
      `CudnnCompatibleLSTMCell` instead.
    """
    if not state_is_tuple:
      tf.logging.warn("%s: Using a concatenated state is slower and will soon be "
                      "deprecated.  Use state_is_tuple=True.", self)

    if not 0.0 <= zoneout_prob_cell <= 1.0:
      raise ValueError("Parameter zoneout_prob_cell must be in [0, 1]")

    if not 0.0 <= zoneout_prob_output <= 1.0:
      raise ValueError("Parameter zoneout_prob_output must be in [0, 1]")

    self._num_units = num_units
    self.is_training = is_training
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh
    self.zoneout_prob_cell = zoneout_prob_cell
    self.zoneout_prob_output = zoneout_prob_output

    if num_proj:
      self._state_size = (
          tf.contrib.rnn.LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          tf.contrib.rnn.LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state, scope=None):
    """Run one step of ZoneoutLSTMCell.
    Args:
      inputs: input Tensor, 2D, `[batch, num_units].
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
    Returns:
      A tuple containing:
      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        ZoneoutLSTMCell after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of ZoneoutLSTMCell after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj

    if self._state_is_tuple:
      (c_prev, h_prev) = state
    else:
      c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
      h_prev = tf.slice(state, [0, self._num_units], [-1, num_proj])

    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = _linear([inputs, h_prev], 4 * self._num_units, True)
    i, j, f, o = array_ops.split(lstm_matrix, 4, axis=1)

    # diagonal connections
    dtype = inputs.dtype
    if self._use_peepholes:
      w_f_diag = tf.get_variable("W_F_diag", shape=[self._num_units], dtype=dtype)
      w_i_diag = tf.get_variable("W_I_diag", shape=[self._num_units], dtype=dtype)
      w_o_diag = tf.get_variable("W_O_diag", shape=[self._num_units], dtype=dtype)

    with tf.name_scope("zoneout"):
      # binary mask tensor for cell
      keep_prob_cell = tf.convert_to_tensor(self.zoneout_prob_cell, dtype=c_prev.dtype)
      random_tensor_cell = keep_prob_cell
      random_tensor_cell += tf.random_uniform(tf.shape(c_prev), seed=None, dtype=c_prev.dtype)
      binary_mask_cell = tf.floor(random_tensor_cell)
      binary_mask_cell_complement = tf.ones(tf.shape(c_prev)) - binary_mask_cell

      # make binary mask tensor for output
      keep_prob_output = tf.convert_to_tensor(self.zoneout_prob_output, dtype=h_prev.dtype)
      random_tensor_output = keep_prob_output
      random_tensor_output += tf.random_uniform(tf.shape(h_prev), seed=None, dtype=h_prev.dtype)
      binary_mask_output = tf.floor(random_tensor_output)
      binary_mask_output_complement = tf.ones(tf.shape(h_prev)) - binary_mask_output

    # apply zoneout for cell
    if self._use_peepholes:
      c_temp = c_prev * tf.sigmoid(f + self._forget_bias + w_f_diag * c_prev) + \
               tf.sigmoid(i + w_i_diag * c_prev) * self._activation(j)
      if self.is_training and self.zoneout_prob_cell > 0.0:
        c = binary_mask_cell * c_prev + binary_mask_cell_complement * c_temp
      else:
        # like dropout, zoneout inference process will use the traditional cell state
        # TODO: fix bug here, inference should use the expectation of Ct-1 and Ct
        c = c_temp
    else:
      c_temp = c_prev * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j)
      if self.is_training and self.zoneout_prob_output > 0.0:
        c = binary_mask_cell * c_prev + binary_mask_cell_complement * c_temp
      else:
        # TODO: fix bug here, inference should use the expectation of Ct-1 and Ct
        c = c_temp

    if self._cell_clip:
        c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)

    # apply zoneout for output
    if self._use_peepholes:
      h_temp = tf.sigmoid(o + w_o_diag * c) * self._activation(c)
      if self.is_training and self.zoneout_prob_output > 0.0:
        h = binary_mask_output * h_prev + binary_mask_output_complement * h_temp
      else:
        # TODO: fix bug here, inference should use the expectation of Ht-1 and Ht
        h = h_temp
    else:
      h_temp = tf.sigmoid(o) * self._activation(c)
      if self.is_training and self.zoneout_prob_output > 0.0:
        h = binary_mask_output * h_prev + binary_mask_output_complement * h_temp
      else:
        # TODO: fix bug here, inference should use the expectation of Ht-1 and Ht
        h = h_temp

    # apply prejection
    if self._num_proj is not None:
      w_proj = tf.get_variable("W_P", [self.num_units, num_proj], dtype=dtype)
      h = tf.matmul(h, w_proj)
      if self._proj_clip is not None:
        h = tf.clip_by_value(h, -self._proj_clip, self._proj_clip)

    new_state = (tf.contrib.rnn.LSTMStateTuple(c, h)
                 if self._state_is_tuple else tf.concat([c, h], axis=1))

    return h, new_state


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(args, axis=1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            initializer=tf.constant_initializer(bias_start))

    return res + bias_term
