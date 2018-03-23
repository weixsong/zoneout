# This is a minimal gist of what you'd have to
# add to TensorFlow code to implement zoneout.

# To see this in action, see zoneout_seq2seq.py
import tensorflow as tf

z_prob_cells = 0.05
z_prob_states = 0


# Wrapper for the TF RNN cell
# For an LSTM, the 'cell' is a tuple containing state and cell
# We use TF's dropout to implement zoneout
class ZoneoutWrapper(tf.nn.rnn_cell.RNNCell):
    """Operator adding zoneout to all states (states+cells) of the given cell."""

    def __init__(self, cell, zoneout_prob, is_training=True, seed=None):
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not an RNNCell.")
        if not isinstance(zoneout_prob, tuple):
            raise TypeError("Paramter zoneout_prob must be a tuple (0.1, 0.2)")

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
            # concern
            # 1. why add (1 - state_part_zoneout_prob) to dropout layer, dorpout on difference of new_state and old state
            #    and then add old state, is enought to represent the zoneout equation.
            # 2. return output, new_state is zoneout, but output still not zoneout, as actual output will be used by next layer
            # 3. paper not mentioned in inference if zoneout still used or as dropout not used.
            #    in inference, why not use the state directly?
            if self.is_training:
                new_state = tuple((1 - state_part_zoneout_prob) * tf.python.nn_ops.dropout(
                    new_state_part - state_part, (1 - state_part_zoneout_prob), seed=self._seed) + state_part
                                  for new_state_part, state_part, state_part_zoneout_prob
                                  in zip(new_state, state, self._zoneout_prob))
            else:
                new_state = tuple(state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
                                  for new_state_part, state_part, state_part_zoneout_prob
                                  in zip(new_state, state, self._zoneout_prob))
        else:
            if self.is_training:
                new_state = (1 - state_part_zoneout_prob) * tf.python.nn_ops.dropout(
                        new_state_part - state_part, (1 - state_part_zoneout_prob), seed=self._seed) + state_part
            else:
                new_state = state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part

        return output, new_state


# Wrap your cells like this
cell = ZoneoutWrapper(tf.nn.rnn_cell.LSTMCell(128),
                      zoneout_prob=z_prob_cells,
                      state_zoneout_prob=z_prob_states)
