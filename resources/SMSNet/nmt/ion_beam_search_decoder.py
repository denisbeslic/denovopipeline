# Copyright 2017 Google Inc. All Rights Reserved.
# Copyright 2019 Korrawe Karunratanakul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A decoder that performs beam search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

import tensorflow as tf # for debugging

from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import nest

from . import input_config as aa_conf
from .utils import iterator_utils

__all__ = [
    "IonBeamSearchDecoder",
]


class IonBeamSearchDecoderState(
    collections.namedtuple("IonBeamSearchDecoderState",
                           ("cell_state", "log_probs", "finished", "lengths", "prefix_mass"))):
  pass


class BeamSearchDecoderOutput(
    collections.namedtuple("BeamSearchDecoderOutput",
                           ("scores", "predicted_ids", "parent_ids", "step_log_probs"))):
  pass


class FinalBeamSearchDecoderOutput(
    collections.namedtuple("FinalBeamDecoderOutput",
                           ["predicted_ids", "beam_search_decoder_output", "step_log_probs"])):
  """Final outputs returned by the beam search after all decoding is finished.
  Args:
    predicted_ids: The final prediction. A tensor of shape
      `[batch_size, T, beam_width]` (or `[T, batch_size, beam_width]` if
      `output_time_major` is True). Beams are ordered from best to worst.
    step_log_probs: log prob of each predicted_ids. A tensor of the same shape
      as predicted_ids.
    beam_search_decoder_output: An instance of `BeamSearchDecoderOutput` that
      describes the state of the beam search.
  """
  pass


class IonBeamSearchDecoder(beam_search_decoder.BeamSearchDecoder):
  
  '''
  Redefine step to match ion input.
  '''
  
  def __init__(self,
               cell,
               embedding,
               start_tokens,
               end_token,
               initial_state,
               beam_width,
               source_spectrum,
               peptide_mass, # <<<<<<<<<<<<<<
               suffix_dp_table, # <<<<<
               aa_weight_table,
               decoder_input_model_fn,
               output_layer=None,
               length_penalty_weight=0.0,
               reorder_tensor_arrays=True):
    """Initialize the BeamSearchDecoder.
    Args:
      cell: An `RNNCell` instance.
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
      beam_width:  Python integer, the number of beams.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
        to storing the result or sampling.
      length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
      reorder_tensor_arrays: If `True`, `TensorArray`s' elements within the cell
        state will be reordered according to the beam search path. If the
        `TensorArray` can be reordered, the stacked form will be returned.
        Otherwise, the `TensorArray` will be returned as is. Set this flag to
        `False` if the cell state contains `TensorArray`s that are not amenable
        to reordering.
    Raises:
      TypeError: if `cell` is not an instance of `RNNCell`,
        or `output_layer` is not an instance of `tf.layers.Layer`.
      ValueError: If `start_tokens` is not a vector or
        `end_token` is not a scalar.
    """
    # if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
    #  raise TypeError("cell must be an RNNCell, received: %s" % type(cell)) ## For tf-1.7
    # rnn_cell_impl.assert_like_rnncell("cell", cell)  # pylint: disable=protected-access
    if (output_layer is not None and
        not isinstance(output_layer, layers_base.Layer)):
      raise TypeError(
          "output_layer must be a Layer, received: %s" % type(output_layer))
      
    self._cell = cell
    self._output_layer = output_layer
    self._reorder_tensor_arrays = reorder_tensor_arrays

    if callable(embedding):
      self._embedding_fn = embedding
    else:
      self._embedding_fn = (
          lambda ids: embedding_ops.embedding_lookup(embedding, ids))

    self._start_tokens = ops.convert_to_tensor(
        start_tokens, dtype=dtypes.int32, name="start_tokens")
    if self._start_tokens.get_shape().ndims != 1:
      raise ValueError("start_tokens must be a vector")
    self._end_token = ops.convert_to_tensor(
        end_token, dtype=dtypes.int32, name="end_token")
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")

    # Change here ##########
    #print("Hellooo")
    ### self._start_tokens = tf.Print(self._start_tokens, [peptide_mass, tf.shape(peptide_mass)], message="Hello") ######### <<<<<<<<<<<<<
    self._peptide_mass = peptide_mass # array_ops.tile(
        # array_ops.expand_dims(peptide_mass, 1), [1, beam_width])
    self._source_spectrum = source_spectrum
    self._suffix_dp_table = suffix_dp_table
    self._aa_weight_table = aa_weight_table
    self._decoder_input_model_fn = decoder_input_model_fn
    
    
    self._batch_size = array_ops.size(start_tokens)
    self._beam_width = beam_width
    self._length_penalty_weight = length_penalty_weight
    self._initial_cell_state = nest.map_structure(
        self._maybe_split_batch_beams, initial_state, self._cell.state_size)
    self._start_tokens = array_ops.tile(
        array_ops.expand_dims(self._start_tokens, 1), [1, self._beam_width])
    self._start_inputs = self._embedding_fn(self._start_tokens)

    self._finished = array_ops.one_hot(
        array_ops.zeros([self._batch_size], dtype=dtypes.int32),
        depth=self._beam_width,
        on_value=False,
        off_value=True,
        dtype=dtypes.bool)


  @property
  def output_size(self):
    # Return the cell output and the id
    return BeamSearchDecoderOutput(
        scores=tensor_shape.TensorShape([self._beam_width]),
        predicted_ids=tensor_shape.TensorShape([self._beam_width]),
        parent_ids=tensor_shape.TensorShape([self._beam_width]),
        step_log_probs=tensor_shape.TensorShape([self._beam_width]))

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and int32 (the id)
    dtype = nest.flatten(self._initial_cell_state)[0].dtype
    return BeamSearchDecoderOutput(
        scores=nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        predicted_ids=dtypes.int32,
        parent_ids=dtypes.int32,
        step_log_probs=dtypes.float32)

  def initialize(self, name=None):
    """Initialize the decoder.
    Args:
      name: Name scope for any created operations.
    Returns:
      `(finished, start_inputs, initial_state)`.
    """
    finished, start_inputs = self._finished, self._start_inputs
    
    # Calculating candidate region inputs
    # set prefix mass to N terminus
    start_prefix = array_ops.ones(
      [self._batch_size, self._beam_width], dtype=dtypes.float32) * tf.constant(aa_conf.mass_N_terminus)
    
    
    with tf.name_scope('fetch_start_input'), tf.device("/gpu:0"):
      candidate_spec = tf.map_fn(
        lambda x: iterator_utils.get_candidate_intensity(x[0], x[1], x[2]), (self._source_spectrum, self._peptide_mass, start_prefix),
        dtype=tf.float32, parallel_iterations=200, back_prop=False)
    
    # apply input model
    next_candidate_input = self._decoder_input_model_fn(candidate_spec)
    # concat with embeded start token
    start_inputs_combined = tf.concat([start_inputs, next_candidate_input], 2)
    
    
    log_probs = array_ops.one_hot(  # shape(batch_sz, beam_sz)
        array_ops.zeros([self._batch_size], dtype=dtypes.int32),
        depth=self._beam_width,
        on_value=0.0,
        off_value=-np.Inf,
        dtype=nest.flatten(self._initial_cell_state)[0].dtype)

    initial_state = IonBeamSearchDecoderState(
        cell_state=self._initial_cell_state,
        log_probs=log_probs,
        finished=finished,
        prefix_mass=start_prefix, #######################
        lengths=array_ops.zeros(
            [self._batch_size, self._beam_width], dtype=dtypes.int64))

    return (finished, start_inputs_combined, initial_state) # start_inputs, initial_state)

  def finalize(self, outputs, final_state, sequence_lengths):
    """Finalize and return the predicted_ids.
    Args:
      outputs: An instance of BeamSearchDecoderOutput.
      final_state: An instance of BeamSearchDecoderState. Passed through to the
        output.
      sequence_lengths: An `int64` tensor shaped `[batch_size, beam_width]`.
        The sequence lengths determined for each beam during decode.
        **NOTE** These are ignored; the updated sequence lengths are stored in
        `final_state.lengths`.
    Returns:
      outputs: An instance of `FinalBeamSearchDecoderOutput` where the
        predicted_ids are the result of calling _gather_tree.
      final_state: The same input instance of `BeamSearchDecoderState`.
    """
    del sequence_lengths
    # Get max_sequence_length across all beams for each batch.
    max_sequence_lengths = math_ops.to_int32(
        math_ops.reduce_max(final_state.lengths, axis=1))
    predicted_ids = beam_search_ops.gather_tree(
        outputs.predicted_ids,
        outputs.parent_ids,
        max_sequence_lengths=max_sequence_lengths,
        end_token=self._end_token)
    # predicted_ids = tf.Print(predicted_ids, [tf.shape(predicted_ids), predicted_ids], summarize=50)
    
    step_probs = gather_tree(
        outputs.step_log_probs,
        outputs.parent_ids)
    # predicted_ids = tf.Print(predicted_ids, [tf.shape(step_probs), step_probs], summarize=50, message="step")
    
    outputs = FinalBeamSearchDecoderOutput(
        beam_search_decoder_output=outputs, predicted_ids=predicted_ids, step_log_probs=step_probs)
    return outputs, final_state

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.
    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.
    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    batch_size = self._batch_size
    beam_width = self._beam_width
    end_token = self._end_token
    length_penalty_weight = self._length_penalty_weight

    with ops.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
      cell_state = state.cell_state
      inputs = nest.map_structure(
          lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
      cell_state = nest.map_structure(self._maybe_merge_batch_beams, cell_state,
                                      self._cell.state_size)
      cell_outputs, next_cell_state = self._cell(inputs, cell_state)
      cell_outputs = nest.map_structure(
          lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
      next_cell_state = nest.map_structure(
          self._maybe_split_batch_beams, next_cell_state, self._cell.state_size)

      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)

      beam_search_output, beam_search_state = _beam_search_step( # beam_search_decoder.
          time=time,
          logits=cell_outputs,
          next_cell_state=next_cell_state,
          beam_state=state,
          peptide_mass=self._peptide_mass,
          batch_size=batch_size,
          beam_width=beam_width,
          end_token=end_token,
          length_penalty_weight=length_penalty_weight,
          suffix_dp_table = self._suffix_dp_table,
          aa_weight_table=self._aa_weight_table)

      finished = beam_search_state.finished
      sample_ids = beam_search_output.predicted_ids
      
      with tf.name_scope('fetch_new_input'), tf.device("/gpu:0"):
        candidate_spec = tf.map_fn(
          lambda x: iterator_utils.get_candidate_intensity(x[0], x[1], x[2]), (self._source_spectrum, self._peptide_mass, beam_search_state.prefix_mass),
          dtype=tf.float32, parallel_iterations=200, back_prop=False)
        
      next_inputs = control_flow_ops.cond(
          math_ops.reduce_all(finished), lambda: self._start_inputs,
          lambda: self._embedding_fn(sample_ids))
      
      next_candidate_input = self._decoder_input_model_fn(candidate_spec)
      
      next_combine_inp = tf.concat([next_inputs, next_candidate_input], 2)
      
    return (beam_search_output, beam_search_state, next_combine_inp, finished)


def gather_tree_py(values, parents):
  """Gathers path through a tree backwards from the leave nodes. Used
  to reconstruct beams given their parents."""
  beam_length = values.shape[0]
  num_beams = values.shape[1]
  res = np.zeros_like(values)
  res[-1, :] = values[-1, :]
  for beam_id in range(num_beams):
    parent = parents[-1][beam_id]
    for level in reversed(range(beam_length - 1)):
      res[level, beam_id] = values[level, beam_id][parent]
      parent = parents[level, beam_id][parent]
  return np.array(res).astype(values.dtype)


def gather_tree(values, parents):
  """Tensor version of gather_tree_py"""
  res = tf.py_func(
      func=gather_tree_py, inp=[values, parents], Tout=values.dtype)
  res.set_shape(values.get_shape().as_list())
  return res


def penalize_invalid_mass(prefix_mass, total_probs, peptide_mass, beam_width, suffix_dp_table, aa_weight_table):
  inv_dp_resolution = aa_conf.inv_dp_resolution
  
  # prefix_mass [batch, beam],    weight [23]
  # want each prefix_mass + each weight [batch, beam, 23]
  new_mass = tf.expand_dims(prefix_mass, 2) + aa_weight_table
  peptide_mass_tile = tf.tile(tf.expand_dims(peptide_mass, 1), [1, beam_width])
  suffix_mass = tf.expand_dims(peptide_mass_tile, 2) - new_mass - (aa_conf.mass_C_terminus + 2.0 * aa_conf.mass_H)
  bounded_suffix_mass = tf.where( tf.logical_or(suffix_mass < 0, suffix_mass > 1499), tf.zeros_like(suffix_mass), suffix_mass)

  # round
  # find positions of each suffix mass on DP table # [batch, beam, 23]
  suffix_mass_idx = tf.cast(tf.round(bounded_suffix_mass * inv_dp_resolution), tf.int32)
  # lookup dp table
  possible_mass = tf.gather(suffix_dp_table, suffix_mass_idx)
  # suffix mass must be higher than 0 (discard ones that have more mass than the peptide)
  possible_mass = tf.logical_and(possible_mass, suffix_mass > 0)
  # allow mass that is more than 1499 as all of them are possible under this config
  possible_mass = tf.logical_or(possible_mass, suffix_mass > 1499)
  # also allow for 10 ppm error in case the prefix use all of the posible mass
  possible_mass = tf.logical_or(possible_mass, tf.abs(suffix_mass) <= 0.03) # 0.03 for 10 ppm // 0.00001 * peptide_mass??

  new_total_probs = tf.where(possible_mass, total_probs, total_probs - 100)
  return new_total_probs


def _beam_search_step(time, logits, next_cell_state, beam_state, peptide_mass, batch_size,
                      beam_width, end_token, length_penalty_weight, suffix_dp_table, aa_weight_table):
  """Performs a single step of Beam Search Decoding.
  Args:
    time: Beam search time step, should start at 0. At time 0 we assume
      that all beams are equal and consider only the first beam for
      continuations.
    logits: Logits at the current time step. A tensor of shape
      `[batch_size, beam_width, vocab_size]`
    next_cell_state: The next state from the cell, e.g. an instance of
      AttentionWrapperState if the cell is attentional.
    beam_state: Current state of the beam search.
      An instance of `IonBeamSearchDecoderState`.
    batch_size: The batch size for this input.
    beam_width: Python int.  The size of the beams.
    end_token: The int32 end token.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
  Returns:
    A new beam state.
  """
  static_batch_size = tensor_util.constant_value(batch_size)

  # Calculate the current lengths of the predictions
  prediction_lengths = beam_state.lengths
  previously_finished = beam_state.finished

  # Calculate the total log probs for the new hypotheses
  # Final Shape: [batch_size, beam_width, vocab_size]
  step_log_probs = nn_ops.log_softmax(logits)
  step_log_probs = beam_search_decoder._mask_probs(step_log_probs, end_token, previously_finished)
  total_probs = array_ops.expand_dims(beam_state.log_probs, 2) + step_log_probs
  
  # Penalize beams with invalid total mass according to dp array of possible mass
  new_total_probs = penalize_invalid_mass(beam_state.prefix_mass, total_probs, peptide_mass, beam_width, suffix_dp_table, aa_weight_table)
  total_probs = new_total_probs

  # Calculate the continuation lengths by adding to all continuing beams.
  vocab_size = logits.shape[-1].value or array_ops.shape(logits)[-1]
  lengths_to_add = array_ops.one_hot(
      indices=array_ops.fill([batch_size, beam_width], end_token),
      depth=vocab_size,
      on_value=np.int64(0),
      off_value=np.int64(1),
      dtype=dtypes.int64)
  add_mask = math_ops.to_int64(math_ops.logical_not(previously_finished))
  lengths_to_add *= array_ops.expand_dims(add_mask, 2)
  new_prediction_lengths = (
      lengths_to_add + array_ops.expand_dims(prediction_lengths, 2))

  # Calculate the scores for each beam
  scores = beam_search_decoder._get_scores(
      log_probs=total_probs,
      sequence_lengths=new_prediction_lengths,
      length_penalty_weight=length_penalty_weight)

  time = ops.convert_to_tensor(time, name="time")
  # During the first time step we only consider the initial beam
  scores_shape = array_ops.shape(scores)
  scores_flat = array_ops.reshape(scores, [batch_size, -1])

  # Pick the next beams according to the specified successors function
  next_beam_size = ops.convert_to_tensor(
      beam_width, dtype=dtypes.int32, name="beam_width")
  next_beam_scores, word_indices = nn_ops.top_k(scores_flat, k=next_beam_size)

  # word_indices = tf.Print(word_indices, [tf.shape(scores_flat)], message="** next beam shape")
  next_beam_scores.set_shape([static_batch_size, beam_width])
  word_indices.set_shape([static_batch_size, beam_width])

  # Pick out the probs, beam_ids, and states according to the chosen predictions
  next_beam_probs = beam_search_decoder._tensor_gather_helper(
      gather_indices=word_indices,
      gather_from=total_probs,
      batch_size=batch_size,
      range_size=beam_width * vocab_size,
      gather_shape=[-1],
      name="next_beam_probs")
    
  # Pick out log probs for each step according to next_beam_id
  next_step_probs = beam_search_decoder._tensor_gather_helper(
    gather_indices=word_indices,
    gather_from=step_log_probs,
    batch_size=batch_size,
    range_size=beam_width * vocab_size,
    gather_shape=[-1],
    name="next_step_probs")
  # Note: just doing the following
  #   math_ops.to_int32(word_indices % vocab_size,
  #       name="next_beam_word_ids")
  # would be a lot cleaner but for reasons unclear, that hides the results of
  # the op which prevents capturing it with tfdbg debug ops.
  raw_next_word_ids = math_ops.mod(
      word_indices, vocab_size, name="next_beam_word_ids")
  next_word_ids = math_ops.to_int32(raw_next_word_ids)
  next_beam_ids = math_ops.to_int32(
      word_indices / vocab_size, name="next_beam_parent_ids")
  
  # Append new ids to current predictions
  previously_finished = beam_search_decoder._tensor_gather_helper(
      gather_indices=next_beam_ids,
      gather_from=previously_finished,
      batch_size=batch_size,
      range_size=beam_width,
      gather_shape=[-1])
  next_finished = math_ops.logical_or(
      previously_finished,
      math_ops.equal(next_word_ids, end_token),
      name="next_beam_finished")

  # Calculate the length of the next predictions.
  # 1. Finished beams remain unchanged.
  # 2. Beams that are now finished (EOS predicted) have their length
  #    increased by 1.
  # 3. Beams that are not yet finished have their length increased by 1.
  lengths_to_add = math_ops.to_int64(math_ops.logical_not(previously_finished))
  next_prediction_len = beam_search_decoder._tensor_gather_helper(
      gather_indices=next_beam_ids,
      gather_from=beam_state.lengths,
      batch_size=batch_size,
      range_size=beam_width,
      gather_shape=[-1])
  next_prediction_len += lengths_to_add

  # Pick prefix_mass according to the next_beam_id
  next_prefix_mass = beam_search_decoder._tensor_gather_helper(
    gather_indices=next_beam_ids,
    gather_from=beam_state.prefix_mass,
    batch_size=batch_size,
    range_size=beam_width,
    gather_shape=[-1])
  next_prefix_mass = next_prefix_mass + tf.gather(aa_weight_table, next_word_ids)
  
  # Pick out the cell_states according to the next_beam_ids. We use a
  # different gather_shape here because the cell_state tensors, i.e.
  # the tensors that would be gathered from, all have dimension
  # greater than two and we need to preserve those dimensions.
  # pylint: disable=g-long-lambda
  next_cell_state = nest.map_structure(
      lambda gather_from: beam_search_decoder._maybe_tensor_gather_helper(
          gather_indices=next_beam_ids,
          gather_from=gather_from,
          batch_size=batch_size,
          range_size=beam_width,
          gather_shape=[batch_size * beam_width, -1]),
      next_cell_state)
  # pylint: enable=g-long-lambda

  next_state = IonBeamSearchDecoderState(
      cell_state=next_cell_state,
      log_probs=next_beam_probs,
      lengths=next_prediction_len,
      prefix_mass=next_prefix_mass,
      finished=next_finished)
  
  output = BeamSearchDecoderOutput(
      scores=next_beam_scores,
      predicted_ids=next_word_ids,
      parent_ids=next_beam_ids,
      step_log_probs=next_step_probs)

  return output, next_state
