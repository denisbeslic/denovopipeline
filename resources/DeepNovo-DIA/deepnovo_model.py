# Copyright 2015 Google Inc. All Rights Reserved.
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

# ==============================================================================
# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
#
# The source code in this file originated from the sequence-to-sequence tutorial
# of TensorFlow, Google Inc. I have modified the entire code to solve the 
# problem of peptide sequencing. The copyright notice of Google is attached 
# above as required by its Apache License.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

import deepnovo_config


class TrainingModel(object):
  """TODO(nh2tran): docstring."""


  def __init__(self, session, training_mode): # TODO(nh2tran): session-unused
    """TODO(nh2tran): docstring."""

    print("TrainingModel: __init__()")

    self.global_step = tf.Variable(0, trainable=False)

    # Dropout probabilities
    self.dropout_keep = {}
    self.dropout_keep["conv"] = tf.placeholder(dtype=tf.float32, name="keep_conv")
    self.dropout_keep["dense"] = tf.placeholder(dtype=tf.float32, name="keep_dense")

    # INPUT PLACEHOLDERS

    # spectrum
    self.spectrum_holder = tf.placeholder(
        dtype=tf.float32,
        shape=[None, deepnovo_config.neighbor_size, deepnovo_config.MZ_SIZE],
        name="spectrum_holder")
    self.ms1_profile = tf.placeholder(
        dtype=tf.float32,
        shape=[None, deepnovo_config.neighbor_size],
        name="ms1_profile")

    # candidate intensity
    self.intensity_inputs_forward = []
    self.intensity_inputs_backward = []
    for x in xrange(deepnovo_config._buckets[-1]): # TODO(nh2tran): _buckets
      self.intensity_inputs_forward.append(tf.placeholder(
          dtype=tf.float32,
          shape=[None,
                 deepnovo_config.vocab_size,
                 deepnovo_config.neighbor_size*deepnovo_config.num_ion,
                 deepnovo_config.WINDOW_SIZE], # TODO(nh2tran): line-too-long, config
          name="intensity_inputs_forward_{0}".format(x)))
      self.intensity_inputs_backward.append(tf.placeholder(
          dtype=tf.float32,
          shape=[None,
                 deepnovo_config.vocab_size,
                 deepnovo_config.neighbor_size*deepnovo_config.num_ion,
                 deepnovo_config.WINDOW_SIZE], # TODO(nh2tran): line-too-long, config
          name="intensity_inputs_backward_{0}".format(x)))

    # decoder inputs
    self.decoder_inputs_forward = []
    self.decoder_inputs_backward = []
    self.target_weights = []
    for x in xrange(deepnovo_config._buckets[-1] + 1): # TODO(nh2tran): _buckets
      self.decoder_inputs_forward.append(tf.placeholder(
          dtype=tf.int32,
          shape=[None],
          name="decoder_inputs_forward_{0}".format(x)))
      self.decoder_inputs_backward.append(tf.placeholder(
          dtype=tf.int32,
          shape=[None],
          name="decoder_inputs_backward_{0}".format(x)))
      self.target_weights.append(tf.placeholder(
          dtype=tf.float32,
          shape=[None],
          name="target_weights_{0}".format(x)))

    # Our targets are decoder inputs shifted by one.
    self.targets_forward = [self.decoder_inputs_forward[x + 1]
                            for x in xrange(len(self.decoder_inputs_forward) - 1)] # TODO(nh2tran): line-too-long
    self.targets_backward = [self.decoder_inputs_backward[x + 1]
                             for x in xrange(len(self.decoder_inputs_backward) - 1)] # TODO(nh2tran): line-too-long

    # OUTPUTS and LOSSES
    self.outputs_forward, self.outputs_backward, self.losses, self.losses_classification = self._build_model(
        self.spectrum_holder,
        self.ms1_profile,
        self.intensity_inputs_forward,
        self.intensity_inputs_backward,
        self.decoder_inputs_forward,
        self.decoder_inputs_backward,
        self.targets_forward,
        self.targets_backward,
        self.target_weights,
        self.dropout_keep)
    #~ (self.outputs_forward,
     #~ self.outputs_backward,
     #~ self.losses) = deepnovo_model_training.train(self.encoder_inputs,
                                           #~ self.intensity_inputs_forward,
                                           #~ self.intensity_inputs_backward,
                                           #~ self.decoder_inputs_forward,
                                           #~ self.decoder_inputs_backward,
                                           #~ self.targets_forward,
                                           #~ self.targets_backward,
                                           #~ self.target_weights,
                                           #~ self.keep_conv_holder,
                                           #~ self.keep_dense_holder)

    # Gradients and SGD update operation for training the model.
    if training_mode:
      params = tf.trainable_variables()
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.AdamOptimizer()
      for b in xrange(len(deepnovo_config._buckets)): # TODO(nh2tran): _buckets
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(
            gradients,
            deepnovo_config.max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params),
            global_step=self.global_step))

      # for TensorBoard
      #~ self.train_writer = tf.train.SummaryWriter(deepnovo_config.FLAGS.train_dir, session.graph)
      #~ self.loss_summaries = [tf.scalar_summary("losses_" + str(b), self.losses[b])
                             #~ for b in xrange(len(deepnovo_config._buckets))]
      #~ dense1_W_penalty = tf.get_default_graph().get_tensor_by_name(
                         #~ "model_with_buckets/embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder_forward/dense1_W_penalty:0")
      #~ self.dense1_W_penalty_summary = tf.scalar_summary("dense1_W_penalty_summary", dense1_W_penalty)

    # Saver
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


  def step(self,
           session,
           spectrum_holder,
           ms1_profile,
           intensity_inputs_forward=None,
           intensity_inputs_backward=None,
           decoder_inputs_forward=None,
           decoder_inputs_backward=None,
           target_weights=None,
           bucket_id=0,
           training_mode=True):
    """TODO(nh2tran): docstring."""

    # Check if the sizes match.
    decoder_size = deepnovo_config._buckets[bucket_id] # TODO(nh2tran): _buckets

    # Input feed
    input_feed = {}
    input_feed[self.spectrum_holder.name] = spectrum_holder
    input_feed[self.ms1_profile.name] = ms1_profile

    # Input feed forward
    if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 2:
      for x in xrange(decoder_size):
        input_feed[self.intensity_inputs_forward[x].name] = intensity_inputs_forward[x] # TODO(nh2tran): line-too-long
        input_feed[self.decoder_inputs_forward[x].name] = decoder_inputs_forward[x] # TODO(nh2tran): line-too-long
      # Since our targets are decoder inputs shifted by one, we need one more.
      last_target_forward = self.decoder_inputs_forward[decoder_size].name
      input_feed[last_target_forward] = np.zeros([spectrum_holder.shape[0]],
                                                 dtype=np.int32)

    # Input feed backward
    if deepnovo_config.FLAGS.direction == 1 or deepnovo_config.FLAGS.direction == 2:
      for x in xrange(decoder_size):
        input_feed[self.intensity_inputs_backward[x].name] = intensity_inputs_backward[x] # TODO(nh2tran): line-too-long
        input_feed[self.decoder_inputs_backward[x].name] = decoder_inputs_backward[x] # TODO(nh2tran): line-too-long
      # Since our targets are decoder inputs shifted by one, we need one more.
      last_target_backward = self.decoder_inputs_backward[decoder_size].name
      input_feed[last_target_backward] = np.zeros([spectrum_holder.shape[0]],
                                                  dtype=np.int32)

    # Input feed target weights
    for x in xrange(decoder_size):
      input_feed[self.target_weights[x].name] = target_weights[x]

    # keeping probability for dropout layers
    if training_mode:
      input_feed[self.dropout_keep["conv"].name] = deepnovo_config.keep_conv
      input_feed[self.dropout_keep["dense"].name] = deepnovo_config.keep_dense
    else:
      input_feed[self.dropout_keep["conv"].name] = 1.0
      input_feed[self.dropout_keep["dense"].name] = 1.0

    # Output feed: depends on whether we do a back-propagation
    if training_mode:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id], # Loss for this batch.
                     self.losses_classification[bucket_id]]

    # Output forward logits
    if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 2:
      for x in xrange(decoder_size):
        output_feed.append(self.outputs_forward[bucket_id][x])

    # Output backward logits
    if deepnovo_config.FLAGS.direction == 1 or deepnovo_config.FLAGS.direction == 2:
      for x in xrange(decoder_size):
        output_feed.append(self.outputs_backward[bucket_id][x])

    # RUN
    outputs = session.run(fetches=output_feed, feed_dict=input_feed)

    # DEBUG
    #~ np.set_printoptions(precision=1)
    #~ tensor_ms1 = "ms1_profile:0"
    #~ tensor_ms2 = "model_with_buckets/embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder_forward/Reshape:0"
    #~ tensor_corr = "model_with_buckets/embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder_forward/Squeeze:0"
    #~ tensor_list = [tensor_ms1, tensor_ms2, tensor_corr]
    #~ tensor_ms1, tensor_ms2, tensor_corr = session.run(tensor_list, feed_dict=input_feed)
    #~ print(tensor_corr[2,:])
    #~ print(tensor_ms1[2,:])
    #~ for aa in xrange(26):
      #~ ms2_profile = np.sum(tensor_ms2[2,aa,:,:,:], axis=(0, 2,))
      #~ print(np.corrcoef(tensor_ms1[2,:], ms2_profile)[0, 1])
    #~ print(abc)

    # for TensorBoard
    #~ if (training_mode and (self.global_step.eval() % deepnovo_config.steps_per_checkpoint == 0)):
      #~ summary_op = tf.merge_summary([self.loss_summaries[bucket_id], self.dense1_W_penalty_summary])
      #~ summary_str = session.run(summary_op, feed_dict=input_feed)
      #~ self.train_writer.add_summary(summary_str, self.global_step.eval())

    if training_mode:
      # loss, [outputs_forward, outputs_backward]
      if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 1:
        return outputs[2], outputs[3:]
      else:
        return outputs[2], outputs[3:(3+decoder_size)], outputs[(3+decoder_size):] # TODO(nh2tran): line-too-long
    else:
      # loss, loss without regularization, [outputs_forward, outputs_backward]
      if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 1:
        return outputs[0], outputs[1], outputs[2:]
      else:
        return outputs[0], outputs[1], outputs[2:(2+decoder_size)], outputs[(2+decoder_size):] # TODO(nh2tran): line-too-long


  def _build_model(self,
                   spectrum_holder,
                   ms1_profile,
                   intensity_inputs_forward,
                   intensity_inputs_backward,
                   decoder_inputs_forward,
                   decoder_inputs_backward,
                   targets_forward,
                   targets_backward,
                   target_weights,
                   dropout_keep):
    """TODO(nh2tran): docstring."""

    all_inputs = ([spectrum_holder]
                  + intensity_inputs_forward
                  + intensity_inputs_backward
                  + decoder_inputs_forward
                  + decoder_inputs_backward
                  + targets_forward
                  + targets_backward
                  + target_weights)
    losses = [] # loss from classification and regularization
    losses_classification = [] # loss from classification only, without regularization
    outputs_forward = []
    outputs_backward = []
    model_network = ModelNetwork()
    #~ with tf.name_scope(name="model_with_buckets", values=all_inputs):
    with ops.op_scope(all_inputs, name="model_with_buckets"):
      for j, bucket in enumerate(deepnovo_config._buckets): # TODO(nh2tran): _buckets
        with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                           reuse=True if j > 0 else None):
          ### build bucket network to calculate output logits
          (bucket_outputs_forward,
           bucket_outputs_backward,
           l2_loss) = model_network.build_network_series(
                spectrum_holder,
                ms1_profile,
                intensity_inputs_forward[:bucket],
                intensity_inputs_backward[:bucket],
                decoder_inputs_forward[:bucket],
                decoder_inputs_backward[:bucket],
                dropout_keep)
          outputs_forward.append(bucket_outputs_forward)
          outputs_backward.append(bucket_outputs_backward)
          ### calculate losses
          if deepnovo_config.FLAGS.direction == 0:
            sequence_loss = self._sequence_loss(outputs_forward[-1],
                                                targets_forward[:bucket],
                                                target_weights[:bucket],
                                                name="sequence_loss_forward")
          elif deepnovo_config.FLAGS.direction == 1:
            sequence_loss = self._sequence_loss(outputs_backward[-1],
                                                targets_backward[:bucket],
                                                target_weights[:bucket],
                                                name="sequence_loss_backward")
          else:
            sequence_loss = ((self._sequence_loss(outputs_forward[-1],
                                                  targets_forward[:bucket],
                                                  target_weights[:bucket],
                                                  name="sequence_loss_forward")
                              + self._sequence_loss(outputs_backward[-1],
                                                    targets_backward[:bucket],
                                                    target_weights[:bucket],
                                                    name="sequence_loss_backward")) / 2)
          ### l2 regularization
          losses_classification.append(sequence_loss)
          losses.append(sequence_loss + l2_loss)
  
    return outputs_forward, outputs_backward, losses, losses_classification


  def _sequence_loss(self,
                     logits,
                     targets,
                     weights,
                     name):
    """TODO(nh2tran): docstring.
    Weighted cross-entropy loss for a sequence of logits, batch-collapsed.
  
    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      average_across_batch: If set, divide the returned cost by the batch size.
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      name: Optional name for this operation, defaults to "sequence_loss".
  
    Returns:
      A scalar float Tensor: The average log-perplexity per symbol (weighted).
  
    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    """

    #~ with tf.name_scope(name=name,
                       #~ values=logits + targets + weights):
    with ops.op_scope(logits + targets + weights, name):
      cost = math_ops.reduce_sum(self._sequence_loss_per_sample(logits,
                                                                targets,
                                                                weights))
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, dtypes.float32)


  def _sequence_loss_per_sample(self,
                                logits,
                                targets,
                                weights):
    """TODO(nh2tran): docstring.
    Weighted cross-entropy loss for a sequence of logits (per example).
  
    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      name: Optional name for this operation, default: "sequence_loss_by_example".
  
    Returns:
      1D batch-sized float Tensor: The log-perplexity for each sequence.
  
    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    """

    #~ with tf.name_scope(name="sequence_loss_by_example",
                       #~ values=logits + targets + weights):
    with ops.op_scope(logits + targets + weights,
                      None,
                      "sequence_loss_by_example"):
      loss_function = self._focal_loss if deepnovo_config.focal_loss else nn_ops.sparse_softmax_cross_entropy_with_logits
      if deepnovo_config.focal_loss:
        print('=' * 80)
        print('USE FOCAL LOSS')
      log_perp_list = []
      for logit, target, weight in zip(logits, targets, weights):
        target = array_ops.reshape(math_ops.to_int64(target), [-1])
        crossent = loss_function(logits=logit, labels=target)
        log_perp_list.append(crossent * weight)
      log_perps = math_ops.add_n(log_perp_list)
      # average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
      return log_perps

  @staticmethod
  def _focal_loss(logits, labels, gamma=2):
    """Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
      logits: A float tensor of shape [batch_size, num_classes],
        representing the predicted logits for each class
      labels: A int32 tensor of shape [batch_size],
        representing one-hot encoded classification targets
      gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A [batch] tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(logits)
    num_classes = tf.shape(logits)[-1]
    target_tensor = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    pos_p_sub = array_ops.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = -  (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent, axis=-1)


class ModelNetwork(object):
  """TODO(nh2tran): docstring.
     Core neural networks to calculate the probability of the next amino acid.
  """


  def __init__(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("ModelNetwork: __init__()")

    # we currently use deepnovo_config to store both const & settings
    # the settings should be shown in __init__() to keep track carefully
    self.batch_size = deepnovo_config.batch_size
    self.neighbor_size = deepnovo_config.neighbor_size
    self.MZ_SIZE = deepnovo_config.MZ_SIZE
    self.SPECTRUM_RESOLUTION = deepnovo_config.SPECTRUM_RESOLUTION
    self.vocab_size = deepnovo_config.vocab_size
    self.num_ion = deepnovo_config.num_ion
    self.WINDOW_SIZE = deepnovo_config.WINDOW_SIZE
    self.l2_weight = deepnovo_config.l2_weight
    self.num_units = deepnovo_config.num_units
    self.embedding_size = deepnovo_config.embedding_size
    self.num_layers = deepnovo_config.num_layers
    self.use_ion = deepnovo_config.FLAGS.use_intensity # TODO(nh2tran): change to "use_ion"
    self.use_lstm = deepnovo_config.FLAGS.use_lstm
    self.lstm_kmer = deepnovo_config.FLAGS.lstm_kmer

    # keep_prob probability of dropout layers, will be defined in build()
    self.dropout_keep = None

    # record the name of variables for l2 regularization
    self.l2_var_name = set()


  def build_network(self, input_dict, dropout_keep):
    """TODO(nh2tran): docstring.
       Build neural networks to calculate the probability of the next amino acid.

       Inputs:
         Input tensors are grouped into a dictionary.
         input_dict["spectrum"]: 3D tensor [batch_size, neighbor_size, MZ_SIZE].
         input_dict["intensity"]: [batch_size, vocab_size, neighbor_size*num_ion, WINDOW_SIZE].
         input_dict["lstm_state"]: tuple of 2 tensors [batch_size, num_units]
         input_dict["AAid"]: list of 2 tensors [batch_size]
         dropout_keep["conv"]: keep_prob of dropout after convolutional layers
         dropout_keep["dense"]: keep_prob of dropout after dense layers

       Outputs:
         Output tensors are grouped into 2 dictionaries, output_forward and
         output_backward, each has 4 tensors:
         ["logit"]: [batch_size, vocab_size], to compute loss in training
         ["logprob"]: [batch_size, vocab_size], to compute score in inference
         ["lstm_state"]: [batch_size, num_units], to compute next iteration
         ["lstm_state0"]: [batch_size, num_units], state from cnn_spectrum
    """

    print("".join(["="] * 80)) # section-separating line
    print("ModelNetwork: build_network()")

    self.dropout_keep = dropout_keep

    ### spectrum and AA embedding spaces do not depend on directions
    if self.use_lstm:
      cnn_spectrum_feature = self._build_cnn_spectrum(input_dict["spectrum_holder"])
    else: # remove heavy cnn_spectrum when not use_lstm
      cnn_spectrum_feature = tf.zeros(shape=[self.batch_size, self.num_units])
    embedding_AAid = self._build_embedding_AAid(input_dict["AAid"])

    ### single-layer LSTM use single tuple (c, h) state, but multi-layer LSTM
    # use a list of tuples. For consistency, we use list to store LSTM state,
    # even for one-item list [(c, h)].
    input_lstm_state0 = input_dict["lstm_state0"]
    input_lstm_state = input_dict["lstm_state"]
    if self.num_layers == 1:
      input_lstm_state0 = input_lstm_state0[0]
      input_lstm_state = input_lstm_state[0]

    ### bi-directional sequencing, each uses a diffferent set of parameters
    output_forward = {}
    output_backward = {}
    for direction, output in zip(["forward", "backward"],
                                 [output_forward, output_backward]):
      scope = "embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder" # TODO(nh2tran): change to "cnn_ion/lstm"
      scope = scope + "_" + direction
      with tf.variable_scope(scope):
        # cnn_ion model
        cnn_ion_feature, cnn_ion_logit = self._build_cnn_ion(input_dict["intensity"],
                                                             input_dict["ms1_profile"])
        # lstm model
        cell, lstm_state0 = self._build_lstm_0(cnn_spectrum_feature)
        lstm_feature, lstm_logit, lstm_state = self._build_lstm_iter(
            cell,
            input_lstm_state0,
            input_lstm_state,
            embedding_AAid)
        # combine cnn_ion and lstm features
        feature_logit = self._combine_feature(cnn_ion_feature, lstm_feature)
        # both ion-lstm models are used together by default
        #   but each can be used separately for investigation
        if self.use_ion and self.use_lstm:
          logit = feature_logit
        elif self.use_ion:
          logit = cnn_ion_logit
        elif self.use_lstm:
          logit = lstm_logit
        else:
          print("Error: wrong ion-lstm model!")
          sys.exit()
        # final softmax layer
        logprob = tf.log(tf.nn.softmax(logit))
        output["logit"] = logit
        output["logprob"] = logprob
        if self.num_layers == 1:
          lstm_state = [lstm_state]
          lstm_state0 = [lstm_state0]
        output["lstm_state"] = lstm_state
        output["lstm_state0"] = lstm_state0

    return output_forward, output_backward


  def build_network_series(self,
                           spectrum_holder,
                           ms1_profile,
                           intensity_inputs_forward,
                           intensity_inputs_backward,
                           decoder_inputs_forward,
                           decoder_inputs_backward,
                           dropout_keep):
    """TODO(nh2tran): docstring."""

    self.dropout_keep = dropout_keep

    ### spectrum and AA embedding spaces do not depend on directions
    if self.use_lstm:
      cnn_spectrum_feature = self._build_cnn_spectrum(spectrum_holder)
    else: # remove heavy cnn_spectrum when not use_lstm
      cnn_spectrum_feature = tf.zeros(shape=[self.batch_size, self.num_units])
    decoder_inputs_forward_emb = self._build_embedding_AAid(decoder_inputs_forward)
    decoder_inputs_backward_emb = self._build_embedding_AAid(decoder_inputs_backward, reuse=True)

    ### bi-directional sequencing
    output_forward = []
    output_backward = []
    for direction, intensity_inputs, decoder_inputs_emb, output in zip(
        ["forward", "backward"],
        [intensity_inputs_forward, intensity_inputs_backward],
        [decoder_inputs_forward_emb, decoder_inputs_backward_emb],
        [output_forward, output_backward]):

      ### each direction uses a diffferent set of parameters
      scope = "embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder" # TODO(nh2tran): change to "cnn_ion/lstm"
      scope = scope + "_" + direction
      with tf.variable_scope(scope):

        ### initialize lstm cell
        cell, lstm_state0 = self._build_lstm_0(cnn_spectrum_feature)
        lstm_state = lstm_state0
        # padding [AA_1, AA_2, ?] with GO/EOS
        AA_1 = decoder_inputs_emb[0]

        ### iterate over the AA sequence
        reuse = False
        for i, AA_2 in enumerate(decoder_inputs_emb):
          # nobi
          if i > 0: # to-do-later: bring variable definitions out of the loop
            variable_scope.get_variable_scope().reuse_variables()
            reuse = True
          # cnn_ion model
          candidate_intensity = intensity_inputs[i] # [128, 27, 2, 10]
          cnn_ion_feature, cnn_ion_logit = self._build_cnn_ion(candidate_intensity,
                                                               ms1_profile)
          # lstm model
          lstm_feature, lstm_logit, lstm_state = self._build_lstm_iter(
              cell,
              lstm_state0,
              lstm_state,
              [AA_1, AA_2],
              reuse=reuse)
          AA_1 = AA_2
          # combine cnn_ion and lstm features
          feature_logit = self._combine_feature(cnn_ion_feature, lstm_feature, reuse=reuse)
          # both ion-lstm models are used together by default
          #   but each can be used separately for investigation
          if self.use_ion and self.use_lstm:
            logit = feature_logit
          elif self.use_ion:
            logit = cnn_ion_logit
          elif self.use_lstm:
            logit = lstm_logit
          else:
            print("Error: wrong ion-lstm model!")
            sys.exit()
          # final softmax layer
          output.append(logit)

    # l2 regularization
    l2_loss = 0.0
    tf_graph = tf.get_default_graph()
    for name in self.l2_var_name:
      tf_var = tf_graph.get_tensor_by_name(name)
      l2_loss += tf.nn.l2_loss(tf_var)
    l2_loss = tf.multiply(l2_loss, self.l2_weight)

    return output_forward, output_backward, l2_loss


  def _build_cnn_ion(self, input_intensity, ms1_profile):
    """TODO(nh2tran): docstring.
       Inputs:
         input_intensity: shape [batch_size, vocab_size, neighbor_size*num_ion, WINDOW_SIZE].
       Outputs:
         cnn_ion: shape [batch_size, num_units]
    """

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("ModelNetwork: _build_cnn_ion()")

    # calculate correlation between ms1 and ms2
    # broadcast ms1 from [128, 5] to [128, 26, 5]
    ms1_profile = tf.stack([ms1_profile]*self.vocab_size, axis=1)
    ms1_mean, ms1_var = tf.nn.moments(ms1_profile, axes=[2], keep_dims=True) # [128, 26, 1]
    ms1_std = tf.sqrt(ms1_var)
    # squeeze ms2 to [128, 26, 5] by summing along ion & window dimensions
    ms2_profile = tf.reshape(input_intensity, [-1, self.vocab_size, self.num_ion, self.neighbor_size, self.WINDOW_SIZE])
    ms2_profile = tf.reduce_sum(ms2_profile, axis=[2, 4]) # [128, 26, 5]
    ms2_mean, ms2_var = tf.nn.moments(ms2_profile, axes=[2], keep_dims=True) # [128, 26, 1]
    ms2_std = tf.sqrt(ms2_var)
    # avoid zero array
    ms1_std += 1e-6
    ms2_std += 1e-6
    # Pearson correlation
    corr = tf.divide( # [128, 26, 1]
        tf.reduce_mean(tf.multiply(ms1_profile - ms1_mean, ms2_profile - ms2_mean), axis=[2], keep_dims=True),
        tf.multiply(ms1_std, ms2_std))
    corr = tf.squeeze(corr, axis=[2]) # [128, 26]

    # reshape [128, 26, 8*5, 10] to [128, 8, 5, 10, 26]
    # TODO(nh2tran): this can be fixed at the input process.
    input_intensity = tf.reshape(input_intensity, [-1, self.vocab_size, self.num_ion, self.neighbor_size, self.WINDOW_SIZE])
    input_intensity = tf.transpose(input_intensity, perm=[0, 2, 3, 4, 1])

    # conv1: [128, 8, 5, 10, 26] >> [128, 8, 5, 10, 64] with kernel [1, 3, 3, 26, 64]
    conv1_weight = tf.get_variable(
        name="conv1_weight",
        shape=[1, 3, 3, self.vocab_size, 64],
        initializer=tf.uniform_unit_scaling_initializer(1.43))
    conv1_bias = tf.get_variable(
        name="conv1_bias",
        shape=[64],
        initializer=tf.constant_initializer(0.1))
    conv1 = tf.nn.relu(tf.nn.conv3d(input_intensity,
                                    conv1_weight,
                                    strides=[1, 1, 1, 1, 1],
                                    padding='SAME')
                       + conv1_bias)
    self.l2_var_name.add(conv1_weight.name)

    # conv2: [128, 8, 5, 10, 64] >> [128, 8, 5, 10, 64] with kernel [1, 2, 2, 64, 64]
    conv2_weight = tf.get_variable(
        name="conv2_weight",
        shape=[1, 2, 2, 64, 64],
        initializer=tf.uniform_unit_scaling_initializer(1.43))
    conv2_bias = tf.get_variable(
        name="conv2_bias",
        shape=[64],
        initializer=tf.constant_initializer(0.1))
    conv2 = tf.nn.relu(tf.nn.conv3d(conv1,
                                    conv2_weight,
                                    strides=[1, 1, 1, 1, 1],
                                    padding='SAME')
                       + conv2_bias)
    self.l2_var_name.add(conv2_weight.name)

    # conv3: [128, 8, 5, 10, 64] >> [128, 8, 5, 10, 64] with kernel [1, 2, 2, 64, 64]
    conv3_weight = tf.get_variable(
        name="conv3_weight",
        shape=[1, 2, 2, 64, 64],
        initializer=tf.uniform_unit_scaling_initializer(1.43))
    conv3_bias = tf.get_variable(
        name="conv3_bias",
        shape=[64],
        initializer=tf.constant_initializer(0.1))
    conv3 = tf.nn.relu(tf.nn.conv3d(conv2,
                                    conv3_weight,
                                    strides=[1, 1, 1, 1, 1],
                                    padding='SAME')
                       + conv3_bias)
    self.l2_var_name.add(conv3_weight.name)

    # model correlation between b-ion & y-ion
    #~ conv2_b = tf.slice(conv2, # [128, 0:0, 5, 10, 64]
                       #~ begin=[0, 0, 0, 0, 0],
                       #~ size=[-1, 1, -1, -1, -1])
    #~ conv2_y = tf.slice(conv2, # [128, 4:4, 5, 10, 64]
                       #~ begin=[0, 4, 0, 0, 0],
                       #~ size=[-1, 1, -1, -1, -1])
    #~ conv2_by = tf.multiply(conv2_b, conv2_y) # [128, 1, 5, 10, 64]
    #~ conv2 = tf.concat([conv2, conv2_by], axis=1) # [128, 9, 5, 10, 64]

    # max pooling on (neighbor, window) dimensions with stride [1, 2, 2, 1]
    # but first reshape conv2: [128, 8, 5, 10, 64] >> [128*8, 5, 10, 64]
    pool1 = tf.reshape(conv3, [-1, self.neighbor_size, self.WINDOW_SIZE, 64])
    # [128*8, 5, 10, 64] >> [128*8, 3, 5, 64]
    pool1 = tf.nn.max_pool(pool1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    # convolution dropout
    dropout_conv = tf.nn.dropout(pool1, self.dropout_keep["conv"])

    # dense1: 4D >> [128, 512]
    dense1_input_size = self.num_ion * (self.neighbor_size // 2 + 1) * (self.WINDOW_SIZE // 2) * 64
    dense1_output_size = self.num_units
    dense1_input = tf.reshape(dropout_conv, [-1, dense1_input_size])
    # add correlation feature
    #~ dense1_input = tf.concat(values=[dense1_input, corr], axis=1)
    #~ dense1_input_size += self.vocab_size
    # add correlation feature
    dense1_weight = tf.get_variable(
        name="dense1_weight",
        shape=[dense1_input_size, dense1_output_size],
        initializer=tf.uniform_unit_scaling_initializer(1.43))
    dense1_bias = tf.get_variable(
        name="dense1_bias",
        shape=[dense1_output_size],
        initializer=tf.constant_initializer(0.1))
    dense1 = tf.nn.relu(tf.matmul(dense1_input, dense1_weight) + dense1_bias)
    self.l2_var_name.add(dense1_weight.name)

    # dense dropout
    dropout_dense = tf.nn.dropout(dense1, self.dropout_keep["dense"])

    cnn_ion_feature = dropout_dense
    # add correlation feature, shape [128, 512 + 26]
    cnn_ion_feature = tf.concat(values=[cnn_ion_feature, corr], axis=1)
    # add correlation feature

    # linear transform to logit [128, 26], in case only cnn_ion model is used
    linear_weight = tf.get_variable(
        name="linear_weight",
        shape=[self.num_units + self.vocab_size, self.vocab_size])
    linear_bias = tf.get_variable(
        name="linear_bias",
        shape=[self.vocab_size],
        initializer=tf.constant_initializer(0.1))
    cnn_ion_logit = tf.matmul(cnn_ion_feature, linear_weight) + linear_bias
    self.l2_var_name.add(linear_weight.name)

    return cnn_ion_feature, cnn_ion_logit


  def _build_cnn_spectrum(self, spectrum_holder):
    """TODO(nh2tran): docstring.

       Inputs:
         input_spectrum: 3D tensor of shape [batch_size, neighbor_size, MZ_SIZE].

       Outputs:
         cnn_spectrum_feature: 2D tensor of shape [batch_size, num_units]
    """

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("ModelNetwork: _build_cnn_spectrum()")

    scope = "embedding_rnn_seq2seq" # TODO(nh2tran): change to "cnn_spectrum"
    with tf.variable_scope(scope):
  
      # reshape the 3D input tensor to common 4D
      layer0 = tf.reshape(tensor=spectrum_holder,
                          shape=[-1, self.neighbor_size, self.MZ_SIZE, 1])
      # use max pooling to reduce resolution to 1.0 Dalton
      reduced_res = self.SPECTRUM_RESOLUTION
      reduced_size = self.MZ_SIZE // reduced_res
      layer0 = tf.nn.max_pool(layer0,
                              ksize=[1, 1, reduced_res, 1],
                              strides=[1, 1, reduced_res, 1],
                              padding='SAME')

      # conv1: filter [neighbor_size, 4, 1, 4] & stride [neighbor_size, 1, 1, 1]
      conv1_weight = tf.get_variable(
          name="conv1_W", # TODO(nh2tran): change to "conv1_weight"
          shape=[self.neighbor_size, 4, 1, 4],
          initializer=tf.uniform_unit_scaling_initializer(1.43))
      conv1_bias = tf.get_variable(
          name="conv1_B", # TODO(nh2tran): change to "conv1_bias"
          shape=[4],
          initializer=tf.constant_initializer(0.1))
      conv1 = tf.nn.relu(tf.nn.conv2d(layer0,
                                      conv1_weight,
                                      strides=[1, self.neighbor_size, 1, 1],
                                      padding='SAME')
                         + conv1_bias)

      # conv2: filter [1, 4, 4, 4] with stride [1, 1, 1, 1]
      conv2_weight = tf.get_variable(
          name="conv2_W", # TODO(nh2tran): change to "conv2_weight"
          shape=[1, 4, 4, 4],
          initializer=tf.uniform_unit_scaling_initializer(1.43))
      conv2_bias = tf.get_variable(
          name="conv2_B", # TODO(nh2tran): change to "conv2_bias"
          shape=[4],
          initializer=tf.constant_initializer(0.1))
      conv2 = tf.nn.relu(tf.nn.conv2d(conv1,
                                      conv2_weight,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
                         + conv2_bias)
      # max pooling [1, 1, 6, 1] with stride [1, 1, 4, 1]
      conv2 = tf.nn.max_pool(conv2,
                             ksize=[1, 1, 6, 1],
                             strides=[1, 1, 4, 1],
                             padding='SAME')
      conv2 = tf.nn.dropout(conv2, self.dropout_keep["conv"])

      # dense1
      dense1_input_size = 1 * (reduced_size // (4)) * 4
      dense1_output_size = self.num_units
      dense1_weight = tf.get_variable(
          name="dense1_W", # TODO(nh2tran): change to "dense1_weight"
          shape=[dense1_input_size, dense1_output_size],
          initializer=tf.uniform_unit_scaling_initializer(1.43))
      dense1_bias = tf.get_variable(
          name="dense1_B", # TODO(nh2tran): change to "dense1_bias"
          shape=[dense1_output_size],
          initializer=tf.constant_initializer(0.1))
      dense1 = tf.reshape(conv2, [-1, dense1_input_size])
      dense1 = tf.nn.relu(tf.matmul(dense1, dense1_weight) + dense1_bias)
      dense1 = tf.nn.dropout(dense1, self.dropout_keep["dense"])

      cnn_spectrum_feature = dense1

    return cnn_spectrum_feature


  def _build_embedding_AAid(self, input_AAid, reuse=False):
    """TODO(nh2tran): docstring.
       Inputs:
         input_AAid: list of 1D tensors of shape [batch_size].
         reuse: boolean, for variable_scope.
       Outputs:
         embedding_AAid: list of 2D tensors [batch_size, embedding_size].
    """

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("ModelNetwork: _build_embedding_AAid()")

    scope = "embedding_rnn_seq2seq/embedding_rnn_decoder" # TODO(nh2tran): to change to "embedding_AAid"
    with tf.variable_scope(scope, reuse=reuse):
      with ops.device("/cpu:0"):
        embedding = tf.get_variable(
            name="embedding",
            shape=[self.vocab_size, self.embedding_size])
      embedding_AAid = [embedding_ops.embedding_lookup(embedding, x)
                        for x in input_AAid]

    return embedding_AAid


  def _build_lstm_0(self, cnn_spectrum):
    """TODO(nh2tran): docstring.
       Inputs:
         cnn_spectrum: shape [batch_size, num_units].
       Outputs:
         cell: lstm cell for later iterations
         lstm_state0: tuple of 2 tensors [batch_size, num_units].
    """

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("ModelNetwork: _build_lstm_0()")

    # BUG rnn_cell tf.1.x: use separate BasicLSTMCell for 2 directions. Ok, fixed.
    single_cell = rnn_cell.BasicLSTMCell(num_units=self.num_units,
                                         state_is_tuple=True)
    if self.num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
    else:
      cell = single_cell
    cell = rnn_cell.DropoutWrapper(cell,
                                   input_keep_prob=self.dropout_keep["dense"],
                                   output_keep_prob=self.dropout_keep["dense"])

    ### initialize lstm cell
    with tf.variable_scope("LSTM_cell"): # TODO(nh2tran): remove
      # cnn_spectrum as input 0 to initialize the lstm cell
      # lstm_state0 is returned for 2 purposes:
      #   (i) initializing several spectra in batch is faster
      #   (ii) using lstm on short 3-mers (nobi model)
      input0 = cnn_spectrum
      batch_size = array_ops.shape(input0)[0]
      zero_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
      # nobi
      _, lstm_state0 = cell(inputs=input0, state=zero_state)

    return cell, lstm_state0


  def _build_lstm_iter(self, cell, input_lstm_state0, input_lstm_state, embedding_AAid, reuse=False):
    """TODO(nh2tran): docstring.
       Inputs:
         cell: lstm cell.
         input_lstm_state0: tuple of 2 tensors [batch_size, num_units], for lstm_kmer
         input_lstm_state: tuple of 2 tensors [batch_size, num_units].
         embedding_AAid: list of 2 tensors [batch_size, embedding_size].
       Outputs:
         lstm_feature: shape [batch_size, num_units].
         lstm_logit: shape [batch_size, num_units].
         lstm_state: tuple of 2 tensors [batch_size, num_units].
    """

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("ModelNetwork: _build_lstm_iter()")

    # project lstm input from embedding_size to num_units
    with tf.variable_scope("LSTM_input_projected", reuse=reuse): # TODO(nh2tran): remove
      project_weight = tf.get_variable(
          name="lstm_input_projected_W", # TODO(nh2tran): change to "project_weight"
          shape=[self.embedding_size, self.num_units])
      project_bias = tf.get_variable(
          name="lstm_input_projected_B", # TODO(nh2tran): change to "project_bias"
          shape=[self.num_units],
          initializer=tf.constant_initializer(0.1))
      # nobi
      AA_1 = embedding_AAid[0]
      AA_2 = embedding_AAid[1]
      AA_1_project = (tf.matmul(AA_1, project_weight) + project_bias)
      AA_2_project = (tf.matmul(AA_2, project_weight) + project_bias)

    # lstm cell's one-iteration
    with tf.variable_scope("LSTM_cell", reuse=reuse): # TODO(nh2tran): remove
      if self.lstm_kmer: # use lstm on k-mers
        _, lstm_state1 = cell(inputs=AA_1_project, state=input_lstm_state0)
        lstm_feature, lstm_state = cell(inputs=AA_2_project, state=lstm_state1)
      else: # use lstm on full sequence
        lstm_feature, lstm_state = cell(inputs=AA_2_project, state=input_lstm_state)

    # linear transform to logit [128, 26], in case only lstm model is used
    # TODO(nh2tran): replace _linear and remove scope
    with tf.variable_scope("lstm_output_projected", reuse=reuse):
      linear_weight = tf.get_variable(
          name="linear_weight",
          shape=[self.num_units, self.vocab_size])
      linear_bias = tf.get_variable(
          name="linear_bias",
          shape=[self.vocab_size],
          initializer=tf.constant_initializer(0.1))
      lstm_logit = tf.matmul(lstm_feature, linear_weight) + linear_bias

    return lstm_feature, lstm_logit, lstm_state


  def _combine_feature(self, cnn_ion_feature, lstm_feature, reuse=False):
    """TODO(nh2tran): docstring.
    """

    feature_weight = tf.get_variable(
        name="dense_concat_W", # TODO(nh2tran): change to "feature_weight"
        shape=[self.num_units*2+self.vocab_size, self.num_units],
        initializer=tf.uniform_unit_scaling_initializer(1.43))
    feature_bias = tf.get_variable(
        name="dense_concat_B", # TODO(nh2tran): change to "feature_bias"
        shape=[self.num_units],
        initializer=tf.constant_initializer(0.1))
    feature_input = tf.concat(values=[cnn_ion_feature, lstm_feature],
                              axis=1)
    feature = tf.nn.relu(tf.matmul(feature_input, feature_weight)
                         + feature_bias)
    feature = tf.nn.dropout(feature, self.dropout_keep["dense"])
    # linear transform to logit [128, 26]
    # TODO(nh2tran): replace _linear and remove scope
    with tf.variable_scope("output_logit", reuse=reuse):
      linear_weight = tf.get_variable(
          name="linear_weight",
          shape=[self.num_units, self.vocab_size])
      linear_bias = tf.get_variable(
          name="linear_bias",
          shape=[self.vocab_size],
          initializer=tf.constant_initializer(0.1))
      feature_logit = tf.matmul(feature, linear_weight) + linear_bias

    return feature_logit


class ModelInference(object):
  """TODO(nh2tran): docstring."""


  def __init__(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("ModelInference: __init__()")

    # we currently use deepnovo_config to store both const & settings
    # the settings should be shown in __init__() to keep track carefully
    self.neighbor_size = deepnovo_config.neighbor_size
    self.MZ_SIZE = deepnovo_config.MZ_SIZE
    self.vocab_size = deepnovo_config.vocab_size
    self.num_ion = deepnovo_config.num_ion
    self.WINDOW_SIZE = deepnovo_config.WINDOW_SIZE
    self.num_layers = deepnovo_config.num_layers
    self.num_units = deepnovo_config.num_units
    self.train_dir = deepnovo_config.FLAGS.train_dir

    # input tensors are grouped into a dictionary
    self.input_dict = {}
    # input spectrum is a 2D tensor of shape [batch_size, , neighbor_size, MZ_SIZE]
    #   for example: [128, 5, 30k]
    self.input_dict["spectrum_holder"] = tf.placeholder(
        dtype=tf.float32,
        shape=[None, self.neighbor_size, self.MZ_SIZE],
        name="spectrum_holder")
    self.input_dict["ms1_profile"] = tf.placeholder(
        dtype=tf.float32,
        shape=[None, self.neighbor_size],
        name="ms1_profile")
    # input intensity profile: [batch_size, vocab_size, neighbor_size*num_ion, WINDOW_SIZE]
    #   for example; [128, 26, 5*8, 10]
    self.input_dict["intensity"] = tf.placeholder(
        dtype=tf.float32,
        shape=[None,
               self.vocab_size,
               self.neighbor_size*self.num_ion,
               self.WINDOW_SIZE],
        name="input_intensity")
    # input lstm state is a tuple of 2 tensors [batch_size, num_units]
    #   for example: [128, 512]
    self.input_dict["lstm_state0"] = [
        (tf.placeholder(dtype=tf.float32,
                        shape=[None, self.num_units],
                        name="layer" + str(x) + "_c_state0"),
         tf.placeholder(dtype=tf.float32,
                        shape=[None, self.num_units],
                        name="layer" + str(x) + "_h_state0"))
        for x in range(self.num_layers)]
    self.input_dict["lstm_state"] = [
        (tf.placeholder(dtype=tf.float32,
                        shape=[None, self.num_units],
                        name="layer" + str(x) + "_c_state"),
         tf.placeholder(dtype=tf.float32,
                        shape=[None, self.num_units],
                        name="layer" + str(x) + "_h_state"))
        for x in range(self.num_layers)]
    # input last 2 amino acids if using lstm for short 3-mers
    #   list of 2 tensors [batch_size]
    #   "AAid" stands for amino acid id
    self.input_dict["AAid"] = [tf.placeholder(dtype=tf.int32,
                                              shape=[None],
                                              name="input_AA_id_1"), # to change to "input_AAid_1"
                               tf.placeholder(dtype=tf.int32,
                                              shape=[None],
                                              name="input_AA_id_2")] # to change to "input_AAid_2"

    # the keep_prob probability of dropout layers
    #   for inference model, they are const 1.0
    #   for train/valid model, they are input tensors
    self.dropout_keep = {}
    self.dropout_keep["conv"] = 1.0
    self.dropout_keep["dense"] = 1.0

    # output tensors are grouped into 2 dictionaries, forward and backward,
    #   each has 4 tensors:
    #   ["logit"]: shape [batch_size, vocab_size], to compute loss in training
    #   ["logprob"]: shape [batch_size, vocab_size], to compute score in inference
    #   ["lstm_state"]: shape [batch_size, num_units], to compute next iteration
    #   ["lstm_state0"]: shape [batch_size, num_units], state from cnn_spectrum
    # they will be built and loaded by build_model() and restore_model()
    self.output_forward = None
    self.output_backward = None


  def build_model(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("ModelInference: build_model()")

    model_network = ModelNetwork()
    self.output_forward, self.output_backward = model_network.build_network(
        self.input_dict,
        self.dropout_keep)


  def restore_model(self, session):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("ModelInference: restore_model()")

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(self.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + ".index"):
      print("restore model from {0:s}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)
    else:
      print("Error: model not found.")
      sys.exit()

