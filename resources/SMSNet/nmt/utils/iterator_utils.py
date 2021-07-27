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
"""For loading data into NMT models."""
from __future__ import print_function

import collections

import tensorflow as tf

import nmt.input_config as input_config

__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# bin_step = 0.01 # 0.01
# inv_bin_step = 1.0/bin_step
# max_posi = 5000
# max_spec_length = int(max_posi * inv_bin_step)
# max_spec_length_cnn = max_spec_length//10

# aa_input_window_size = 0.2 #1.0
# # full_aa_window = int(aa_input_window_size * inv_bin_step)
# half_aa_window = int(aa_input_window_size * inv_bin_step/2)
# full_aa_window = half_aa_window * 2

class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "encoder_input", "target_input",
                            "target_output", "source_spectrum_length",
                            "target_sequence_length", "aa_spectrum", "peptide_mass"))):
  pass



def _get_2nd_step_ion(spec, b_ion, peptide_mass):
  # b_ion shape is [length, 20(vocab)]
  b_ion = tf.expand_dims(b_ion, 2) # [length, 20, 1]
  b_ion = tf.tile(b_ion, [1, 1, input_config.vocab_size]) # [length, 1st_vocab(20), 2nd_vocab(20)]
  
  aa_mass = tf.convert_to_tensor(input_config.mass_ID_np, tf.float32) # [20]
  aa_mass = tf.reshape(aa_mass, [1, 1, -1]) # [1, 1, 20]
  
  b_ion = b_ion + aa_mass # [length, 1st_vocab(20), 2nd_vocab(20)] + [1, 1, 20] -> [len, 20, 20]
  
  # Calculate b2+, y, y2+
  y_ion = peptide_mass - b_ion
  b_plus2_charge1 = (b_ion + input_config.mass_H) / 2 # [length, 20, 20]
  y_plus2_charge1 = (y_ion + input_config.mass_H) / 2 # [length, 20, 20]

  # [length, 20, 20] x 4 -> [length, 20, 80]
  candidate_position = tf.concat([b_ion, y_ion, b_plus2_charge1, y_plus2_charge1], 2)

  candidate_idx = tf.scalar_mul(input_config.inv_bin_step, candidate_position)
  candidate_idx = tf.cast(tf.round(candidate_idx), tf.int32)

  candidate_idx_shape = tf.shape(candidate_idx)
  candidate_idx = tf.reshape(candidate_idx, [-1])

  candidate_spec = tf.map_fn(
    lambda x: tf.cond(tf.logical_and(x - input_config.half_aa_window > 0, x+input_config.half_aa_window < input_config.max_spec_length),
                      lambda: spec[x-input_config.half_aa_window : x+input_config.half_aa_window],
                      lambda: tf.ones([2*input_config.half_aa_window]) * -0.1), 
    candidate_idx, dtype=(tf.float32), parallel_iterations=200, back_prop=False)
  candidate_spec = tf.reshape(candidate_spec, tf.concat([candidate_idx_shape, [2*input_config.half_aa_window]], axis=0))
  # [length, 20, 80] -> [length, 20, 80, input_config.full_aa_window(20)]

  lookahead = tf.reduce_max(candidate_spec, axis=2)
  # [length, 20, *80*, 20] -> [length, 20, 20]
  return lookahead # tf.zeros(tf.concat([tf.shape(b_ion), [2*input_config.half_aa_window]], axis=0))


def get_batch_candidate_intensity(spectrum_original,
                                  peptide_mass,
                                  prefix_mass):
#                             direction):
  """
  This function recieve exactly one `spectrum_original` and `peptide_mass` 
  and a list of `prefix_mass`
  Args:
    spectrum_original: 2-D Tensor of original spectrum of size [batch_size, max_spec_length]
    peptide_mass: 1-D Tensor. Peptide masses for each spectrum in the batch
    prefix_mass: 2-D Tensor. Sums of predicted amino acids mass of size [batch_size, beam_width]
  Returns:
    candidate_spec: 4-D Tensor of size [batch_size, len(prefix_mass), 20 * ion_type, full_aa_window]
  """
#   total_aa = tf.shape(prefix_mass)[0]
#   prefix_mass = tf.tile(prefix_mass, [input_config.vocab_size])

#   prefix_mass = tf.reshape(prefix_mass, [input_config.vocab_size, total_aa]) # [20 * beam]
#   prefix_mass = tf.transpose(prefix_mass, [1, 0]) # [beam * 20]
  # [batch, beam_width] -> [batch, beam_width, 20]
  prefix_mass = tf.tile(tf.expand_dims(prefix_mass, 2), [1, 1, input_config.vocab_size]) # [batch, beam, 20]

  
  # Add aa mass to each position
  mass_ID_tensor = tf.convert_to_tensor(input_config.mass_ID_np, tf.float32) # [20]
  mass_ID_tensor = tf.reshape(mass_ID_tensor, [1, 1, -1]) # [1, 1, 20]
  
  b_ion = prefix_mass + mass_ID_tensor # [batch, beam, 20] + [1, 1, 20] -> [batch, beam, 20]
  y_ion = tf.reshape(peptide_mass, [-1, 1, 1]) - b_ion # [batch, 1, 1] - [batch, beam, 20] -> [batch, beam, 20]
  
  # Mod b-ion
  b_H2O = b_ion - input_config.mass_H2O # [batch, beam, 20]
  b_NH3 = b_ion - input_config.mass_NH3 # [batch, beam, 20]
  b_plus2_charge1 = (b_ion + input_config.mass_H) / 2 # [batch, beam, 20]

  # Mod y-ion
  y_H2O = y_ion - input_config.mass_H2O # [batch, beam, 20]
  y_NH3 = y_ion - input_config.mass_NH3 # [batch, beam, 20]
  y_plus2_charge1 = (y_ion + input_config.mass_H) / 2 # [batch, beam, 20]

  candidate_position = tf.concat([b_ion, b_H2O, b_NH3, b_plus2_charge1,
                                  y_ion, y_H2O, y_NH3, y_plus2_charge1], 2)
  # [batch, beam, 20*8] -> [batch, beam, 160]
  
  # candidate_position = tf.concat([b_ion, b_plus2_charge1,
  #                                 y_ion, y_plus2_charge1], 1)

  candidate_idx = tf.scalar_mul(input_config.inv_bin_step, candidate_position)
  candidate_idx = tf.round(candidate_idx)
  candidate_idx = tf.cast(candidate_idx, tf.int32)

  # new way
  candidate_positions = tf.range(0, input_config.full_aa_window, dtype=tf.int32) - input_config.half_aa_window # [full_aa_window]
  candidate_positions = tf.reshape(candidate_positions, [1, 1, 1, -1]) # [1, 1, 1, full_aa_window]
  candidate_positions = tf.tile(candidate_positions, tf.concat([tf.shape(candidate_idx), [1]], axis=0)) # [batch, beam, 160, full_aa_window]

  # [batch, beam, 160, 1] + [batch, beam, 160, full_aa_window] -> [batch, beam, 160, full_aa_window]
  candidate_positions = tf.expand_dims(candidate_idx, 3) + candidate_positions
  # point all invalid position to max_spec_length + 1 so that we can change it to -0.1 later
  candidate_positions = tf.where(tf.logical_and(candidate_positions >= 0, 
                                                candidate_positions < input_config.max_spec_length), #tf.minimum(tf.cast(peptide_mass * inv_bin_step, tf.int32),
                                                                                 #input_config.max_spec_length)), # input_config.max_spec_length)
                                 candidate_positions,
                                 tf.zeros_like(candidate_positions) + input_config.max_spec_length)

  candidate_positions = tf.expand_dims(candidate_positions, 4) # [batch, beam, 160, full_aa_window, *1*]
  # want [batch, beam, 160, full_aa_window, *2*]
  batch_size = tf.shape(spectrum_original)[0]
  spec_num = tf.range(0, batch_size, dtype=tf.int32) # [batch] [0,1,2,3,...,batch]
  spec_num = tf.reshape(spec_num, [-1, 1, 1, 1, 1]) # [batch, 1, 1, 1, 1]
  spec_num = tf.tile(spec_num, tf.concat([[1], tf.shape(candidate_positions)[1:]], axis=0)) # [batch, beam, 160, full_aa_window, 1]
  
  candidate_positions = tf.concat([spec_num, candidate_positions], axis=4) # [batch, beam, 160, full_aa_window, *2*]
  candidate_spec = tf.gather_nd(tf.concat([spectrum_original, tf.ones([batch_size, 1])* -0.1], axis=1), candidate_positions)
  # tf.Print(candidate_spec, [tf.shape(candidate_spec)])

  return candidate_spec


def get_candidate_intensity(spectrum_original,
                            peptide_mass,
                            prefix_mass):
#                             direction):
  """
  This function recieve exactly one `spectrum_original` and `peptide_mass` 
  and a list of `prefix_mass`
  Args:
    spectrum_original: 1-D Tensor of original spectrum
    prefix_mass: 1-D Tensor, sums of predicted amino acid mass for each beam
    peptide_mass: peptide mass for calculating y-ion position
  Returns:
    candidate_spec: 3-D Tensor of size [len(prefix_mass), vocab_size * ion_type, full_aa_window]
  """
  # For calling map_fn without lambda
  # spectrum_original = input_list[0]
  # peptide_mass = input_list[1]
  # prefix_mass = input_list[2]
  
#   total_aa = tf.shape(prefix_mass)[0]
#   prefix_mass = tf.tile(prefix_mass, [input_config.vocab_size])
  
#   prefix_mass = tf.reshape(prefix_mass, [input_config.vocab_size, total_aa]) # [20 * beam]
#   prefix_mass = tf.transpose(prefix_mass, [1, 0]) # [beam * 20]
  prefix_mass = tf.tile(tf.expand_dims(prefix_mass, 1), [1, input_config.vocab_size])

  # Add aa mass to each position
  b_ion = prefix_mass + input_config.mass_ID_np # [beam, 20] + [1, 20] -> [length, 20]
  y_ion = peptide_mass - b_ion
  
  # Mod b-ion
  b_H2O = b_ion - input_config.mass_H2O # [length, 20]
  b_NH3 = b_ion - input_config.mass_NH3 # [length, 20]
  b_plus2_charge1 = (b_ion + input_config.mass_H) / 2 # [length, 20]

  # Mod y-ion
  y_H2O = y_ion - input_config.mass_H2O # [length, 20]
  y_NH3 = y_ion - input_config.mass_NH3 # [length, 20]
  y_plus2_charge1 = (y_ion + input_config.mass_H) / 2 # [length, 20]

  candidate_position = tf.concat([b_ion, b_H2O, b_NH3, b_plus2_charge1,
                                  y_ion, y_H2O, y_NH3, y_plus2_charge1], 1)
  
  # candidate_position = tf.concat([b_ion, b_plus2_charge1,
  #                                 y_ion, y_plus2_charge1], 1)

  candidate_idx = tf.scalar_mul(input_config.inv_bin_step, candidate_position)
  candidate_idx = tf.round(candidate_idx)
  candidate_idx = tf.cast(candidate_idx, tf.int32)

  # new way
  candidate_positions = tf.range(0, input_config.full_aa_window, dtype=tf.int32) - input_config.half_aa_window # [full_aa_window]
  candidate_positions = tf.reshape(candidate_positions, [1, 1, -1]) # [1, 1, full_aa_window]
  candidate_positions = tf.tile(candidate_positions, tf.concat([tf.shape(candidate_idx), [1]], axis=0)) # [length, 160, full_aa_window]

  # [length, 160, 1] + [length, 160, full_aa_window] -> [length, 160, full_aa_window]
  candidate_positions = tf.expand_dims(candidate_idx, 2) + candidate_positions
  # point all invalid position to max_spec_length + 1 so that we can change it to -0.1 later
  candidate_positions = tf.where(tf.logical_and(candidate_positions >= 0, 
                                                candidate_positions < input_config.max_spec_length), #tf.minimum(tf.cast(peptide_mass * input_config.inv_bin_step, tf.int32),
                                                                                 #input_config.max_spec_length)), # input_config.max_spec_length)
                                 candidate_positions,
                                 tf.zeros_like(candidate_positions) + input_config.max_spec_length) 

  candidate_positions = tf.expand_dims(candidate_positions, 3)
  candidate_spec = tf.gather_nd(tf.concat([spectrum_original, [-0.1]], axis=0), candidate_positions)
  # tf.Print(candidate_spec, [tf.shape(candidate_spec)])
    
  
#   candidate_idx_shape = tf.shape(candidate_idx)
#   candidate_idx = tf.reshape(candidate_idx, [-1])
  
#   candidate_spec = tf.map_fn(
#     lambda x: tf.cond(tf.logical_and(x - input_config.half_aa_window > 0, x + input_config.half_aa_window < input_config.max_spec_length),
#                       lambda: spectrum_original[x-input_config.half_aa_window : x+input_config.half_aa_window],
#                       lambda: tf.ones([2*input_config.half_aa_window]) * -0.1), 
#     candidate_idx, dtype=(tf.float32), parallel_iterations=200, back_prop=False)
#   candidate_spec = tf.reshape(candidate_spec, tf.concat([candidate_idx_shape, [2*input_config.half_aa_window]], axis=0))

  # lookahead_ion = _get_2nd_step_ion(spectrum_original, b_ion, peptide_mass) # [length, 20, 20]
  # tf.Print(candidate_spec, [tf.shape(lookahead_ion)])

  # candidate_spec = tf.concat([candidate_spec, lookahead_ion], axis=1) # [length, 160 + 20, 20]
  # tf.Print(candidate_spec, [tf.shape(candidate_spec)])
  return candidate_spec


def get_encoder_lookahead(encoder_after_cnn,
                          peptide_mass,
                          prefix_mass):
  """
  This function recieve exactly one `encoder_after_cnn` and `peptide_mass` 
  and a list of `prefix_mass`
  Args:
    encoder_after_cnn: 2-D Tensor of size [max_spec_length/pool_size, last_cnn_fileter_size]. Encoder state after 1x1 CNN layers
    prefix_mass: 1-D Tensor, sums of predicted amino acid mass for each beam
    peptide_mass: peptide mass for calculating y 
  Returns:
    candidate_spec: 2-D Tensor of size [len(prefix_mass), vocab_size * last_cnn_fileter_size]
  """
  
#   prefix_mass = tf.reshape(prefix_mass, [input_config.vocab_size, total_aa]) # [20 * beam]
#   prefix_mass = tf.transpose(prefix_mass, [1, 0]) # [beam * 20]
  prefix_mass = tf.tile(tf.expand_dims(prefix_mass, 1), [1, input_config.vocab_size])

  # Add aa mass to each position
  candidate_masses = prefix_mass + input_config.mass_ID_np # [beam, 20] + [1, 20] -> [length, 20]

  candidate_mass_idx = tf.scalar_mul(input_config.inv_bin_step / 10, candidate_masses)
  candidate_mass_idx = tf.floor(candidate_mass_idx)
  candidate_mass_idx = tf.cast(candidate_mass_idx, tf.int32) # [length, 20]

  # Prepare gather structure [length, 20, 0 1]
  # gather_positions = tf.range(0, 2, dtype=tf.int32) # [last_cnn_fileter_size]
  # gather_positions = tf.reshape(gather_positions, [1, 1, -1]) # [1, 1, full_aa_window]
  # gather_positions = tf.tile(gather_positions, tf.concat([tf.shape(candidate_mass_idx), [1]], axis=0)) # [length, 20, last_cnn_fileter_size]

  # Add mass_idx to gather_structure
  # [length, 160, 1] + [length, 160, full_aa_window] -> [length, 160, full_aa_window]
  # candidate_positions = tf.expand_dims(candidate_mass_idx, 2) + candidate_positions
  
  # point all invalid position to max_spec_length + 1 so that we can change it to -0.1 later
  # [length, 20]
  candidate_positions = tf.where(tf.logical_and(candidate_mass_idx >= 0, 
                                                candidate_mass_idx < input_config.max_spec_length//10),
                                 candidate_mass_idx,
                                 tf.zeros_like(candidate_mass_idx) + input_config.max_spec_length//10) 

  # [vocab_size, vocab_size(20), 1]
  candidate_positions = tf.expand_dims(candidate_positions, 2)
  # [length, vocab_size, 2]
  candidate_spec = tf.gather_nd(tf.concat([encoder_after_cnn, [-0.1, -0.1]], axis=0), candidate_positions)
  # Reshape to [length, vocab_size * 2]
  candidate_spec = tf.reshape(candidate_spec, [tf.shape(candidate_spec)[0], input_config.vocab_size * 2])
  return candidate_spec


def get_infer_iterator(src_dataset,
                       #src_vocab_table,
                       tgt_vocab_table, # <<<<<<<<<<<< delete this for real dataset !!!
                       batch_size,
                       aa_weight_table,
                       #eos,
                      ): #src_max_len=None):
  #src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  #src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)
  
  src_dataset = src_dataset.map(
    lambda src: (tf.string_split([src], delimiter='|').values))
  
  # split by '|' -> sequence|charge|mass/charge|score1|score2|spectrum
  src_dataset = src_dataset.map(
    lambda src: (src[-1], src[1], src[2]))

  # Convert charge and mass/charge to float
  src_dataset = src_dataset.map(
    lambda spec, charge, mass_p_charge: (spec,
                                         tf.string_to_number(charge, tf.float32),
                                         tf.string_to_number(mass_p_charge, tf.float32)))
  
  # Calculate total mass at 2+
  src_dataset = src_dataset.map(
    lambda spec, charge, mass_p_charge: (spec,
                                         charge * mass_p_charge + (2.0 - charge) * input_config.mass_H))
  # Edit line above to remove seq from input
  # This should be the peptide mass from dataset but we will use this for now
  # Split sequence to characters.
#   src_dataset = src_dataset.map(
#     lambda spec, seq: (spec, tf.string_split([seq], delimiter='').values))
  
#   src_dataset = src_dataset.map(
#   lambda spec, seq: (
#      spec, tf.cast(tgt_vocab_table.lookup(seq), tf.int32)))

#   # src_dataset = src_dataset.map(
#   #   lambda spec, seq: (spec, tf.reduce_sum(
#   #     tf.string_to_number(aa_weight_table.lookup(tf.cast(seq, tf.int64))))))
  
#   src_dataset = src_dataset.map(
#     lambda spec, seq: (spec, tf.reduce_sum( tf.gather(aa_weight_table, seq))))
  
#   src_dataset = src_dataset.map(
#     lambda spec, peptide_mass: (spec, peptide_mass + input_config.mass_N_terminus + input_config.mass_C_terminus + 2*input_config.mass_H))
#   ##### Edit above to use peptide mass from dataset ######
  
  # split values by ',' -> key,value,key,value,...
  src_dataset = src_dataset.map(
    lambda spec, peptide_mass: (tf.string_split([spec], delimiter=',').values, peptide_mass))

  #if src_max_len:
  #  src_dataset = src_dataset.map(lambda src: src[:src_max_len])
  # Convert the word strings to ids
  #src_dataset = src_dataset.map(
  #    lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
  # Add in the word counts.
  #src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

  
  # There is a bug in tf.bincount tensorflow 1.5 as of 13/2/18 that
  # if weights are specified, maxlength will be ignored, thus it will generate error
  # for the position that greater than max_length
  def _bin_value(positions, values, inv_bin_step_for_bin, max_length):
    positions = tf.scalar_mul(inv_bin_step_for_bin, positions)
    positions = tf.round(positions)
    positions = tf.cast(positions, tf.int32)

    # Because of the bug mentioned above, we will mask every position greater than max_spec_length
    # then remove them from positions and values.
    mask = tf.less(positions, max_length)
    mask.set_shape([None])
    positions = tf.boolean_mask(positions, mask)
    values = tf.boolean_mask(values, mask)

    spec = tf.bincount(positions, weights=values, minlength=max_length, maxlength=max_length)
    return spec
  
  
  def _bin_value_high_and_low_res(positions, values, peptide_mass):
    spec = _bin_value(positions, values, input_config.inv_bin_step, input_config.max_spec_length)
    low_res_spec = _bin_value(positions, values, input_config.inv_bin_step / 10, input_config.max_spec_length // 10)
    return spec, low_res_spec, peptide_mass


  def _seperate_posi_value(spec, peptide_mass):
    pair_count = tf.size(spec)
    mask = tf.tile([True,False], [tf.cast(tf.divide(pair_count, 2), tf.int32)])
    inv_mask = tf.logical_not(mask)
    
    positions = tf.boolean_mask(spec, mask)
    positions = tf.string_to_number(positions, tf.float32)
    values = tf.boolean_mask(spec, inv_mask)
    values = tf.string_to_number(values, tf.float32)
    return positions, values, peptide_mass
  
    
  def _shift_and_mask(spec, shift):
    rolled = tf.manip.roll(spec , shift=-shift, axis=0)
    mask = tf.concat([tf.ones([input_config.max_spec_length//10 - shift]), tf.zeros([shift])], 0)
    return tf.multiply(rolled, mask)
  
  
  def _get_cnn_encoder_input(spec, low_res_spec, peptide_mass):
    shifts = tf.convert_to_tensor(input_config.mass_ID_np, tf.float32)
    shifts = tf.scalar_mul(input_config.inv_bin_step / 10, shifts)
    shifts = tf.round(shifts)
    shifts = tf.cast(shifts, tf.int32)
    cnn_input = tf.map_fn(
      lambda x: _shift_and_mask(low_res_spec, x), shifts, dtype=(tf.float32))
    cnn_input = tf.concat([tf.expand_dims(low_res_spec, 0), cnn_input], axis=0)
    # Add position number to input
    pos_number = (tf.range(0, input_config.max_spec_length // 10, dtype=tf.float32) - 2499.5) / 1443.3756441065507
    cnn_input = tf.concat([tf.expand_dims(pos_number, 0), cnn_input], axis=0)
    
    cnn_input = tf.transpose(cnn_input, [1, 0])
    return spec, cnn_input, peptide_mass
  

  # Seperate positions and values.
  src_dataset = src_dataset.map(_seperate_posi_value)

  # Convert values to percentages.
  src_dataset = src_dataset.map(
    lambda positions, values, peptide_mass: (
      positions, tf.divide(values * 100.0, tf.reduce_sum(values)), peptide_mass))

  # Bin value according to bin_step.
  src_dataset = src_dataset.map(_bin_value_high_and_low_res)
  
  # Generate input for CNN encoder
  src_dataset = src_dataset.map(_get_cnn_encoder_input)

  # Add in sequence lengths.
  src_dataset = src_dataset.map(
    lambda src, cnn_input, peptide_mass: ( src, cnn_input, tf.size(src), peptide_mass))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None,None]),  # src
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])), # peptide_mass
            
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            tf.cast(0, tf.float32),  # src
            tf.cast(0, tf.float32),  # src
            0,   # src_len -- unused
            tf.cast(0, tf.float32)))  # peptide_mass -- unused

  batched_dataset = batching_func(src_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_spec, cnn_input_spec, src_seq_len, peptide_mass) = batched_iter.get_next()
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_spec,
      encoder_input=cnn_input_spec,
      target_input=None,
      target_output=None,
      source_spectrum_length=src_seq_len,
      target_sequence_length=None,
      aa_spectrum=None,
      peptide_mass=peptide_mass)


def get_iterator(src_files, #src_tgt_dataset,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 aa_weight_table,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=20,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 reshuffle_each_iteration=True):
  if not output_buffer_size:
    # output_buffer_size = batch_size * 1000
    output_buffer_size = 200000
  
  # src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.from_tensor_slices(src_files).shuffle(len(src_files))
  
  src_tgt_dataset = src_tgt_dataset.interleave(lambda filename:
    tf.data.TextLineDataset(filename), cycle_length=len(src_files)) ####
  
  #src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

  #src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index) 
  #if skip_count is not None:
  #  src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed, reshuffle_each_iteration)

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src: (tf.string_split([src], delimiter='|').values),
    num_parallel_calls=num_parallel_calls).prefetch(batch_size) #output_buffer_size
  
  # split by '|' -> file_name|number|seq|ion|values
  # split by '|' -> sequence|charge|mass/charge|score1|score2|spectrum
  src_tgt_dataset = src_tgt_dataset.map(
    lambda src: (src[-1], src[0], src[1], src[2]),
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)
  
  # Convert charge and mass/charge to float
  src_tgt_dataset = src_tgt_dataset.map(
    lambda spec, seq, charge, mass_p_charge: (spec,
                                              seq,
                                              tf.string_to_number(charge, tf.float32),
                                              tf.string_to_number(mass_p_charge, tf.float32)),
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)
  
  # Calculate total mass at 2+
  src_tgt_dataset = src_tgt_dataset.map(
    lambda spec, seq, charge, mass_p_charge: (spec,
                                              seq,
                                              charge * mass_p_charge + (2.0 - charge) * input_config.mass_H),
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)
  
  # filter out row with >30 AA for ease of training
  src_tgt_dataset = src_tgt_dataset.filter(lambda spec, seq, total_mass: tf.less(tf.size(tf.string_split([seq],"")), 30))
  
  # split values by ',' -> key,value,key,value,...
  src_tgt_dataset = src_tgt_dataset.map(
    lambda spec, seq, total_mass: (tf.string_split([spec], delimiter=',').values , seq, total_mass),
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)

  # There is a bug in tf.bincount tensorflow 1.5 as of 13/2/18 that
  # if weights are specified, maxlength will be ignored, thus it will generate error
  # for the position that greater than max_length
  def _bin_value(positions, values, inv_bin_step_for_bin, max_length):
    positions = tf.scalar_mul(inv_bin_step_for_bin, positions)
    positions = tf.round(positions)
    positions = tf.cast(positions, tf.int32)

    # Because of the bug mentioned above, we will mask every position greater than max_spec_length
    # then remove them from positions and values.
    mask = tf.less(positions, max_length)
    mask.set_shape([None])
    positions = tf.boolean_mask(positions, mask)
    values = tf.boolean_mask(values, mask)

    spec = tf.bincount(positions, weights=values, minlength=max_length, maxlength=max_length)
    return spec
  
  
  def _bin_value_high_and_low_res(positions, values, seq, total_mass):
    spec = _bin_value(positions, values, input_config.inv_bin_step, input_config.max_spec_length)
    low_res_spec = _bin_value(positions, values, input_config.inv_bin_step / 10, input_config.max_spec_length // 10)
    return spec, low_res_spec, seq, total_mass

  
  def _seperate_posi_value(spec, seq, total_mass):
    pair_count = tf.size(spec)
    mask = tf.tile([True,False], [tf.cast(tf.divide(pair_count, 2), tf.int32)])
    inv_mask = tf.logical_not(mask)
    
    positions = tf.boolean_mask(spec, mask)
    positions = tf.string_to_number(positions, tf.float32)
    values = tf.boolean_mask(spec, inv_mask)
    values = tf.string_to_number(values, tf.float32)
    return positions, values, seq, total_mass


  def _get_ion_intensity(spec, low_res_spec, seq, peptide_mass):
    # weight = tf.string_to_number(aa_weight_table.lookup(tf.cast(seq, tf.int64)))
    weight = tf.gather(aa_weight_table, seq)
    weight = tf.concat(([0.0], weight), 0)
    cum_sum = tf.cumsum(weight) #, exclusive=True)
    
    # Replace this with real data in the future
    # peptide_mass = cum_sum[-1] + input_config.mass_N_terminus + input_config.mass_C_terminus + 2*input_config.mass_H
    
    total_aa = tf.shape(cum_sum)[0]
#     cum_sum = tf.tile(cum_sum, [input_config.vocab_size])
    
#     cum_sum = tf.reshape(cum_sum, [input_config.vocab_size, total_aa]) # [20 * length]
#     cum_sum = tf.transpose(cum_sum, [1, 0]) # [length * 20]
    cum_sum = tf.tile(tf.expand_dims(cum_sum, 1), [1, input_config.vocab_size]) # [length * 20]
    
    # Add aa mass to each position
    b_no_terminus = cum_sum + input_config.mass_ID_np # [length, 20] + [20] -> [length, 20]
    b_ion = b_no_terminus + input_config.mass_N_terminus
    y_ion = peptide_mass - b_ion
    
    # Mod b-ion
    b_H2O = b_ion - input_config.mass_H2O # [length, 20]
    b_NH3 = b_ion - input_config.mass_NH3 # [length, 20]
    b_plus2_charge1 = (b_ion + input_config.mass_H) / 2 # [length, 20]
    
    # Mod y-ion
    y_H2O = y_ion - input_config.mass_H2O # [length, 20]
    y_NH3 = y_ion - input_config.mass_NH3 # [length, 20]
    y_plus2_charge1 = (y_ion + input_config.mass_H) / 2 # [length, 20]
    
    candidate_position = tf.concat([b_ion, b_H2O, b_NH3, b_plus2_charge1,
                                    y_ion, y_H2O, y_NH3, y_plus2_charge1], 1)
    
    # candidate_position = tf.concat([b_ion, b_plus2_charge1,
    #                                 y_ion, y_plus2_charge1], 1)
    
    # cum_sum = tf.Print(cum_sum, [tf.shape(cum_sum)])
    candidate_idx = tf.scalar_mul(input_config.inv_bin_step, candidate_position) # input_config.inv_bin_step
    candidate_idx = tf.round(candidate_idx)
    candidate_idx = tf.cast(candidate_idx, tf.int32)
    
    candidate_positions = tf.range(0, input_config.full_aa_window, dtype=tf.int32) - input_config.half_aa_window # [full_aa_window]
    candidate_positions = tf.reshape(candidate_positions, [1, 1, -1]) # [1, 1, full_aa_window]
    candidate_positions = tf.tile(candidate_positions, tf.concat([tf.shape(candidate_idx), [1]], axis=0)) # [length, 160, full_aa_window]
    
    # [length, 160, 1] + [length, 160, full_aa_window] -> [length, 160, full_aa_window]
    candidate_positions = tf.expand_dims(candidate_idx, 2) + candidate_positions
    # point all invalid position to max_spec_length so that we can change it to -0.1 later
    candidate_positions = tf.where(tf.logical_and(candidate_positions >= 0, 
                                                  candidate_positions < input_config.max_spec_length), #tf.minimum(tf.cast(peptide_mass * input_config.inv_bin_step, tf.int32),
                                                                                   #max_spec_length)), # max_spec_length)
                                   candidate_positions,
                                   tf.zeros_like(candidate_positions) + input_config.max_spec_length)
    
    candidate_positions = tf.expand_dims(candidate_positions, 3)
    candidate_spec = tf.gather_nd(tf.concat([spec, [-0.1]], axis=0), candidate_positions)
    # tf.Print(spec, [tf.shape(candidate_spec)])
    
    # candidate_idx_shape = tf.shape(candidate_idx)
    # candidate_idx = tf.reshape(candidate_idx, [-1])
    # # candidate_spec = tf.map_fn(lambda x: tf.ones([2 * input_config.half_aa_window]) * tf.to_float(x), candidate_idx, dtype=(tf.float32))
    # candidate_spec = tf.map_fn(
    #   lambda x: tf.cond(tf.logical_and(x - input_config.half_aa_window > 0, x+input_config.half_aa_window < max_spec_length),
    #                     lambda: spec[x-input_config.half_aa_window : x+input_config.half_aa_window],
    #                     lambda: tf.ones([2*input_config.half_aa_window]) * -0.1), 
    #   candidate_idx, dtype=(tf.float32))
    # candidate_spec = tf.reshape(candidate_spec, tf.concat([candidate_idx_shape, [2*input_config.half_aa_window]], axis=0))
    
    # lookahead_ion = _get_2nd_step_ion(spec, b_ion, peptide_mass) # [length, 20, 20]
    # tf.Print(spec, [tf.shape(lookahead_ion)])
    
    # candidate_spec = tf.concat([candidate_spec, lookahead_ion], axis=1) # [length, 160 + 20, 20]
    # tf.Print(spec, [tf.shape(candidate_spec)])

    # cum_sum = tf.Print(cum_sum, [tf.concat([cum_sum_shape, [3]], axis=0)])
    return spec, low_res_spec, candidate_spec, seq
  
  
  def _shift_and_mask(spec, shift):
    rolled = tf.manip.roll(spec , shift=-shift, axis=0)
    mask = tf.concat([tf.ones([input_config.max_spec_length//10 - shift]), tf.zeros([shift])], 0)
    return tf.multiply(rolled, mask)
  
  
  def _get_cnn_encoder_input(spec, low_res_spec, candidate_spec, seq):
    shifts = tf.convert_to_tensor(input_config.mass_ID_np, tf.float32)
    shifts = tf.scalar_mul(input_config.inv_bin_step / 10, shifts)
    shifts = tf.round(shifts)
    shifts = tf.cast(shifts, tf.int32)
    
    cnn_input = tf.map_fn(
      lambda x: _shift_and_mask(low_res_spec, x), shifts, dtype=(tf.float32))
    cnn_input = tf.concat([tf.expand_dims(low_res_spec, 0), cnn_input], axis=0)
    # Add position number to input
    pos_number = (tf.range(0, input_config.max_spec_length // 10, dtype=tf.float32) - 2499.5) / 1443.3756441065507
    cnn_input = tf.concat([tf.expand_dims(pos_number, 0), cnn_input], axis=0) 
    
    cnn_input = tf.transpose(cnn_input, [1, 0])
    return spec, cnn_input, candidate_spec, seq
  
  
  # Seperate positions and values.
  src_tgt_dataset = src_tgt_dataset.map(_seperate_posi_value,
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)

  # Convert values to percentages.
  src_tgt_dataset = src_tgt_dataset.map(
    lambda positions, values, seq, total_mass: (
      positions, tf.divide(values * 100.0, tf.reduce_sum(values)), seq, total_mass),
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)

  # Bin value according to bin_step.
  src_tgt_dataset = src_tgt_dataset.map(_bin_value_high_and_low_res,
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)

  # Split sequence to characters.
  src_tgt_dataset = src_tgt_dataset.map(
    lambda spec, low_res_spec, seq, total_mass: (spec, low_res_spec, tf.string_split([seq], delimiter='').values, total_mass),
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)
  
  # Convert the characters to ids. Characters that are not in the
  # vocab get the lookup table's default_value integer.
  src_tgt_dataset = src_tgt_dataset.map(
    lambda spec, low_res_spec, seq, total_mass: (
       spec, low_res_spec, tf.cast(tgt_vocab_table.lookup(seq), tf.int32), total_mass),
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)
  
  # Change all 'L' to 'I' ##### WARNING: hard code!!! *12*
  src_tgt_dataset = src_tgt_dataset.map(
    lambda spec, low_res_spec, seq, total_mass: (spec, low_res_spec,
                                                 tf.where(tf.equal(seq, 12), tf.ones_like(seq)*10, seq), total_mass),
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)

  # Retrieve amino acid intensity based on target prefix mass
  src_tgt_dataset = src_tgt_dataset.map(_get_ion_intensity,
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)
  
  # Generate input for CNN encoder
  src_tgt_dataset = src_tgt_dataset.map(_get_cnn_encoder_input,
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)
  
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
    lambda spec, cnn_input, aa_spec, tgt: (spec,
                                           cnn_input,
                                           aa_spec,
                                           tf.concat(([tgt_sos_id], tgt), 0),
                                           tf.concat((tgt, [tgt_eos_id]), 0)),
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)

  # Add in sequence lengths.
  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, cnn_input, aa_spec, tgt_in, tgt_out: (
      src, cnn_input, aa_spec, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
    num_parallel_calls=num_parallel_calls).prefetch(batch_size)

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([None]),  # source spec of fixed size
            tf.TensorShape([None,None]),  # source spec for CNN encoder of fixed size
            tf.TensorShape([None,None,None]),  # source aa spec of fixed size
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            tf.cast(0, tf.float32),  # src
            tf.cast(0, tf.float32),  # src CNN encoder input
            tf.cast(0, tf.float32),  # src aa
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,  # src_len -- unused
            0))  # tgt_len -- unused

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, unused_4, unused_5, src_len, tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if tgt_max_len:
        bucket_width = (tgt_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tgt_len // bucket_width
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

  else:
    batched_dataset = batching_func(src_tgt_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_spec, cnn_input_spec, aa_spec, tgt_input_ids, tgt_output_ids, src_seq_len,
   tgt_seq_len) = (batched_iter.get_next())
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_spec,
      encoder_input = cnn_input_spec,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      source_spectrum_length=src_seq_len,
      target_sequence_length=tgt_seq_len,
      aa_spectrum=aa_spec,
      peptide_mass=None)
