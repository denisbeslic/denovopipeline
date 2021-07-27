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

import math
import os
import random
# ~ random.seed(4)
import sys
import time
import re
import resource

import numpy as np
# ~ cimport numpy as np
# ~ ctypedef np.float32_t C_float32
# ~ ctypedef np.int32_t C_int32
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import traceback
import gc
import deepnovo_config
import deepnovo_model
from deepnovo_worker_io import WorkerIO, WorkerI
import deepnovo_worker_io
from deepnovo_cython_modules import process_spectrum, get_candidate_intensity


def inspect_file_location(data_format, input_file):
  """TODO(nh2tran): docstring."""

  print("inspect_file_location(), input_file = ", input_file)

  if data_format == "msp":
    keyword = "Name"
  elif data_format == "mgf":
    keyword = "BEGIN IONS"

  spectra_file_location = []
  with open(input_file, mode="r") as file_handle:
    line = True
    while line:
      file_location = file_handle.tell()
      line = file_handle.readline()
      if keyword in line:
        spectra_file_location.append(file_location)

  return spectra_file_location


def read_single_spectrum(feature_index, worker_i, feature_fr, spectrum_fr):

  spectrum_list = worker_i.get_spectrum([feature_index], feature_fr, spectrum_fr)
  if not spectrum_list:
    return None
  spectrum = spectrum_list[0]
  feature_id = spectrum["feature_id"]
  raw_sequence = spectrum["raw_sequence"]
  precursor_mass = spectrum["precursor_mass"]
  spectrum_holder = spectrum["spectrum_holder"] if deepnovo_config.FLAGS.use_lstm else None
  spectrum_original_forward = spectrum["spectrum_original_forward"]
  spectrum_original_backward = spectrum["spectrum_original_backward"]
  ms1_profile = spectrum["ms1_profile"]

  ### parse peptide sequence
  # unlabelled spectra with empty raw_sequence can be used as neighbors,
  #   but not as main spectrum for training >> skip empty raw_sequence
  if not raw_sequence:
    status = 'empty'
    return None, None, status
  # parse peptide sequence, skip if unknown_modification
  raw_sequence_len = len(raw_sequence)
  peptide = []
  index = 0
  unknown_modification = False
  while index < raw_sequence_len:
    if raw_sequence[index] == "(":
      if peptide[-1] == "C" and raw_sequence[index:index + 8] == "(+57.02)":
        peptide[-1] = "C(Carbamidomethylation)"
        index += 8
      elif peptide[-1] == 'M' and raw_sequence[index:index + 8] == "(+15.99)":
        peptide[-1] = 'M(Oxidation)'
        index += 8
      elif peptide[-1] == 'N' and raw_sequence[index:index + 6] == "(+.98)":
        peptide[-1] = 'N(Deamidation)'
        index += 6
      elif peptide[-1] == 'Q' and raw_sequence[index:index + 6] == "(+.98)":
        peptide[-1] = 'Q(Deamidation)'
        index += 6
      else:  # unknown modification
        # ~ elif ("".join(raw_sequence[index:index+8])=="(+42.01)"):
        # ~ print("ERROR: unknown modification!")
        # ~ print("raw_sequence = ", raw_sequence)
        # ~ sys.exit()
        unknown_modification = True
        break
    else:
      peptide.append(raw_sequence[index])
      index += 1
  if unknown_modification:
    status = 'mod'
    return None, None, status
  # skip if peptide length > MAX_LEN (train: 30; decode:50)
  peptide_len = len(peptide)
  if peptide_len > deepnovo_config.MAX_LEN:
    status = 'length'
    return None, None, status
  # DEPRECATED quality control: precursor_mass & sequence_mass
  #   this can only be applied in train/valid/test, not in real application;
  #   but training data is from PEAKS DB with ppm control (except DIA),
  #   so this quality control is not needed.
  # ~ sequence_mass = sum(deepnovo_config.mass_AA[aa] for aa in peptide)
  # ~ sequence_mass += deepnovo_config.mass_N_terminus + deepnovo_config.mass_C_terminus
  # ~ print(str(abs(precursor_mass-sequence_mass)/sequence_mass), file=test_handle, end="\n")
  # ~ if (abs(precursor_mass-sequence_mass)/sequence_mass > deepnovo_config.precursor_mass_ppm):
  # ~ counter_skipped_mass_precision += 1
  # ~ print(abs(precursor_mass-sequence_mass))
  # ~ print(sequence_mass)

  # all mass and sequence filters passed
  # counter_read += 1

  ### prepare forward, backward, and padding
  for bucket_id, target_size in enumerate(deepnovo_config._buckets):
    if peptide_len + 2 <= target_size:  # +2 to include GO and EOS
      break
  decoder_size = deepnovo_config._buckets[bucket_id]
  # parse peptide AA sequence to list of ids
  peptide_ids = [deepnovo_config.vocab[x] for x in peptide]
  # padding
  pad_size = decoder_size - (len(peptide_ids) + 2)
  # forward
  if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 2:
    peptide_ids_forward = peptide_ids[:]
    peptide_ids_forward.insert(0, deepnovo_config.GO_ID)
    peptide_ids_forward.append(deepnovo_config.EOS_ID)
    peptide_ids_forward += [deepnovo_config.PAD_ID] * pad_size
  # backward
  if deepnovo_config.FLAGS.direction == 1 or deepnovo_config.FLAGS.direction == 2:
    peptide_ids_backward = peptide_ids[::-1]
    peptide_ids_backward.insert(0, deepnovo_config.EOS_ID)
    peptide_ids_backward.append(deepnovo_config.GO_ID)
    peptide_ids_backward += [deepnovo_config.PAD_ID] * pad_size

  ### retrieve candidate_intensity for test/decode_true_feeding
  if not deepnovo_config.FLAGS.beam_search:
    # forward
    if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 2:
      candidate_intensity_list_forward = []
      prefix_mass = 0.0
      for index in xrange(decoder_size):
        prefix_mass += deepnovo_config.mass_ID[peptide_ids_forward[index]]
        candidate_intensity = get_candidate_intensity(
          spectrum_original_forward,
          precursor_mass,
          prefix_mass,
          0)
        candidate_intensity_list_forward.append(candidate_intensity)
    # backward
    if deepnovo_config.FLAGS.direction == 1 or deepnovo_config.FLAGS.direction == 2:
      candidate_intensity_list_backward = []
      suffix_mass = 0.0
      for index in xrange(decoder_size):
        suffix_mass += deepnovo_config.mass_ID[peptide_ids_backward[index]]
        candidate_intensity = get_candidate_intensity(
          spectrum_original_backward,
          precursor_mass,
          suffix_mass,
          1)
        candidate_intensity_list_backward.append(candidate_intensity)

  ### assign data to buckets
  if deepnovo_config.FLAGS.beam_search:
    if deepnovo_config.FLAGS.direction == 0:
      data = [feature_id,
              spectrum_holder,
              spectrum_original_forward,
              precursor_mass,
              peptide_ids_forward]

    elif deepnovo_config.FLAGS.direction == 1:
      data = [feature_id,
              spectrum_holder,
              spectrum_original_backward,
              precursor_mass,
              peptide_ids_backward]

    else:
      data = [feature_id,
              spectrum_holder,
              spectrum_original_forward,
              spectrum_original_backward,
              precursor_mass,
              peptide_ids_forward,
              peptide_ids_backward]

  else:
    if deepnovo_config.FLAGS.direction == 0:
      data = [spectrum_holder,
              candidate_intensity_list_forward,
              peptide_ids_forward]
    elif deepnovo_config.FLAGS.direction == 1:
      data = [spectrum_holder,
              candidate_intensity_list_backward,
              peptide_ids_backward]
    else:
      data = [spectrum_holder,
              candidate_intensity_list_forward,
              candidate_intensity_list_backward,
              peptide_ids_forward,
              peptide_ids_backward,
              ms1_profile]
  return data, bucket_id, 'OK'


def _prepare_data(feature_index, worker_i):
  """

  :param feature_index:
  :param get_spectrum: a callable, takes in [feature)index] and result spectrum_list
  :return: None if the input feature is not valid
  data, bucket_id, status_code
  """
  ### retrieve spectrum information
  # read spectrum, skip if precursor_mass > MZ_MAX, pre-process spectrum
  try:
    with open(worker_i.input_feature_file, 'r') as feature_fr :
      with open(worker_i.input_spectrum_file, 'r') as spectrum_fr:
        return read_single_spectrum(feature_index, worker_i, feature_fr, spectrum_fr)
  except Exception:
    print("exception in _prepare_data: ")
    traceback.print_exc()
    raise


def read_spectra(worker_io, feature_index_list):
  """TODO(nh2tran): docstring."""

  print("".join(["="] * 80))  # section-separating line
  print("read_spectra()")

  # assign spectrum into buckets according to peptide length
  data_set = [[] for _ in deepnovo_config._buckets]

  # use single/multi processor to read data during training
  worker_i = WorkerI(worker_io)
  if deepnovo_config.FLAGS.multiprocessor == 1:
    with open(worker_i.input_feature_file, 'r') as feature_fr :
      with open(worker_i.input_spectrum_file, 'r') as spectrum_fr:
        result_list = [read_single_spectrum(feature_index, worker_i, feature_fr, spectrum_fr)
                       for feature_index in feature_index_list]
  else:
    mp_func = partial(_prepare_data, worker_i=worker_i)
    gc.collect()
    pool = Pool(processes=deepnovo_config.FLAGS.multiprocessor)
    try:
      result_list = pool.map_async(mp_func, feature_index_list).get(9999)
      pool.close()
      pool.join()
    except KeyboardInterrupt:
      pool.terminate()
      pool.join()
      sys.exit(1)

  counter = len(feature_index_list)
  # worker_io is designed for both prediction and training, hence it does not
  #   check raw_sequence for empty/mod/len because raw_sequence is only provided
  #   in training.
  counter_skipped_empty = 0
  counter_skipped_mod = 0
  counter_skipped_len = 0
  counter_read = 0
  # ~ counter_skipped_mass_precision = 0
  for result in result_list:
    if result is None:
        continue
    data, bucket_id, status = result
    if data:
      counter_read += 1
      data_set[bucket_id].append(data)
    elif status == 'empty':
      counter_skipped_empty += 1
    elif status == 'mod':
      counter_skipped_mod += 1
    elif status == 'length':
      counter_skipped_len += 1
  worker_io.feature_count["read"] += len(result_list)

  del result_list
  del worker_i
  gc.collect()

  counter_skipped_mass = worker_io.feature_count["skipped_mass"]
  counter_skipped = counter_skipped_mass + counter_skipped_empty + counter_skipped_mod + counter_skipped_len
  print("  total peptide %d" % counter)
  print("    peptide read %d" % counter_read)
  print("    peptide skipped %d" % counter_skipped)
  print("    peptide skipped by mass %d" % counter_skipped_mass)
  print("    peptide skipped by empty %d" % counter_skipped_empty)
  print("    peptide skipped by mod %d" % counter_skipped_mod)
  print("    peptide skipped by len %d" % counter_skipped_len)
  # ~ print(counter_skipped_mass_precision)
  # ~ print(abc)

  return data_set, counter_read


def read_random_stack(worker_io, feature_index_list, stack_size):
  """TODO(nh2tran): docstring."""

  print("read_random_stack()")
  random_index_list = random.sample(feature_index_list,
                                    min(stack_size, len(feature_index_list)))
  return read_spectra(worker_io, random_index_list)


def get_batch_01(index_list, data_set, bucket_id):
  """TODO(nh2tran): docstring."""

  # ~ print("get_batch()")

  batch_size = len(index_list)
  spectra_list = []
  candidate_intensity_lists = []
  decoder_inputs = []
  for index in index_list:

    # Get a random entry of encoder and decoder inputs from data,
    (spectrum_holder,
     candidate_intensity_list,
     decoder_input) = data_set[bucket_id][index]

    if spectrum_holder is None:  # spectrum_holder is not provided if not use_lstm
      spectrum_holder = np.zeros(shape=(deepnovo_config.neighbor_size, deepnovo_config.MZ_SIZE),
                                 dtype=np.float32)
    spectra_list.append(spectrum_holder)
    candidate_intensity_lists.append(candidate_intensity_list)
    decoder_inputs.append(decoder_input)

  batch_encoder_inputs = [np.array(spectra_list)]
  batch_intensity_inputs = []
  batch_decoder_inputs = []
  batch_weights = []
  decoder_size = deepnovo_config._buckets[bucket_id]
  for length_idx in xrange(decoder_size):

    # batch_intensity_inputs and batch_decoder_inputs are just re-indexed.
    batch_intensity_inputs.append(
      np.array([candidate_intensity_lists[batch_idx][length_idx]
                for batch_idx in xrange(batch_size)], dtype=np.float32))
    batch_decoder_inputs.append(
      np.array([decoder_inputs[batch_idx][length_idx]
                for batch_idx in xrange(batch_size)], dtype=np.int32))

    # Create target_weights to be 0 for targets that are padding.
    batch_weight = np.ones(batch_size, dtype=np.float32)
    for batch_idx in xrange(batch_size):
      # The corresponding target is decoder_input shifted by 1 forward.
      if length_idx < decoder_size - 1:
        target = decoder_inputs[batch_idx][length_idx + 1]
      # We set weight to 0 if the corresponding target is a PAD symbol.
      if (length_idx == decoder_size - 1
          or target == deepnovo_config.EOS_ID
          or target == deepnovo_config.GO_ID
          or target == deepnovo_config.PAD_ID):
        batch_weight[batch_idx] = 0.0
    batch_weights.append(batch_weight)

  return (batch_encoder_inputs,
          batch_intensity_inputs,
          batch_decoder_inputs,
          batch_weights)


def get_batch_2(index_list, data_set, bucket_id):
  """TODO(nh2tran): docstring."""

  # ~ print("get_batch()")

  batch_size = len(index_list)
  spectrum_holder_list = []
  ms1_profile_list = []
  candidate_intensity_lists_forward = []
  candidate_intensity_lists_backward = []
  decoder_inputs_forward = []
  decoder_inputs_backward = []
  for index in index_list:

    # Get a random entry of encoder and decoder inputs from data,
    (spectrum_holder,
     candidate_intensity_list_forward,
     candidate_intensity_list_backward,
     decoder_input_forward,
     decoder_input_backward,
     ms1_profile) = data_set[bucket_id][index]

    if spectrum_holder is None:  # spectrum_holder is not provided if not use_lstm
      spectrum_holder = np.zeros(shape=(deepnovo_config.neighbor_size, deepnovo_config.MZ_SIZE),
                                 dtype=np.float32)
    spectrum_holder_list.append(spectrum_holder)
    ms1_profile_list.append(ms1_profile)
    candidate_intensity_lists_forward.append(candidate_intensity_list_forward)
    candidate_intensity_lists_backward.append(candidate_intensity_list_backward)
    decoder_inputs_forward.append(decoder_input_forward)
    decoder_inputs_backward.append(decoder_input_backward)

  batch_spectrum_holder = np.array(spectrum_holder_list)
  batch_ms1_profile = np.array(ms1_profile_list)
  batch_intensity_inputs_forward = []
  batch_intensity_inputs_backward = []
  batch_decoder_inputs_forward = []
  batch_decoder_inputs_backward = []
  batch_weights = []
  decoder_size = deepnovo_config._buckets[bucket_id]
  for length_idx in xrange(decoder_size):

    # batch_intensity_inputs and batch_decoder_inputs are re-indexed.
    batch_intensity_inputs_forward.append(
      np.array([candidate_intensity_lists_forward[batch_idx][length_idx]
                for batch_idx in xrange(batch_size)], dtype=np.float32))
    batch_intensity_inputs_backward.append(
      np.array([candidate_intensity_lists_backward[batch_idx][length_idx]
                for batch_idx in xrange(batch_size)], dtype=np.float32))
    batch_decoder_inputs_forward.append(
      np.array([decoder_inputs_forward[batch_idx][length_idx]
                for batch_idx in xrange(batch_size)], dtype=np.int32))
    batch_decoder_inputs_backward.append(
      np.array([decoder_inputs_backward[batch_idx][length_idx]
                for batch_idx in xrange(batch_size)], dtype=np.int32))

    # Create target_weights to be 0 for targets that are padding.
    batch_weight = np.ones(batch_size, dtype=np.float32)
    for batch_idx in xrange(batch_size):
      # The corresponding target is decoder_input shifted by 1 forward.
      if length_idx < decoder_size - 1:
        target = decoder_inputs_forward[batch_idx][length_idx + 1]
      # We set weight to 0 if the corresponding target is a PAD symbol.
      if (length_idx == decoder_size - 1
          or target == deepnovo_config.EOS_ID
          or target == deepnovo_config.GO_ID
          or target == deepnovo_config.PAD_ID):
        batch_weight[batch_idx] = 0.0
    batch_weights.append(batch_weight)

  return (batch_spectrum_holder,
          batch_ms1_profile,
          batch_intensity_inputs_forward,
          batch_intensity_inputs_backward,
          batch_decoder_inputs_forward,
          batch_decoder_inputs_backward,
          batch_weights)


def trim_decoder_input(decoder_input, direction):
  """TODO(nh2tran): docstring."""

  if direction == 0:
    LAST_LABEL = deepnovo_config.EOS_ID
  elif direction == 1:
    LAST_LABEL = deepnovo_config.GO_ID

  # excluding FIRST_LABEL, LAST_LABEL & PAD
  return decoder_input[1:decoder_input.index(LAST_LABEL)]


def print_AA_basic(output_file_handle,
                   feature_id,
                   decoder_input,
                   output,
                   direction,
                   accuracy_AA,
                   len_AA,
                   exact_match):
  """TODO(nh2tran): docstring."""

  if direction == 0:
    LAST_LABEL = deepnovo_config.EOS_ID
  elif direction == 1:
    LAST_LABEL = deepnovo_config.GO_ID

  decoder_input = trim_decoder_input(decoder_input, direction)
  decoder_input_AA = [deepnovo_config.vocab_reverse[x] for x in decoder_input]

  output_seq, output_score = output
  output_len = output_seq.index(LAST_LABEL) if LAST_LABEL in output_seq else 0
  output_AA = [deepnovo_config.vocab_reverse[x] for x in output_seq[:output_len]]

  if direction == 1:
    decoder_input_AA = decoder_input_AA[::-1]
    output_AA = output_AA[::-1]

  print("%s\t%s\t%s\t%.2f\t%d\t%d\t%d\n"
        % (feature_id,
           ",".join(decoder_input_AA),
           ",".join(output_AA),
           output_score,
           accuracy_AA,
           len_AA,
           exact_match),
        file=output_file_handle,
        end="")


def test_AA_match_1by1(decoder_input, output):
  """TODO(nh2tran): docstring."""

  decoder_input_len = len(decoder_input)

  num_match = 0
  index_aa = 0
  while index_aa < decoder_input_len:
    # ~ if  decoder_input[index_aa]==output[index_aa]:
    if (abs(deepnovo_config.mass_ID[decoder_input[index_aa]]
            - deepnovo_config.mass_ID[output[index_aa]])
        < deepnovo_config.AA_MATCH_PRECISION):
      num_match += 1
    index_aa += 1

  return num_match


def test_AA_match_novor(decoder_input, output):
  """TODO(nh2tran): docstring."""

  decoder_input_len = len(decoder_input)
  output_len = len(output)
  decoder_input_mass = [deepnovo_config.mass_ID[x] for x in decoder_input]
  decoder_input_mass_cum = np.cumsum(decoder_input_mass)
  output_mass = [deepnovo_config.mass_ID[x] for x in output]
  output_mass_cum = np.cumsum(output_mass)

  num_match = 0
  i = 0
  j = 0
  while i < decoder_input_len and j < output_len:

    if abs(decoder_input_mass_cum[i] - output_mass_cum[j]) < 0.5:
      if abs(decoder_input_mass[i] - output_mass[j]) < 0.1:
        num_match += 1
      i += 1
      j += 1
    elif decoder_input_mass_cum[i] < output_mass_cum[j]:
      i += 1
    else:
      j += 1

  return num_match


def test_AA_decode_single(decoder_input, output, direction):
  """TODO(nh2tran): docstring."""

  accuracy_AA = 0.0
  len_AA = 0.0
  exact_match = 0.0
  len_match = 0.0

  if direction == 0:
    LAST_LABEL = deepnovo_config.EOS_ID
  elif direction == 1:
    LAST_LABEL = deepnovo_config.GO_ID

  # decoder_input = [AA]; output = [AA]
  decoder_input = trim_decoder_input(decoder_input, direction)
  decoder_input_len = len(decoder_input)
  output_len = output.index(LAST_LABEL) if LAST_LABEL in output else 0
  output = output[:output_len]

  # measure accuracy
  num_match = test_AA_match_novor(decoder_input, output)
  # ~ accuracy_AA = num_match / decoder_input_len
  accuracy_AA = num_match
  len_AA = decoder_input_len
  len_decode = output_len
  if num_match == decoder_input_len:
    exact_match = 1.0
  if output_len == decoder_input_len:
    len_match = 1.0

  # testing
  # ~ print(decoder_input_AA)
  # ~ print(output_AA)
  # ~ print(num_match)
  # ~ print(batch_accuracy_AA)
  # ~ print(num_exact_match)
  # ~ print(num_len_match)
  # ~ sys.exit()

  return accuracy_AA, len_AA, len_decode, exact_match, len_match


def test_AA_true_feeding_single(decoder_input, output, direction):
  """TODO(nh2tran): docstring."""

  accuracy_AA = 0.0
  len_AA = 0.0
  exact_match = 0.0
  len_match = 0.0

  # decoder_input = [AA]; output = [AA...]
  decoder_input = trim_decoder_input(decoder_input, direction)
  decoder_input_len = len(decoder_input)

  # measure accuracy
  num_match = test_AA_match_1by1(decoder_input, output)
  # ~ accuracy_AA = num_match / decoder_input_len
  accuracy_AA = num_match
  len_AA = decoder_input_len
  if num_match == decoder_input_len:
    exact_match = 1.0
  # ~ if output_len == decoder_input_len:
  # ~ len_match = 1.0

  return accuracy_AA, len_AA, exact_match, len_match


def test_AA_decode_batch(scans,
                         decoder_inputs,
                         outputs,
                         direction,
                         output_file_handle):
  """TODO(nh2tran): docstring."""

  batch_accuracy_AA = 0.0
  batch_len_AA = 0.0
  batch_len_decode = 0.0
  num_exact_match = 0.0
  num_len_match = 0.0

  # ~ batch_size = len(decoder_inputs)

  for index in xrange(len(scans)):
    # ~ # for testing
    # ~ for index in xrange(15,20):

    scan = scans[index]
    decoder_input = decoder_inputs[index]
    output = outputs[index]
    output_seq, _ = output

    (accuracy_AA,
     len_AA,
     len_decode,
     exact_match,
     len_match) = test_AA_decode_single(decoder_input, output_seq, direction)

    # ~ # for testing
    # ~ print([deepnovo_config.vocab_reverse[x] for x in decoder_input[1:]])
    # ~ print([deepnovo_config.vocab_reverse[x] for x in output_seq])
    # ~ print(accuracy_AA)
    # ~ print(exact_match)
    # ~ print(len_match)

    # print to output file
    print_AA_basic(output_file_handle,
                   scan,
                   decoder_input,
                   output,
                   direction,
                   accuracy_AA,
                   len_AA,
                   exact_match)

    batch_accuracy_AA += accuracy_AA
    batch_len_AA += len_AA
    batch_len_decode += len_decode
    num_exact_match += exact_match
    num_len_match += len_match

  # ~ return (batch_accuracy_AA/batch_size,
  # ~ num_exact_match/batch_size,
  # ~ num_len_match/batch_size)
  # ~ return (batch_accuracy_AA/batch_len_AA,
  # ~ num_exact_match/batch_size,
  # ~ num_len_match/batch_size)
  return (batch_accuracy_AA,
          batch_len_AA,
          batch_len_decode,
          num_exact_match,
          num_len_match)


def test_logit_single_01(decoder_input, output_logit):
  """TODO(nh2tran): docstring."""

  output = [np.argmax(x) for x in output_logit]

  return test_AA_true_feeding_single(decoder_input,
                                     output,
                                     deepnovo_config.FLAGS.direction)


def test_logit_single_2(decoder_input_forward,
                        decoder_input_backward,
                        output_logit_forward,
                        output_logit_backward):
  """TODO(nh2tran): docstring."""

  # length excluding FIRST_LABEL & LAST_LABEL
  decoder_input_len = decoder_input_forward[1:].index(deepnovo_config.EOS_ID)

  # average forward-backward prediction logit
  logit_forward = output_logit_forward[:decoder_input_len]
  logit_backward = output_logit_backward[:decoder_input_len]
  logit_backward = logit_backward[::-1]
  output = []
  for x, y in zip(logit_forward, logit_backward):
    prob_forward = np.exp(x) / np.sum(np.exp(x))
    prob_backward = np.exp(y) / np.sum(np.exp(y))
    output.append(np.argmax(prob_forward * prob_backward))

  output.append(deepnovo_config.EOS_ID)

  return test_AA_true_feeding_single(decoder_input_forward, output, direction=0)


def test_logit_batch_01(decoder_inputs, output_logits):
  """TODO(nh2tran): docstring."""

  batch_accuracy_AA = 0.0
  batch_len_AA = 0.0
  num_exact_match = 0.0
  num_len_match = 0.0
  batch_size = len(decoder_inputs[0])
  for batch in xrange(batch_size):
    decoder_input = [x[batch] for x in decoder_inputs]
    output_logit = [x[batch] for x in output_logits]
    accuracy_AA, len_AA, exact_match, len_match = test_logit_single_01(
      decoder_input,
      output_logit)

    # for testing
    # ~ if (exact_match==0):
    # ~ print(batch)
    # ~ print([deepnovo_config.vocab_reverse[x] for x in decoder_input[1:]])
    # ~ print([deepnovo_config.vocab_reverse[np.argmax(x)] for x in output_logit])
    # ~ print(accuracy_AA)
    # ~ print(exact_match)
    # ~ print(len_match)
    # ~ sys.exit()

    batch_accuracy_AA += accuracy_AA
    batch_len_AA += len_AA
    num_exact_match += exact_match
    num_len_match += len_match

  # ~ return (batch_accuracy_AA/batch_size,
  # ~ num_exact_match/batch_size,
  # ~ num_len_match/batch_size)
  # ~ return (batch_accuracy_AA/batch_len_AA,
  # ~ num_exact_match/batch_size,
  # ~ num_len_match/batch_size)
  return batch_accuracy_AA, batch_len_AA, num_exact_match, num_len_match


def test_logit_batch_2(decoder_inputs_forward,
                       decoder_inputs_backward,
                       output_logits_forward,
                       output_logits_backward):
  """TODO(nh2tran): docstring."""

  batch_accuracy_AA = 0.0
  batch_len_AA = 0.0
  num_exact_match = 0.0
  num_len_match = 0.0
  batch_size = len(decoder_inputs_forward[0])
  for batch in xrange(batch_size):
    decoder_input_forward = [x[batch] for x in decoder_inputs_forward]
    decoder_input_backward = [x[batch] for x in decoder_inputs_backward]
    output_logit_forward = [x[batch] for x in output_logits_forward]
    output_logit_backward = [x[batch] for x in output_logits_backward]

    accuracy_AA, len_AA, exact_match, len_match = test_logit_single_2(
      decoder_input_forward,
      decoder_input_backward,
      output_logit_forward,
      output_logit_backward)

    batch_accuracy_AA += accuracy_AA
    batch_len_AA += len_AA
    num_exact_match += exact_match
    num_len_match += len_match

  # ~ return (batch_accuracy_AA/batch_size,
  # ~ num_exact_match/batch_size,
  # ~ num_len_match/batch_size)
  # ~ return (batch_accuracy_AA/batch_len_AA,
  # ~ num_exact_match/batch_size,
  # ~ num_len_match/batch_size)
  return batch_accuracy_AA, batch_len_AA, num_exact_match, num_len_match


def test_accuracy(sess, model, data_set, bucket_id, print_summary=True):
  """TODO(nh2tran): docstring."""

  spectrum_time = 0.0
  avg_loss = 0.0
  avg_loss_classification = 0.0
  avg_accuracy_AA = 0.0
  avg_len_AA = 0.0
  avg_accuracy_peptide = 0.0
  avg_accuracy_len = 0.0

  data_set_len = len(data_set[bucket_id])
  data_set_index_list = range(data_set_len)
  data_set_index_chunk_list = [data_set_index_list[i:i + deepnovo_config.batch_size]
                               for i in range(0,
                                              data_set_len,
                                              deepnovo_config.batch_size)]

  for chunk in data_set_index_chunk_list:

    start_time = time.time()

    # get_batch_01/2
    if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 1:
      (encoder_inputs,
       intensity_inputs,
       decoder_inputs,
       target_weights) = get_batch_01(chunk, data_set, bucket_id)
    else:
      (spectrum_holder,
       ms1_profile,
       intensity_inputs_forward,
       intensity_inputs_backward,
       decoder_inputs_forward,
       decoder_inputs_backward,
       target_weights) = get_batch_2(chunk, data_set, bucket_id)

    # model_step
    if deepnovo_config.FLAGS.direction == 0:
      loss, loss_classification, output_logits = model.step(
        sess,
        encoder_inputs,
        intensity_inputs_forward=intensity_inputs,
        decoder_inputs_forward=decoder_inputs,
        target_weights=target_weights,
        bucket_id=bucket_id,
        training_mode=False)
    elif deepnovo_config.FLAGS.direction == 1:
      loss, loss_classification, output_logits = model.step(
        sess,
        encoder_inputs,
        intensity_inputs_backward=intensity_inputs,
        decoder_inputs_backward=decoder_inputs,
        target_weights=target_weights,
        bucket_id=bucket_id,
        training_mode=False)
    else:
      loss, loss_classification, output_logits_forward, output_logits_backward = model.step(
        sess,
        spectrum_holder,
        ms1_profile,
        intensity_inputs_forward=intensity_inputs_forward,
        intensity_inputs_backward=intensity_inputs_backward,
        decoder_inputs_forward=decoder_inputs_forward,
        decoder_inputs_backward=decoder_inputs_backward,
        target_weights=target_weights,
        bucket_id=bucket_id,
        training_mode=False)

    spectrum_time += time.time() - start_time

    # test_logit_batch_01/2
    if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 1:
      (batch_accuracy_AA,
       batch_len_AA,
       num_exact_match,
       num_len_match) = test_logit_batch_01(decoder_inputs, output_logits)
    else:
      (batch_accuracy_AA,
       batch_len_AA,
       num_exact_match,
       num_len_match) = test_logit_batch_2(decoder_inputs_forward,
                                           decoder_inputs_backward,
                                           output_logits_forward,
                                           output_logits_backward)

    avg_loss += loss * len(chunk)  # because the loss was averaged by batch_size
    avg_loss_classification += loss_classification * len(chunk)
    avg_accuracy_AA += batch_accuracy_AA
    avg_len_AA += batch_len_AA
    avg_accuracy_peptide += num_exact_match
    avg_accuracy_len += num_len_match

  spectrum_time /= data_set_len
  avg_loss /= data_set_len
  avg_loss_classification /= data_set_len
  avg_accuracy_AA /= avg_len_AA
  avg_accuracy_peptide /= data_set_len
  avg_accuracy_len /= data_set_len
  eval_ppx = math.exp(avg_loss) if avg_loss < 300 else float('inf')
  eval_ppx_classification = math.exp(avg_loss_classification) if avg_loss_classification < 300 else float('inf')

  if print_summary:
    print("test_accuracy()")
    print("  bucket %d spectrum_time %.4f" % (bucket_id, spectrum_time))
    print("  bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
    print("  bucket %d perplexity classification %.2f" % (bucket_id, eval_ppx_classification))
    print("  bucket %d avg_accuracy_AA %.4f" % (bucket_id, avg_accuracy_AA))
    print("  bucket %d avg_accuracy_peptide %.4f"
          % (bucket_id, avg_accuracy_peptide))
    print("  bucket %d avg_accuracy_len %.4f" % (bucket_id, avg_accuracy_len))

  return avg_loss, avg_accuracy_AA, avg_accuracy_peptide, avg_accuracy_len


def knapsack_example():
  """TODO(nh2tran): docstring."""

  peptide_mass = 11
  print("peptide_mass = ", peptide_mass)
  mass_aa = [2, 3, 4, 5]
  print("mass_aa = ", mass_aa)
  knapsack_matrix = np.zeros(shape=(4, 11), dtype=bool)

  for aa_id in xrange(4):
    for col in xrange(peptide_mass):

      current_mass = col + 1

      if current_mass < mass_aa[aa_id]:
        knapsack_matrix[aa_id, col] = False

      if current_mass == mass_aa[aa_id]:
        knapsack_matrix[aa_id, col] = True

      if current_mass > mass_aa[aa_id]:
        sub_mass = current_mass - mass_aa[aa_id]
        sub_col = sub_mass - 1
        if np.sum(knapsack_matrix[:, sub_col]) > 0:
          knapsack_matrix[aa_id, col] = True
          knapsack_matrix[:, col] = np.logical_or(knapsack_matrix[:, col],
                                                  knapsack_matrix[:, sub_col])
        else:
          knapsack_matrix[aa_id, col] = False

    print("mass_aa[{0}] = {1}".format(aa_id, mass_aa[aa_id]))
    print(knapsack_matrix)


def knapsack_build():
  """TODO(nh2tran): docstring."""

  peptide_mass = deepnovo_config.MZ_MAX
  peptide_mass = peptide_mass - (deepnovo_config.mass_C_terminus + deepnovo_config.mass_H)
  print("peptide_mass = ", peptide_mass)

  peptide_mass_round = int(round(peptide_mass
                                 * deepnovo_config.KNAPSACK_AA_RESOLUTION))
  print("peptide_mass_round = ", peptide_mass_round)

  # ~ peptide_mass_upperbound = (peptide_mass_round
  # ~ + deepnovo_config.KNAPSACK_MASS_PRECISION_TOLERANCE)
  peptide_mass_upperbound = (peptide_mass_round
                             + deepnovo_config.KNAPSACK_AA_RESOLUTION)

  knapsack_matrix = np.zeros(shape=(deepnovo_config.vocab_size,
                                    peptide_mass_upperbound),
                             dtype=bool)

  for aa_id in xrange(3, deepnovo_config.vocab_size):  # excluding PAD, GO, EOS

    mass_aa_round = int(round(deepnovo_config.mass_ID[aa_id]
                              * deepnovo_config.KNAPSACK_AA_RESOLUTION))
    print(deepnovo_config.vocab_reverse[aa_id], mass_aa_round)

    for col in xrange(peptide_mass_upperbound):

      # col 0 ~ mass 1
      # col + 1 = mass
      # col = mass - 1
      current_mass = col + 1

      if current_mass < mass_aa_round:
        knapsack_matrix[aa_id, col] = False

      if current_mass == mass_aa_round:
        knapsack_matrix[aa_id, col] = True

      if current_mass > mass_aa_round:
        sub_mass = current_mass - mass_aa_round
        sub_col = sub_mass - 1
        if np.sum(knapsack_matrix[:, sub_col]) > 0:
          knapsack_matrix[aa_id, col] = True
          knapsack_matrix[:, col] = np.logical_or(knapsack_matrix[:, col],
                                                  knapsack_matrix[:, sub_col])
        else:
          knapsack_matrix[aa_id, col] = False

  np.save("knapsack.npy", knapsack_matrix)


def knapsack_search(knapsack_matrix, peptide_mass, mass_precision_tolerance):
  """TODO(nh2tran): docstring."""

  # ~ knapsack_matrix = np.load("knapsack.npy")

  peptide_mass_round = int(round(peptide_mass
                                 * deepnovo_config.KNAPSACK_AA_RESOLUTION))
  # 100 x 0.0001 Da
  peptide_mass_upperbound = peptide_mass_round + mass_precision_tolerance
  peptide_mass_lowerbound = peptide_mass_round - mass_precision_tolerance

  # [peptide_mass_lowerbound, peptide_mass_upperbound] will NOT be less than
  #   mass_AA_min_round.
  if peptide_mass_upperbound < deepnovo_config.mass_AA_min_round:  # 57.0215
    return []
  # peptide_mass_upperbound may exceed column 2982.9895,
  #   but numpy will ignore the extra indices.
  # not necessary, because peptide_mass_upperbound > 57.0215
  # ~ if (peptide_mass_lowerbound < 0):
  # ~ return []

  # col 0 ~ mass 1
  # col + 1 = mass
  # col = mass - 1
  # [)
  peptide_mass_lowerbound_col = peptide_mass_lowerbound - 1
  peptide_mass_upperbound_col = peptide_mass_upperbound - 1
  # Search for any nonzero col
  candidate_AA_id = np.flatnonzero(
    np.any(knapsack_matrix[:, peptide_mass_lowerbound_col:peptide_mass_upperbound_col + 1],
           # pylint: disable=line-too-long
           axis=1))

  # Search for closest col
  # ~ m = knapsack_matrix[:,peptide_mass_lowerbound_col:peptide_mass_upperbound_col+1] # pylint: disable=line-too-long
  # ~ m_any = np.any(m, axis=0)
  # ~ m_nonzero_col = np.flatnonzero(m_any)
  # ~ if (m_nonzero_col.size == 0):
  # ~ return []
  # ~ m_closest_col = m_nonzero_col[np.argmin(np.absolute(
  # ~ m_nonzero_col-mass_precision_tolerance))]
  # ~ candidate_AA_id = np.flatnonzero(m[:, m_closest_col])

  return candidate_AA_id.tolist()


def knapsack_search_mass(knapsack_matrix,
                         peptide_mass,
                         mass_precision_tolerance):
  """TODO(nh2tran): docstring."""

  # ~ knapsack_matrix = np.load("knapsack.npy")

  # 100 x 0.0001 Da
  peptide_mass_round = int(round(peptide_mass
                                 * deepnovo_config.KNAPSACK_AA_RESOLUTION))
  peptide_mass_upperbound = peptide_mass_round + mass_precision_tolerance
  peptide_mass_lowerbound = peptide_mass_round - mass_precision_tolerance

  # [peptide_mass_lowerbound, peptide_mass_upperbound] will NOT be less than
  #   mass_AA_min_round.
  if peptide_mass_upperbound < deepnovo_config.mass_AA_min_round:  # 57.0215
    return []
  # peptide_mass_upperbound may exceed column 2982.9895, but numpy will ignore
  #   the extra indices.
  # not necessary, because peptide_mass_upperbound > 57.0215
  # ~ if (peptide_mass_lowerbound < 0):
  # ~ return []

  # col 0 ~ mass 1
  # col + 1 = mass
  # col = mass - 1
  # [)
  peptide_mass_lowerbound_col = peptide_mass_lowerbound - 1
  peptide_mass_upperbound_col = peptide_mass_upperbound - 1

  # Search for any nonzero col
  # ~ candidate_AA_id = np.flatnonzero(np.any(knapsack_matrix[:,peptide_mass_lowerbound_col:peptide_mass_upperbound_col+1], # pylint: disable=line-too-long
  # ~ axis=1))

  # Search for closest col
  m = knapsack_matrix[:, peptide_mass_lowerbound_col:peptide_mass_upperbound_col + 1]
  m_any = np.any(m, axis=0)
  m_nonzero_col = np.flatnonzero(m_any)
  if m_nonzero_col.size == 0:
    return []
  m_closest_col = m_nonzero_col[np.argmin(
    np.absolute(m_nonzero_col - mass_precision_tolerance))]
  m_closest_col_mass = ((peptide_mass_lowerbound_col + m_closest_col + 1)
                        / float(deepnovo_config.KNAPSACK_AA_RESOLUTION))
  # ~ print(m_closest_col_mass)
  # ~ sys.exit()

  return [m_closest_col_mass]


def create_model(session, training_mode):
  """TODO(nh2tran): docstring."""

  print("".join(["="] * 80))  # section-separating line
  print("create_model()")

  model = deepnovo_model.TrainingModel(session, training_mode)

  # folder of training state
  ckpt = tf.train.get_checkpoint_state(deepnovo_config.FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + ".index"):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    # ~ train_writer = tf.train.SummaryWriter("train_log", session.graph)
    # ~ train_writer.close()
    session.run(tf.global_variables_initializer())
  return model


def decode_true_feeding_01(sess, model, direction, data_set):
  """TODO(nh2tran): docstring."""

  # FORWARD/BACKWARD setting
  if direction == 0:
    model_lstm_state0 = model.output_forward["lstm_state0"]
    model_output_log_prob = model.output_forward["logprob"]
    model_lstm_state = model.output_forward["lstm_state"]
  elif direction == 1:
    model_lstm_state0 = model.output_backward["lstm_state0"]
    model_output_log_prob = model.output_backward["logprob"]
    model_lstm_state = model.output_backward["lstm_state"]

  data_set_len = len(data_set)
  # recall that a data_set[spectrum_id] includes the following
  #     spectrum_holder                     # 0
  #     candidate_intensity_list[MAX_LEN]   # 1
  #     decoder_input[MAX_LEN]              # 2

  # process in stacks
  decode_stack_size = 128
  data_set_index_list = range(data_set_len)
  data_set_index_stack_list = [data_set_index_list[i:i + decode_stack_size]
                               for i in range(0,
                                              data_set_len,
                                              decode_stack_size)]

  # spectrum_holder >> lstm_state0;
  block_c_state0 = []
  block_h_state0 = []

  for stack in data_set_index_stack_list:
    block_spectrum = np.array([data_set[x][0] for x in stack])
    input_feed = {}
    input_feed[model.input_dict["spectrum"].name] = block_spectrum
    output_feed = model_lstm_state0
    stack_c_state0, stack_h_state0 = sess.run(fetches=output_feed,
                                              feed_dict=input_feed)
    block_c_state0.append(stack_c_state0)
    block_h_state0.append(stack_h_state0)

  # ~ block_state0 = np.vstack(block_state0)

  # MAIN decoding LOOP in STACKS
  output_log_probs = [[] for x in xrange(len(data_set[0][2]))]

  for stack_index, stack in enumerate(data_set_index_stack_list):

    stack_c_state = block_c_state0[stack_index]
    stack_h_state = block_h_state0[stack_index]

    for index in xrange(len(data_set[0][2])):

      block_candidate_intensity = np.array([data_set[x][1][index]
                                            for x in stack])
      block_AA_id_2 = np.array([data_set[x][2][index] for x in stack])  # nobi
      if index - 1 >= 0:
        block_AA_id_1 = np.array([data_set[x][2][index - 1] for x in stack])
      else:
        block_AA_id_1 = block_AA_id_2  # nobi

      # FEED & RUN
      input_feed = {}
      # nobi
      input_feed[model.input_dict["AAid"][0].name] = block_AA_id_1
      input_feed[model.input_dict["AAid"][1].name] = block_AA_id_2
      input_feed[model.input_dict["intensity"].name] = block_candidate_intensity
      # lstm.len_full
      input_feed[model.input_dict["lstm_state"][0].name] = stack_c_state
      input_feed[model.input_dict["lstm_state"][1].name] = stack_h_state
      # nobi
      # ~ input_feed[model.input_dict["lstm_state"][0].name] = block_c_state0[stack_index]
      # ~ input_feed[model.input_dict["lstm_state"][1].name] = block_h_state0[stack_index]
      #
      # lstm.len_full
      output_feed = [model_output_log_prob, model_lstm_state]
      # nobi
      # ~ output_feed = model_output_log_prob
      #
      # lstm.len_full
      stack_log_prob, (stack_c_state, stack_h_state) = sess.run(
        fetches=output_feed,
        feed_dict=input_feed)
      # nobi
      # ~ stack_log_prob = sess.run(fetches=output_feed, feed_dict=input_feed)
      #
      output_log_probs[index].append(stack_log_prob)

  output_log_probs = [np.vstack(x) for x in output_log_probs]

  return output_log_probs


def decode_true_feeding_2(sess, model, data_set):
  """TODO(nh2tran): docstring."""

  data_set_forward = [[x[0], x[1], x[3]] for x in data_set]
  data_set_backward = [[x[0], x[2], x[4]] for x in data_set]

  output_log_prob_forwards = decode_true_feeding_01(sess,
                                                    model,
                                                    direction=0,
                                                    data_set=data_set_forward)

  output_log_prob_backwards = decode_true_feeding_01(sess,
                                                     model,
                                                     direction=1,
                                                     data_set=data_set_backward)

  return output_log_prob_forwards, output_log_prob_backwards


def decode_true_feeding(sess, model, data_set):
  """TODO(nh2tran): docstring."""

  spectrum_time = time.time()

  if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 1:

    output_log_probs = decode_true_feeding_01(sess,
                                              model,
                                              deepnovo_config.FLAGS.direction,
                                              data_set)

    spectrum_time = time.time() - spectrum_time

    decoder_inputs = [x[-1] for x in data_set]
    (batch_accuracy_AA,
     batch_len_AA,
     num_exact_match,
     num_len_match) = test_logit_batch_01(zip(*decoder_inputs),
                                          output_log_probs)
  else:

    (output_log_prob_forwards,
     output_log_prob_backwards) = decode_true_feeding_2(sess, model, data_set)

    spectrum_time = time.time() - spectrum_time

    decoder_inputs_forward = [x[-2] for x in data_set]
    decoder_inputs_backward = [x[-1] for x in data_set]
    (batch_accuracy_AA,
     batch_len_AA,
     num_exact_match,
     num_len_match) = test_logit_batch_2(zip(*decoder_inputs_forward),
                                         zip(*decoder_inputs_backward),
                                         output_log_prob_forwards,
                                         output_log_prob_backwards)

  return (batch_accuracy_AA / batch_len_AA,
          num_exact_match / len(data_set),
          num_len_match / len(data_set),
          spectrum_time / len(data_set))


def decode_beam_select_01(output_top_paths, direction):
  """TODO(nh2tran): docstring."""

  # LAST_LABEL
  if direction == 0:
    LAST_LABEL = deepnovo_config.EOS_ID
  elif direction == 1:
    LAST_LABEL = deepnovo_config.GO_ID

  outputs = []
  for entry in xrange(len(output_top_paths)):

    top_paths = output_top_paths[entry]

    if not top_paths:  # cannot find the peptide
      output_seq = [LAST_LABEL]
      output_score = float("inf")
      # ~ print("ERROR: no path found ", entry)
      # ~ sys.exit()
    else:
      # path format: [path, score, direction]
      path_scores = np.array([x[1] for x in top_paths])
      top_path = top_paths[np.argmax(path_scores)]
      output_seq = top_path[0][1:]
      output_score = top_path[1]

    # output sequence with score
    outputs.append([output_seq, output_score])

  return outputs


def decode_beam_select_2(output_top_paths):
  """TODO(nh2tran): docstring."""

  outputs = []
  for top_paths in output_top_paths:

    if not top_paths:  # cannot find the peptide
      output_seq = [deepnovo_config.EOS_ID]
      output_score = float("inf")
      # ~ print("ERROR: no path found ", entry)
      # ~ sys.exit()
    else:
      # path format: [path, score, direction]
      path_scores = np.array([x[1] for x in top_paths])
      path_lengths = np.array([len(x[0]) for x in top_paths])
      # ~ top_path = top_paths[np.argmax(path_scores)]
      top_path = top_paths[np.argmax(path_scores / path_lengths)]
      output_seq = top_path[0]
      output_score = top_path[1] / len(output_seq)
      output_seq.append(deepnovo_config.EOS_ID)
      # ~ output_score = top_path[1]

    # output sequence with score
    outputs.append([output_seq, output_score])

  return outputs


def decode_beam_search_01(sess,
                          model,
                          knapsack_matrix,
                          direction,
                          prefix_mass_list,
                          precursor_mass_precision,
                          knapsack_precision,
                          data_set):
  """TODO(nh2tran): docstring."""

  print("decode_beam_search_01(), direction={0}".format(direction))

  # for testing
  test_time_decode = 0.0
  test_time_tf = 0.0
  test_time = 0.0

  # for testing
  start_time_decode = time.time()

  # FORWARD/BACKWARD setting
  if direction == 0:

    model_lstm_state0 = model.output_forward["lstm_state0"]
    model_output_log_prob = model.output_forward["logprob"]
    model_lstm_state = model.output_forward["lstm_state"]

    FIRST_LABEL = deepnovo_config.GO_ID
    LAST_LABEL = deepnovo_config.EOS_ID

  elif direction == 1:

    model_lstm_state0 = model.output_backward["lstm_state0"]
    model_output_log_prob = model.output_backward["logprob"]
    model_lstm_state = model.output_backward["lstm_state"]

    FIRST_LABEL = deepnovo_config.EOS_ID
    LAST_LABEL = deepnovo_config.GO_ID

  data_set_len = len(data_set)
  # recall that a data_set[spectrum_id] includes the following
  #     feature_id                # 0
  #     spectrum_holder     # 1
  #     spectrum_original   # 2
  #     peptide_mass        # 3

  # our TARGET
  output_top_paths = [[] for x in xrange(data_set_len)]

  # how many spectra to process at 1 block-run
  decode_block_size = deepnovo_config.batch_size

  # for testing
  start_time_tf = time.time()

  # spectrum_holder >> lstm_state0; process in stacks
  data_set_index_list = range(data_set_len)
  data_set_index_stack_list = [data_set_index_list[i:i + decode_block_size]
                               for i in range(0,
                                              data_set_len,
                                              decode_block_size)]

  block_c_state0 = []
  block_h_state0 = []

  for stack in data_set_index_stack_list:

    block_spectrum = []
    for x in stack:
      spectrum_holder = data_set[x][1]
      if spectrum_holder is None:
        spectrum_holder = np.zeros(shape=(deepnovo_config.neighbor_size, deepnovo_config.MZ_SIZE),
                                   dtype=np.float32)
        block_spectrum.append(spectrum_holder)
    block_spectrum = np.array(block_spectrum)
    input_feed = {}
    input_feed[model.input_dict["spectrum"].name] = block_spectrum
    output_feed = model_lstm_state0
    stack_c_state0, stack_h_state0 = sess.run(fetches=output_feed,
                                              feed_dict=input_feed)
    block_c_state0.append(stack_c_state0)
    block_h_state0.append(stack_h_state0)

  block_c_state0 = np.vstack(block_c_state0)
  block_h_state0 = np.vstack(block_h_state0)

  # for testing
  test_time_tf += time.time() - start_time_tf

  # hold the spectra & their paths under processing
  active_search = []

  # fill in the first entries of active_search
  for spectrum_id in xrange(decode_block_size):
    active_search.append([])
    active_search[-1].append(spectrum_id)
    active_search[-1].append([[[FIRST_LABEL],  # current_paths
                               prefix_mass_list[spectrum_id],
                               0.0,
                               block_c_state0[spectrum_id],
                               block_h_state0[spectrum_id]]])

  # how many spectra that have been put into active_search
  spectrum_count = decode_block_size

  # MAIN LOOP; break when the block-run is empty
  while True:

    # block-data for model-step-run
    block_AA_ID_1 = []  # nobi
    block_AA_ID_2 = []  # nobi
    block_c_state = []
    block_h_state = []
    block_candidate_intensity = []

    # data to construct new_paths
    block_path_0 = []
    block_prefix_mass = []
    block_score = []
    block_mass_filter_candidate = []

    # store the number of paths of each entry (spectrum) in the big blocks
    entry_block_size = []

    # gather data into blocks
    for entry in active_search:

      spectrum_id = entry[0]
      current_paths = entry[1]
      peptide_mass = data_set[spectrum_id][3]
      spectrum_original = data_set[spectrum_id][2]

      path_count = 0

      for path in current_paths:

        AA_ID_2 = path[0][-1]  # nobi
        if len(path[0]) > 1:
          AA_ID_1 = path[0][-2]
        else:
          AA_ID_1 = AA_ID_2  # nobi

        prefix_mass = path[1]
        score = path[2]
        c_state = path[3]
        h_state = path[4]

        # reach LAST_LABEL >> check mass
        if AA_ID_2 == LAST_LABEL:  # nobi
          if (abs(prefix_mass - peptide_mass)
              <= deepnovo_config.PRECURSOR_MASS_PRECISION_TOLERANCE):
            output_top_paths[spectrum_id].append([path[0], path[2], direction])
          continue

        # for testing
        start_time = time.time()

        # CANDIDATE INTENSITY
        candidate_intensity = get_candidate_intensity(spectrum_original,
                                                      peptide_mass,
                                                      prefix_mass,
                                                      direction)

        # for testing
        test_time += time.time() - start_time

        # SUFFIX MASS filter
        suffix_mass = (peptide_mass - prefix_mass
                       - deepnovo_config.mass_ID[LAST_LABEL])
        mass_filter_candidate = knapsack_search(knapsack_matrix,
                                                suffix_mass,
                                                knapsack_precision)

        if not mass_filter_candidate:  # not enough mass left to extend
          mass_filter_candidate.append(LAST_LABEL)  # try to end the sequence

        # gather BLOCK-data
        block_AA_ID_1.append(AA_ID_1)  # nobi
        block_AA_ID_2.append(AA_ID_2)  # nobi
        block_c_state.append(c_state)
        block_h_state.append(h_state)
        block_candidate_intensity.append(candidate_intensity)

        block_path_0.append(path[0])
        block_prefix_mass.append(prefix_mass)
        block_score.append(score)
        block_mass_filter_candidate.append(mass_filter_candidate)

        # record the block size for each entry
        path_count += 1

      entry_block_size.append(path_count)

    # RUN tf blocks if not empty
    if block_AA_ID_1:
      # for testing
      start_time_tf = time.time()

      # FEED and RUN TensorFlow-model
      block_AA_ID_1 = np.array(block_AA_ID_1)  # nobi
      block_AA_ID_2 = np.array(block_AA_ID_2)  # nobi
      block_c_state = np.array(block_c_state)
      block_h_state = np.array(block_h_state)
      block_candidate_intensity = np.array(block_candidate_intensity)

      input_feed = {}
      input_feed[model.input_dict["AAid"][0].name] = block_AA_ID_1  # nobi
      input_feed[model.input_dict["AAid"][1].name] = block_AA_ID_2  # nobi
      input_feed[model.input_dict["intensity"].name] = block_candidate_intensity
      input_feed[model.input_dict["lstm_state"][0].name] = block_c_state
      input_feed[model.input_dict["lstm_state"][1].name] = block_h_state

      output_feed = [model_output_log_prob, model_lstm_state]  # lstm.len_full
      # ~ output_feed = model_output_log_prob # nobi

      current_log_prob, (current_c_state, current_h_state) = sess.run(
        output_feed,
        input_feed)  # lstm.len_full
      # ~ current_log_prob = sess.run(output_feed,input_feed) # nobi

      # for testing
      test_time_tf += time.time() - start_time_tf

    # find new_paths for each entry
    block_index = 0
    for entry_index, entry in enumerate(active_search):

      new_paths = []

      for index in xrange(block_index,
                          block_index + entry_block_size[entry_index]):

        for aa_id in block_mass_filter_candidate[index]:

          new_paths.append([])
          new_paths[-1].append(block_path_0[index] + [aa_id])
          new_paths[-1].append(block_prefix_mass[index]
                               + deepnovo_config.mass_ID[aa_id])

          if aa_id > 2:  # do NOT add score of GO, EOS, PAD
            new_paths[-1].append(block_score[index]
                                 + current_log_prob[index][aa_id])
          else:
            new_paths[-1].append(block_score[index])

          new_paths[-1].append(current_c_state[index])  # lstm.len_full
          new_paths[-1].append(current_h_state[index])  # lstm.len_full
          # ~ new_paths[-1].append(block_c_state[index]) # nobi
          # ~ new_paths[-1].append(block_h_state[index]) # nobi

      # pick the top BEAM_SIZE
      if len(new_paths) > deepnovo_config.FLAGS.beam_size:
        new_path_scores = np.array([x[2] for x in new_paths])
        # ~ new_path_lengths = np.array([len(x[0])-1 for x in new_paths])
        top_k_indices = np.argpartition(-new_path_scores, deepnovo_config.FLAGS.beam_size)[
                        :deepnovo_config.FLAGS.beam_size]  # pylint: disable=line-too-long
        # ~ top_k_indices = np.argpartition(-new_path_scores/new_path_lengths,deepnovo_config.FLAGS.beam_size)[:deepnovo_config.FLAGS.beam_size] # pylint: disable=line-too-long
        entry[1] = [new_paths[top_k_indices[x]]
                    for x in xrange(deepnovo_config.FLAGS.beam_size)]
      else:
        entry[1] = new_paths[:]

      # update the accumulated block_index
      block_index += entry_block_size[entry_index]

    # update active_search
    #   by removing empty entries
    active_search = [entry for entry in active_search if entry[1]]
    #   and adding new entries
    active_search_len = len(active_search)
    if active_search_len < decode_block_size and spectrum_count < data_set_len:

      new_spectrum_count = min(spectrum_count
                               + decode_block_size
                               - active_search_len,
                               data_set_len)

      for spectrum_id in xrange(spectrum_count, new_spectrum_count):
        active_search.append([])
        active_search[-1].append(spectrum_id)
        active_search[-1].append([[[FIRST_LABEL],  # current_paths
                                   prefix_mass_list[spectrum_id],
                                   0.0,
                                   block_c_state0[spectrum_id],
                                   block_h_state0[spectrum_id]]])

      spectrum_count = new_spectrum_count

    # STOP decoding if no path
    if not active_search:
      break

  # for testing
  test_time_decode += time.time() - start_time_decode

  # for testing
  print("  test_time_tf = %.2f" % (test_time_tf))
  print("  test_time_decode = %.2f" % (test_time_decode))
  print("  test_time = %.2f" % (test_time))

  return output_top_paths


def decode_beam_search_2(sess, model, data_set, knapsack_matrix):
  """TODO(nh2tran): docstring."""

  print("decode_beam_search_2()")

  # ignore x[5] (peptide_ids_forward)
  # ignore x[6] (peptide_ids_backward)
  data_set_forward = [[x[0], x[1], x[2], x[4]] for x in data_set]
  data_set_backward = [[x[0], x[1], x[3], x[4]] for x in data_set]
  data_set_len = len(data_set_forward)
  peptide_mass_list = [x[4] for x in data_set]

  candidate_mass_list = []

  # Pick GO mass
  # GO has only one option: prefix_mass
  mass_GO = deepnovo_config.mass_ID[deepnovo_config.GO_ID]
  prefix_mass_list = [mass_GO] * data_set_len
  suffix_mass_list = [(x - mass_GO) for x in peptide_mass_list]
  candidate_mass_list.append([prefix_mass_list,
                              suffix_mass_list,
                              0.01,  # precursor_mass_precision
                              100])  # knapsack_precision

  # Pick EOS mass
  # EOS has only one option: suffix_mass
  mass_EOS = deepnovo_config.mass_ID[deepnovo_config.EOS_ID]
  prefix_mass_list = [(x - mass_EOS) for x in peptide_mass_list]
  suffix_mass_list = [mass_EOS] * data_set_len
  candidate_mass_list.append([prefix_mass_list,
                              suffix_mass_list,
                              0.01,  # precursor_mass_precision
                              100])  # knapsack_precision

  # Pick a middle mass
  num_position = deepnovo_config.num_position
  argmax_mass_list = []
  argmax_mass_complement_list = []

  # by choosing the location of max intensity from (0, peptide_mass_C_location)
  for spectrum_id in xrange(data_set_len):
    peptide_mass = peptide_mass_list[spectrum_id]
    peptide_mass_C = peptide_mass - mass_EOS
    peptide_mass_C_location = int(round(peptide_mass_C
                                        * deepnovo_config.SPECTRUM_RESOLUTION))
    spectrum_forward = data_set[spectrum_id][2]
    argmax_location = np.argpartition(-spectrum_forward[deepnovo_config.neighbor_size // 2, :peptide_mass_C_location],
                                      num_position)[:num_position]  # pylint: disable=line-too-long
    # !!! LOWER precision 0.1 Da !!!
    argmax_mass = argmax_location / deepnovo_config.SPECTRUM_RESOLUTION

    # Find the closest theoretical mass to argmax_mass
    # ~ aprox_mass = []
    # ~ for x in argmax_mass:
    # ~ y = knapsack_search_mass(knapsack_matrix, x, 1000)
    # ~ if (not y):
    # ~ aprox_mass.append(x)
    # ~ else:
    # ~ aprox_mass.append(y[0])
    # ~ argmax_mass = aprox_mass

    argmax_mass_complement = [(peptide_mass - x) for x in argmax_mass]
    argmax_mass_list.append(argmax_mass)
    argmax_mass_complement_list.append(argmax_mass_complement)

  # Add the mass and its complement to candidate_mass_list
  for position in xrange(num_position):
    prefix_mass_list = [x[position] for x in argmax_mass_list]
    suffix_mass_list = [x[position] for x in argmax_mass_complement_list]
    candidate_mass_list.append([prefix_mass_list,
                                suffix_mass_list,
                                0.1,  # precursor_mass_precision
                                1000])  # knapsack_precision

    prefix_mass_list = [x[position] for x in argmax_mass_complement_list]
    suffix_mass_list = [x[position] for x in argmax_mass_list]
    candidate_mass_list.append([prefix_mass_list,
                                suffix_mass_list,
                                0.1,  # precursor_mass_precision
                                1000])  # knapsack_precision

  # Start decoding for each candidate_mass
  output_top_paths = [[] for x in xrange(data_set_len)]
  for candidate_mass in candidate_mass_list:

    top_paths_forward = decode_beam_search_01(
      sess,
      model,
      knapsack_matrix,
      0,
      candidate_mass[0],  # prefix_mass_list
      candidate_mass[2],  # precursor_mass_precision
      candidate_mass[3],  # knapsack_precision
      data_set_forward)

    top_paths_backward = decode_beam_search_01(
      sess,
      model,
      knapsack_matrix,
      1,
      candidate_mass[1],  # suffix_mass_list
      candidate_mass[2],  # precursor_mass_precision
      candidate_mass[3],  # knapsack_precision
      data_set_backward)

    for spectrum_id in xrange(data_set_len):

      if ((not top_paths_forward[spectrum_id])
          or (not top_paths_backward[spectrum_id])):  # any list is empty
        continue
      else:
        for x_path in top_paths_forward[spectrum_id]:
          for y_path in top_paths_backward[spectrum_id]:
            seq_forward = x_path[0][1:-1]
            seq_backward = y_path[0][1:-1]
            seq_backward = seq_backward[::-1]
            seq = seq_backward + seq_forward
            score = x_path[1] + y_path[1]
            direction = candidate_mass[0][0]
            output_top_paths[spectrum_id].append([seq, score, direction])

  # ~ return output_top_paths

  # Refinement using peptide_mass_list, especially for middle mass
  output_top_paths_refined = [[] for x in xrange(data_set_len)]
  for spectrum_id in xrange(data_set_len):
    top_paths = output_top_paths[spectrum_id]
    for path in top_paths:
      seq = path[0]
      seq_mass = sum(deepnovo_config.mass_ID[x] for x in seq)
      seq_mass += mass_GO + mass_EOS
      if abs(seq_mass - peptide_mass_list[spectrum_id]) <= 0.01:
        output_top_paths_refined[spectrum_id].append(path)

  return output_top_paths_refined


def decode_beam_search(sess,
                       model,
                       data_set,
                       knapsack_matrix,
                       output_file_handle):
  """TODO(nh2tran): docstring."""

  print("decode_beam_search()")

  spectrum_time = time.time()

  scans = [x[0] for x in data_set]

  if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 1:

    output_top_paths = decode_beam_search_01(sess,
                                             model,
                                             knapsack_matrix,
                                             deepnovo_config.FLAGS.direction,
                                             data_set)

    outputs = decode_beam_select_01(output_top_paths,
                                    deepnovo_config.FLAGS.direction)

    spectrum_time = time.time() - spectrum_time

    decoder_inputs = [x[-1] for x in data_set]
    (batch_accuracy_AA,
     batch_len_AA,
     batch_len_decode,
     num_exact_match,
     num_len_match) = test_AA_decode_batch(scans,
                                           decoder_inputs,
                                           outputs,
                                           deepnovo_config.FLAGS.direction,
                                           output_file_handle)
  else:

    output_top_paths = decode_beam_search_2(sess,
                                            model,
                                            data_set,
                                            knapsack_matrix)

    outputs = decode_beam_select_2(output_top_paths)

    spectrum_time = time.time() - spectrum_time

    decoder_inputs_forward = [x[-2] for x in data_set]
    (batch_accuracy_AA,
     batch_len_AA,
     batch_len_decode,
     num_exact_match,
     num_len_match) = test_AA_decode_batch(scans,
                                           decoder_inputs_forward,
                                           outputs,
                                           0,
                                           output_file_handle)

  return (batch_accuracy_AA,
          batch_len_AA,
          batch_len_decode,
          num_exact_match,
          num_len_match,
          spectrum_time)


def decode():
  """TODO(nh2tran): docstring."""

  print("".join(["="] * 80))  # section-separating line
  print("decode()")

  ### input data
  worker_io_decode = deepnovo_worker_io.WorkerIO(
    input_spectrum_file=deepnovo_config.input_spectrum_file_test,
    input_feature_file=deepnovo_config.input_feature_file_test)
  worker_io_decode.open_input()
  worker_io_decode.get_location()
  feature_index_list = worker_io_decode.feature_index_list
  data_set_len = len(feature_index_list)
  print("Total number of spectra = {0:d}".format(data_set_len))

  ### build and load model
  session = tf.Session()
  model = deepnovo_model.ModelInference()
  model.build_model()
  model.restore_model(session)

  ### decode with beam_search
  if deepnovo_config.FLAGS.beam_search:
    print("Decode with beam_search")

    print("Load knapsack_matrix from default: knapsack.npy")
    knapsack_matrix = np.load("knapsack.npy")

    ### split a large list of locations into stacks
    decode_stack_size = deepnovo_config.decode_stack_size
    feature_index_stack_list = [
      feature_index_list[i:i + decode_stack_size]
      for i in range(0, data_set_len, decode_stack_size)]

    ### open output file and print headers
    decode_output_file = deepnovo_config.FLAGS.train_dir + "/decode_output.tab"
    output_file_handle = open(decode_output_file, 'w')
    print("feature_id\ttarget_seq\toutput_seq\toutput_score\taccuracy_AA\tlen_AA"
          "\texact_match\n",
          file=output_file_handle,
          end="")

    ### Read & decode in stacks
    print("Read & decode in stacks")
    total_accuracy_AA = 0.0
    total_len_AA = 0.0
    total_len_decode = 0.0
    total_exact_match = 0.0
    total_len_match = 0.0
    total_spectrum_time = 0.0
    total_peptide_decode = 0.0
    counter_peptide = 0
    for stack in feature_index_stack_list:
      start_time = time.time()
      stack_data_set, _ = read_spectra(worker_io_decode, stack)
      counter_peptide += len(stack)
      print("Read {0:d}/{1:d} spectra, reading time = {2:.2f}".format(
        counter_peptide,
        data_set_len,
        time.time() - start_time))
      # concatenate data buckets
      stack_data_set = sum(stack_data_set, [])
      # decode_beam_search
      (batch_accuracy_AA,
       batch_len_AA,
       batch_len_decode,
       num_exact_match,
       num_len_match,
       spectrum_time) = decode_beam_search(session,
                                           model,
                                           stack_data_set,
                                           knapsack_matrix,
                                           output_file_handle)
      # update accuracy
      total_accuracy_AA += batch_accuracy_AA
      total_len_AA += batch_len_AA
      total_len_decode += batch_len_decode
      total_exact_match += num_exact_match
      total_len_match += num_len_match
      total_spectrum_time += spectrum_time
      total_peptide_decode += len(stack_data_set)

    ### close handlers
    output_file_handle.close()

    print("ACCURACY SUMMARY")
    print("  recall_AA %.4f" % (total_accuracy_AA / total_len_AA))
    print("  precision_AA %.4f" % (total_accuracy_AA / total_len_decode))
    print("  recall_peptide %.4f" % (total_exact_match / total_peptide_decode))
    print("  recall_len %.4f" % (total_len_match / total_peptide_decode))
    print("  spectrum_time %.4f" % (total_spectrum_time / total_peptide_decode))

  ### decode with true feeding
  else:
    print("Decode with true feeding")

    ### read spectra
    print("Read spectra")
    start_time = time.time()
    data_set, _ = read_spectra(worker_io_decode, feature_index_list)
    print("Reading time = {0:.2f}".format(time.time() - start_time))

    ### decode_true_feeding each bucket separately, like in training/validation
    for bucket_id in xrange(len(deepnovo_config._buckets)):
      if not data_set[bucket_id]:  # empty bucket
        continue
      (batch_accuracy_AA,
       num_exact_match,
       num_len_match,
       spectrum_time) = decode_true_feeding(session, model, data_set[bucket_id])
      print("ACCURACY SUMMARY - bucket {0}".format(bucket_id))
      print("  accuracy_AA %.4f" % (batch_accuracy_AA))
      print("  accuracy_peptide %.4f" % (num_exact_match))
      print("  accuracy_len %.4f" % (num_len_match))
      print("  spectrum_time %.4f" % (spectrum_time))

  ### close handlers
  session.close()
  worker_io_decode.close_input()


def train_cycle(model,
                sess,
                worker_io_train,
                feature_index_list_train,
                valid_set,
                valid_set_len,
                valid_bucket_len,
                valid_bucket_pos_id,
                checkpoint_path,
                log_file_handle,
                step_time,
                loss,
                current_step,
                best_valid_perplexity):
  """TODO(nh2tran): docstring."""

  ### for running time profiling
  run_time = 0.0
  read_time = 0.0
  train_time = 0.0
  train_getBatch_time = 0.0
  train_step_time = 0.0
  valid_time = 0.0
  valid_test_time = 0.0
  valid_save_time = 0.0
  run_time_start = time.time()

  ### Read a RANDOM stack from the train file to train_set
  read_time_start = time.time()
  # need to reset feature_count, otherwise it will be accumulated
  worker_io_train.feature_count = dict.fromkeys(worker_io_train.feature_count, 0)
  train_set, _ = read_random_stack(worker_io_train,
                                   feature_index_list_train,
                                   deepnovo_config.train_stack_size)
  read_time = time.time() - read_time_start

  ### for uniform-sample of data from buckets
  # to select a bucket, length of [scale[i], scale[i+1]] is proportional to
  #   the size if i-th training bucket, as used later.
  train_bucket_sizes = [len(train_set[b])
                        for b in xrange(len(deepnovo_config._buckets))]
  train_total_size = float(sum(train_bucket_sizes))
  print("train_bucket_sizes ", train_bucket_sizes)
  print("train_total_size ", train_total_size)
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in xrange(len(train_bucket_sizes))]
  print("train_buckets_scale ", train_buckets_scale)
  # to monitor the number of spectra in the current stack
  # that have been used for training
  train_current_spectra = [0 for b in xrange(len(deepnovo_config._buckets))]

  ### Get a batch and train
  train_bucket_id = 0
  while True:

    start_time = time.time()
    train_time_start = time.time()

    ### Extract a batch from buckets
    # if not enough spectra left in the current bucket, check next bucket
    if (train_current_spectra[train_bucket_id] + deepnovo_config.batch_size
        > train_bucket_sizes[train_bucket_id]):
      train_bucket_id += 1
      # if no bucket left, break, go up, and load a new stack to train_set
      if (train_bucket_id == len(deepnovo_config._buckets) or
          train_bucket_sizes[train_bucket_id] < deepnovo_config.batch_size):
        print("train_current_spectra ", train_current_spectra)
        print("train_bucket_sizes ", train_bucket_sizes)
        break

    ### get a batch from the current bucket
    index_list = range(train_current_spectra[train_bucket_id],
                       train_current_spectra[train_bucket_id] + deepnovo_config.batch_size)
    # get_batch_01/2
    train_getBatch_time_start = time.time()
    if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 1:
      (encoder_inputs,
       intensity_inputs,
       decoder_inputs,
       target_weights) = get_batch_01(index_list, train_set, train_bucket_id)
    else:
      (spectrum_holder,
       ms1_profile,
       intensity_inputs_forward,
       intensity_inputs_backward,
       decoder_inputs_forward,
       decoder_inputs_backward,
       target_weights) = get_batch_2(index_list, train_set, train_bucket_id)
    # monitor the number of spectra that have been processed
    train_current_spectra[train_bucket_id] += deepnovo_config.batch_size
    train_getBatch_time += time.time() - train_getBatch_time_start

    ### make a training step on the data batch
    train_step_time_start = time.time()
    if deepnovo_config.FLAGS.direction == 0:
      step_loss, output_logits = model.step(
        sess,
        encoder_inputs,
        intensity_inputs_forward=intensity_inputs,
        decoder_inputs_forward=decoder_inputs,
        target_weights=target_weights,
        bucket_id=train_bucket_id,
        training_mode=True)
    elif deepnovo_config.FLAGS.direction == 1:
      step_loss, output_logits = model.step(
        sess,
        encoder_inputs,
        intensity_inputs_backward=intensity_inputs,
        decoder_inputs_backward=decoder_inputs,
        target_weights=target_weights,
        bucket_id=train_bucket_id,
        training_mode=True)
    else:
      step_loss, output_logits_forward, output_logits_backward = model.step(
        sess,
        spectrum_holder,
        ms1_profile,
        intensity_inputs_forward=intensity_inputs_forward,
        intensity_inputs_backward=intensity_inputs_backward,
        decoder_inputs_forward=decoder_inputs_forward,
        decoder_inputs_backward=decoder_inputs_backward,
        target_weights=target_weights,
        bucket_id=train_bucket_id,
        training_mode=True)
    train_step_time += time.time() - train_step_time_start

    ### update time, loss
    step_time[0] += (time.time() - start_time) / deepnovo_config.steps_per_checkpoint
    loss[0] += step_loss / deepnovo_config.steps_per_checkpoint
    current_step[0] += 1
    train_time += time.time() - train_time_start

    ### checkpoint: training statistics & evaluation on valid_set
    if current_step[0] % deepnovo_config.steps_per_checkpoint == 0:

      valid_time_start = time.time()

      ### Training statistics for the last round
      #     step_time & loss are averaged over the steps
      #     accuracy is only for the last batch
      perplexity = math.exp(loss[0]) if loss[0] < 300 else float('inf')
      # test_logit_batch_01/2
      if deepnovo_config.FLAGS.direction == 0 or deepnovo_config.FLAGS.direction == 1:
        (batch_accuracy_AA,
         batch_len_AA,
         num_exact_match,
         _) = test_logit_batch_01(decoder_inputs, output_logits)
      else:
        (batch_accuracy_AA,
         batch_len_AA,
         num_exact_match,
         _) = test_logit_batch_2(decoder_inputs_forward,
                                 decoder_inputs_backward,
                                 output_logits_forward,
                                 output_logits_backward)
      # print update of accuracy, time, loss
      accuracy_AA = batch_accuracy_AA / batch_len_AA
      accuracy_peptide = num_exact_match / deepnovo_config.batch_size
      epoch = (model.global_step.eval()
               * deepnovo_config.batch_size
               / len(feature_index_list_train))
      print("global step %d epoch %.1f step-time %.2f perplexity %.4f"
            % (model.global_step.eval(),
               epoch,
               step_time[0],
               perplexity))
      print("%d \t %.1f \t %.2f \t %.4f \t %.4f \t %.4f \t"
            % (model.global_step.eval(),
               epoch,
               step_time[0],
               perplexity,
               accuracy_AA,
               accuracy_peptide),
            file=log_file_handle,
            end="")
      # Zero timer and loss.
      step_time[0], loss[0] = 0.0, 0.0

      ### Evaluation on valid_set.
      # sum up validation loss of buckets
      valid_test_time_start = time.time()
      valid_loss = 0.0
      for bucket_id in valid_bucket_pos_id:
        eval_loss, accuracy_AA, accuracy_peptide, _ = test_accuracy(
          sess,
          model,
          valid_set,
          bucket_id,
          print_summary=False)
        valid_loss += eval_loss * valid_bucket_len[bucket_id]
        eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
        print("%d \t %.4f \t %.4f \t %.4f \t"
              % (bucket_id, eval_ppx, accuracy_AA, accuracy_peptide),
              file=log_file_handle,
              end="")
      valid_test_time += time.time() - valid_test_time_start
      # normalize loss of validation set
      valid_loss /= valid_set_len
      valid_ppx = math.exp(valid_loss) if valid_loss < 300 else float('inf')
      # Save model with best_valid_perplexity
      valid_save_time_start = time.time()
      if valid_ppx < best_valid_perplexity[0]:
        best_valid_perplexity[0] = valid_ppx
        model.saver.save(sess,
                         checkpoint_path,
                         global_step=model.global_step,
                         write_meta_graph=False)
        print("best model: valid_ppx %.4f" % (valid_ppx))
      valid_save_time += time.time() - valid_save_time_start

      valid_time += time.time() - valid_time_start
      print("\n", file=log_file_handle, end="")
      log_file_handle.flush()

  # for testing
  run_time += time.time() - run_time_start
  print("read_time = {0:.2f}".format(read_time))
  print("train_time = {0:.2f}".format(train_time))
  print("   train_getBatch_time = {0:.2f}".format(train_getBatch_time))
  print("   train_step_time = {0:.2f}".format(train_step_time))
  print("valid_time = {0:.2f}".format(valid_time))
  print("   valid_test_time = {0:.2f}".format(valid_test_time))
  print("   valid_save_time = {0:.2f}".format(valid_save_time))
  print("run_time = {0:.2f}".format(run_time))


def train():
  """TODO(nh2tran): docstring."""

  print("".join(["="] * 80))  # section-separating line
  print("train()")

  ### input train and valid data
  worker_io_train = deepnovo_worker_io.WorkerIO(
    input_spectrum_file=deepnovo_config.input_spectrum_file_train,
    input_feature_file=deepnovo_config.input_feature_file_train)
  worker_io_valid = deepnovo_worker_io.WorkerIO(
    input_spectrum_file=deepnovo_config.input_spectrum_file_valid,
    input_feature_file=deepnovo_config.input_feature_file_valid)
  worker_io_train.open_input()
  worker_io_valid.open_input()
  # get location_list for random access
  worker_io_train.get_location()
  worker_io_valid.get_location()
  feature_index_list_train = worker_io_train.feature_index_list
  feature_index_list_valid = worker_io_valid.feature_index_list
  # read random stacks to valid_set
  # ~ print("read random stacks to valid_set")
  valid_set, valid_set_len = read_random_stack(
    worker_io_valid,
    feature_index_list_valid,
    deepnovo_config.valid_stack_size)
  valid_bucket_len = [len(x) for x in valid_set]
  assert valid_set_len == sum(valid_bucket_len), "Error: valid_set_len"
  valid_bucket_pos_id = np.nonzero(valid_bucket_len)[0]

  ### log_file to record perplexity/accuracy during training
  log_file = deepnovo_config.FLAGS.train_dir + "/log_file_caption_2dir.tab"
  print("Open log_file: ", log_file)
  log_file_handle = open(log_file, 'w')
  print("global step \t epoch \t step-time \t"
        "perplexity \t last_accuracy_AA \t last_accuracy_peptide \t"
        "valid_bucket_id \t perplexity \t accuracy_AA \t accuracy_peptide \n",
        file=log_file_handle,
        end="")

  ### TRAINING
  with tf.Session() as sess:

    ### create model with fresh parameters or load them if exist
    print("Create model for training")
    model = create_model(sess, training_mode=True)
    checkpoint_path = os.path.join(deepnovo_config.FLAGS.train_dir, "translate.ckpt")
    print("Model directory: ", checkpoint_path)

    ### LOOP: Read stacks of spectra, Train, Update record
    best_valid_perplexity = [float("inf")]
    loss = [0.0]
    step_time = [0.0]
    current_step = [0]
    print("Training loop")
    while True:
      train_cycle(model,
                  sess,
                  worker_io_train,
                  feature_index_list_train,
                  valid_set,
                  valid_set_len,
                  valid_bucket_len,
                  valid_bucket_pos_id,
                  checkpoint_path,
                  log_file_handle,
                  step_time,
                  loss,
                  current_step,
                  best_valid_perplexity)
      # stop training if >= 50 epochs
      epoch = (model.global_step.eval()
               * deepnovo_config.batch_size
               / len(feature_index_list_train))
      if epoch >= deepnovo_config.epoch_stop:
        print("EPOCH: {0:.1f}, EXCEED {1:d}, STOP TRAINING LOOP".format(
          epoch,
          deepnovo_config.epoch_stop))
        break
      # monitor memory during training
      print("RESOURCE-train_cycle: ",
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)

  ### close handlers
  log_file_handle.close()
  worker_io_train.close_input()
  worker_io_valid.close_input()


def test_true_feeding():
  """TODO(nh2tran): docstring."""

  print("".join(["="] * 80))  # section-separating line
  print("test_true_feeding()")

  ### input data
  worker_io_test = deepnovo_worker_io.WorkerIO(
    input_spectrum_file=deepnovo_config.input_spectrum_file_test,
    input_feature_file=deepnovo_config.input_feature_file_test)
  worker_io_test.open_input()
  # get spectrum locations
  worker_io_test.get_location()
  feature_index_list_test = worker_io_test.feature_index_list
  # read spectrum, pre-process and assign to buckets
  # ~ test_set, _ = read_spectra(worker_io_test, spectrum_index_list_test)
  test_set, _ = read_spectra(worker_io_test, feature_index_list_test[:deepnovo_config.test_stack_size])

  ### create, load, and test model
  session = tf.Session()
  print("Create model for testing")
  model = create_model(session, training_mode=False)
  # export filter weights to study their effects
  # ~ conv1_weight_forward = [session.run(v) for v in tf.global_variables() if v.name == "embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder_forward/conv1_weights:0"]
  # ~ conv1_weight_backward = [session.run(v) for v in tf.global_variables() if v.name == "embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder_backward/conv1_weights:0"]
  # ~ np.save("conv1_weight_forward.npy", conv1_weight_forward[0])
  # ~ np.save("conv1_weight_backward.npy", conv1_weight_backward[0])
  # ~ print(abc)
  for bucket_id in xrange(len(deepnovo_config._buckets)):
    # ~ if valid_set[bucket_id]: # bucket not empty
    # ~ print("valid_set - bucket {0}".format(bucket_id))
    # ~ test_accuracy(session, model, valid_set, bucket_id)
    if test_set[bucket_id]:  # bucket not empty
      print("test_set - bucket {0}".format(bucket_id))
      test_accuracy(session, model, test_set, bucket_id)

  ### close handlers
  session.close()
  worker_io_test.close_input()
