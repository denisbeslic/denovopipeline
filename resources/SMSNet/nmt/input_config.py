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

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import lookup_ops

# ==============================================================================
# MODEL HYPERPARAMETERS
# ==============================================================================

bin_step = 0.01 # 0.01
inv_bin_step = 1.0/bin_step
max_posi = 5000
max_spec_length = int(max_posi * inv_bin_step)
max_spec_length_cnn = max_spec_length//10

aa_input_window_size = 0.2 #1.0
# full_aa_window = int(aa_input_window_size * inv_bin_step)
half_aa_window = int(aa_input_window_size * inv_bin_step/2)
full_aa_window = half_aa_window * 2

# knapsack parameters
max_dp_sz = 1500
dp_resolution = 0.0005
inv_dp_resolution = 1.0/dp_resolution
max_dp_array_size = int(round(max_dp_sz * inv_dp_resolution))

# ==============================================================================
# GLOBAL VARIABLES for VOCABULARY
# ==============================================================================


# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _GO, _EOS]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2

vocab_reverse = ['<unk>',
                 '<s>',
                 '</s>',
                 'A',
                 'C',
                 # 'Cmod',
                 'D',
                 'E',
                 'F',
                 'G',
                 'H',
                 'I',
                 'K',
                 'L',
                 'M',
                 'm', # 'Mmod',
                 'N',
                 # 'n', # 'Nmod',
                 'P',
                 'Q',
                 # 'q', # 'Qmod',
                 'R',
                 'S',
                 's',
                 'T',
                 't',
                 'V',
                 'W',
                 'Y',
                 'y',
                ]

# vocab_reverse = _START_VOCAB + vocab_reverse
# print("vocab_reverse ", vocab_reverse) #
vocab_reverse_np = np.array(vocab_reverse)
# vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
# print("vocab ", vocab)

vocab_size_with_eos = len(vocab_reverse)
vocab_size = len(vocab_reverse) - 3
# print("vocab_size ", vocab_size) #


# ==============================================================================
# GLOBAL VARIABLES for THEORETICAL MASS
# ==============================================================================


mass_H = 1.007825
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.007825
mass_C_terminus = 17.00274
mass_CO = 27.9949

mass_AA = {# '_PAD': 0.0,
           # '_GO': mass_N_terminus-mass_H,
           # '</s>': mass_C_terminus+mass_H,
           '<unk>': 0.0,
           '<s>': 0.0,
           '</s>': 0.0,
           'A': 71.03711, # 0
           # 'C': 103.00919, # 4
           'C': 160.03065, # C(+57.02)
           #~ 'Cmod': 161.01919, # C(+58.01) # orbi
           'R': 156.10111, # 1
           'N': 114.04293, # 2
           # 'n': 115.02695, # N mod, N(+.98)
           'D': 115.02694, # 3

           'E': 129.04259, # 5
           'Q': 128.05858, # 6
           # 'q': 129.0426, # Q mod, Q(+.98)
           'G': 57.02146, # 7
           'H': 137.05891, # 8
           'I': 113.08406, # 9
           'L': 113.08406, # 10
           'K': 128.09496, # 11
           'M': 131.04049, # 12 
           'm': 147.0354, # M mod, M(+15.99)
           'F': 147.06841, # 13
           'P': 97.05276, # 14
           'S': 87.03203, # 15
           's': 166.99836, # S mod, S(ph), S + 79.96633
           'T': 101.04768, # 16
           't': 181.01401, # T mod, T(ph), T + 79.96633
           'W': 186.07931, # 17
           'Y': 163.06333, # 18
           'y': 243.02966, # Y mod, y(ph), Y + 79.96633
           'V': 99.06841, # 19
          }

vocab_ID = np.array(list(range(vocab_size_with_eos)), dtype=np.int32)
# print(vocab_ID) #
mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size_with_eos)]
mass_ID_np_with_eos = np.array(mass_ID, dtype=np.float32)
# print(mass_ID_np_with_eos) #
mass_ID_np = mass_ID_np_with_eos[3:]
# print(mass_ID_np) #

mass_AA_min = mass_AA["G"] # 57.02146

def create_aa_tables():
  """Creates amino acid tables."""
#   mass_ID_tf = tf.convert_to_tensor(mass_ID_np_with_eos, dtype=tf.float32)
#   vocab_tf = tf.convert_to_tensor(vocab_ID, dtype=tf.int64)#int32)

#   AA_weight_table = lookup_ops.HashTable(
#     lookup_ops.KeyValueTensorInitializer(vocab_tf, mass_ID_tf), -1)
  with tf.device("/gpu:0"):
    # AA_weight_table = lookup_ops.index_to_string_table_from_file(
    #       "nmt/vocab/mass.txt", default_value="0.0")
    AA_weight_table = tf.constant([0.0, 0.0, 0.0, 71.03711, 160.03065,
                                   115.02694, 129.04259, 147.06841, 57.02146, 137.05891,
                                   113.08406, 128.09496, 113.08406, 131.04049, 147.0354,
                                   114.04293, 97.05276, 128.05858, 156.10111, 87.03203,
                                   166.99836, 101.04768, 181.01401, 99.06841, 186.07931,
                                   163.06333, 243.02966]) # M mod + ph mod

    # AA_weight_table = tf.constant([0.0, 0.0, 0.0, 71.03711, 160.03065,
    #                                115.02694, 129.04259, 147.06841, 57.02146, 137.05891,
    #                                113.08406, 128.09496, 113.08406, 131.04049, 147.0354,
    #                                114.04293, 97.05276, 128.05858, 156.10111, 87.03203,
    #                                101.04768, 99.06841, 186.07931, 163.06333]) # M mod
    
    # AA_weight_table = tf.constant([0.0, 0.0, 0.0, 71.03711, 160.03065,
    #                                115.02694, 129.04259, 147.06841, 57.02146, 137.05891,
    #                                113.08406, 128.09496, 113.08406, 131.04049,
    #                                114.04293, 97.05276, 128.05858, 156.10111, 87.03203,
    #                                101.04768, 99.06841, 186.07931, 163.06333]) # No mod
  return AA_weight_table
