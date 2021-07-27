from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

import numpy as np

import tensorflow as tf












########################################################################
# FLAGS (options) for this app
########################################################################

tf.app.flags.DEFINE_string("train_dir", "train", "Training directory.")

tf.app.flags.DEFINE_integer("direction", 2, "Set to 0/1/2 for Forward/Backward/Bi-directional.")

tf.app.flags.DEFINE_boolean("use_intensity", True, "Set to True to use intensity-model.")

tf.app.flags.DEFINE_boolean("shared", False, "Set to True to use shared weights.")

tf.app.flags.DEFINE_boolean("use_lstm", True, "Set to True to use lstm-model.")

tf.app.flags.DEFINE_boolean("knapsack_build", False, "Set to True to build knapsack matrix.")

tf.app.flags.DEFINE_boolean("train", False, "Set to True for training.")

tf.app.flags.DEFINE_boolean("test_true_feeding", False, "Set to True for testing.")

tf.app.flags.DEFINE_boolean("decode", False, "Set to True for decoding.")

tf.app.flags.DEFINE_boolean("beam_search", False, "Set to True for beam search.")

tf.app.flags.DEFINE_integer("beam_size", 1, "Number of optimal paths to search during decoding.")

FLAGS = tf.app.flags.FLAGS
########################################################################











########################################################################
# VOCABULARY 
########################################################################

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _GO, _EOS]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2

vocab_reverse = ['A',
         'R',
         'N',
         'Nmod',
         'D',
         #~ 'C',
         'Cmod',
         'E',
         'Q',
         'Qmod',
         'G',
         'H',
         'I',
         'L',
         'K',
         'M',
         'Mmod',
         'F',
         'P',
         'S',
         'T',
         'W',
         'Y',
         'V',
        ]
#
vocab_reverse = _START_VOCAB + vocab_reverse
print("vocab_reverse ", vocab_reverse)
#
vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
print("vocab ", vocab)
#
vocab_size = len(vocab_reverse)
print("vocab_size ", vocab_size)





########################################################################
# MASS
########################################################################

mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949

mass_AA = {'_PAD':0.0,
         '_GO':mass_N_terminus-mass_H,
         '_EOS':mass_C_terminus+mass_H,
         'A':71.03711, # 0
         'R':156.10111, # 1
         'N':114.04293, # 2
         'Nmod':115.02695,
         'D':115.02694, # 3
         #~ 'C':103.00919, # 4
         'Cmod':160.03065, # C(+57.02)
         #~ 'Cmod':161.01919, # C(+58.01) # orbi
         'E':129.04259, # 5
         'Q':128.05858, # 6
         'Qmod':129.0426,
         'G':57.02146, # 7
         'H':137.05891, # 8
         'I':113.08406, # 9
         'L':113.08406, # 10
         'K':128.09496, # 11
         'M':131.04049, # 12
         'Mmod':147.0354,
         'F':147.06841, # 13
         'P':97.05276, # 14
         'S':87.03203, # 15
         'T':101.04768, # 16
         'W':186.07931, # 17
         'Y':163.06333, # 18
         'V':99.06841, # 19
        }

mass_ID = [mass_AA[vocab_reverse[x]] for x in xrange(vocab_size)]
mass_ID_np = np.array(mass_ID, dtype=np.float32)

mass_AA_min = mass_AA["G"] # 57.02146






########################################################################
# PRECISION & RESOLUTION & temp-Limits of MASS & LEN
########################################################################

# if change, need to re-compile cython_speedup
#~ SPECTRUM_RESOLUTION = 10 # bins for 1.0 Da = precision 0.1 Da
#~ SPECTRUM_RESOLUTION = 20 # bins for 1.0 Da = precision 0.05 Da
#~ SPECTRUM_RESOLUTION = 40 # bins for 1.0 Da = precision 0.025 Da
SPECTRUM_RESOLUTION = 50 # bins for 1.0 Da = precision 0.02 Da
#~ SPECTRUM_RESOLUTION = 80 # bins for 1.0 Da = precision 0.0125 Da
print("SPECTRUM_RESOLUTION ", SPECTRUM_RESOLUTION)
WINDOW_SIZE = 10 # bins
print("WINDOW_SIZE ", WINDOW_SIZE)

# if change, need to re-compile cython_speedup
MZ_MAX = 4500.0
MZ_SIZE = int(MZ_MAX * SPECTRUM_RESOLUTION) # 30k

KNAPSACK_AA_RESOLUTION = 10000 # 0.0001 Da
mass_AA_min_round = int(round(mass_AA_min * KNAPSACK_AA_RESOLUTION)) # 57.0215 # 57.02146
KNAPSACK_MASS_PRECISION_TOLERANCE = 100 # 0.01 Da
num_position = 0

PRECURSOR_MASS_PRECISION_TOLERANCE = 0.01

# ONLY for accuracy evaluation
#~ PRECURSOR_MASS_PRECISION_INPUT_FILTER = 0.01
PRECURSOR_MASS_PRECISION_INPUT_FILTER = 1000
AA_MATCH_PRECISION = 0.1

# skip (x > MZ_MAX,MAX_LEN) ~ 7% of data; also skip not enough spectra, N-terminal mod
MAX_LEN = 30
if (FLAGS.decode): # for decode 
  MAX_LEN = 50
print("MAX_LEN ", MAX_LEN)

# We use a number of buckets and pad to the closest one for efficiency.
_buckets = [12,22,32] 
#~ _buckets = [12,22,32,42,52] # for decode
print("_buckets ", _buckets)






########################################################################
# TRAINING PARAMETERS
########################################################################

num_ion = 8 # 2
print("num_ion ", num_ion)

l2_loss_weight = 0.0 # 0.0
print("l2_loss_weight ", l2_loss_weight)

#~ encoding_cnn_size = 4 * (RESOLUTION//10) # 4 # proportion to RESOLUTION
#~ encoding_cnn_filter = 4
#~ print("encoding_cnn_size ", encoding_cnn_size)
#~ print("encoding_cnn_filter ", encoding_cnn_filter)

embedding_size = 512
print("embedding_size ", embedding_size)

num_layers = 1
num_units = 512
print("num_layers ", num_layers)
print("num_units ", num_units)

keep_conv = 0.75
keep_dense = 0.5
print("keep_conv ", keep_conv)
print("keep_dense ", keep_dense)

batch_size = 128
print("batch_size ", batch_size)

epoch_stop = 5 #20 # 50
print("epoch_stop ", epoch_stop)

train_stack_size = 4500
valid_stack_size = 10000 # 10%
test_stack_size = 5000
print("train_stack_size ", train_stack_size)
print("valid_stack_size ", valid_stack_size)
print("test_stack_size ", test_stack_size)

steps_per_checkpoint = 100 # 20 # 100 # 2 # 4 # 200
random_test_batches = 10
print("steps_per_checkpoint ", steps_per_checkpoint)
print("random_test_batches ", random_test_batches)

max_gradient_norm = 5.0
print("max_gradient_norm ", max_gradient_norm)






########################################################################
# DATASETS
########################################################################

# YEAST-FULL-PEAKS-DB-FRAC_123-DUP
#~ data_format = "mgf"
#~ input_file_train = "data/yeast.full/peaks.db.frac_123.mgf.train.dup"
#~ input_file_valid = "data/yeast.full/peaks.db.frac_123.mgf.valid.dup"
#~ input_file_test = "data/yeast.full/peaks.db.frac_123.mgf.test.dup"

# YEAST-FULL-PEAKS-DB-FRAC_123-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/yeast.full/peaks.db.frac_123.mgf.train.repeat"
#~ input_file_valid = "data/yeast.full/peaks.db.frac_123.mgf.valid.repeat"
#~ input_file_test = "data/yeast.full/peaks.db.frac_123.mgf.test.repeat"

#~ decode_test_file = "data/yeast.full/peaks.db.frac_123.mgf.test.dup"
#~ decode_test_file = "data/yeast.full/peaks.db.frac_123.mgf.test.repeat"






# MIX-7SPECIES_50k.EXCLUDE_YEAST-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/mix.7species_50k.exclude_yeast/mix.cat.mgf.train.repeat"
#~ input_file_valid = "data/mix.7species_50k.exclude_yeast/mix.cat.mgf.valid.repeat"
#~ input_file_test = "data/mix.7species_50k.exclude_yeast/mix.cat.mgf.test.repeat"






# CROSS-7SPECIES_50k.EXCLUDE_ARABIDOPSI-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.7species_50k.exclude_arabidopsis/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.7species_50k.exclude_arabidopsis/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.7species_50k.exclude_arabidopsis/cross.cat.mgf.test.repeat"

# CROSS-7SPECIES_50k.EXCLUDE_CELEGANS-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.7species_50k.exclude_celegans/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.7species_50k.exclude_celegans/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.7species_50k.exclude_celegans/cross.cat.mgf.test.repeat"

# CROSS-7SPECIES_50k.EXCLUDE_ECOLI-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.7species_50k.exclude_ecoli/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.7species_50k.exclude_ecoli/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.7species_50k.exclude_ecoli/cross.cat.mgf.test.repeat"

# CROSS-7SPECIES_50k.EXCLUDE_FRUITFLY-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.7species_50k.exclude_fruitfly/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.7species_50k.exclude_fruitfly/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.7species_50k.exclude_fruitfly/cross.cat.mgf.test.repeat"

# CROSS-7SPECIES_50k.EXCLUDE_HUMAN-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.7species_50k.exclude_human/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.7species_50k.exclude_human/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.7species_50k.exclude_human/cross.cat.mgf.test.repeat"

# CROSS-7SPECIES_50k.EXCLUDE_MOUSE-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.7species_50k.exclude_mouse/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.7species_50k.exclude_mouse/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.7species_50k.exclude_mouse/cross.cat.mgf.test.repeat"

# CROSS-7SPECIES_50k.EXCLUDE_PSEUDOMONAS-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.7species_50k.exclude_pseudomonas/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.7species_50k.exclude_pseudomonas/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.7species_50k.exclude_pseudomonas/cross.cat.mgf.test.repeat"

# CROSS-7SPECIES_50k.EXCLUDE_YEAST-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/cross.7species_50k.exclude_yeast/cross.cat.mgf.train.repeat"
#~ input_file_valid = "data/cross.7species_50k.exclude_yeast/cross.cat.mgf.valid.repeat"
#~ input_file_test = "data/cross.7species_50k.exclude_yeast/cross.cat.mgf.test.repeat"

# YEAST-FULL-PEAKS-DB-10k
#~ input_file_test = "data/yeast.full/peaks.db.10k.mgf"
#~ decode_test_file = "data/yeast.full/peaks.db.10k.mgf"






# AB-TRAINING-REPEAT
#~ data_format = "mgf"
#~ input_file_train = "data/ab.training/peaks.db.mgf.train.repeat"
#~ input_file_valid = "data/ab.training/peaks.db.mgf.valid.repeat"
#~ input_file_test = "data/ab.training/.peaks.db.mgf.test.repeat"

# AB-TRAINING-MOUSE-REPEAT
data_format = "mgf"
input_file_train = "/home/dbeslic/master/DeepLearning_TrainingData/03_NIST_HCD/NIST_transformed_train.mgf"
input_file_valid = "/home/dbeslic/master/DeepLearning_TrainingData/03_NIST_HCD/NIST_transformed_valid.mgf"
input_file_test = "/home/dbeslic/master/DeepLearning_TrainingData/03_NIST_HCD/NIST_transformed_test.mgf"

# Assem-TESTING-MOUSE-WATERS-LIGHT
#~ input_file_test = "data/ab.testing/assem.waters.mouse.light/peaks.refine.mgf"
#~ decode_test_file = "data/ab.testing/assem.waters.mouse.light/peaks.refine.mgf"

# Assem-TESTING-PUBLIC-MOUSE-LIGHT
#~ input_file_test = "data/ab.testing/assem.public.mouse.light/peaks.refine.mgf"
#~ decode_test_file = "data/ab.testing/assem.public.mouse.light/peaks.refine.mgf"

# Assem-TESTING-PUBLIC-HUMAN-LIGHT
#~ input_file_test = "data/ab.testing/assem.public.human.light/peaks.refine.mgf"
#~ decode_test_file = "data/ab.testing/assem.public.human.light/peaks.refine.mgf"

# Assem-TESTING-MOUSE-WATERS-HEAVY
#~ input_file_test = "data/ab.testing/assem.public.mouse.waters.heavy/peaks.refine.mgf"
#~ decode_test_file = "data/ab.testing/assem.public.mouse.waters.heavy/peaks.refine.mgf"

# Assem-TESTING-PUBLIC-MOUSE-HEAVY
#~ input_file_test = "data/ab.testing/assem.public.mouse.waters.heavy/peaks.refine.mgf"
#~ decode_test_file = "data/ab.testing/assem.public.mouse.waters.heavy/peaks.refine.mgf"

# Assem-TESTING-PUBLIC-HUMAN-HEAVY
#~ input_file_test = "data/ab.testing/assem.public.human.heavy/peaks.refine.mgf"
#~ decode_test_file = "data/ab.testing/assem.public.human.heavy/peaks.refine.mgf"

# Assem-TESTING-PUBLIC-MOUSE-HEAVY
input_file_test = "/home/dbeslic/master/antibody-de-novo-sequencing/example_dataset/01-raw-data/waters.mouse.light_reformatted_deepnovo.mgf"
decode_test_file = "/home/dbeslic/master/antibody-de-novo-sequencing/example_dataset/01-raw-data/waters.mouse.light_reformatted_deepnovo.mgf"

# AB-TESTING-MOUSE-WATERS-HEAVY
#~ input_file_test = "data/ab.testing/ab.mouse.waters.heavy/peaks.db.mgf"
#~ decode_test_file = "data/ab.testing/ab.mouse.waters.heavy/peaks.db.mgf"

# AB-TESTING-HUMAN-BOB
#~ input_file_test = "data/ab.testing/ab.human.bob/peaks.db.mgf"
#~ decode_test_file = "data/ab.testing/ab.human.bob/peaks.db.mgf"




    
  

# DATA-test

#~ decode_test_file = "data/arabidopsis.PXD004742/peaks.db.10k.mgf"
#~ decode_test_file = "data/celegans.PXD000636/peaks.db.10k.mgf"
#~ decode_test_file = "data/ecoli.PXD002912/peaks.db.10k.mgf"
#~ decode_test_file = "data/fruitfly.PXD004120/peaks.db.10k.mgf"
#~ decode_test_file = "data/human.PXD002179.sds/peaks.db.10k.mgf"
#~ decode_test_file = "data/mouse.PXD002247/peaks.db.10k.mgf"
#~ decode_test_file = "data/yeast.full/peaks.db.10k.mgf"

#~ decode_test_file = "data/arabidopsis.PXD004742/peaks.db.mgf"
#~ decode_test_file = "data/celegans.PXD000636/peaks.db.mgf"
#~ decode_test_file = "data/ecoli.PXD002912/peaks.db.mgf"
#~ decode_test_file = "data/fruitfly.PXD004120/peaks.db.mgf"
#~ decode_test_file = "data/human.PXD002179.sds/peaks.db.mgf"
#~ decode_test_file = "data/mouse.PXD002247/peaks.db.mgf"
#~ decode_test_file = "data/pseudomonas.PXD004560/peaks.db.mgf"
#~ decode_test_file = "data/yeast.full/peaks.db.mgf"
########################################################################
























