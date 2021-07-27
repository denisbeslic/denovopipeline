# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# ==============================================================================
# FLAGS (options) for this app
# ==============================================================================


tf.app.flags.DEFINE_string("train_dir", # flag_name
                           "train", # default_value
                           "Training directory.") # docstring

tf.app.flags.DEFINE_integer("direction",
                            2,
                            "Set to 0/1/2 for Forward/Backward/Bi-directional.")

tf.app.flags.DEFINE_boolean("use_intensity",
                            True,
                            "Set to True to use intensity-model.")

tf.app.flags.DEFINE_boolean("shared",
                            False,
                            "Set to True to use shared weights.")

tf.app.flags.DEFINE_boolean("use_lstm",
                            True,
                            "Set to True to use lstm-model.")

tf.app.flags.DEFINE_boolean("lstm_kmer",
                            False,
                            "Set to True to use lstm model on k-mers instead of full sequence.")

tf.app.flags.DEFINE_boolean("knapsack_build",
                            False,
                            "Set to True to build knapsack matrix.")

tf.app.flags.DEFINE_boolean("train",
                            False,
                            "Set to True for training.")

tf.app.flags.DEFINE_string("train_spectrum",
                           "train_spectrum",
                           "Spectrum mgf file to train a new model.")

tf.app.flags.DEFINE_string("train_feature",
                           "train_feature",
                           "Feature csv file to train a new model.")

tf.app.flags.DEFINE_string("valid_spectrum",
                           "valid_spectrum",
                           "Spectrum mgf file for validation during training.")

tf.app.flags.DEFINE_string("valid_feature",
                           "valid_feature",
                           "Feature csv file for validation during training.")

tf.app.flags.DEFINE_boolean("test_true_feeding",
                            False,
                            "Set to True for testing.")

tf.app.flags.DEFINE_boolean("decode",
                            False,
                            "Set to True for decoding.")

tf.app.flags.DEFINE_boolean("beam_search",
                            False,
                            "Set to True for beam search.")

tf.app.flags.DEFINE_integer("beam_size",
                            5,
                            "Number of optimal paths to search during decoding.")

tf.app.flags.DEFINE_boolean("search_db",
                            False,
                            "Set to True to do a database search.")

tf.app.flags.DEFINE_boolean("search_denovo",
                            False,
                            "Set to True to do a denovo search.")

tf.app.flags.DEFINE_string("denovo_spectrum",
                           "denovo_spectrum",
                           "Spectrum mgf file to perform de novo sequencing.")

tf.app.flags.DEFINE_string("denovo_feature",
                           "denovo_feature",
                           "Feature csv file to perform de novo sequencing.")

tf.app.flags.DEFINE_boolean("search_hybrid",
                            False,
                            "Set to True to do a hybrid, db+denovo, search.")

tf.app.flags.DEFINE_boolean("test",
                            False,
                            "Set to True to test the prediction accuracy.")

tf.app.flags.DEFINE_string("target_file",
                           "target_file",
                           "Target file to calculate the prediction accuracy.")

tf.app.flags.DEFINE_string("predicted_file",
                           "predicted_file",
                           "Predicted file to calculate the prediction accuracy.")

tf.app.flags.DEFINE_boolean("header_seq",
                            True,
                            "Set to False if peptide sequence is not provided.")

tf.app.flags.DEFINE_boolean("decoy",
                            False,
                            "Set to True to search decoy database.")

tf.app.flags.DEFINE_integer("multiprocessor",
                            1,
                            "Use multi processors to read data during training.")

FLAGS = tf.app.flags.FLAGS


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

vocab_reverse = ['A',
                 'R',
                 'N',
                 'N(Deamidation)',
                 'D',
                 #~ 'C',
                 'C(Carbamidomethylation)',
                 'E',
                 'Q',
                 'Q(Deamidation)',
                 'G',
                 'H',
                 'I',
                 'L',
                 'K',
                 'M',
                 'M(Oxidation)',
                 'F',
                 'P',
                 'S',
                 'T',
                 'W',
                 'Y',
                 'V',
                ]

vocab_reverse = _START_VOCAB + vocab_reverse
print("vocab_reverse ", vocab_reverse)

vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
print("vocab ", vocab)

vocab_size = len(vocab_reverse)
print("vocab_size ", vocab_size)


# ==============================================================================
# GLOBAL VARIABLES for THEORETICAL MASS
# ==============================================================================


mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949

mass_AA = {'_PAD': 0.0,
           '_GO': mass_N_terminus-mass_H,
           '_EOS': mass_C_terminus+mass_H,
           'A': 71.03711, # 0
           'R': 156.10111, # 1
           'N': 114.04293, # 2
           'N(Deamidation)': 115.02695,
           'D': 115.02694, # 3
           #~ 'C(Carbamidomethylation)': 103.00919, # 4
           'C(Carbamidomethylation)': 160.03065, # C(+57.02)
           #~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
           'E': 129.04259, # 5
           'Q': 128.05858, # 6
           'Q(Deamidation)': 129.0426,
           'G': 57.02146, # 7
           'H': 137.05891, # 8
           'I': 113.08406, # 9
           'L': 113.08406, # 10
           'K': 128.09496, # 11
           'M': 131.04049, # 12
           'M(Oxidation)': 147.0354,
           'F': 147.06841, # 13
           'P': 97.05276, # 14
           'S': 87.03203, # 15
           'T': 101.04768, # 16
           'W': 186.07931, # 17
           'Y': 163.06333, # 18
           'V': 99.06841, # 19
          }

mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]
mass_ID_np = np.array(mass_ID, dtype=np.float32)

mass_AA_min = mass_AA["G"] # 57.02146


# ==============================================================================
# GLOBAL VARIABLES for PRECISION, RESOLUTION, temp-Limits of MASS & LEN
# ==============================================================================


# if change, need to re-compile cython_speedup << NO NEED
#~ SPECTRUM_RESOLUTION = 10 # bins for 1.0 Da = precision 0.1 Da
#~ SPECTRUM_RESOLUTION = 20 # bins for 1.0 Da = precision 0.05 Da
#~ SPECTRUM_RESOLUTION = 40 # bins for 1.0 Da = precision 0.025 Da
SPECTRUM_RESOLUTION = 50 # bins for 1.0 Da = precision 0.02 Da
#~ SPECTRUM_RESOLUTION = 100 # bins for 1.0 Da = precision 0.01 Da
print("SPECTRUM_RESOLUTION ", SPECTRUM_RESOLUTION)

# if change, need to re-compile cython_speedup << NO NEED
WINDOW_SIZE = 10 # 10 bins
print("WINDOW_SIZE ", WINDOW_SIZE)

MZ_MAX = 3000.0
MZ_SIZE = int(MZ_MAX * SPECTRUM_RESOLUTION) # 30k

KNAPSACK_AA_RESOLUTION = 10000 # 0.0001 Da
mass_AA_min_round = int(round(mass_AA_min * KNAPSACK_AA_RESOLUTION)) # 57.02146
KNAPSACK_MASS_PRECISION_TOLERANCE = 100 # 0.01 Da
num_position = 0

PRECURSOR_MASS_PRECISION_TOLERANCE = 0.01

# ONLY for accuracy evaluation
AA_MATCH_PRECISION = 0.1

# skip (x > MZ_MAX,MAX_LEN)
MAX_LEN = 50 if FLAGS.decode else 30
print("MAX_LEN ", MAX_LEN)

# We use a number of buckets and pad to the closest one for efficiency.
_buckets = [12,22,32,42,52] if FLAGS.decode else [12, 22, 32]
print("_buckets ", _buckets)


# ==============================================================================
# HYPER-PARAMETERS of the NEURAL NETWORKS
# ==============================================================================


num_ion = 8 # 2
print("num_ion ", num_ion)

l2_weight = 0.0
print("l2_weight ", l2_weight)

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

batch_size = 32
print("batch_size ", batch_size)

epoch_stop = 49 + 10#10 # 50 # 31800*32/(counter_train = 20568)
print("epoch_stop ", epoch_stop)

train_stack_size = 500 # 3000 # 5000
valid_stack_size = 1500#1000 # 3000 # 5000
test_stack_size = 5000
decode_stack_size = 1000 # 3000
print("train_stack_size ", train_stack_size)
print("valid_stack_size ", valid_stack_size)
print("test_stack_size ", test_stack_size)
print("decode_stack_size ", decode_stack_size)

steps_per_checkpoint = 10 # 100 # 2 # 4 # 200
random_test_batches = 10
print("steps_per_checkpoint ", steps_per_checkpoint)
print("random_test_batches ", random_test_batches)

max_gradient_norm = 5.0
print("max_gradient_norm ", max_gradient_norm)

# DIA model parameters
neighbor_size = 5 # allow up to ? spectra, including the main spectrum
dia_window = 20.0 # the window size of MS2 scan in Dalton
focal_loss = True


# ==============================================================================
# DB SEARCH PARAMETERS
# ==============================================================================


data_format = "mgf"
cleavage_rule = "trypsin"
num_missed_cleavage = 2
fixed_mod_list = ['C']
var_mod_list = ['N', 'Q', 'M']
num_mod = 3
precursor_mass_tolerance = 0.01 # Da
precursor_mass_ppm = 15.0/1000000 # ppm (20 better) # instead of absolute 0.01 Da
topk_output = 1


# ==============================================================================
# INPUT/OUTPUT FILES
# ==============================================================================


# pre-built knapsack matrix
knapsack_file = "knapsack.npy"

# train/valid/test files
input_spectrum_file_train = FLAGS.train_spectrum
input_feature_file_train = FLAGS.train_feature
input_spectrum_file_valid = FLAGS.valid_spectrum
input_feature_file_valid = FLAGS.valid_feature
input_spectrum_file_test = "data.training/dia.pecan.plasma.2018_03_29/testing_gs.spectrum.mgf"
input_feature_file_test = "data.training/dia.pecan.plasma.2018_03_29/testing_gs.feature.csv"

# denovo files
denovo_input_spectrum_file = FLAGS.denovo_spectrum
denovo_input_feature_file = FLAGS.denovo_feature
denovo_output_file = denovo_input_feature_file + ".deepnovo_denovo"

# db files
#~ db_fasta_file = "data/uniprot_sprot.human.db_decoy.fasta"
#~ db_input_spectrum_file = "data.training/dia.pecan.hela.2018_03_29/testing.spectrum.mgf"
#~ db_input_feature_file = "data.training/dia.abrf.2018_03_27/testing.feature.csv.2k"
#~ db_output_file = db_input_feature_file + ".deepnovo_db"
#~ if FLAGS.decoy:
  #~ db_output_file += ".decoy"

# hybrid files
#~ hybrid_fasta_file = "data/uniprot_sprot.human.db_decoy.fasta"
#~ hybrid_input_spectrum_file = "data.training/dia.abrf.2018_03_27/prediction.spectrum.mgf"
#~ hybrid_input_feature_file = "data.training/dia.abrf.2018_03_27/prediction.feature.csv.part1"
#~ hybrid_denovo_file = hybrid_input_feature_file + ".deepnovo_hybrid_denovo"
#~ hybrid_output_file = hybrid_input_feature_file + ".deepnovo_hybrid"
#~ if FLAGS.decoy:
  #~ hybrid_output_file += ".decoy"

# test accuracy files
predicted_format = "deepnovo"
target_file = FLAGS.target_file
predicted_file = FLAGS.predicted_file
# ~ predicted_file = "data.training/dia.pecan.plasma.2018_03_29/testing_plasma.unlabeled.csv.deepnovo_denovo.top90"
#~ predicted_format = "peaks"
#~ target_file = "data.training/dia.urine.2018_04_23/testing_gs.feature.csv"
#~ predicted_file = "data.training/dia.urine.2018_04_23/peaks.denovo.csv.uti"
accuracy_file = predicted_file + ".accuracy"
denovo_only_file = predicted_file + ".denovo_only"
scan2fea_file = predicted_file + ".scan2fea"
multifea_file = predicted_file + ".multifea"


# ==============================================================================
# ==============================================================================


# feature file column format
col_feature_id = 0
col_precursor_mz = 1
col_precursor_charge = 2
col_rt_mean = 3
col_raw_sequence = 4
col_scan_list = 5
col_ms1_list = 6
col_feature_area = 7
col_num = 8
# predicted file column format
pcol_feature_id = 0
pcol_feature_area = 1
pcol_sequence = 2
pcol_score = 3
pcol_position_score = 4
pcol_precursor_mz = 5
pcol_precursor_charge = 6
pcol_protein_id = 7
pcol_scan_list_middle = 8
pcol_scan_list_original = 9
pcol_score_max = 10

