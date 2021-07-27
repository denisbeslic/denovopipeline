# PointNovo is publicly available for non-commercial uses.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
from itertools import combinations
# ==============================================================================
# FLAGS (options) for this app
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, default="train_NIST_HCD_5epochs")
parser.add_argument("--beam_size", type=int, default="5")
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--search_denovo", dest="search_denovo", action="store_true")
parser.add_argument("--search_db", dest="search_db", action="store_true")
parser.add_argument("--valid", dest="valid", action="store_true")
parser.add_argument("--test", dest="test", action="store_true")

parser.set_defaults(train=False)
parser.set_defaults(search_denovo=False)
parser.set_defaults(search_db=False)
parser.set_defaults(valid=False)
parser.set_defaults(test=False)

args = parser.parse_args()

FLAGS = args
train_dir = FLAGS.train_dir
use_lstm = False

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
assert PAD_ID == 0




vocab_reverse = ['A',
                 'R',
                 'N',
                 'N(Deamidation)',
                 'D',
                 # 'C',
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
                 # 'S(Phosphorylation)',
                 'T',
                 # 'T(Phosphorylation)',
                 'W',
                 'Y',
                 # 'Y(Phosphorylation)',
                 'V',
                 ]

vocab_reverse = _START_VOCAB + vocab_reverse
print("Training vocab_reverse ", vocab_reverse)

vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
print("Training vocab ", vocab)

vocab_size = len(vocab_reverse)
print("Training vocab_size ", vocab_size)

# database search parameter
## the PTMs to be included in the database search
fix_mod_dict = {"C": "C(Carbamidomethylation)"}
# var_mod_dict = {"N": "N(Deamidation)", 'Q': 'Q(Deamidation)', 'M': 'M(Oxidation)'}
var_mod_dict = {'M': 'M(Oxidation)'}
max_num_mod = 3
db_ppm_tolenrance = 20.
semi_cleavage = False

normalizing_std_n = 150
normalizing_mean_n = 10

inference_value_max_batch_size = 20
num_psm_per_scan_for_percolator = 10
db_fasta_file = "fasta_files/uniprot_sprot_human_with_decoy.fasta"
num_db_searcher_worker = 8
fragment_ion_mz_diff_threshold = 0.02
quick_scorer = "num_matched_ions"


def _fix_transform(aa: str):
    def trans(peptide: list):
        return [x if x != aa else fix_mod_dict[x] for x in peptide]
    return trans


def fix_mod_peptide_transform(peptide: list):
    """
    apply fix modification transform on a peptide
    :param peptide:
    :return:
    """
    for aa in fix_mod_dict.keys():
        trans = _fix_transform(aa)
        peptide = trans(peptide)
    return peptide


def _find_all_ptm(peptide, position_list):
    if len(position_list) == 0:
        return [peptide]
    position = position_list[0]
    aa = peptide[position]
    result = []
    temp = peptide[:]
    temp[position] = var_mod_dict[aa]
    result += _find_all_ptm(temp, position_list[1:])
    return result


def var_mod_peptide_transform(peptide: list):
    """
    apply var modification transform on a peptide, the max number of var mod is max_num_mod
    :param peptide:
    :return:
    """
    position_list = [position for position, aa in enumerate(peptide) if aa in var_mod_dict]
    position_count = len(position_list)
    num_mod = min(position_count, max_num_mod)
    position_combination_list = []
    for x in range(1, num_mod+1):
        position_combination_list += combinations(position_list, x)
    # find all ptm peptides
    ptm_peptide_list = []
    for position_combination in position_combination_list:
        ptm_peptide_list += _find_all_ptm(peptide, position_combination)
    return ptm_peptide_list


# mass value
mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949
mass_Phosphorylation = 79.96633

# mass_AA should be comprehensive, including the mass for all common ptm
mass_AA = {'_PAD': 0.0,
           '_GO': mass_N_terminus - mass_H,
           '_EOS': mass_C_terminus + mass_H,
           'A': 71.03711,  # 0
           'R': 156.10111,  # 1
           'N': 114.04293,  # 2
           'N(Deamidation)': 115.02695,
           'D': 115.02694,  # 3
           'C': 103.00919,  # 4
           'C(Carbamidomethylation)': 160.03065,  # C(+57.02)
           # ~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
           'E': 129.04259,  # 5
           'Q': 128.05858,  # 6
           'Q(Deamidation)': 129.0426,
           'G': 57.02146,  # 7
           'H': 137.05891,  # 8
           'I': 113.08406,  # 9
           'L': 113.08406,  # 10
           'K': 128.09496,  # 11
           'M': 131.04049,  # 12
           'M(Oxidation)': 147.0354,
           'F': 147.06841,  # 13
           'P': 97.05276,  # 14
           'S': 87.03203,  # 15
           'S(Phosphorylation)': 87.03203 + mass_Phosphorylation,
           'T': 101.04768,  # 16
           'T(Phosphorylation)': 101.04768 + mass_Phosphorylation,
           'W': 186.07931,  # 17
           'Y': 163.06333,  # 18
           'Y(Phosphorylation)': 163.06333 + mass_Phosphorylation,
           'V': 99.06841,  # 19
           }

mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]
mass_ID_np = np.array(mass_ID, dtype=np.float32)

mass_AA_min = mass_AA["G"]  # 57.02146

# ==============================================================================
# GLOBAL VARIABLES for PRECISION, RESOLUTION, temp-Limits of MASS & LEN
# ==============================================================================


MZ_MAX = 5000.0 if FLAGS.search_db else 3000.0

MAX_NUM_PEAK = 1000

KNAPSACK_AA_RESOLUTION = 10000  # 0.0001 Da
mass_AA_min_round = int(round(mass_AA_min * KNAPSACK_AA_RESOLUTION))  # 57.02146
KNAPSACK_MASS_PRECISION_TOLERANCE = 100  # 0.01 Da
num_position = 0

PRECURSOR_MASS_PRECISION_TOLERANCE = 0.01

# ONLY for accuracy evaluation
# ~ PRECURSOR_MASS_PRECISION_INPUT_FILTER = 0.01
# ~ PRECURSOR_MASS_PRECISION_INPUT_FILTER = 1000
AA_MATCH_PRECISION = 0.1

# skip (x > MZ_MAX,MAX_LEN)
MAX_LEN = 60 if FLAGS.search_denovo or FLAGS.search_db else 30
print("MAX_LEN ", MAX_LEN)

# ==============================================================================
# HYPER-PARAMETERS of the NEURAL NETWORKS
# ==============================================================================

num_ion = 12
print("num_ion ", num_ion)

weight_decay = 0.0  # no weight decay lead to better result.
print("weight_decay ", weight_decay)

# ~ encoding_cnn_size = 4 * (RESOLUTION//10) # 4 # proportion to RESOLUTION
# ~ encoding_cnn_filter = 4
# ~ print("encoding_cnn_size ", encoding_cnn_size)
# ~ print("encoding_cnn_filter ", encoding_cnn_filter)

embedding_size = 512
print("embedding_size ", embedding_size)

num_lstm_layers = 1
num_units = 64
lstm_hidden_units = 512
print("num_lstm_layers ", num_lstm_layers)
print("num_units ", num_units)

dropout_rate = 0.25

batch_size = 16
num_workers = 6
print("batch_size ", batch_size)

num_epoch = 5

init_lr = 1e-3

steps_per_validation = 300  # 100 # 2 # 4 # 200
print("steps_per_validation ", steps_per_validation)

max_gradient_norm = 5.0
print("max_gradient_norm ", max_gradient_norm)

# ==============================================================================
# DATASETS
# ==============================================================================


data_format = "mgf"
cleavage_rule = "trypsin"
num_missed_cleavage = 2
knapsack_file = "knapsack.npy"

input_spectrum_file_train = "/home/dbeslic/master/DeepLearning_TrainingData/03_NIST_HCD/spectrum.mgf"
input_feature_file_train = "/home/dbeslic/master/DeepLearning_TrainingData/03_NIST_HCD/features.train.csv"
input_spectrum_file_valid = "/home/dbeslic/master/DeepLearning_TrainingData/03_NIST_HCD/spectrum.mgf"
input_feature_file_valid = "/home/dbeslic/master/DeepLearning_TrainingData/03_NIST_HCD/features.valid.csv"
input_spectrum_file_test = "/home/dbeslic/master/DeepLearning_TrainingData/03_NIST_HCD/spectrum.mgf"
input_feature_file_test = "/home/dbeslic/master/DeepLearning_TrainingData/03_NIST_HCD/features.test.csv"
# denovo files
denovo_input_spectrum_file = "/home/dbeslic/master/antibody-de-novo-sequencing/example_dataset/01-raw-data/WIgG1-Light-AspN_reformatted.mgf"
denovo_input_feature_file  = "/home/dbeslic/master/antibody-de-novo-sequencing/example_dataset/01-raw-data/features.csv"
denovo_output_file = "/home/dbeslic/master/antibody-de-novo-sequencing/example_dataset/04-results_modsQMN_MassiveTrainedModels/IgG1_Waters_Mouse/LC/MouseLC_AspN/PointNovo/features.csv.deepnovo_denovo"

# db search files
search_db_input_spectrum_file = "Lumos_data/PXD008999/export_0.mgf"
search_db_input_feature_file = "Lumos_data/PXD008999/export_0.csv"
db_output_file = search_db_input_feature_file + '.pin'

# test accuracy
predicted_format = "deepnovo"
target_file = denovo_input_feature_file
predicted_file = denovo_output_file

accuracy_file = predicted_file + ".accuracy"
denovo_only_file = predicted_file + ".denovo_only"
scan2fea_file = predicted_file + ".scan2fea"
multifea_file = predicted_file + ".multifea"
# ==============================================================================
# feature file column format
col_feature_id = "spec_group_id"
col_precursor_mz = "m/z"
col_precursor_charge = "z"
col_rt_mean = "rt_mean"
col_raw_sequence = "seq"
col_scan_list = "scans"
col_feature_area = "feature area"

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

distance_scale_factor = 100.
sinusoid_base = 30000.
spectrum_reso = 10
n_position = int(MZ_MAX) * spectrum_reso
