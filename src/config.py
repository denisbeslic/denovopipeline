import numpy as np
from pyteomics import mgf, mass
from collections import OrderedDict

tools_list = [
    "Novor",
    "pNovo",
    "DeepNovo",
    "SMSNet",
    "PointNovo",
]

figure_colors = ['#488f31', '#8aac49', '#c6c96a', '#ffe792', '#f8b267', '#eb7a52', '#de425b', '#ffa600', '#488f31',
                 '#8aac49', '#c6c96a', '#ffe792', '#f8b267', '#eb7a52', '#de425b',
                 '#ffa600']



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
                 'n',
                 'D',
                 'C',
                 # 'C(Carbamidomethylation)',
                 'E',
                 'Q',
                 'q',
                 'G',
                 'H',
                 'I',
                 'L',
                 'K',
                 'M',
                 'm',
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

vocab_reverse_nomods = ['a',
                        'r',
                        'd',
                        'c',
                        'e',
                        'g',
                        'h',
                        'i',
                        'l',
                        'k',
                        'f',
                        'p',
                        's',
                        't',
                        'w',
                        'y',
                        'v',
                        ]

'''novor_config_mod_dict = OrderedDict([('M(0)', 'm'), ('Q(2)', 'q'), ('N(1)', 'n'), (' ', ''), ('C(3)', 'C')])
pepnovo_config_mod_dict = OrderedDict()
smsnet_config_mod_dict = OrderedDict()
deepnovo_config_mod_dict = OrderedDict()
pointnovo_config_mod_dict = OrderedDict()
pnovo_config_mod_dict = OrderedDict()
peptideshaker_congif_mod_dict = OrderedDict()'''

vocab_reverse = _START_VOCAB + vocab_reverse
vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
vocab_size = len(vocab_reverse)

# mass value
mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949
mass_Phosphorylation = 79.96633

mass_AA = {'_PAD': 0.0,
           '_GO': mass_N_terminus - mass_H,
           '_EOS': mass_C_terminus + mass_H,
           'A': 71.03711,  # 0
           'R': 156.10111,  # 1
           'N': 114.04293,  # 2
           'n': 115.02695,
           'D': 115.02694,  # 3
           'C': 160.03065,  # 103.00919,  # 4
           # 'C(Carbamidomethylation)': 160.03065,  # C(+57.02)
           # ~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
           'E': 129.04259,  # 5
           'Q': 128.05858,  # 6
           'q': 129.0426,
           'G': 57.02146,  # 7
           'H': 137.05891,  # 8
           'I': 113.08406,  # 9
           'L': 113.08406,  # 10
           'K': 128.09496,  # 11
           'M': 131.04049,  # 12
           'm': 147.0354,
           'F': 147.06841,  # 13
           'P': 97.05276,  # 14
           'S': 87.03203,  # 15
           # 'S(Phosphorylation)': 87.03203 + mass_Phosphorylation,
           'T': 101.04768,  # 16
           # 'T(Phosphorylation)': 101.04768 + mass_Phosphorylation,
           'W': 186.07931,  # 17
           'Y': 163.06333,  # 18
           # 'Y(Phosphorylation)': 163.06333 + mass_Phosphorylation,
           'V': 99.06841,  # 19
           }

mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]
mass_ID_np = np.array(mass_ID, dtype=np.float32)
mass_AA_min = mass_AA["G"]  # 57.02146


def arePermutation(str1, str2):
    # Get lenghts of both strings
    n1 = len(str1)
    n2 = len(str2)
    # If length of both strings is not same,
    # then they cannot be Permutation
    if (str1 == str2):
        return False
    if (n1 != n2):
        return False
    # Sort both strings
    a = sorted(str1)
    str1 = " ".join(a)
    b = sorted(str2)
    str2 = " ".join(b)
    # Compare sorted strings
    for i in range(0, n1, 1):
        if (str1[i] != str2[i]):
            return False
    return True


# Function from DeepNovo to calculate correct match between two sequences
def _match_AA_novor(target, predicted):
    """TODO(nh2tran): docstring."""

    # ~ print("".join(["="] * 80)) # section-separating line
    # ~ print("WorkerTest._test_AA_match_novor()")
    num_match = 0
    target_len = len(target)
    predicted_len = len(predicted)
    target_mass = [mass_ID[x] for x in target]
    target_mass_cum = np.cumsum(target_mass)
    predicted_mass = [mass_ID[x] for x in predicted]
    predicted_mass_cum = np.cumsum(predicted_mass)

    i = 0
    j = 0
    while i < target_len and j < predicted_len:
        if abs(target_mass_cum[i] - predicted_mass_cum[j]) < 0.5:
            if abs(target_mass[i] - predicted_mass[j]) < 0.1:
                # ~ if  decoder_input[index_aa] == output[index_aa]:
                num_match += 1
            i += 1
            j += 1
        elif target_mass_cum[i] < predicted_mass_cum[j]:
            i += 1
        else:
            j += 1

    return num_match


def norm(x): return np.linalg.norm(x)


def cosine(u, v): return np.dot(u, v) / max(norm(u) * norm(v), 1e-16)


def dotproduct(u, v): return np.dot(u, v)


def pearson(u, v): return np.dot(np.mean(u), np.mean(v)) / max(norm(u) * norm(v), 1e-16)


def dotbias(u, v): return (np.sqrt(np.dot(u ** 2, v ** 2) / max(norm(u) * norm(v), 1e-16))) / (max(cosine(u, v), 1e-16))


def dotbias_dotproduct(u, v): return (np.sqrt(np.dot(u ** 2, v ** 2))) / (max(dotproduct(u, v), 1e-16))


def similarity_scoring(u, v): return cosine(u, v) * (1 - dotbias(u, v))


def similarity_scoring_dot(u, v): return dotproduct(u, v) * (1 - dotbias_dotproduct(u, v))


DIMENSION = 90000
BIN_SIZE = 0.1


def spectrum2vector(mz_list, itensity_list, mass, bin_size, charge):
    itensity_list = itensity_list / np.max(itensity_list)

    vector = np.zeros(DIMENSION, dtype='float32')

    mz_list = np.asarray(mz_list)

    indexes = mz_list / bin_size
    indexes = np.around(indexes).astype('int32')

    for i, index in enumerate(indexes):
        vector[index] += itensity_list[i]

    # normalize
    vector = np.sqrt(vector)

    # remove precursors, including isotropic peaks
    for delta in (0, 1, 2):
        precursor_mz = mass + delta / charge
        if precursor_mz > 0 and precursor_mz < 2000:
            vector[round(precursor_mz / bin_size)] = 0

    return vector


def spectrum2vectorWeighted(mz_list, itensity_list, mass, bin_size, charge):
    a = 1
    b = 1
    itensity_list = itensity_list / np.max(itensity_list)

    vector = np.zeros(DIMENSION, dtype='float32')

    mz_list = np.asarray(mz_list)
    indexes = mz_list / bin_size
    indexes = np.around(indexes).astype('int32')
    # print(itensity_list[1:10], mz_list[1:10], indexes[1:10])

    for i, index in enumerate(indexes):
        # vector[index] += itensity_list[i]*mz_list[i]
        vector[index] += (itensity_list[i] ** a) * (mz_list[i] ** b)

    # normalize
    vector = np.sqrt(vector)

    # remove precursors, including isotropic peaks
    for delta in (0, 1, 2):
        precursor_mz = mass + delta / charge
        if precursor_mz > 0 and precursor_mz < 2000:
            vector[round(precursor_mz / bin_size)] = 0

    return vector


def parse_spectra(sps):
    db = []
    for sp in sps:
        param = sp['params']

        c = int(str(param['charge'][0])[0])

        if 'seq' in param:
            pep = param['seq']
        else:
            pep = param['title']

        if 'pepmass' in param:
            mass = param['pepmass'][0]
        else:
            mass = float(param['parent'])

        if 'hcd' in param:
            try:
                hcd = param['hcd']
                if hcd[-1] == '%':
                    hcd = float(hcd)
                elif hcd[-2:] == 'eV':
                    hcd = float(hcd[:-2])
                    hcd = hcd * 500 * cr[c] / mass
                else:
                    raise Exception("Invalid type!")
            except:
                hcd = 0
        else:
            hcd = 0
        mz = sp['m/z array']
        it = sp['intensity array']

        db.append({'pep': pep, 'charge': c,
                   'mass': mass, 'mz': mz, 'it': it, 'nce': hcd})

    return db


def readmgf(fn):
    file = open(fn, "r")
    data = mgf.read(file, convert_arrays=1, read_charges=False,
                    dtype='float32', use_index=False)
    codes = parse_spectra(data)
    return codes
