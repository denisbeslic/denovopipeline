import numpy as np
from pyteomics import mgf, mass
from collections import OrderedDict

mass_tol = 0.5 # in Da

tools_list = [
    "Novor",
    "pNovo",
    "DeepNovo",
    "SMSNet",
    "PointNovo",
]

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
                 'T',
                 'W',
                 'Y',
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
           'T': 101.04768,  # 16
           'W': 186.07931,  # 17
           'Y': 163.06333,  # 18
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

DIMENSION = 90000
BIN_SIZE = 0.1

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
