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

### CONSTANTS
aa_list = 'GASPVTCLINDQKEMHFRYWmnq'
aa_mass_list = [57.02146,71.03711,87.03203,
                97.05276,99.06841,101.04768,
                103.00919 + 57.02146,113.08406,113.08406, ## C are C mod
                114.04293,115.02694,128.05858,
                128.09496,129.04259,131.04049,
                137.05891,147.06841,156.10111,
                163.06333,186.07931,131.04049 + 15.99491, ## m = M(ox)
                114.04293 + 0.98402,128.05858 + 0.98402] ## n = N(de), q = Q(de)

aa_mass = {}
for i in range(len(aa_list)):
    aa_mass[aa_list[i]] = aa_mass_list[i]

proton = 1.007276
water = 18.010565
ammonia = 17.026549
aion = 27.994915
mass_tol = 20 * 0.000001 # ppm

############################################
### FUNCTION FOR COMPUTING EVIDENCE
### MAJOR = +1, +2
### MINOR = -H2O, -NH3, -CO
def find_evidence(seq, spectrum, tol):
    evidence_major = [0] * len(seq)
    evidence_minor = [0] * len(seq)
    
    ladder = seq_to_ladder(seq)
    ion_major = [0] * len(ladder)
    ion_minor = [0] * len(ladder)
    
    for pos in range(len(ladder)):
        found_major, found_minor = find_evidence_helper(ladder, pos, spectrum, tol)
        
        if found_major:
            ion_major[pos] = 1
            ion_minor[pos] = 1
        elif found_minor:
            ion_minor[pos] = 1
    
    for i in range(len(seq)):
        if ion_major[i] == 1 and ion_major[i + 1] == 1:
            evidence_major[i] = 1
        if ion_minor[i] == 1 and ion_minor[i + 1] == 1:
            evidence_minor[i] = 1
    
    return evidence_major, evidence_minor

def find_evidence_helper(ladder, pos, spectrum, tol): # look for evidence for a ladder position in a spectrum
    if pos == 0 or pos == len(ladder) - 1:
        return True, True # always return True for 0 and total mass
    
    found_major = False
    found_minor = False
    
    major_masses, minor_masses = get_fragment_ion_masses(ladder[pos], ladder[-1])
    
    for m in major_masses:
        matched_loc, matched_error = mass_matching_init(m, spectrum[:, 0], tol)
        
        if len(matched_loc) > 0:
            found_major = True
            break
    
    if not found_major:
        for m in minor_masses:
            matched_loc, matched_error = mass_matching_init(m, spectrum[:, 0], tol)
        
            if len(matched_loc) > 0:
                found_minor = True
                break
    
    return found_major, found_minor

def seq_to_ladder(seq): # convert amino acid sequence to mass ladder
    ladder = [0] * (len(seq) + 1)
    ladder[1] = aa_mass[seq[0]]

    for i in range(2, len(ladder)):
        ladder[i] = ladder[i - 1] + aa_mass[seq[i - 1]]

    return ladder

def get_fragment_ion_masses(aa_mass, total_mass): # compute b- and y-ion masses from sum of aa mass
    major_masses = sorted([aa_mass + proton, aa_mass / 2.0 + proton, \
                    total_mass - aa_mass + water + proton, (total_mass - aa_mass + water) / 2.0 + proton])
    minor_masses = sorted([major_masses[0] - water, major_masses[0] - ammonia, major_masses[0] - aion, \
                    major_masses[2] - water, major_masses[2] - ammonia])
    return major_masses, minor_masses

def get_mass_error(observed, expected):
    if expected == 0:
        return observed - expected
    else:
        return (observed - expected) * 1000000.0 / expected

def mass_matching_init(target, mass_list, tol):
    return mass_matching(target, mass_list, tol, 0, len(mass_list))

def mass_matching(target, mass_list, tol, start_loc, end_loc):
    if target * (1.0 + tol) < mass_list[start_loc] or target * (1.0 - tol) > mass_list[end_loc - 1]:
        return [], []

    if end_loc - start_loc == 1:
        return [start_loc], [get_mass_error(mass_list[start_loc], target)]

    mid_loc = int((start_loc + end_loc) / 2)

    if target * (1.0 + tol) < mass_list[mid_loc]:
        return mass_matching(target, mass_list, tol, start_loc, mid_loc)

    if target * (1.0 - tol) > mass_list[mid_loc]:
        if mid_loc == len(mass_list) - 1:
            return False
        return mass_matching(target, mass_list, tol, mid_loc + 1, end_loc)

    hit_loc = [mid_loc]
    hit_error = [get_mass_error(mass_list[mid_loc], target)]
    next_loc = mid_loc + 1

    while next_loc < end_loc and mass_list[next_loc] <= target * (1.0 + tol):
        hit_loc.append(next_loc)
        hit_error.append(get_mass_error(mass_list[next_loc], target))
        next_loc += 1

    next_loc = mid_loc - 1

    while next_loc >= start_loc and mass_list[next_loc] >= target * (1.0 - tol):
        hit_loc.append(next_loc)
        hit_error.append(get_mass_error(mass_list[next_loc], target))
        next_loc -= 1

    return hit_loc, hit_error

############################################
### MAIN
input_files = []
for i in range(22):
    input_files.append("train_" + str(i))
# print(input_files)
input_files = ["test_no_dup", "val_no_dup"]

# data_path = '../data_best_compat/val_no_dup'
total_seq_count = 0
keep_count = 0

for input_file in input_files:
  print(input_file)
  data_path = '../data_best_compat/' + input_file
  output_path = '../data_best_compat_clean/' + input_file
  tgt_output_path = '../data_best_compat_clean/' + input_file
  
  with open(data_path + '.csv', 'rt') as fin, open(output_path + '.csv', 'w') as fout, open(output_path + '_tgt.csv', 'w') as tgt_out:
      line = fin.readline()

      i = 0
      for line in fin:
          content = line.rstrip('\n').split('|')
          seq = content[0]
          heder = content[:5]
          # print(seq)
          if 'U' in seq:
            continue
          spectrum = np.reshape([float(x) for x in content[-1].split(',')], (-1, 2))       
          evidence_major, evidence_minor = find_evidence(seq, spectrum, mass_tol)
          # print('--')
          ### DO WHATEVER YOU WANT WITH EVIDENCE DATA HERE
          if np.sum(evidence_minor) / len(evidence_minor) >= 0.4:
            keep_count += 1
            # print(np.sum(evidence_minor), len(evidence_minor))
            # print(' '.join(seq))
            fout.write(line)
            tgt_out.write(' '.join(seq) + '\n')
          
          
          total_seq_count += 1
          # i += 1
          # if i > 5: break

print('keep: %d, total: %d' % (keep_count, total_seq_count))
