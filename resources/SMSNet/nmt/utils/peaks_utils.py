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

import collections
import multiprocessing
import numpy as np
import sys
import time

# sys.path.append('..')
from nmt import input_config

def parse_row(input_row):
  '''Parse input string to data entry'''
  # header = ["sequence", "charge", "mass/charge", "score1", "score2", "spectrum"]
  input_row = input_row.strip()
  data = input_row.split('|')
  
  charge = int(float(data[1]))
  mass_p_charge = float(data[2])
  peptide_mass = charge * mass_p_charge + (2.0 - charge) * input_config.mass_H

  spec = data[-1].split(',')

  positions = spec[::2]
  values = spec[1::2]
  seq = data[0]
  
  positions = [float(posi) for posi in positions]
  values = [float(val) for val in values]

  return seq, charge, mass_p_charge, peptide_mass, positions, values


def find_nearest_peak(array, value):
  idx = (np.abs(array - value)).argmin()
  return array[idx]


def has_peak_within_threshold(relevant_peaks, value, threshold=0.1):
  return (np.abs(relevant_peaks - value)).min() < threshold


def get_sorted_relevant_peak_masses(sequence, peptide_mass, charge):
  '''Compute relevant peak masses from the input sequence
  Input:
  sequence: A list of strings. Each entry contain a string representing an amino acid
  peptide_mass: Float. total mass of the peptide
  charge: Int.
  
  Output:
  A list of relevant masses. For each amino acid, 8 locations are considered: b,b(2+),b-H2O,b-NH3,
    y,y(2+),y-H2O,y-NH3.
  '''
  amino_acid_masses = []
  
  for amino_acid in sequence:
    assert amino_acid in input_config.mass_AA
    amino_acid_masses.append(input_config.mass_AA[amino_acid])
  
  amino_acid_masses = np.array(amino_acid_masses)
  cum_sum = np.cumsum(amino_acid_masses)
  
  b_ion = cum_sum + input_config.mass_N_terminus
  y_ion = peptide_mass - b_ion
  
  b_H2O = b_ion - input_config.mass_H2O
  b_NH3 = b_ion - input_config.mass_NH3
  b_plus2_charge1 = (b_ion + input_config.mass_H) / 2
  
  y_H2O = y_ion - input_config.mass_H2O
  y_NH3 = y_ion - input_config.mass_NH3
  y_plus2_charge1 = (y_ion + input_config.mass_H) / 2
  
  relevant_masses = np.concatenate((b_ion, b_H2O, b_NH3, b_plus2_charge1, 
                                    y_ion, y_H2O, y_NH3, y_plus2_charge1), axis=0)
  sorted_relevant_masses = np.sort(relevant_masses)
  return sorted_relevant_masses


def get_relevant_peak_mass_dict(sequence, peptide_mass, charge):
  '''Compute relevant peak masses from the input sequence
  Input:
  sequence: A list of strings. Each entry contain a string representing an amino acid
  peptide_mass: Float. total mass of the peptide
  charge: Int.
  
  Output:
  A dictionary of lists of relevant masses. For each amino acid, 8 locations are considered: 
    b,b(2+),b-H2O,b-NH3,y,y(2+),y-H2O,y-NH3.
  '''
  amino_acid_masses = []
  
  for amino_acid in sequence:
    assert amino_acid in input_config.mass_AA
    amino_acid_masses.append(input_config.mass_AA[amino_acid])
  
  amino_acid_masses = np.array(amino_acid_masses)
  cum_sum = np.cumsum(amino_acid_masses)
  
  relevant_peak_dict = {}
  b_ion = cum_sum + input_config.mass_N_terminus
  y_ion = peptide_mass - b_ion
  relevant_peak_dict['b_ion'] = b_ion
  relevant_peak_dict['y_ion'] = y_ion
  
  relevant_peak_dict['b_H2O'] = b_ion - input_config.mass_H2O
  relevant_peak_dict['b_NH3'] = b_ion - input_config.mass_NH3
  relevant_peak_dict['b_plus2_charge1'] = (b_ion + input_config.mass_H) / 2
  
  relevant_peak_dict['y_H2O'] = y_ion - input_config.mass_H2O
  relevant_peak_dict['y_NH3'] = y_ion - input_config.mass_NH3
  relevant_peak_dict['y_plus2_charge1'] = (y_ion + input_config.mass_H) / 2
  return relevant_peak_dict


def compute_evidence_correlation(source_file_name, result_file_name):
  '''Compute correlation between amino acid eviden and correctness of prediction
  Args:
    source_file_name: data file
    result_file_name: result file, same number of line as data file.
      each line consisting of 1 or 0 indicating the correctness of the predictions
  '''
  with open(source_file_name, 'r') as source_file, open(result_file_name, 'r') as result_file:
    count = 0
    amino_acid_count, amino_acid_with_evidence_count = 0., 0.
    has_evidence_correct_count, has_evidence_incorrect_count = 0., 0.
    no_evidence_correct_count, no_evidence_incorrect_count = 0., 0.
    for row in source_file:
      result = result_file.readline().strip()
      result = result.split(',')

      (seq, charge, mass_p_charge, peptide_mass, positions, values) = parse_row(row)
      # print(seq, charge, mass_p_charge, peptide_mass, positions, values)
      assert len(seq) == len(result)

      relevant_peak_dict = get_relevant_peak_mass_dict(seq, peptide_mass, charge)
      # considered_ion = ['b_ion','y_ion']
      considered_ion = ['b_ion','y_ion','b_plus2_charge1','y_plus2_charge1']
      # considered_ion = ['b_ion','y_ion','b_H2O','y_H2O','b_NH3','y_NH3','b_plus2_charge1','y_plus2_charge1']
      
      # positions must be sorted
      positions = np.array([float(posi) for posi in positions])
      amino_acid_count += len(seq)
      has_evidence_in_previous_step = True
      for pos in range(len(seq)):
        has_evidence = False
        for ion_type in considered_ion:
          has_evidence |= has_peak_within_threshold(positions, relevant_peak_dict[ion_type][pos])
        evidence_for_next_step = has_evidence
        has_evidence = has_evidence_in_previous_step and has_evidence
        
        if result[pos] == '1' and has_evidence:
          has_evidence_correct_count += 1
        elif result[pos] == '1' and not has_evidence:
          no_evidence_correct_count += 1
        elif result[pos] == '0' and has_evidence:
          has_evidence_incorrect_count += 1
        elif result[pos] == '0' and not has_evidence:
          no_evidence_incorrect_count += 1
        else:
          print("Something is worng here")

        if has_evidence:
          amino_acid_with_evidence_count += 1
        has_evidence_in_previous_step = evidence_for_next_step
      count += 1
      if count % 5000 == 0: print(count) # break

  print("Amino acid with evidence: %.2f %%" % (100 * amino_acid_with_evidence_count/amino_acid_count))
  print("Amino acid with evidence: %.0f, Total amino acid: %0.f" % (amino_acid_with_evidence_count,amino_acid_count))
  print("Has evidence and correct: %.0f (%.2f %%), Has evidence but incorrect: %.0f (%.2f %%)" % 
        (has_evidence_correct_count, 100 * has_evidence_correct_count/amino_acid_count,
         has_evidence_incorrect_count, 100 * has_evidence_incorrect_count/amino_acid_count))
  print("No evidence but correct: %.0f (%.2f %%), No evidence and incorrect: %.0f (%.2f %%)" % 
        (no_evidence_correct_count, 100 * no_evidence_correct_count/amino_acid_count,
         no_evidence_incorrect_count, 100 * no_evidence_incorrect_count/amino_acid_count))
  print("Of all incorrect amino acids, %.2f %% has no evidence" % 
        (100 * no_evidence_incorrect_count / (no_evidence_incorrect_count + has_evidence_incorrect_count)))
  return
