# Copyright 2019 Sira Sriswasdi
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

### FDR THRESHOLD
fdr_target = 5 ## choices are 5 or 10, right now
min_predicted_aa = 4
min_predicted_frac = 0
model = 'm-mod'

output_folder = 'CUSB_antibody_May2019_output'
mgf_folder = 'CUSB_antibody_May2019'

##########################################################
### LIBRARY
import os, math

##########################################################
### INTERNAL PARAMETERS AND CONSTANTS
fdr_threshold_map = {'m-mod': {5: 0.83, 10: 0.56},
                     'p-mod': {5: 0.81, 10: 0.57}}

aa_list = 'GASPVTCLINDQKEMHFRYWmsty'
aa_mass_list = [57.02146,71.03711,87.03203,
                97.05276,99.06841,101.04768,
                103.00919 + 57.02146,113.08406,113.08406, ## C are C mod
                114.04293,115.02694,128.05858,
                128.09496,129.04259,131.04049,
                137.05891,147.06841,156.10111,
                163.06333,186.07931,131.04049 + 15.99491, ## m = M(ox)
                87.03203 + 79.96633,101.04768 + 79.96633,163.06333 + 79.96633] ## s = S(ph), t = T(ph), y = Y(ph)

aa_mass = {}
for i in range(len(aa_list)):
    aa_mass[aa_list[i]] = aa_mass_list[i]

proton = 1.007276
water = 18.010565

##########################################################
### MAIN SCRIPT
def main(output_folder, mgf_folder, model):
  rescore_threshold = fdr_threshold_map[model][fdr_target]
  predictions = {}

  spectra_count = 0
  predicted_seq_count = 0
  predicted_full_seq_count = 0
  predicted_aa_count = 0
  predicted_mask_count = 0

  for f in os.listdir(output_folder):
      if f.endswith('_rescore'):
          fname = f[:-8]
          predictions[fname] = {}

          with open(os.path.join(output_folder, fname), 'rt') as seq_in, open(os.path.join(output_folder, f), 'rt') as score_in:
              seq_line = seq_in.readline()
              score_line = score_in.readline()
              current_id = 0 ## use line number to map between MGF and OUTPUT

              while seq_line and score_line:
                  seq_list = seq_line.strip().split()
                  score_list = score_line.strip().split()

                  if not '<s>' in seq_list and not '<unk>' in seq_list and len(seq_list) == len(score_list): ## valid prediction
                      score_list = [math.exp(float(x)) for x in score_list]
                      mask_list = ['Y'] * len(seq_list)
                      mask_count = 0

                      for i in range(len(score_list)):
                          if score_list[i] < rescore_threshold:
                              mask_list[i] = 'N'
                              mask_count += 1

                          temp = str(score_list[i]).split('.')
                          score_list[i] = temp[0] + '.' + temp[1][:2]

                      ## count as predicted only if all thresholds are satisfied
                      if len(seq_list) - mask_count >= min_predicted_aa and (len(seq_list) - mask_count) / len(seq_list) >= min_predicted_frac:
                          predicted_seq_count += 1
                          predicted_aa_count += len(seq_list)
                          predicted_mask_count += mask_count

                          predicted_seq = ''
                          total_mass = 0
                          unknown_mass = 0

                          for i in range(len(score_list)):
                              total_mass += aa_mass[seq_list[i]]

                              if mask_list[i] == 'Y':
                                  if unknown_mass > 0: ## preceded by masked positions
                                      temp = str(unknown_mass).split('.')
                                      predicted_seq += '(' + temp[0] + '.' + temp[1][:min(5, len(temp[1]))] + ')'
                                      unknown_mass = 0

                                  predicted_seq += seq_list[i]
                              else:
                                  unknown_mass += aa_mass[seq_list[i]]

                          if unknown_mass > 0: ## ends with unknown mass
                              temp = str(unknown_mass).split('.')
                              predicted_seq += '(' + temp[0] + '.' + temp[1][:min(5, len(temp[1]))] + ')'

                          if mask_count == 0:
                              predicted_full_seq_count += 1

                          theoretical_mhp = str(total_mass + proton + water)                        
                          predictions[fname][current_id] = [predicted_seq, ';'.join([str(x) for x in score_list]), theoretical_mhp]

                  current_id += 1 ## update ID
                  seq_line = seq_in.readline()
                  score_line = score_in.readline()

          spectra_count += current_id

  print('total spectra:                     ', spectra_count)
  print('predicted sequences:               ', predicted_seq_count)
  print('predicted full sequences:          ', predicted_full_seq_count)
  print('amino acids in predicted sequences:', predicted_aa_count)
  print('amino acids after masking:         ', predicted_aa_count - predicted_mask_count)
  print('masks:                             ', predicted_mask_count)

  for f in os.listdir(mgf_folder):
      if f.endswith('.mgf'):
          fname = f[:-4]

          if fname in predictions:
              with open(os.path.join(mgf_folder, f), 'rt') as fin:
                  current_id = 0
                  line = fin.readline()

                  while line:
                      if line.startswith('BEGIN'):
                          if current_id in predictions[fname]:                       
                              scan_num = 'UNK'
                              charge = 'UNK'
                              ret_time = 'UNK'
                              precursor_mass = 'UNK'
                              precursor_mhp = 'UNK'
                              precursor_int = 'UNK'
                              mass_error = 'UNK' 
                              line = fin.readline()

                              while not line.startswith('END'):
                                  if line.startswith('TITLE'):
                                      scan_num = line.strip().split('scan=')[1].replace('"', '')

                                  elif line.startswith('RTINSECONDS'):
                                      ret_time = str(float(line.strip().split('=')[1]) / 60.0)
                                  elif line.startswith('PEPMASS'):
                                      content = line.strip().split('=')[1].split(' ')

                                      if len(content) == 2:
                                          precursor_mass = content[0]
                                          precursor_int = content[1]
                                      elif len(content) == 1:
                                          precursor_mass = content[0]

                                  elif line.startswith('CHARGE'):
                                      charge = line.strip().split('=')[1][:-1]

                                  try:
                                      precursor_mhp = float(precursor_mass) * float(charge) - (float(charge) - 1) * proton
                                      mass_error = str((precursor_mhp - float(predictions[fname][current_id][2])) * 1000000.0 / float(predictions[fname][current_id][2]))
                                      precursor_mhp = str(precursor_mhp)
                                  except:
                                      pass

                                  line = fin.readline()

                              predictions[fname][current_id].extend([fname, scan_num, charge, ret_time, precursor_mass, precursor_mhp, mass_error, precursor_int])

                          current_id += 1

                      line = fin.readline()

  with open('_'.join([mgf_folder, model, 'fdr' + str(fdr_target)]) + '.tsv', 'w') as fout:
      fout.write('\t'.join(['MS File', 'ScanNum', 'Charge', 'RT(min)', 'ObservedM/Z', 'ObservedM+H', 'TheoreticalM+H', 'MassError(ppm)', 'ObservedInt', 'Prediction', 'Scores']) + '\n')

      for fname in predictions:
          for spectrum in predictions[fname]:
              if not len(predictions[fname][spectrum]) == 11:
                  print(fname, spectrum, predictions[fname][spectrum])
              else:
                  fout.write('\t'.join([predictions[fname][spectrum][i] for i in [3, 4, 5, 6, 7, 8, 2, 9, 10, 0, 1]]) + '\n')

                
if __name__ == "__main__":
  main(sys.argv[1:])
