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

import os
import sys

# file_names = [] # 'evidence_16.txt'
# for i in range(0, 28, 1): # range(0,212,1):
#     strr = "train_" + str(i) # + '.csv'
#     file_names.append(strr)

# file_names = [
# '/data/deepnovo/high_all.mgf',
# ]
# # file_names = ['test_MGF/DS13HipH_RP_CE27_EqMass_1_formatted.mgf']
# # file_names = ['wu_peptidome/' + f for f in file_names]
# file_names = ['test_no_dup_1000.mgf']
# print(file_names)



def main(args):
  # args = [output_dir, file1, file2, file3, ...]
  out_dir = args[0]
  # header = ["sequence", "charge", "mass/charge", "score1", "score2", "spectrum"]
  i = 0
  m_mod_count = 0
  n_mod_count = 0
  q_mod_count = 0
  for file_name in args[1:]:
      print(file_name)
      out_filename = os.path.basename(file_name)[:-4] + '.csv'
      out_filename = os.path.join(out_dir, out_filename)
      # file_name = file_name
      with open(file_name, 'r') as input_file, open(out_filename, 'w') as output_file:

          for row in input_file:
              row = row.strip()
              if row == 'BEGIN IONS':
                # print('begin')
                mass_p_charge = ''
                seq = ''
                charge = ''
                spectrum = []

              elif row == 'END IONS':
                if seq == '':
                  seq = 'SEQ'
                spectrum_str = ','.join(spectrum)
                output_str = '|'.join([seq, charge, mass_p_charge, spectrum_str])
                # print(output_str)
                output_file.write(output_str + '\n')
                i = i + 1
                # if i % 1000 == 0: print(i)

              elif row.startswith('TITLE'):
                # print('title')
                pass
              elif row.startswith('PEPMASS'):
                mass_p_charge = row.split('=')[1]
                # for wu_peptidome data, the pepmass contains a pair of space-seperated mass and sum of abundance
                mass_p_charge = mass_p_charge.split(' ')[0]
                # print(mass_p_charge)
              elif row.startswith('CHARGE'):
                charge = row.split('=')[1][0]
                # print(charge)
              elif row.startswith('SCANS'):
                pass
              elif row.startswith('RTINSECONDS'):
                pass
              elif row.startswith('SEQ'):
                seq = row.split('=')[1]

                # Edit modification
                # C mod +57 -> normal C
                seq = seq.replace('C(+57.02)', 'C')
                # M mod -> m
                seq = seq.replace('M(+15.99)', 'm')
                # Q mod -> q
                seq = seq.replace('Q(+.98)', 'q')
                # N mod -> n
                seq = seq.replace('N(+.98)', 'n')

                if '(' in seq:
                  print(seq)
                  break
                if 'm' in seq: m_mod_count += 1
                if 'q' in seq: q_mod_count += 1
                if 'n' in seq: n_mod_count += 1

              else:
                spectrum += row.split(' ')
              # i = i + 1
              # if i >= 2: 
                # break


  print('total seq:', i)
  print('M mod:', m_mod_count, 100.*m_mod_count/i)
  print('N mod:', n_mod_count, 100.*n_mod_count/i)
  print('Q mod:', q_mod_count, 100.*q_mod_count/i)
# BEGIN IONS
# TITLE=DS13HipH_RP_CE27_EqMass_1.7.7.2 File:"DS13HipH_RP_CE27_EqMass_1.raw", NativeID:"controllerType=0 controllerNumber=1 scan=7"
# PEPMASS=469.424499511719
# CHARGE=2+
# SCANS=7
# RTINSECONDS=2.44445604
# SEQ=
# 65.09106445 7391.3022460938

if __name__ == "__main__":
  main(sys.argv[1:])
