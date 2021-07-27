import numpy as np

from nmt import input_config


def read_empirical_mass(file_data, source_file_name):
  with open(source_file_name, 'r') as source_file:
    file_data["ref_empirical_mass"] = []
    for row in source_file:
      row = row.strip()
      data = row.split('|')

      charge = int(float(data[1]))
      mass_p_charge = float(data[2])
      peptide_mass = charge * mass_p_charge + (2.0 - charge) * input_config.mass_H
      sum_mass = peptide_mass - 2*input_config.mass_H - input_config.mass_H2O
      file_data["ref_empirical_mass"].append(sum_mass)


def read_output_file(input_filename, prob_filename, ref_filename=None, mass_spec_filename=None):
  file_entry = {"ref_seqs": [],
                "nmt_seqs": [],
                "ref_weights": [],
                "nmt_weights": [],
                "probs": [],
                "length": 0}
  if ref_filename:
    with open(ref_filename, "r") as ref_file:
      for line in ref_file:
        ref_seq = line.strip()
        file_entry["ref_seqs"].append(ref_seq)

  with open(input_filename, "r") as input_file:
    for line in input_file:
      nmt_seq = line.strip()
      file_entry["nmt_seqs"].append(nmt_seq)

  with open(prob_filename, "r") as prob_file:
    for line in prob_file:
      probs = line.strip()
      probs = probs.split(" ")
      # if probs[0] != '':
      #   probs = [np.exp(float(prob)) for prob in probs]
      # else:
      #   probs = []
      file_entry["probs"].append(probs)
      file_entry["length"] += 1
      
  read_empirical_mass(file_entry, mass_spec_filename)
  return file_entry
  
  
def read_compare_file(input_filename):
  file_entry = {"ref_seqs": [],
                "nmt_seqs": [],
                "ref_weights": [],
                "nmt_weights": [],
                "probs": [],
                "length": 0}

  with open(input_filename, "r") as input_file:
    while True:
      ref_seq = input_file.readline()
      if not ref_seq: break

      nmt_seq = input_file.readline()
      nmt_seq = nmt_seq.strip()
      ref_seq = ref_seq.strip()

      total_weight_ref = input_file.readline().strip()
      total_weight_nmt = input_file.readline().strip()

      probs = input_file.readline().strip()
      probs = probs.split(" ")
      tmp = input_file.readline()

      file_entry["ref_seqs"].append(ref_seq)
      file_entry["nmt_seqs"].append(nmt_seq)
      file_entry["ref_weights"].append(float(total_weight_ref))
      file_entry["nmt_weights"].append(float(total_weight_nmt))
      file_entry["probs"].append(probs)
      file_entry["length"] += 1
  return file_entry


def read_deepnovo_file(input_filename, ref_filename, mass_spec_filename):
  file_entry = {"ref_seqs": [],
                "nmt_seqs": [],
                "ref_weights": [],
                "nmt_weights": [],
                "probs": [],
                "length": 0}
  with open(ref_filename, "r") as ref_file:
    last = 0
    for line in ref_file:
      ref_seq = line.strip()
      file_entry["ref_seqs"].append(ref_seq)

  # scan predicted_sequence predicted_score predicted_position_score
  # 0	Y,E,E,I,Q,I,T,Q,R	-0.45	-1.24,-0.01,-0.00,-0.01,-0.05,-0.01,-2.70,-0.05
  with open(input_filename, "r") as input_file:
    header = input_file.readline()
    last = 0
    for line in input_file:
      line = line.strip().split('\t')
      scan_num = int(line[0])

      while scan_num > last:
        file_entry["nmt_seqs"].append('')
        file_entry["probs"].append('')
        file_entry["length"] += 1
        last += 1

      deepnovo_seq = line[1]
      deepnovo_seq = deepnovo_seq.replace('Cmod', 'C')
      deepnovo_seq = deepnovo_seq.replace('Mmod', 'm')
      deepnovo_seq = deepnovo_seq.replace('Qmod', 'q')
      deepnovo_seq = deepnovo_seq.replace('Nmod', 'n')
      deepnovo_seq = deepnovo_seq.replace(',', ' ')
      # print(deepnovo_seq)
      # print(file_entry["ref_seqs"][last])
      # print('--------')

      if len(line[1]) > 0:
        probs = line[3].split(",")
      else:
        probs = []
      probs.append('0')

      file_entry["nmt_seqs"].append(deepnovo_seq)
      file_entry["probs"].append(probs)
      file_entry["length"] += 1
      last += 1

  read_empirical_mass(file_entry, mass_spec_filename)
  return file_entry  