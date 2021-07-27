# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys
import re

from Bio import SeqIO
from Bio.SeqIO import FastaIO

import numpy as np
import math
import deepnovo_config


def compute_peptide_mass(peptide):
  """TODO(nh2tran): docstring.
  """

  #~ print("".join(["="] * 80)) # section-separating line ===
  #~ print("WorkerDB: _compute_peptide_mass()")

  peptide_mass = (deepnovo_config.mass_N_terminus
                  + sum(deepnovo_config.mass_AA[aa] for aa in peptide)
                  + deepnovo_config.mass_C_terminus)

  return peptide_mass

#~ peptide = 'AAAAAAALQAK'
#~ print(peptide)
#~ print(compute_peptide_mass(peptide))


def read_feature_accuracy(input_file, split_char):

  feature_list = []
  with open(input_file, 'r') as handle:
    header_line = handle.readline()
    for line in handle:
      line = re.split(split_char, line)
      feature = {}
      feature["feature_id"] = line[0]
      feature["feature_area"] = math.log10(float(line[1]))
      feature["predicted_score"] = float(line[4])
      feature["recall_AA"] = float(line[5])
      feature["predicted_len"] = float(line[6])
      feature_list.append(feature)
  return feature_list


def find_score_cutoff(accuracy_file, accuracy_cutoff):
  """TODO(nh2tran): docstring."""

  print("".join(["="] * 80)) # section-separating line
  print("find_score_cutoff()")

  feature_list = read_feature_accuracy(accuracy_file, '\t|\r|\n')
  feature_list_sorted = sorted(feature_list, key=lambda k: k['predicted_score'], reverse=True)
  recall_cumsum = np.cumsum([f['recall_AA'] for f in feature_list_sorted])
  predicted_len_cumsum = np.cumsum([f['predicted_len'] for f in feature_list_sorted])
  accuracy_cumsum = recall_cumsum / predicted_len_cumsum
  cutoff_index = np.flatnonzero(accuracy_cumsum < accuracy_cutoff)[0]
  cutoff_score = feature_list_sorted[cutoff_index]['predicted_score']
  print('cutoff_index = ', cutoff_index)
  print('cutoff_score = ', cutoff_score)
  print('cutoff_score = ', 100*math.exp(cutoff_score))

  return cutoff_score


def select_top_score(input_file, output_file, split_char, col_score, score_cutoff):
  """TODO(nh2tran): docstring."""

  print("".join(["="] * 80)) # section-separating line
  print("select_top_score()")

  print('input_file = ', input_file)
  print('output_file = ', output_file)
  print('score_cutoff = ', score_cutoff)

  total_feature = 0
  select_feature = 0
  with open(input_file, 'r') as input_handle:
    with open(output_file, 'w') as output_handle:
      # header
      header_line = input_handle.readline()
      print(header_line, file=output_handle, end="")
      predicted_list = []
      for line in input_handle:
        total_feature += 1
        line_split = re.split(split_char, line)
        predicted = {}
        predicted["line"] = line
        predicted["score"] = float(line_split[col_score]) if line_split[col_score] else -999
        if predicted["score"] >= score_cutoff:
          select_feature += 1
          print(predicted["line"], file=output_handle, end="")
  print('total_feature = ', total_feature)
  print('select_feature = ', select_feature)
          
#~ accuracy_cutoff = 0.90
#~ input_file = "data.training/dia.pecan.plasma.2018_03_29/testing_plasma.unlabeled.csv.deepnovo_denovo"
#~ accuracy_file = "data.training/dia.pecan.plasma.2018_03_29/testing_plasma.unlabeled.csv.deepnovo_denovo.accuracy"
#~ output_file = input_file + ".top90"
#~ split_char = '\t|\n'
#~ col_score = deepnovo_config.pcol_score_max
#~ score_cutoff = find_score_cutoff(accuracy_file, accuracy_cutoff)
#~ select_top_score(input_file, output_file, split_char, col_score, score_cutoff)


def database_lookup(input_fasta_file, input_denovo_file, output_file, split_char, col_sequence):

  print("".join(["="] * 80)) # section-separating line
  print("database_lookup()")

  print('input_fasta_file = ', input_fasta_file)
  print('input_denovo_file = ', input_denovo_file)
  print('output_file = ', output_file)

  with open(input_fasta_file, 'r') as input_fasta_handle:
    record_list = list(SeqIO.parse(input_fasta_handle, "fasta"))
    print("Number of protein sequences: ", len(record_list))

  total_count = 0 
  db_count = 0
  denovo_count = 0
  with open(input_denovo_file, 'r') as input_denovo_handle:
    with open(output_file, 'w') as output_handle:
      # header
      header_line = input_denovo_handle.readline()
      print(header_line, file=output_handle, end="")
      for line in input_denovo_handle:
        total_count += 1
        line_split = re.split(split_char, line)
        line_split = line_split[:-1] # exclude the last empty ""
        predicted_sequence = line_split[col_sequence]
        predicted_sequence = predicted_sequence.replace(',', '')
        predicted_sequence = predicted_sequence.replace('C(Carbamidomethylation)', 'C')
        indb = False
        for record in record_list:
          if predicted_sequence in record.seq:
            indb = True
            break
        if indb:
          db_count += 1
          line_split.append("db")
        else:
          denovo_count += 1
          line_split.append("denovo")
        print('\t'.join(line_split), file=output_handle, end="\n")
  print('total_count = ', total_count)
  print('db_count = ', db_count)
  print('denovo_count = ', denovo_count)

#~ input_fasta_file = "data/uniprot_sprot.human.fasta"
#~ input_denovo_file = "data.training/dia.pecan.plasma.2018_03_29/testing_plasma.unlabeled.csv.deepnovo_denovo.top90.denovo_only"
#~ output_file = input_denovo_file + ".lookup"
#~ split_char = '\t|\n'
#~ col_sequence = 2
#~ database_lookup(input_fasta_file, input_denovo_file, output_file, split_char, col_sequence)


def select_top_k(input_file, output_file, top_k, split_char, col_score):
  """TODO(nh2tran): docstring."""

  print("".join(["="] * 80)) # section-separating line
  print("select_top_k()")

  print('input_file = ', input_file)
  print('output_file = ', output_file)
  print('top_k = ', top_k)

  with open(input_file, 'r') as input_handle:
    with open(output_file, 'w') as output_handle:
      # header
      header_line = input_handle.readline()
      print(header_line, file=output_handle, end="")
      predicted_list = []
      for line in input_handle:
        line_split = re.split(split_char, line)
        predicted = {}
        predicted["line"] = line
        predicted["score"] = float(line_split[col_score]) if line_split[col_score] else -999
        predicted_list.append(predicted)
      sorted_list = sorted(predicted_list, key=lambda k: k['score'], reverse=True) 
      for entry in sorted_list[:top_k]:
        print(entry["line"], file=output_handle, end="")
          
#~ top_k = 7673
#~ split_char = '\t|\n'
#~ col_score = deepnovo_config.pcol_score_max
#~ input_file = "data.training/dia.pecan.plasma.2018_03_29/testing.unlabeled.csv.deepnovo_denovo"
#~ output_file = input_file + ".topk"
#~ select_top_k(input_file, output_file, top_k, split_char, col_score)
#~ split_char = ',|\n'
#~ col_score = 5
#~ input_file = "data.training/dia.urine.2018_03_29/peaks.denovo.csv"


def filter_min_len(input_file, output_file, min_len):
  """TODO(nh2tran): docstring."""

  print("".join(["="] * 80)) # section-separating line
  print("filter_min_len()")
  print('input_file = ', input_file)
  print('output_file = ', output_file)
  print('min_len = ', min_len)

  total_count = 0
  min_len_count = 0
  with open(input_file, 'r') as input_handle:
    with open(output_file, 'w') as output_handle:
      # header
      header_line = input_handle.readline()
      print(header_line, file=output_handle, end="")
      col_sequence = deepnovo_config.pcol_sequence
      for line in input_handle:
        total_count += 1
        line_split = re.split('\t|\n', line)
        predicted_sequence = line_split[col_sequence]
        if predicted_sequence and len(re.split(',', predicted_sequence)) >= min_len:
          print(line, file=output_handle, end="")
          min_len_count += 1
  print('min_len_count = ', min_len_count)
  print('total_count = ', total_count)
          
#~ min_len = 5
#~ input_file = "data.training/dia.abrf.2018_03_27/testing.unlabeled.csv.deepnovo_denovo"
#~ output_file = input_file + ".minlen_" + str(min_len)
#~ filter_min_len(input_file, output_file, min_len)


# filter features of single-feature (DDA-like) scan or multi-feature scan (DIA)
def filter_multifeature(input_file):
  """TODO(nh2tran): docstring."""

  print("".join(["="] * 80)) # section-separating line
  print("filter_multifeature()")

  print('input_file = ', input_file)
  output_file_1 = input_file + '.1fea'
  output_file_2 = input_file + '.2fea'
  print('output_file_1 = ', output_file_1)
  print('output_file_2 = ', output_file_2)

  # read feature and record feature_dict, scan_dict
  with open(input_file, 'r') as input_handle:
    # header
    header_line = input_handle.readline()
    col_feature_id = deepnovo_config.col_feature_id
    col_scan_list = deepnovo_config.col_scan_list
    feature_dict = {}
    scan_dict = {}
    # read feature and record feature_dict, scan_dict
    for line in input_handle:
      line_split = re.split(',|\n', line)
      feature_id = line_split[col_feature_id]
      scan_list = re.split(';', line_split[col_scan_list])
      feature_dict[feature_id] = {}
      feature_dict[feature_id]['line'] = line
      feature_dict[feature_id]['scan_list'] = scan_list
      for scan_id in scan_list:
        if scan_id in scan_dict:
          scan_dict[scan_id]['feature_list'].append(feature_id)
        else:
          scan_dict[scan_id] = {}
          scan_dict[scan_id]['feature_list'] = [feature_id]

  print('Total scan count = ', len(scan_dict))
  print('  Scan with single-feature = ',
        sum([1 if (len(scan['feature_list'])==1) else 0 for _, scan in scan_dict.iteritems()]))
  print('  Scan with multi-feature = ',
        sum([1 if (len(scan['feature_list'])>=2) else 0 for _, scan in scan_dict.iteritems()]))

  # write feature to separate files,
  # depending on its scan is single-feature (DDA-like) or multi-feature (DIA)
  single_feature_count = 0
  multi_feature_count = 0
  with open(output_file_1, 'w') as output_handle_1:
    with open(output_file_2, 'w') as output_handle_2:
      # header
      print(header_line, file=output_handle_1, end="")
      print(header_line, file=output_handle_2, end="")
      for feature_id, feature in feature_dict.iteritems():
        # assuming all scans are single-feature
        output_handle = output_handle_1
        single_feature_count += 1
        # at least 1 scan is multi-feature
        #~ for scan_id in feature['scan_list']:
          #~ if len(scan_dict[scan_id]['feature_list']) >= 2:
            #~ output_handle = output_handle_2
            #~ multi_feature_count += 1
            #~ single_feature_count -= 1
            #~ break
        # average feature count of scans
        feature_count = sum([len(scan_dict[scan_id]['feature_list']) for scan_id in feature['scan_list']])
        feature_count /= float(len(feature['scan_list']))
        if feature_count >= 2:
          output_handle = output_handle_2
          multi_feature_count += 1
          single_feature_count -= 1
        print(feature['line'], file=output_handle, end="")

  print('Total feature count = ', len(feature_dict))
  print('Feature with single-feature scans = ', single_feature_count)
  print('Feature with at least 1 multi-feature scans = ', multi_feature_count)

#~ input_file = "data.training/dia.urine.2018_03_29/testing_12.feature.csv"
#~ filter_multifeature(input_file)

