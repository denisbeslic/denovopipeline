# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import re
from random import shuffle
from itertools import combinations

from Bio import SeqIO
from pyteomics import parser
import numpy as np
import tensorflow as tf

import deepnovo_config
from deepnovo_cython_modules import get_candidate_intensity


class WorkerDB(object):
  """TODO(nh2tran): docstring.
     This class contains the database search module.
     We use "db" for "database".
     We use "pepmod" to refer to a modified version of a "peptide"
  """


  def __init__(self, db_fasta_file):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerDB: __init__()")

    # search_db and search_hybrid could use different fasta files for their
    #   worker_db objects. So it's better to have fasta files as input.
    self.db_fasta_file = db_fasta_file

    # we currently use deepnovo_config to store both const & settings
    # the settings should be shown in __init__() to keep track carefully
    # input info to build a db
    self.cleavage_rule = deepnovo_config.cleavage_rule
    self.num_missed_cleavage = deepnovo_config.num_missed_cleavage
    self.fixed_mod_list = deepnovo_config.fixed_mod_list
    self.var_mod_list = deepnovo_config.var_mod_list
    self.num_mod = deepnovo_config.num_mod
    self.precursor_mass_tolerance = deepnovo_config.precursor_mass_tolerance
    self.precursor_mass_ppm = deepnovo_config.precursor_mass_ppm
    self.decoy = deepnovo_config.FLAGS.decoy
    print("db_fasta_file = {0:s}".format(self.db_fasta_file))
    print("cleavage_rule = {0:s}".format(self.cleavage_rule))
    print("num_missed_cleavage = {0:d}".format(self.num_missed_cleavage))
    print("fixed_mod_list = {0}".format(self.fixed_mod_list))
    print("var_mod_list = {0}".format(self.var_mod_list))
    print("num_mod = {0}".format(self.num_mod))
    print("precursor_mass_tolerance = {0:.4f}".format(self.precursor_mass_tolerance))
    print("precursor_mass_ppm = {0:.6f}".format(self.precursor_mass_ppm))

    # data structure to store a db
    # all attributes will be built/loaded by build_db()
    self.peptide_count = None
    self.peptide_list = None
    self.peptide_mass_array = None
    self.pepmod_maxmass_array = None

    self.test_time = 0.0


  def build_db(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerDB: build_db()")

    # parse the input fasta file into a list of sequences
    # more about SeqIO and SeqRecord: http://biopython.org/wiki/SeqRecord
    with open(self.db_fasta_file, "r") as handle:
      record_iterator = SeqIO.parse(handle, "fasta")
      record_list = list(record_iterator)
      print("Number of protein sequences: {0:d}".format(len(record_list)))

    # cleave protein sequences into a list of unique peptides
    # more about pyteomics.parser.cleave and cleavage rules:
    # https://pythonhosted.org/pyteomics/api/parser.html

    # create a peptide to protein accession id map.
    peptide_2_protein_id = {}
    for record in record_list:
      protein_sequence = str(record.seq)
      protein_id = str(record.id)
      cleaved_peptide_set = parser.cleave(
        sequence=protein_sequence,
        rule=parser.expasy_rules[self.cleavage_rule],
        missed_cleavages=self.num_missed_cleavage)
      for peptide in cleaved_peptide_set:
        if any(x in peptide for x in ['X', 'B', 'U', 'Z']):
          # skip peptides with undetermined amino acid ['X', 'B', 'U', 'Z']
          continue
        if peptide not in peptide_2_protein_id:
          peptide_2_protein_id[peptide] = {protein_id}
        else:
          peptide_2_protein_id[peptide].add(protein_id)

    peptide_list = [list(peptide) for peptide in peptide_2_protein_id.keys()]
    peptide_list = [[x + 'mod' if x in self.fixed_mod_list else x for x in peptide] for peptide in peptide_list ]

    peptide_count = len(peptide_list)
    print("Number of peptides: {0:d}".format(peptide_count))

    # for each peptide, find the mass and the max modification mass
    peptide_mass_array = np.zeros(peptide_count)
    pepmod_maxmass_array = np.zeros(peptide_count)
    for index, peptide in enumerate(peptide_list):
      peptide_mass_array[index] = self._compute_peptide_mass(peptide)
      pepmod = [x + 'mod' if x in self.var_mod_list else x for x in peptide]
      pepmod_maxmass_array[index] = self._compute_peptide_mass(pepmod)

    self.peptide_count = peptide_count
    self.peptide_list = peptide_list
    self.peptide_mass_array = peptide_mass_array
    self.pepmod_maxmass_array = pepmod_maxmass_array
    self.peptide_2_protein_id = peptide_2_protein_id


  def search_db(self, model, worker_io, predicted_denovo_list=None):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerDB: search_db()")

    # move load/build db here?

    # if provided, convert predicted_denovo_list to dictionary for easy lookup
    denovo_peptide_dict = None
    if predicted_denovo_list is not None:
      print("WorkerDB: search_db() - read denovo peptides")
      denovo_peptide_dict = {}
      for predicted in predicted_denovo_list:
        feature_id = predicted["feature_id"]
        sequence = predicted["sequence"]
        denovo_peptide_dict[feature_id] = sequence

    print("WorkerDB: search_db() - open tensorflow session")
    session = tf.Session()
    model.restore_model(session)

    worker_io.open_input()
    worker_io.get_location()
    worker_io.split_feature_index()
    worker_io.open_output()

    print("".join(["="] * 80)) # section-separating line
    print("WorkerDB: search_db() - search loop")

    for index, feature_index_batch in enumerate(worker_io.feature_index_batch_list):
      print("Read {0:d}/{1:d} batches".format(index + 1,
                                              worker_io.feature_index_batch_count))
      spectrum_batch = worker_io.get_spectrum(feature_index_batch)
      predicted_batch = self._search_db_batch(spectrum_batch,
                                              model,
                                              session,
                                              denovo_peptide_dict)
      worker_io.write_prediction(predicted_batch)

    print("Total spectra: {0:d}".format(worker_io.feature_count["total"]))
    print("  read: {0:d}".format(worker_io.feature_count["read"]))
    print("  skipped: {0:d}".format(worker_io.feature_count["skipped"]))
    print("    by mass: {0:d}".format(worker_io.feature_count["skipped_mass"]))

    worker_io.close_input()
    worker_io.close_output()
    session.close()


  def _compute_peptide_mass(self, peptide):
    """TODO(nh2tran): docstring.
    """

    #~ print("".join(["="] * 80)) # section-separating line ===
    #~ print("WorkerDB: _compute_peptide_mass()")

    peptide_mass = (deepnovo_config.mass_N_terminus
                    + sum(deepnovo_config.mass_AA[aa] for aa in peptide)
                    + deepnovo_config.mass_C_terminus)

    return peptide_mass


  def _expand_peptide_modification(self, peptide):
    """TODO(nh2tran): docstring.
       May also use parser.isoforms
    """

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerDB: _expand_peptide_modification()")

    # all possible positions for modification
    position_list = [position for position, aa in enumerate(peptide)
                     if aa in self.var_mod_list]
    position_count = len(position_list)
    # max number of modifications allowed
    num_mod = min(position_count, self.num_mod)
    # find all combinations upto num_mod
    position_combination_list = []
    for x in xrange(1, num_mod+1):
      position_combination_list += combinations(position_list, x)
    # find all pepmod
    pepmod_list = [peptide]
    for position_combination in position_combination_list:
      pepmod = peptide[:]
      for position in position_combination:
        pepmod[position] += 'mod'
      pepmod_list.append(pepmod)
    
    return pepmod_list


  def _filter_by_mass(self, precursor_mass):
    """TODO(nh2tran): docstring.
    """

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerDB: _filter_by_mass()")

    # use precursor_mass_ppm instead of absolute precursor_mass_tolerance
    #~ precursor_mass_tolerance = self.precursor_mass_tolerance
    precursor_mass_tolerance = self.precursor_mass_ppm * precursor_mass

    # 1st filter by the peptide mass and the max pepmod mass
    filter1_index = np.flatnonzero(np.logical_and(
        np.less_equal(self.peptide_mass_array,
                      precursor_mass + precursor_mass_tolerance),
        np.greater_equal(self.pepmod_maxmass_array,
                         precursor_mass - precursor_mass_tolerance)))

    # find all possible modifications
    pepmod_list = []
    for index in filter1_index:
      peptide = self.peptide_list[index]
      pepmod_list += self._expand_peptide_modification(peptide)
    pepmod_mass_array = np.array([self._compute_peptide_mass(pepmod)
                                  for pepmod in pepmod_list])

    # 2nd filter by exact pepmod mass
    filter2_index = np.flatnonzero(np.logical_and(
        np.less_equal(pepmod_mass_array,
                      precursor_mass + precursor_mass_tolerance),
        np.greater_equal(pepmod_mass_array,
                         precursor_mass - precursor_mass_tolerance)))

    candidate_list = [pepmod_list[x] for x in filter2_index]

    return candidate_list


  def _score_spectrum(self,
                      precursor_mass,
                      spectrum_original,
                      state0_c,
                      state0_h,
                      candidate_list,
                      model,
                      model_output_logprob,
                      model_lstm_state,
                      session,
                      direction):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerDB: _score()")

    # convert symbols into id
    candidate_list = [[deepnovo_config.vocab[x] for x in candidate] 
                      for candidate in candidate_list]

    # we shall group candidates into minibatches
    # === candidate_len ===
    # s
    # i
    # z
    # e
    # =====================
    minibatch_size = len(candidate_list) # number of candidates
    candidate_len = len(candidate_list[0]) # length of each candidate

    # candidates share the same state0, so repeat into [minibatch_size, 512]
    # the states will also be updated after every iteration
    state0_c = state0_c.reshape((1, -1)) # reshape to [1, 512]
    state0_h = state0_h.reshape((1, -1))
    minibatch_state_c = np.repeat(state0_c, minibatch_size, axis=0)
    minibatch_state_h = np.repeat(state0_h, minibatch_size, axis=0)

    # mass of each candidate, will be accumulated everytime an AA is appended
    minibatch_prefix_mass = np.zeros(minibatch_size)

    # output is a list of candidate_len arrays of shape [minibatch_size, 26]
    # each row is log of probability distribution over 26 classes/symbols
    output_logprob_list = []

    # recurrent iterations
    for position in range(candidate_len):

      # gather minibatch data
      minibatch_AA_id = np.zeros(minibatch_size)
      for index, candidate in enumerate(candidate_list):
        AA = candidate[position]
        minibatch_AA_id[index] = AA
        minibatch_prefix_mass[index] += deepnovo_config.mass_ID[AA]

      # this is the most time-consuming ~70-75%
      minibatch_intensity = [get_candidate_intensity(spectrum_original,
                                                     precursor_mass,
                                                     prefix_mass,
                                                     direction)
                             for prefix_mass in np.nditer(minibatch_prefix_mass)]

      # final shape [minibatch_size, 26, 8, 10]
      minibatch_intensity = np.array(minibatch_intensity)

      # model feed
      input_feed = {}
      input_feed[model.input_dict["AAid"][1].name] = minibatch_AA_id
      input_feed[model.input_dict["intensity"].name] = minibatch_intensity
      input_feed[model.input_dict["lstm_state"][0].name] = minibatch_state_c
      input_feed[model.input_dict["lstm_state"][1].name] = minibatch_state_h
      # and run
      output_feed = [model_output_logprob, model_lstm_state]
      output_logprob, (minibatch_state_c, minibatch_state_h) = session.run(
          fetches=output_feed,
          feed_dict=input_feed)

      output_logprob_list.append(output_logprob)

    return output_logprob_list


  def _search_db_batch(self,
                       spectrum_batch,
                       model,
                       session,
                       denovo_peptide_dict):
    """TODO(nh2tran): docstring.
       Inputs:
         spectrum_batch: a list of spectrum, each is a dictionary
           spectrum["feature_id"]
           spectrum["precursor_mass"]
           spectrum["spectrum_holder"]
           spectrum["spectrum_original_forward"]
           spectrum["spectrum_original_backward"]
       Outputs:
         predicted_batch: a list of predicted, each is a dictionary
           predicted["feature_id"]
           predicted["sequence"]
           predicted["score"]
           predicted["position_score"]
    """

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerDB: _search_db_batch()")

    # initialize the lstm using the spectrum
    # for faster speed, we initialize the whole spectrum_batch instead of 1-by-1
    input_feed = {}
    spectrum_holder = np.array([spectrum["spectrum_holder"]
                                for spectrum in spectrum_batch])
    input_feed[model.input_dict["spectrum"].name] = spectrum_holder
    output_feed = [model.output_forward["lstm_state0"],
                   model.output_backward["lstm_state0"]]
    ((state0_c_forward, state0_h_forward),
     (state0_c_backward, state0_h_backward)) = session.run(fetches=output_feed,
                                                           feed_dict=input_feed)

    predicted_batch = []
    # we search spectrum by spectrum
    # a faster way is to process them in parallel, but hard to debug
    #~ test_id = "F12:7420"
    for spectrum_index, spectrum in enumerate(spectrum_batch):
      #~ if spectrum["feature_id"] != test_id:
        #~ continue

      predicted = {"feature_id": spectrum["feature_id"],
                   "sequence": [],
                   "score": -float("inf"),
                   "position_score": [],
                   "precursor_mz": spectrum["precursor_mz"],
                   "precursor_charge": spectrum["precursor_charge"],
                   "protein_access_id": "",
                   "scan_list_middle": spectrum["scan_list_middle"]}

      # filter by precursor mass
      # example: [['M', 'D', 'K', 'F', 'Nmod', 'K', 'K']]
      precursor_mass = spectrum["precursor_mass"]
      candidate_list = self._filter_by_mass(precursor_mass)

      # add denovo peptide if provided
      feature_id = spectrum["feature_id"]
      if denovo_peptide_dict is not None and feature_id in denovo_peptide_dict:
        sequence = denovo_peptide_dict[feature_id]
        # TODO(nh2tran): change the precursor_mass_tolerance of denovo
        sequence_mass = self._compute_peptide_mass(sequence)
        precursor_mass_tolerance = precursor_mass * self.precursor_mass_ppm
        if abs(precursor_mass - sequence_mass) <= precursor_mass_tolerance:
          candidate_list.append(sequence)

      # if no candidate found, return empty sequence for this spectrum.
      if not candidate_list:
        predicted_batch.append(predicted)
        continue

      # if decoy is activated, randomly shuffle amino acids to form decoy db.
      if self.decoy:
        for x in candidate_list:
          shuffle(x) # this function works in place and returns None.

      # add special GO/EOS and reverse
      # example: [['_GO', 'M', 'D', 'K', 'F', 'Nmod', 'K', 'K', '_EOS']]
      candidate_forward_list = [[deepnovo_config._GO] + x + [deepnovo_config._EOS]
                                for x in candidate_list]
      candidate_backward_list = [x[::-1] for x in candidate_forward_list]

      # add PAD to all candidates to the same max length
      # [['_GO', 'M', 'D', 'K', 'F', 'Nmod', 'K', 'K', '_EOS', '_PAD', '_PAD']]
      # due to the same precursor mass, candidates have very similar lengths
      candidate_len_list = [len(x) for x in candidate_list]
      candidate_maxlen = max(candidate_len_list)
      for index, length in enumerate(candidate_len_list):
        if length < candidate_maxlen:
          pad_size = candidate_maxlen - length
          candidate_forward_list[index] += [deepnovo_config._PAD] * pad_size
          candidate_backward_list[index] += [deepnovo_config._PAD] * pad_size
      
      # score the spectrum against its candidates
      #   using the forward model
      logprob_forward_list = self._score_spectrum(
          spectrum["precursor_mass"],
          spectrum["spectrum_original_forward"],
          state0_c_forward[spectrum_index],
          state0_h_forward[spectrum_index],
          candidate_forward_list,
          model,
          model.output_forward["logprob"],
          model.output_forward["lstm_state"],
          session,
          direction=0)
      #   and using the backward model
      logprob_backward_list = self._score_spectrum(
          spectrum["precursor_mass"],
          spectrum["spectrum_original_backward"],
          state0_c_backward[spectrum_index],
          state0_h_backward[spectrum_index],
          candidate_backward_list,
          model,
          model.output_backward["logprob"],
          model.output_backward["lstm_state"],
          session,
          direction=1)

      # note that the candidates are grouped into minibatches
      # === candidate_len ===
      # s
      # i
      # z
      # e
      # =====================
      # logprob_forward_list is a list of candidate_maxlen arrays of shape
      #   [minibatch_size, 26]
      # each row is log of probability distribution over 26 classes/symbols

      # find the best scoring candidate
      #~ test_handle = open("test_file", 'w')
      for index, candidate in enumerate(candidate_list):

        # only calculate score on the actual length, not on GO/EOS/PAD
        candidate_len = candidate_len_list[index]

        # align forward and backward logprob
        logprob_forward = [logprob_forward_list[position][index]
                           for position in range(candidate_len)]
        logprob_backward = [logprob_backward_list[position][index]
                            for position in range(candidate_len)]
        logprob_backward = logprob_backward[::-1]

        # score is the sum of logprob(AA) of the candidate in both directions
        #   averaged by the candidate length
        position_score = []
        for position in range(candidate_len):
          AA = candidate[position]
          AA_id = deepnovo_config.vocab[AA]
          position_score.append(logprob_forward[position][AA_id]
                                + logprob_backward[position][AA_id])
        score = sum(position_score) / candidate_len
        if score > predicted["score"]:
          predicted["sequence"] = candidate
          predicted["score"] = score
          predicted["position_score"] = position_score
          protein_access_id = self.peptide_2_protein_id.get(
              ''.join(candidate).replace('mod', ''),
              'DENOVO')
          if isinstance(protein_access_id, set):
            protein_access_id = ','.join(list(protein_access_id))
          predicted["protein_access_id"] = protein_access_id

        #~ if spectrum["feature_id"] == test_id:
          #~ print_candidate = ",".join(candidate)
          #~ print_score = "{0:.2f}".format(score)
          #~ print_row = "\t".join([print_candidate, print_score])
          #~ print(print_row, file=test_handle, end="\n")
      #~ test_handle.close()
      #~ print(abc)
      predicted_batch.append(predicted)

    return predicted_batch


