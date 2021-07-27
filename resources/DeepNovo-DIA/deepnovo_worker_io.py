# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import numpy as np
import pickle

import deepnovo_config
from deepnovo_cython_modules import process_spectrum


class WorkerIO(object):
  """TODO(nh2tran): docstring.
  """


  def __init__(self, input_spectrum_file, input_feature_file, output_file=None):
    """TODO(nh2tran): docstring.
       The input_file could be input_file or input_file_train/valid/test.
       The output_file is None for train/valid/test cases.
       During training we use two separate WorkerIO objects for train and valid.
    """

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO: __init__()")

    # we currently use deepnovo_config to store both const & settings
    # the settings should be shown in __init__() to keep track carefully
    self.MZ_MAX = deepnovo_config.MZ_MAX
    self.MZ_SIZE = deepnovo_config.MZ_SIZE
    self.batch_size = deepnovo_config.batch_size
    self.header_seq = deepnovo_config.FLAGS.header_seq
    self.neighbor_size = deepnovo_config.neighbor_size
    print("neighbor_size = {0:d}".format(self.neighbor_size))
    self.dia_window = deepnovo_config.dia_window

    self.input_spectrum_file = input_spectrum_file
    self.input_feature_file = input_feature_file
    self.output_file = output_file
    print("input_spectrum_file = {0:s}".format(self.input_spectrum_file))
    print("input_feature_file = {0:s}".format(self.input_feature_file))
    print("output_file = {0:s}".format(self.output_file))
    # keep the file handles open throughout the process to read/write batches
    self.input_spectrum_handle = None
    self.input_feature_handle = None
    self.output_handle = None

    # split data into batches
    self.feature_index_list = []
    self.feature_index_batch_list = []
    self.feature_index_batch_count = 0

    ### store file location of each feature for random access
    self.feature_location_list = []

    # store the file location of all spectra for random access
    self.spectrum_location_dict = {}
    self.spectrum_rtinseconds_dict = {}

    # record the status of spectra that have been read
    self.feature_count = {"total": 0,
                          "read": 0,
                          "skipped": 0,
                          "skipped_mass": 0}
    self.spectrum_count = 0


  def close_input(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO: close_input()")

    self.input_spectrum_handle.close()
    self.input_feature_handle.close()


  def close_output(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO: close_output()")

    self.output_handle.close()


  def get_spectrum(self, feature_index_batch):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO: get_spectrum()")

    spectrum_list = []
    for feature_index in feature_index_batch:
      # parse a feature
      feature_location = self.feature_location_list[feature_index]
      feature_id, feature_area, precursor_mz, precursor_charge, rt_mean, raw_sequence, scan_list, ms1_list = self._parse_feature(feature_location)
      # skip if precursor_mass > MZ_MAX
      precursor_mass = precursor_mz * precursor_charge - deepnovo_config.mass_H * precursor_charge
      if precursor_mass > self.MZ_MAX:
        self.feature_count["skipped"] += 1
        self.feature_count["skipped_mass"] += 1
        continue
      self.feature_count["read"] += 1
      # parse and process spectrum
      (spectrum_holder,
       spectrum_original_forward,
       spectrum_original_backward,
       scan_list_middle,
       scan_list_original,
       ms1_profile) = self._parse_spectrum(precursor_mz, precursor_mass, rt_mean, scan_list, ms1_list)
      # update dataset
      spectrum = {"feature_id": feature_id,#str(feature_index),#scan,
                  "feature_area": feature_area,
                  "raw_sequence": raw_sequence,
                  "precursor_mass": precursor_mass,
                  "spectrum_holder": spectrum_holder,
                  "spectrum_original_forward": spectrum_original_forward,
                  "spectrum_original_backward": spectrum_original_backward,
                  "precursor_mz": precursor_mz,
                  "precursor_charge": precursor_charge,
                  "scan_list_middle": scan_list_middle,
                  "scan_list_original": scan_list_original,
                  "ms1_profile": ms1_profile}
      spectrum_list.append(spectrum)

    return spectrum_list


  def get_location(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO: get_location()")

    ### store file location of each spectrum for random access {scan:location}
    ### since mgf file can be rather big, cache the locations for each spectrum mgf file.
    spectrum_location_file = self.input_spectrum_file + '.locations.pkl'
    if os.path.exists(spectrum_location_file):
      print("WorkerIO: read cached spectrum locations")
      with open(spectrum_location_file, 'rb') as fr:
        data = pickle.load(fr)
        self.spectrum_location_dict, self.spectrum_rtinseconds_dict, self.spectrum_count = data
    else:
      print("WorkerIO: build spectrum location from scratch")
      spectrum_location_dict = {}
      spectrum_rtinseconds_dict = {}
      line = True
      while line:
        current_location = self.input_spectrum_handle.tell()
        line = self.input_spectrum_handle.readline()
        if "BEGIN IONS" in line:
          spectrum_location = current_location
        elif "SCANS=" in line:
          scan = re.split('=|\r\n', line)[1]
          spectrum_location_dict[scan] = spectrum_location
        elif "RTINSECONDS=" in line:
          rtinseconds = float(re.split('=|\r\n', line)[1])
          spectrum_rtinseconds_dict[scan] = rtinseconds
      self.spectrum_location_dict = spectrum_location_dict
      self.spectrum_rtinseconds_dict = spectrum_rtinseconds_dict
      self.spectrum_count = len(spectrum_location_dict)
      with open(spectrum_location_file, 'wb') as fw:
        pickle.dump((self.spectrum_location_dict, self.spectrum_rtinseconds_dict, self.spectrum_count), fw)

    ### store file location of each feature for random access
    feature_location_list = []
    # skip header line
    _ = self.input_feature_handle.readline()
    line = True
    while line:
      feature_location = self.input_feature_handle.tell()
      feature_location_list.append(feature_location)
      line = self.input_feature_handle.readline()
    feature_location_list = feature_location_list[:-1]
    self.feature_location_list = feature_location_list
    self.feature_count["total"] = len(feature_location_list)
    self.feature_index_list = range(self.feature_count["total"])

    print("spectrum_count = {0:d}".format(self.spectrum_count))
    print("feature_count[total] = {0:d}".format(self.feature_count["total"]))


  def open_input(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO: open_input()")

    self.input_spectrum_handle = open(self.input_spectrum_file, 'r')
    self.input_feature_handle = open(self.input_feature_file, 'r')


  def open_output(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO: open_output()")

    self.output_handle = open(self.output_file, 'w')
    self._print_prediction_header()


  def split_feature_index(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO: split_index()")

    index_batch_list = [self.feature_index_list[i:(i+self.batch_size)]
                            for i in range(0,
                                           self.feature_count["total"],
                                           self.batch_size)]

    self.feature_index_batch_list = index_batch_list
    self.feature_index_batch_count = len(self.feature_index_batch_list)


  def write_prediction(self, predicted_batch):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO: write_prediction()")

    for predicted in predicted_batch:
      feature_id = predicted["feature_id"]
      feature_area = str(predicted["feature_area"])
      precursor_mz = str(predicted["precursor_mz"])
      precursor_charge = str(predicted["precursor_charge"])
      scan_list_middle = ";".join(predicted["scan_list_middle"])
      scan_list_original = ";".join(predicted["scan_list_original"])
      if predicted["sequence"]:
        predicted_sequence = ';'.join([','.join(x) for x in predicted["sequence"]])
        predicted_score = ';'.join(['{0:.2f}'.format(x) for x in predicted["score"]])
        predicted_score_max = '{0:.2f}'.format(np.max(predicted["score"]))
        predicted_position_score = ';'.join([
            ','.join(['{0:.2f}'.format(y) for y in x])
            for x in predicted["position_score"]])
        if "protein_access_id" in predicted:
          # predicted_batch is returned from search_db
          protein_access_id = predicted['protein_access_id']
        else:
          # predicted_batch is returned from search_denovo
          protein_access_id = 'DENOVO'
      else: # if no peptide found, write empty sequence to the output file
        predicted_sequence = ""
        predicted_score = ""
        predicted_score_max = ""
        predicted_position_score = ""
        protein_access_id = ""
      predicted_row = "\t".join([feature_id,
                                 feature_area,
                                 predicted_sequence,
                                 predicted_score,
                                 predicted_position_score,
                                 precursor_mz,
                                 precursor_charge,
                                 protein_access_id,
                                 scan_list_middle,
                                 scan_list_original,
                                 predicted_score_max])
      print(predicted_row, file=self.output_handle, end="\n")


  def _parse_spectrum(self, precursor_mz, precursor_mass, rt_mean, scan_list, ms1_list):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO: _parse_spectrum()")

    spectrum_holder_list = []
    spectrum_original_forward_list = []
    spectrum_original_backward_list = []

    ### select best neighbors from the scan_list by their distance to rt_mean
    # probably move this selection to get_location(), run once rather than repeating
    neighbor_count = len(scan_list)
    best_scan_index = None
    best_distance = float('inf')
    for scan_index, scan in enumerate(scan_list):
      distance = abs(self.spectrum_rtinseconds_dict[scan] - rt_mean)
      if distance < best_distance:
        best_distance = distance
        best_scan_index = scan_index
    neighbor_center = best_scan_index
    neighbor_left_count = neighbor_center
    neighbor_right_count = neighbor_count - neighbor_left_count - 1
    neighbor_size_half = self.neighbor_size // 2
    neighbor_left_count = min(neighbor_left_count, neighbor_size_half)
    neighbor_right_count = min(neighbor_right_count, neighbor_size_half)

    ### padding zero arrays to the left if not enough neighbor spectra
    if neighbor_left_count < neighbor_size_half:
      for x in range(neighbor_size_half - neighbor_left_count):
        spectrum_holder_list.append(np.zeros(
            shape=(1, self.MZ_SIZE),
            dtype=np.float32))
        spectrum_original_forward_list.append(np.zeros(
            shape=(1, self.MZ_SIZE),
            dtype=np.float32))
        spectrum_original_backward_list.append(np.zeros(
            shape=(1, self.MZ_SIZE),
            dtype=np.float32))

    ### parse and add neighbor spectra
    scan_list_middle = []
    ms1_intensity_list_middle = []
    for index in range(neighbor_center - neighbor_left_count, neighbor_center + neighbor_right_count + 1):
      scan = scan_list[index]
      scan_list_middle.append(scan)
      ms1_entry = ms1_list[index]
      ms1_intensity = float(re.split(':', ms1_entry)[1])
      ms1_intensity_list_middle.append(ms1_intensity)
    ms1_intensity_max = max(ms1_intensity_list_middle)
    assert ms1_intensity_max > 0.0, "Error: Zero ms1_intensity_max"
    ms1_intensity_list_middle = [x/ms1_intensity_max for x in ms1_intensity_list_middle]
    for scan, ms1_intensity in zip(scan_list_middle, ms1_intensity_list_middle):
      spectrum_location = self.spectrum_location_dict[scan]
      self.input_spectrum_handle.seek(spectrum_location)
      # parse header lines
      line = self.input_spectrum_handle.readline()
      assert "BEGIN IONS" in line, "Error: wrong input BEGIN IONS"
      line = self.input_spectrum_handle.readline()
      assert "TITLE=" in line, "Error: wrong input TITLE="
      line = self.input_spectrum_handle.readline()
      assert "PEPMASS=" in line, "Error: wrong input PEPMASS="
      line = self.input_spectrum_handle.readline()
      assert "CHARGE=" in line, "Error: wrong input CHARGE="
      line = self.input_spectrum_handle.readline()
      assert "SCANS=" in line, "Error: wrong input SCANS="
      line = self.input_spectrum_handle.readline()
      assert "RTINSECONDS=" in line, "Error: wrong input RTINSECONDS="
      # parse fragment ions
      mz_list, intensity_list = self._parse_spectrum_ion()
      # pre-process spectrum
      (spectrum_holder,
       spectrum_original_forward,
       spectrum_original_backward) = process_spectrum(mz_list,
                                                      intensity_list,
                                                      precursor_mass)
      # normalize by each individual spectrum
      #~ spectrum_holder /= np.max(spectrum_holder)
      #~ spectrum_original_forward /= np.max(spectrum_original_forward)
      #~ spectrum_original_backward /= np.max(spectrum_original_backward)
      # weight by ms1 profile
      #~ spectrum_holder *= ms1_intensity
      #~ spectrum_original_forward *= ms1_intensity
      #~ spectrum_original_backward *= ms1_intensity
      # add spectrum to the neighbor list
      spectrum_holder_list.append(spectrum_holder)
      spectrum_original_forward_list.append(spectrum_original_forward)
      spectrum_original_backward_list.append(spectrum_original_backward)
    ### padding zero arrays to the right if not enough neighbor spectra
    if neighbor_right_count < neighbor_size_half:
      for x in range(neighbor_size_half - neighbor_right_count):
        spectrum_holder_list.append(np.zeros(
            shape=(1, self.MZ_SIZE),
            dtype=np.float32))
        spectrum_original_forward_list.append(np.zeros(
            shape=(1, self.MZ_SIZE),
            dtype=np.float32))
        spectrum_original_backward_list.append(np.zeros(
            shape=(1, self.MZ_SIZE),
            dtype=np.float32))

    spectrum_holder = np.vstack(spectrum_holder_list)
    spectrum_original_forward = np.vstack(spectrum_original_forward_list)
    spectrum_original_backward = np.vstack(spectrum_original_backward_list)
    assert spectrum_holder.shape == (self.neighbor_size,
                                     self.MZ_SIZE), "Error:shape"
    # spectrum-CNN normalization: by feature
    spectrum_holder /= np.max(spectrum_holder)

    # ms1_profile 
    for x in range(neighbor_size_half - neighbor_left_count):
      ms1_intensity_list_middle = [0.0] + ms1_intensity_list_middle
    for x in range(neighbor_size_half - neighbor_right_count):
      ms1_intensity_list_middle = ms1_intensity_list_middle + [0.0]
    assert len(ms1_intensity_list_middle) == self.neighbor_size, "Error: ms1 profile"
    ms1_profile = np.array(ms1_intensity_list_middle)

    return spectrum_holder, spectrum_original_forward, spectrum_original_backward, scan_list_middle, scan_list, ms1_profile


  def _parse_feature(self, feature_location):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO: _parse_feature()")

    self.input_feature_handle.seek(feature_location)
    line = self.input_feature_handle.readline()
    line = re.split(',|\r|\n', line)
    feature_id = line[deepnovo_config.col_feature_id]
    feature_area_str = line[deepnovo_config.col_feature_area]
    feature_area = float(feature_area_str) if feature_area_str else 1.0
    precursor_mz = float(line[deepnovo_config.col_precursor_mz])
    precursor_charge = float(line[deepnovo_config.col_precursor_charge])
    rt_mean = float(line[deepnovo_config.col_rt_mean])
    raw_sequence = line[deepnovo_config.col_raw_sequence]
    scan_list = re.split(';', line[deepnovo_config.col_scan_list])
    ms1_list = re.split(';', line[deepnovo_config.col_ms1_list])
    assert len(scan_list) == len(ms1_list), "Error: scan_list and ms1_list not matched."

    return feature_id, feature_area, precursor_mz, precursor_charge, rt_mean, raw_sequence, scan_list, ms1_list


  def _parse_spectrum_ion(self):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO: _parse_spectrum_ion()")

    # ion
    mz_list = []
    intensity_list = []
    line = self.input_spectrum_handle.readline()
    while not "END IONS" in line:
      mz, intensity = re.split(' |\n', line)[:2]
      mz_float = float(mz)
      intensity_float = float(intensity)
      # skip an ion if its mass > MZ_MAX
      if mz_float > self.MZ_MAX:
        line = self.input_spectrum_handle.readline()
        continue
      mz_list.append(mz_float)
      intensity_list.append(intensity_float)
      line = self.input_spectrum_handle.readline()

    return mz_list, intensity_list


  def _print_prediction_header(self):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80)) # section-separating line
    print("WorkerIO: _print_prediction_header()")

    header_list = ["feature_id",
                   "feature_area",
                   "predicted_sequence",
                   "predicted_score",
                   "predicted_position_score",
                   "precursor_mz",
                   "precursor_charge",
                   "protein_access_id",
                   "scan_list_middle",
                   "scan_list_original",
                   "predicted_score_max"]
    header_row = "\t".join(header_list)
    print(header_row, file=self.output_handle, end="\n")

class WorkerI(object):
  """
  This is a helper class designed for multi-process get_spectrum
  """
  def __init__(self, worker_io):
    self.MZ_MAX = worker_io.MZ_MAX
    self.MZ_SIZE = worker_io.MZ_SIZE
    self.batch_size = worker_io.batch_size
    self.header_seq = worker_io.header_seq
    self.neighbor_size = worker_io.neighbor_size

    self.dia_window = worker_io.dia_window

    self.input_spectrum_file = worker_io.input_spectrum_file
    self.input_feature_file = worker_io.input_feature_file
    self.output_file = worker_io.output_file

    # split data into batches
    self.feature_index_list = worker_io.feature_index_list
    self.feature_index_batch_list = worker_io.feature_index_batch_list
    self.feature_index_batch_count = worker_io.feature_index_batch_count

    ### store file location of each feature for random access
    self.feature_location_list = worker_io.feature_location_list

    # store the file location of all spectra for random access
    self.spectrum_location_dict = worker_io.spectrum_location_dict
    self.spectrum_rtinseconds_dict = worker_io.spectrum_rtinseconds_dict

    # record the status of spectra that have been read
    self.feature_count = worker_io.feature_count
    self.spectrum_count = worker_io.spectrum_count

  def get_spectrum(self, feature_index_batch, input_feature_file_handle, input_spectrum_file_handle):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO: get_spectrum()")

    spectrum_list = []
    for feature_index in feature_index_batch:
      # parse a feature
      feature_location = self.feature_location_list[feature_index]
      feature_id, feature_area, precursor_mz, precursor_charge, rt_mean, raw_sequence, scan_list, ms1_list = self._parse_feature(feature_location, input_feature_file_handle)
      # skip if precursor_mass > MZ_MAX
      precursor_mass = precursor_mz * precursor_charge - deepnovo_config.mass_H * precursor_charge
      if precursor_mass > self.MZ_MAX:
        continue

      # parse and process spectrum
      (spectrum_holder,
       spectrum_original_forward,
       spectrum_original_backward,
       scan_list_middle,
       scan_list_original,
       ms1_profile) = self._parse_spectrum(precursor_mz, precursor_mass, rt_mean, scan_list, ms1_list, input_spectrum_file_handle)
      # update dataset
      spectrum = {"feature_id": feature_id,#str(feature_index),#scan,
                  "feature_area": feature_area,
                  "raw_sequence": raw_sequence,
                  "precursor_mass": precursor_mass,
                  "spectrum_holder": spectrum_holder,
                  "spectrum_original_forward": spectrum_original_forward,
                  "spectrum_original_backward": spectrum_original_backward,
                  "precursor_mz": precursor_mz,
                  "precursor_charge": precursor_charge,
                  "scan_list_middle": scan_list_middle,
                  "scan_list_original": scan_list_original,
                  "ms1_profile": ms1_profile}
      spectrum_list.append(spectrum)

    return spectrum_list

  def _parse_feature(self, feature_location, input_file_handle):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO: _parse_feature()")

    input_file_handle.seek(feature_location)
    line = input_file_handle.readline()
    line = re.split(',|\r|\n', line)
    feature_id = line[deepnovo_config.col_feature_id]
    feature_area = 0#float(line[deepnovo_config.col_feature_area])
    precursor_mz = float(line[deepnovo_config.col_precursor_mz])
    precursor_charge = float(line[deepnovo_config.col_precursor_charge])
    rt_mean = float(line[deepnovo_config.col_rt_mean])
    raw_sequence = line[deepnovo_config.col_raw_sequence]
    scan_list = re.split(';', line[deepnovo_config.col_scan_list])
    ms1_list = re.split(';', line[deepnovo_config.col_ms1_list])
    assert len(scan_list) == len(ms1_list), "Error: scan_list and ms1_list not matched."

    return feature_id, feature_area, precursor_mz, precursor_charge, rt_mean, raw_sequence, scan_list, ms1_list

  def _parse_spectrum(self, precursor_mz, precursor_mass, rt_mean, scan_list, ms1_list, input_file_handle):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO: _parse_spectrum()")

    spectrum_holder_list = []
    spectrum_original_forward_list = []
    spectrum_original_backward_list = []

    ### select best neighbors from the scan_list by their distance to rt_mean
    # probably move this selection to get_location(), run once rather than repeating
    neighbor_count = len(scan_list)
    best_scan_index = None
    best_distance = float('inf')
    for scan_index, scan in enumerate(scan_list):
      print(rt_mean)
      distance = abs(self.spectrum_rtinseconds_dict[scan] - rt_mean)
      if distance < best_distance:
        best_distance = distance
        best_scan_index = scan_index
    neighbor_center = best_scan_index
    neighbor_left_count = neighbor_center
    neighbor_right_count = neighbor_count - neighbor_left_count - 1
    neighbor_size_half = self.neighbor_size // 2
    neighbor_left_count = min(neighbor_left_count, neighbor_size_half)
    neighbor_right_count = min(neighbor_right_count, neighbor_size_half)

    ### padding zero arrays to the left if not enough neighbor spectra
    if neighbor_left_count < neighbor_size_half:
      for x in range(neighbor_size_half - neighbor_left_count):
        spectrum_holder_list.append(np.zeros(
            shape=(1, self.MZ_SIZE),
            dtype=np.float32))
        spectrum_original_forward_list.append(np.zeros(
            shape=(1, self.MZ_SIZE),
            dtype=np.float32))
        spectrum_original_backward_list.append(np.zeros(
            shape=(1, self.MZ_SIZE),
            dtype=np.float32))

    ### parse and add neighbor spectra
    scan_list_middle = []
    ms1_intensity_list_middle = []
    for index in range(neighbor_center - neighbor_left_count, neighbor_center + neighbor_right_count + 1):
      scan = scan_list[index]
      scan_list_middle.append(scan)
      ms1_entry = ms1_list[index]
      ms1_intensity = float(re.split(':', ms1_entry)[1])
      ms1_intensity_list_middle.append(ms1_intensity)
    ms1_intensity_max = max(ms1_intensity_list_middle)
    assert ms1_intensity_max > 0.0, "Error: Zero ms1_intensity_max"
    ms1_intensity_list_middle = [x/ms1_intensity_max for x in ms1_intensity_list_middle]
    for scan, ms1_intensity in zip(scan_list_middle, ms1_intensity_list_middle):
      spectrum_location = self.spectrum_location_dict[scan]
      input_file_handle.seek(spectrum_location)
      # parse header lines
      line = input_file_handle.readline()
      assert "BEGIN IONS" in line, "Error: wrong input BEGIN IONS"
      line = input_file_handle.readline()
      assert "TITLE=" in line, "Error: wrong input TITLE="
      line = input_file_handle.readline()
      assert "PEPMASS=" in line, "Error: wrong input PEPMASS="
      line = input_file_handle.readline()
      assert "CHARGE=" in line, "Error: wrong input CHARGE="
      line = input_file_handle.readline()
      assert "SCANS=" in line, "Error: wrong input SCANS="
      line = input_file_handle.readline()
      assert "RTINSECONDS=" in line, "Error: wrong input RTINSECONDS="
      # parse fragment ions
      mz_list, intensity_list = self._parse_spectrum_ion(input_file_handle)
      # pre-process spectrum
      (spectrum_holder,
       spectrum_original_forward,
       spectrum_original_backward) = process_spectrum(mz_list,
                                                      intensity_list,
                                                      precursor_mass)
      # normalize by each individual spectrum
      #~ spectrum_holder /= np.max(spectrum_holder)
      #~ spectrum_original_forward /= np.max(spectrum_original_forward)
      #~ spectrum_original_backward /= np.max(spectrum_original_backward)
      # weight by ms1 profile
      #~ spectrum_holder *= ms1_intensity
      #~ spectrum_original_forward *= ms1_intensity
      #~ spectrum_original_backward *= ms1_intensity
      # add spectrum to the neighbor list
      spectrum_holder_list.append(spectrum_holder)
      spectrum_original_forward_list.append(spectrum_original_forward)
      spectrum_original_backward_list.append(spectrum_original_backward)
    ### padding zero arrays to the right if not enough neighbor spectra
    if neighbor_right_count < neighbor_size_half:
      for x in range(neighbor_size_half - neighbor_right_count):
        spectrum_holder_list.append(np.zeros(
            shape=(1, self.MZ_SIZE),
            dtype=np.float32))
        spectrum_original_forward_list.append(np.zeros(
            shape=(1, self.MZ_SIZE),
            dtype=np.float32))
        spectrum_original_backward_list.append(np.zeros(
            shape=(1, self.MZ_SIZE),
            dtype=np.float32))

    spectrum_holder = np.vstack(spectrum_holder_list)
    spectrum_original_forward = np.vstack(spectrum_original_forward_list)
    spectrum_original_backward = np.vstack(spectrum_original_backward_list)
    assert spectrum_holder.shape == (self.neighbor_size,
                                     self.MZ_SIZE), "Error:shape"
    # spectrum-CNN normalization: by feature
    spectrum_holder /= np.max(spectrum_holder)

    # ms1_profile
    for x in range(neighbor_size_half - neighbor_left_count):
      ms1_intensity_list_middle = [0.0] + ms1_intensity_list_middle
    for x in range(neighbor_size_half - neighbor_right_count):
      ms1_intensity_list_middle = ms1_intensity_list_middle + [0.0]
    assert len(ms1_intensity_list_middle) == self.neighbor_size, "Error: ms1 profile"
    ms1_profile = np.array(ms1_intensity_list_middle)

    return spectrum_holder, spectrum_original_forward, spectrum_original_backward, scan_list_middle, scan_list, ms1_profile

  def _parse_spectrum_ion(self, input_file_handle):
    """TODO(nh2tran): docstring."""

    #~ print("".join(["="] * 80)) # section-separating line
    #~ print("WorkerIO: _parse_spectrum_ion()")

    # ion
    mz_list = []
    intensity_list = []
    line = input_file_handle.readline()
    while not "END IONS" in line:
      mz, intensity = re.split(' |\n', line)[:2]
      mz_float = float(mz)
      intensity_float = float(intensity)
      # skip an ion if its mass > MZ_MAX
      if mz_float > self.MZ_MAX:
        line = input_file_handle.readline()
        continue
      mz_list.append(mz_float)
      intensity_list.append(intensity_float)
      line = input_file_handle.readline()

    return mz_list, intensity_list
