import os
import torch
from torch.utils.data import Dataset
from db_searcher import DataBaseSearcher
import time
import numpy as np
from typing import List
import pickle
import csv
import re
import logging
from dataclasses import dataclass

import config
from deepnovo_cython_modules import get_ion_index, process_peaks

logger = logging.getLogger(__name__)


def parse_raw_sequence(raw_sequence: str):
    raw_sequence_len = len(raw_sequence)
    peptide = []
    index = 0
    while index < raw_sequence_len:
        if raw_sequence[index] == "(":
            if peptide[-1] == "C" and raw_sequence[index:index + 8] == "(+57.02)":
                peptide[-1] = "C(Carbamidomethylation)"
                index += 8
            elif peptide[-1] == 'M' and raw_sequence[index:index + 8] == "(+15.99)":
                peptide[-1] = 'M(Oxidation)'
                index += 8
            elif peptide[-1] == 'N' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'N(Deamidation)'
                index += 6
            elif peptide[-1] == 'Q' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'Q(Deamidation)'
                index += 6
            elif peptide[-1] == 'S' and raw_sequence[index:index + 8] == "(+79.97)":
                peptide[-1] = "S(Phosphorylation)"
                index += 8
            elif peptide[-1] == 'T' and raw_sequence[index:index + 8] == "(+79.97)":
                peptide[-1] = "T(Phosphorylation)"
                index += 8
            elif peptide[-1] == 'Y' and raw_sequence[index:index + 8] == "(+79.97)":
                peptide[-1] = "Y(Phosphorylation)"
                index += 8
            else:  # unknown modification
                logger.warning(f"unknown modification in seq {raw_sequence}")
                return False, peptide
        else:
            peptide.append(raw_sequence[index])
            index += 1

    for aa in peptide:
        if aa not in config.vocab:
            logger.warning(f"unknown modification in seq {raw_sequence}")
            return False, peptide
    return True, peptide


def to_tensor(data_dict: dict) -> dict:
    temp = [(k, torch.from_numpy(v)) for k, v in data_dict.items()]
    return dict(temp)


def pad_to_length(input_data: list, pad_token, max_length: int) -> list:
    assert len(input_data) <= max_length
    result = input_data[:]
    for i in range(max_length - len(result)):
        result.append(pad_token)
    return result


@dataclass
class DDAFeature:
    feature_id: str
    mz: float
    z: float
    rt_mean: float
    peptide: list
    scan: str
    mass: float
    feature_area: str


@dataclass
class _DenovoData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    spectrum_representation: np.ndarray
    original_dda_feature: DDAFeature


@dataclass
class BatchDenovoData:
    peak_location: torch.Tensor
    peak_intensity: torch.Tensor
    spectrum_representation: torch.Tensor
    original_dda_feature_list: List[DDAFeature]


@dataclass
class TrainData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    spectrum_representation: np.ndarray
    forward_id_target: list
    backward_id_target: list
    forward_ion_location_index_list: list
    backward_ion_location_index_list: list
    forward_id_input: list
    backward_id_input: list


class BaseDataset(Dataset):
    def __init__(self, feature_filename, spectrum_filename, transform=None):
        """
        An abstract class, read all feature information and store in memory,
        :param feature_filename:
        :param spectrum_filename:
        """
        logger.info(f"input spectrum file: {spectrum_filename}")
        logger.info(f"input feature file: {feature_filename}")
        self.spectrum_filename = spectrum_filename
        self.input_spectrum_handle = None
        self.feature_list = []
        self.spectrum_location_dict = {}
        self.transform = transform
        # read spectrum location file
        spectrum_location_file = spectrum_filename + '.location.pytorch.pkl'
        if os.path.exists(spectrum_location_file):
            logger.info(f"read cached spectrum locations")
            with open(spectrum_location_file, 'rb') as fr:
                self.spectrum_location_dict = pickle.load(fr)
        else:
            logger.info("build spectrum location from scratch")
            spectrum_location_dict = {}
            line = True
            with open(spectrum_filename, 'r') as f:
                while line:
                    current_location = f.tell()
                    line = f.readline()
                    if "BEGIN IONS" in line:
                        spectrum_location = current_location
                    elif "SCANS=" in line:
                        scan = re.split('[=\r\n]', line)[1]
                        spectrum_location_dict[scan] = spectrum_location
            self.spectrum_location_dict = spectrum_location_dict
            with open(spectrum_location_file, 'wb') as fw:
                pickle.dump(self.spectrum_location_dict, fw)

        # read feature file
        skipped_by_mass = 0
        skipped_by_ptm = 0
        skipped_by_length = 0
        with open(feature_filename, 'r') as fr:
            reader = csv.reader(fr, delimiter=',')
            header = next(reader)
            feature_id_index = header.index(config.col_feature_id)
            mz_index = header.index(config.col_precursor_mz)
            z_index = header.index(config.col_precursor_charge)
            rt_mean_index = header.index(config.col_rt_mean)
            seq_index = header.index(config.col_raw_sequence)
            scan_index = header.index(config.col_scan_list)
            feature_area_index = header.index(config.col_feature_area)
            for line in reader:
                mass = (float(line[mz_index]) - config.mass_H) * float(line[z_index])
                ok, peptide = parse_raw_sequence(line[seq_index])
                if not ok:
                    skipped_by_ptm += 1
                    logger.debug(f"{line[seq_index]} skipped by ptm")
                    continue
                if mass > config.MZ_MAX:
                    skipped_by_mass += 1
                    logger.debug(f"{line[seq_index]} skipped by mass")
                    continue
                if len(peptide) > config.MAX_LEN - 2:
                    skipped_by_length += 1
                    logger.debug(f"{line[seq_index]} skipped by length")
                    continue
                new_feature = DDAFeature(feature_id=line[feature_id_index],
                                         mz=float(line[mz_index]),
                                         z=float(line[z_index]),
                                         rt_mean=float(line[rt_mean_index]),
                                         peptide=peptide,
                                         scan=line[scan_index],
                                         mass=mass,
                                         feature_area=line[feature_area_index])
                self.feature_list.append(new_feature)
        logger.info(f"read {len(self.feature_list)} features, {skipped_by_mass} skipped by mass, "
                    f"{skipped_by_ptm} skipped by unknown modification, {skipped_by_length} skipped by length")

    def __len__(self):
        return len(self.feature_list)

    def close(self):
        self.input_spectrum_handle.close()

    def _parse_spectrum_ion(self):
        mz_list = []
        intensity_list = []
        line = self.input_spectrum_handle.readline()
        while not "END IONS" in line:
            mz, intensity = re.split(' |\r|\n', line)[:2]
            mz_float = float(mz)
            intensity_float = float(intensity)
            # skip an ion if its mass > MZ_MAX
            if mz_float > config.MZ_MAX:
                line = self.input_spectrum_handle.readline()
                continue
            mz_list.append(mz_float)
            intensity_list.append(intensity_float)
            line = self.input_spectrum_handle.readline()
        return mz_list, intensity_list

    def _get_feature(self, feature: DDAFeature):
        raise NotImplementedError("subclass should implement _get_feature method")

    def __getitem__(self, idx):
        if self.input_spectrum_handle is None:
            self.input_spectrum_handle = open(self.spectrum_filename, 'r')
        feature = self.feature_list[idx]
        return self._get_feature(feature)


class DeepNovoTrainDataset(BaseDataset):
    def _get_feature(self, feature: DDAFeature) -> TrainData:
        spectrum_location = self.spectrum_location_dict[feature.scan]
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
        mz_list, intensity_list = self._parse_spectrum_ion()
        peak_location, peak_intensity, spectrum_representation = process_peaks(mz_list, intensity_list, feature.mass)

        assert np.max(peak_intensity) < 1.0 + 1e-5

        peptide_id_list = [config.vocab[x] for x in feature.peptide]
        forward_id_input = [config.GO_ID] + peptide_id_list
        forward_id_target = peptide_id_list + [config.EOS_ID]
        forward_ion_location_index_list = []
        prefix_mass = 0.
        for i, id in enumerate(forward_id_input):
            prefix_mass += config.mass_ID[id]
            ion_location = get_ion_index(feature.mass, prefix_mass, 0)
            forward_ion_location_index_list.append(ion_location)

        backward_id_input = [config.EOS_ID] + peptide_id_list[::-1]
        backward_id_target = peptide_id_list[::-1] + [config.GO_ID]
        backward_ion_location_index_list = []
        suffix_mass = 0
        for i, id in enumerate(backward_id_input):
            suffix_mass += config.mass_ID[id]
            ion_location = get_ion_index(feature.mass, suffix_mass, 1)
            backward_ion_location_index_list.append(ion_location)

        return TrainData(peak_location=peak_location,
                         peak_intensity=peak_intensity,
                         spectrum_representation=spectrum_representation,
                         forward_id_target=forward_id_target,
                         backward_id_target=backward_id_target,
                         forward_ion_location_index_list=forward_ion_location_index_list,
                         backward_ion_location_index_list=backward_ion_location_index_list,
                         forward_id_input=forward_id_input,
                         backward_id_input=backward_id_input)


def collate_func(train_data_list):
    """

    :param train_data_list: list of TrainData
    :return:
        peak_location: [batch, N]
        peak_intensity: [batch, N]
        forward_target_id: [batch, T]
        backward_target_id: [batch, T]
        forward_ion_index_list: [batch, T, 26, 8]
        backward_ion_index_list: [batch, T, 26, 8]
    """
    # sort data by seq length (decreasing order)
    train_data_list.sort(key=lambda x: len(x.forward_id_target), reverse=True)
    batch_max_seq_len = len(train_data_list[0].forward_id_target)
    ion_index_shape = train_data_list[0].forward_ion_location_index_list[0].shape
    assert ion_index_shape == (config.vocab_size, config.num_ion)

    peak_location = [x.peak_location for x in train_data_list]
    peak_location = np.stack(peak_location) # [batch_size, N]
    peak_location = torch.from_numpy(peak_location)

    peak_intensity = [x.peak_intensity for x in train_data_list]
    peak_intensity = np.stack(peak_intensity) # [batch_size, N]
    peak_intensity = torch.from_numpy(peak_intensity)

    spectrum_representation = [x.spectrum_representation for x in train_data_list]
    spectrum_representation = np.stack(spectrum_representation)  # [batch_size, embed_size]
    spectrum_representation = torch.from_numpy(spectrum_representation)

    batch_forward_ion_index = []
    batch_forward_id_target = []
    batch_forward_id_input = []
    for data in train_data_list:
        ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                               np.float32)
        forward_ion_index = np.stack(data.forward_ion_location_index_list)
        ion_index[:forward_ion_index.shape[0], :, :] = forward_ion_index
        batch_forward_ion_index.append(ion_index)

        f_target = np.zeros((batch_max_seq_len,), np.int64)
        forward_target = np.array(data.forward_id_target, np.int64)
        f_target[:forward_target.shape[0]] = forward_target
        batch_forward_id_target.append(f_target)

        f_input = np.zeros((batch_max_seq_len,), np.int64)
        forward_input = np.array(data.forward_id_input, np.int64)
        f_input[:forward_input.shape[0]] = forward_input
        batch_forward_id_input.append(f_input)



    batch_forward_id_target = torch.from_numpy(np.stack(batch_forward_id_target))  # [batch_size, T]
    batch_forward_ion_index = torch.from_numpy(np.stack(batch_forward_ion_index))  # [batch, T, 26, 8]
    batch_forward_id_input = torch.from_numpy(np.stack(batch_forward_id_input))

    batch_backward_ion_index = []
    batch_backward_id_target = []
    batch_backward_id_input = []
    for data in train_data_list:
        ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                             np.float32)
        backward_ion_index = np.stack(data.backward_ion_location_index_list)
        ion_index[:backward_ion_index.shape[0], :, :] = backward_ion_index
        batch_backward_ion_index.append(ion_index)

        b_target = np.zeros((batch_max_seq_len,), np.int64)
        backward_target = np.array(data.backward_id_target, np.int64)
        b_target[:backward_target.shape[0]] = backward_target
        batch_backward_id_target.append(b_target)

        b_input = np.zeros((batch_max_seq_len,), np.int64)
        backward_input = np.array(data.backward_id_input, np.int64)
        b_input[:backward_input.shape[0]] = backward_input
        batch_backward_id_input.append(b_input)

    batch_backward_id_target = torch.from_numpy(np.stack(batch_backward_id_target))  # [batch_size, T]
    batch_backward_ion_index = torch.from_numpy(np.stack(batch_backward_ion_index))  # [batch, T, 26, 8]
    batch_backward_id_input = torch.from_numpy(np.stack(batch_backward_id_input))

    return (peak_location,
            peak_intensity,
            spectrum_representation,
            batch_forward_id_target,
            batch_backward_id_target,
            batch_forward_ion_index,
            batch_backward_ion_index,
            batch_forward_id_input,
            batch_backward_id_input
            )


# helper functions
def chunks(l, n: int):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class DeepNovoDenovoDataset(DeepNovoTrainDataset):
    # override _get_feature method
    def _get_feature(self, feature: DDAFeature) -> _DenovoData:
        spectrum_location = self.spectrum_location_dict[feature.scan]
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
        mz_list, intensity_list = self._parse_spectrum_ion()
        peak_location, peak_intensity, spectrum_representation = process_peaks(mz_list, intensity_list, feature.mass)

        return _DenovoData(peak_location=peak_location,
                           peak_intensity=peak_intensity,
                           spectrum_representation=spectrum_representation,
                           original_dda_feature=feature)


def denovo_collate_func(data_list: List[_DenovoData]):
    batch_peak_location = np.array([x.peak_location for x in data_list])
    batch_peak_intensity = np.array([x.peak_intensity for x in data_list])
    batch_spectrum_representation = np.array([x.spectrum_representation for x in data_list])

    batch_peak_location = torch.from_numpy(batch_peak_location)
    batch_peak_intensity = torch.from_numpy(batch_peak_intensity)
    batch_spectrum_representation = torch.from_numpy(batch_spectrum_representation)

    original_dda_feature_list = [x.original_dda_feature for x in data_list]

    return BatchDenovoData(batch_peak_location, batch_peak_intensity, batch_spectrum_representation,
                           original_dda_feature_list)


@dataclass
class DBSearchData:
    peak_location: np.ndarray  # (N, )
    peak_intensity: np.ndarray  # (N, )
    forward_id_target: np.ndarray  # (num_candidate, T)
    backward_id_target: np.ndarray  # (num_candidate, T)
    forward_ion_location_index: np.ndarray  # (num_candidate, T, 26, 12)
    backward_ion_location_index: np.ndarray  # (num_candidate, T, 26, 12)
    ppm: np.ndarray  # (num_seq_per_sample,)
    num_var_mod: np.ndarray  # (num_seq_per_sample)
    charge: np.ndarray  # (num_seq_per_sample)
    precursor_mass: float
    peptide_candidates: list  # list of PeptideCandidate
    dda_feature: DDAFeature


class DBSearchDataset(BaseDataset):
    def __init__(self, feature_filename, spectrum_filename, db_searcher: DataBaseSearcher):
        super(DBSearchDataset, self).__init__(feature_filename, spectrum_filename)
        self.db_searcher = db_searcher
        if config.quick_scorer == "num_matched_ions":
            self.quick_scorer = self.get_num_matched_fragment_ions
        elif config.quick_scorer == "peaks_scorer":
            self.quick_scorer = self.peaks_quick_scorer
        else:
            raise ValueError(f"unknown quick_scorer attribute: {config.quick_scorer}")

    @staticmethod
    def peptide_to_aa_id_seq(peptide: list, direction=0):
        """

        :param peptide:
        :param direction: 0 for forward, 1 for backward
        :return:
        """
        if len(peptide) > config.MAX_LEN - 2:
            raise ValueError(f"received a peptide longer than {config.MAX_LEN}")
        aa_id_seq = [config.vocab[aa] for aa in peptide]
        aa_id_seq.insert(0, config.GO_ID)
        aa_id_seq.append(config.EOS_ID)
        if direction != 0:
            aa_id_seq = aa_id_seq[::-1]
        aa_id_seq = pad_to_length(aa_id_seq, config.PAD_ID, config.MAX_LEN)
        return aa_id_seq

    def _get_feature(self, feature: DDAFeature):
        start_time = time.time()
        spectrum_location = self.spectrum_location_dict[feature.scan]
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
        mz_list, intensity_list = self._parse_spectrum_ion()
        ## empty spectrum
        if not mz_list:
            return None
        peak_location, peak_intensity, _ = process_peaks(mz_list, intensity_list, feature.mass)
        precursor_mass = feature.mass

        candidate_list = self.db_searcher.search_peptide_by_mass(precursor_mass, pad_with_random_permutation=True)

        if len(candidate_list) == 0:
            #  no candidates
            return None

        if len(candidate_list) > config.normalizing_std_n:
            quick_scores = [self.quick_scorer(feature.mass, peak_location, peak_intensity, pc.seq) for pc in candidate_list]
            quick_scores = np.array(quick_scores)
            top_k_ind = np.argpartition(quick_scores, -config.normalizing_std_n)[-config.normalizing_std_n:]

            top_candidate_list = []
            for ind in top_k_ind:
                top_candidate_list.append(candidate_list[ind])
            candidate_list = top_candidate_list

        assert len(candidate_list) == config.normalizing_std_n

        forward_id_target_arr = []
        backward_id_target_arr = []
        forward_ion_location_index_arr = []
        backward_ion_location_index_arr = []
        ppm_arr = []
        num_var_mod_arr = []
        charge_arr = feature.z * np.ones(len(candidate_list), dtype=np.float32)

        for pc in candidate_list:
            ppm_arr.append(pc.ppm)
            num_var_mod_arr.append(pc.num_var_mod)

            peptide_id_list = [config.vocab[x] for x in pc.seq]
            forward_id_input = [config.GO_ID] + peptide_id_list
            forward_id_target = peptide_id_list + [config.EOS_ID]
            forward_ion_location_index_list = []
            prefix_mass = 0.
            for i, id in enumerate(forward_id_input):
                prefix_mass += config.mass_ID[id]
                ion_location = get_ion_index(feature.mass, prefix_mass, 0)
                forward_ion_location_index_list.append(ion_location)

            backward_id_input = [config.EOS_ID] + peptide_id_list[::-1]
            backward_id_target = peptide_id_list[::-1] + [config.GO_ID]
            backward_ion_location_index_list = []
            suffix_mass = 0
            for i, id in enumerate(backward_id_input):
                suffix_mass += config.mass_ID[id]
                ion_location = get_ion_index(feature.mass, suffix_mass, 1)
                backward_ion_location_index_list.append(ion_location)

            forward_id_target_arr.append(forward_id_target)
            backward_id_target_arr.append(backward_id_target)
            forward_ion_location_index_arr.append(forward_ion_location_index_list)
            backward_ion_location_index_arr.append(backward_ion_location_index_list)  # nested_list

        # assemble data in the way in collate function.
        ion_index_shape = forward_ion_location_index_list[0].shape
        assert ion_index_shape == (config.vocab_size, config.num_ion)
        batch_max_seq_len = max([len(x) for x in forward_id_target_arr])
        num_candidates = len(forward_id_target_arr)
        if batch_max_seq_len > 50:
            logger.warning(f"feature {feature} has sequence candidate longer than 50")

        # batch forward data
        batch_forward_ion_index = []
        batch_forward_id_target = []
        for ii, forward_id_target in enumerate(forward_id_target_arr):
            ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                                 np.float32)
            forward_ion_index = np.stack(forward_ion_location_index_arr[ii])
            ion_index[:forward_ion_index.shape[0], :, :] = forward_ion_index
            batch_forward_ion_index.append(ion_index)

            f_target = np.zeros((batch_max_seq_len,), np.int64)
            forward_target = np.array(forward_id_target, np.int64)
            f_target[:forward_target.shape[0]] = forward_target
            batch_forward_id_target.append(f_target)

        # batch backward data
        batch_backward_ion_index = []
        batch_backward_id_target = []
        for ii, backward_id_target in enumerate(backward_id_target_arr):
            ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                                 np.float32)
            backward_ion_index = np.stack(backward_ion_location_index_arr[ii])
            ion_index[:backward_ion_index.shape[0], :, :] = backward_ion_index
            batch_backward_ion_index.append(ion_index)

            b_target = np.zeros((batch_max_seq_len,), np.int64)
            backward_target = np.array(backward_id_target, np.int64)
            b_target[:backward_target.shape[0]] = backward_target
            batch_backward_id_target.append(b_target)

        batch_forward_id_target = np.stack(batch_forward_id_target)
        batch_forward_ion_index = np.stack(batch_forward_ion_index)
        batch_backward_id_target = np.stack(batch_backward_id_target)
        batch_backward_ion_index = np.stack(batch_backward_ion_index)


        ppm_arr = np.array(ppm_arr, dtype=np.float32)
        num_var_mod_arr = np.array(num_var_mod_arr, dtype=np.float32)

        duration = time.time() - start_time
        logger.debug(f"a feature takes {duration} seconds to process.")
        return DBSearchData(peak_location, peak_intensity, batch_forward_id_target, batch_backward_id_target,
            batch_forward_ion_index, batch_backward_ion_index, ppm_arr,
            num_var_mod_arr, charge_arr, precursor_mass, candidate_list, feature)

    @staticmethod
    def get_fragment_ion_location(precursor_neutral_mass, prefix_mass):
        """

        :param precursor_neutral_mass: float number
        :param prefix_mass: float number
        :return: theoretical mass location, an array of 3 elems.
        """
        b_mass = prefix_mass
        y_mass = precursor_neutral_mass - b_mass
        a_mass = b_mass - config.mass_CO

        ion_list = [b_mass, y_mass, a_mass]
        theoretical_location = np.asarray(ion_list, dtype=np.float32)
        return theoretical_location

    @classmethod
    def get_num_matched_fragment_ions(cls, precursor_mass, peaks_location, peaks_intensity, seq):
        """

        :param peaks_location: the spectrum m/z location array
        :param seq: attribute of PeptideCandidate
        :return: num_matched_ions
        """
        peaks_location = np.expand_dims(peaks_location, axis=1)
        peptide_id_list = [config.vocab[x] for x in seq]
        forward_id_input = [config.GO_ID] + peptide_id_list
        prefix_mass = 0.
        num_matched_ions = 0
        for i, id in enumerate(forward_id_input):
            prefix_mass += config.mass_ID[id]
            ion_location = cls.get_fragment_ion_location(precursor_mass, prefix_mass)  # [3]
            ion_location = np.expand_dims(ion_location, axis=0)

            # diff matrix
            mz_diff = np.abs(peaks_location - ion_location)
            mz_diff = np.any(mz_diff < config.fragment_ion_mz_diff_threshold)
            num_matched_ions += mz_diff.astype(np.int32)
        return num_matched_ions

    @classmethod
    def peaks_quick_scorer(cls, precursor_mass, peaks_location, peaks_intensity, seq):
        peaks_location = np.expand_dims(peaks_location, axis=1)
        peaks_intensity = np.log(1 + 10 * peaks_intensity)  # use log intensity
        peptide_id_list = [config.vocab[x] for x in seq]
        forward_id_input = [config.GO_ID] + peptide_id_list
        prefix_mass = 0.
        score = 0
        for i, id in enumerate(forward_id_input):
            prefix_mass += config.mass_ID[id]
            ion_location = cls.get_fragment_ion_location(precursor_mass, prefix_mass)  # [3]
            ion_location = np.expand_dims(ion_location, axis=0)

            # diff matrix
            mz_diff = np.abs(peaks_location - ion_location)  ## [N, 3]
            score_vec = np.max(np.exp(-np.square(mz_diff*100)), axis=1)  # each observed peak can only be explained by one ion type
            score += np.sum(score_vec * peaks_intensity)
        return score
