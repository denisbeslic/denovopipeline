from dataclasses import dataclass
from data_reader import DDAFeature
import config
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BeamSearchedSequence:
    sequence: list  # list of aa id
    position_score: list
    score: float  # average by length score


class DenovoWriter(object):
    def __init__(self, denovo_output_file):
        self.output_handle = open(denovo_output_file, 'w')
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
        print(header_row, file=self.output_handle, end='\n')

    def close(self):
        self.output_handle.close()

    def write(self, dda_original_feature: DDAFeature, searched_sequence: BeamSearchedSequence):
        """
        keep the output in the same format with the tensorflow version
        :param dda_original_feature:
        :param searched_sequence:
        :return:
        """
        feature_id = dda_original_feature.feature_id
        feature_area = dda_original_feature.feature_area
        precursor_mz = str(dda_original_feature.mz)
        precursor_charge = str(dda_original_feature.z)
        scan_list_middle = dda_original_feature.scan
        scan_list_original = dda_original_feature.scan
        if searched_sequence.sequence:
            predicted_sequence = ','.join([config.vocab_reverse[aa_id] for
                                           aa_id in searched_sequence.sequence])
            predicted_score = "{:.2f}".format(searched_sequence.score)
            predicted_score_max = predicted_score
            predicted_position_score = ','.join(['{0:.2f}'.format(x) for x in searched_sequence.position_score])
            protein_access_id = 'DENOVO'
        else:
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

    def __del__(self):
        self.close()


@dataclass
class PSM:
    feature_id: str
    scan: str
    num_id: int
    exp_mass: float
    calc_mass: float
    charge: int
    peptide_str: str
    accession_id: str
    is_decoy: bool
    length_score: float
    log_length_score: float
    length_normalized_score: float
    log_length_normalized_score: float
    ppm: float
    peptide_length: int
    num_var_mod: int


class PercolatorWriter(object):
    def __init__(self, denovo_output_file):
        self.output_handle = open(denovo_output_file, 'w')
        header_list = ["FeatureID",  # 0
                       "Label",  # 1
                       "ScanNr",  # 2
                       "ExpMass",  # 3
                       "CalcMass",  # 4
                       "LengthScore",  # 5
                       "LengthNormalizedScore",  # 6
                       "LogLengthScore",  # 7
                       "LogLengthNormalizedScore",  # 8
                       "PpmAbsDiff",  # 9
                       "PepLen",  # 10
                       "Charge1",  # 11
                       "Charge2",  # 12
                       "Charge3",  # 13
                       "Charge4",  # 14
                       "Charge5",  # 15
                       "Charge6",  # 16
                       "NumVarMod",  # 17
                       "Peptide",  # 18
                       "Proteins",  # 19
                       ]
        header_row = "\t".join(header_list)
        print(header_row, file=self.output_handle, end='\n')
        self.scan_nr_counter = 1

    def close(self):
        self.output_handle.close()

    def write(self, psm: PSM):
        feature_id = psm.feature_id + '_' + str(psm.num_id)
        if psm.is_decoy:
            label = '-1'
        else:
            label= '1'
        # label = str(psm.is_decoy is False)
        scan_nr = f"{self.scan_nr_counter}"
        self.scan_nr_counter += 1
        exp_mass = "{:.4f}".format(psm.exp_mass)
        calc_mass = "{:.4f}".format(psm.calc_mass)
        length_score = "{:.4f}".format(psm.length_score)
        length_normalized_score = "{:.4f}".format(psm.length_normalized_score)
        log_length_score = "{:.4f}".format(psm.log_length_score)
        log_length_normalized_score = "{:.4f}".format(psm.log_length_normalized_score)
        ppm = "{:.4f}".format(np.abs(psm.ppm))
        pep_len = str(psm.peptide_length)

        print_list = [feature_id,
                      label,
                      scan_nr,
                      exp_mass,
                      calc_mass,
                      length_score,
                      length_normalized_score,
                      log_length_score,
                      log_length_normalized_score,
                      ppm,
                      pep_len]

        charge = ["0"] * 6
        charge_index = min(psm.charge - 1, 5)
        charge[charge_index] = "1"
        print_list += charge

        print_list.append(str(psm.num_var_mod))
        print_list.append(psm.peptide_str)
        print_list.append(psm.accession_id)
        print("\t".join(print_list), file=self.output_handle, end="\n")
