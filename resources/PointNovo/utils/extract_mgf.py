from data_reader import BaseDataset, DDAFeature
from typing import List
import csv


class SimpleDataset(BaseDataset):
    """
    a simple implementation of the abstract class
    """
    def _get_feature(self, feature: DDAFeature):
        return None


def extract_mgf_file(feature_file_name, spectrum_file_name, output_spectrum_file_name):
    """
    read all valid features from feature_file, location their spectrum in the input mgf file and output them into
    a new mgf
    :param feature_file_name:
    :param spectrum_file_name:
    :param output_spectrum_file_name:
    :return:
    """
    dataset = SimpleDataset(feature_file_name, spectrum_file_name)
    feature_list: List[DDAFeature] = dataset.feature_list
    with open(dataset.spectrum_filename, 'r') as fr:
        with open(output_spectrum_file_name, 'w') as fw:
            for feature in feature_list:
                spectrum_location = dataset.spectrum_location_dict[feature.scan]
                fr.seek(spectrum_location)
                for line in fr:
                    fw.write(line)
                    if "END IONS" in line:
                        fw.write("\n")
                        break


if __name__ == '__main__':
    data_path = "/home/rui/work/DeepNovo-pytorch/Lumos_data/PXD010559/"
    feature_file_name = data_path + "features.csv.mass_corrected.identified.test.nodup"
    spectrum_file_name = data_path + "spectrum.mgf"
    output_spectrum_file_name = data_path + "test_features_spectrum.mgf"
    extract_mgf_file(feature_file_name, spectrum_file_name, output_spectrum_file_name)



