import sys
import os
from utils.merge import cat_file_mgf, cat_file_feature, split_identified_and_unidentified_features
import csv
import numpy as np
import config
from data_reader import parse_raw_sequence


def compute_neutral_peptide_mass(peptide: list):
    peptide_neutral_mass = config.mass_N_terminus + config.mass_C_terminus
    for aa in peptide:
        peptide_neutral_mass += config.mass_AA[aa]
    return peptide_neutral_mass


def feature_file_mass_correction(feature_filename: str):
    """
    read feature file, find out mass shift then correct
    :param feature_filename:
    :return:
    """
    output_feature_filename = feature_filename + '.mass_corrected'
    ppm_shift = []
    with open(feature_filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        seq_index = header.index("seq")
        mz_index = header.index("m/z")
        z_index = header.index("z")
        for line in reader:
            mz = float(line[mz_index])
            z = float(line[z_index])
            observed_mass = mz * z - z * config.mass_H
            if not line[seq_index]:
                continue
            okay, peptide = parse_raw_sequence(line[seq_index])
            if not okay:
                # unknown mods
                continue
            theoretical_mass = compute_neutral_peptide_mass(peptide)
            ppm = (observed_mass - theoretical_mass) / theoretical_mass * 1e6
            ppm_shift.append(ppm)
    if len(ppm_shift) < 100:
        raise ValueError("too less identified feature for mass correction")
    ppm_shift = np.median(ppm_shift)
    print(f"ppm shift: {ppm_shift}")
    with open(feature_filename, 'r') as fr:
        with open(output_feature_filename, 'w') as fw:
            reader = csv.reader(fr, delimiter=',')
            writer = csv.writer(fw, delimiter=',')
            writer.writerow(next(reader))
            for line in reader:
                mz = float(line[mz_index])
                mz = mz * (1 - ppm_shift * 1e-6)
                line[mz_index] = "{}".format(mz)
                writer.writerow(line)


if __name__ == '__main__':
    prefix_folder_name = sys.argv[1]
    num_fractions = int(sys.argv[2])

    input_mgf_file_list = [os.path.join(prefix_folder_name, 'export_{}.mgf'.format(i)) for i in range(num_fractions)]
    input_feature_file_list = [os.path.join(prefix_folder_name, 'export_{}.csv'.format(i)) for i in range(num_fractions)]
    fraction_list = list(range(num_fractions))
    mgf_output_file = os.path.join(prefix_folder_name, 'spectrum.mgf')
    feature_output_file = os.path.join(prefix_folder_name, 'features.csv')

    # concatenate different fractions
    cat_file_mgf(input_mgf_file_list, fraction_list, mgf_output_file)
    cat_file_feature(input_feature_file_list, fraction_list, feature_output_file)

    # mass correction
    feature_file_mass_correction(feature_output_file)

    # split identified and unidentified features
    split_identified_and_unidentified_features(feature_output_file + '.mass_corrected')
