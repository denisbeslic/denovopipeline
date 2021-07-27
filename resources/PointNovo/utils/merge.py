import os
import re
import numpy as np

col_feature_id = 0
col_scan_list = 5
col_raw_sequence = 4


def cat_file_mgf(input_file_list, fraction_list, output_file):
    print("cat_file_mgf()")

    counter = 0
    with open(output_file, mode="w") as output_handle:
        for index, input_file in enumerate(input_file_list):
            print("input_file = ", os.path.join(input_file))
            with open(input_file, mode="r") as input_handle:
                line = input_handle.readline()
                while line:
                    if "SCANS=" in line:  # a spectrum found
                        counter += 1
                        scan = re.split('=|\n', line)[1]
                        # re-number scan id
                        output_handle.write("SCANS=F{0}:{1}\n".format(fraction_list[index], scan))
                    else:
                        output_handle.write(line)
                    line = input_handle.readline()

    print("output_file = {0:s}".format(output_file))
    print("counter = {0:d}".format(counter))


def cat_file_feature(input_file_list, fraction_list, output_file):
    print("cat_file_feature()")

    counter = 0
    with open(output_file, mode="w") as output_handle:
        for index, input_file in enumerate(input_file_list):
            print("input_file = ", os.path.join(input_file))
            with open(input_file, mode="r") as input_handle:
                header_line = input_handle.readline()
                if index == 0:
                    output_handle.write(header_line)
                line = input_handle.readline()
                while line:
                    counter += 1
                    line = re.split(',|\r|\n', line)
                    # add fraction to feature id
                    feature_id = line[col_feature_id]
                    feature_id = "F" + str(fraction_list[index]) + ":" + feature_id
                    line[col_feature_id] = feature_id
                    # add fraction to scan id
                    scan_list = re.split(';', line[col_scan_list])
                    scan_list = ["F" + str(fraction_list[index]) + ":" + x for x in scan_list]
                    line[col_scan_list] = ";".join(scan_list)
                    # join the line back together and write to output
                    output_handle.write(",".join(line) + "\n")
                    line = input_handle.readline()

    print("output_file = {0:s}".format(output_file))
    print("counter = {0:d}".format(counter))


def split_identified_and_unidentified_features(feature_file_name: str,
                                               output_identified_file_name=None,
                                               output_unidentified_file_name=None):
    if output_identified_file_name is None:
        output_identified_file_name = feature_file_name + '.identified'
    if output_unidentified_file_name is None:
        output_unidentified_file_name = feature_file_name + '.unidentified'
    id_handle = open(output_identified_file_name, 'w')
    unid_handle = open(output_unidentified_file_name, 'w')

    with open(feature_file_name, 'r') as f:
        line = f.readline()
        # write header
        id_handle.write(line)
        unid_handle.write(line)
        for line in f:
            seq = line.split(',')[col_raw_sequence]
            if seq:
                id_handle.write(line)
            else:
                unid_handle.write(line)


def partition_feature_file_nodup(input_feature_file, prob):
    print("partition_feature_file_nodup()")

    print("input_feature_file = ", os.path.join(input_feature_file))
    print("prob = ", prob)

    output_file_train = input_feature_file + ".train" + ".nodup"
    output_file_valid = input_feature_file + ".valid" + ".nodup"
    output_file_test = input_feature_file + ".test" + ".nodup"

    peptide_train_list = []
    peptide_valid_list = []
    peptide_test_list = []

    with open(input_feature_file, mode="r") as input_handle:
        with open(output_file_train, mode="w") as output_handle_train:
            with open(output_file_valid, mode="w") as output_handle_valid:
                with open(output_file_test, mode="w") as output_handle_test:
                    counter = 0
                    counter_train = 0
                    counter_valid = 0
                    counter_test = 0
                    counter_unique = 0
                    # header line
                    line = input_handle.readline()
                    output_handle_train.write(line)
                    output_handle_valid.write(line)
                    output_handle_test.write(line)
                    # first feature
                    line = input_handle.readline()
                    while line:
                        counter += 1
                        # check if the peptide already exists in any of the three lists
                        # if yes, this new feature will be assigned to that list
                        peptide = re.split(',|\r|\n', line)[4]
                        if (peptide in peptide_train_list):
                            output_handle = output_handle_train
                            counter_train += 1
                        elif (peptide in peptide_valid_list):
                            output_handle = output_handle_valid
                            counter_valid += 1
                        elif (peptide in peptide_test_list):
                            output_handle = output_handle_test
                            counter_test += 1
                        # if not, this new peptide and its spectrum will be randomly assigned
                        else:
                            counter_unique += 1
                            set_num = np.random.choice(a=3, size=1, p=prob)
                            if set_num == 0:
                                peptide_train_list.append(peptide)
                                output_handle = output_handle_train
                                counter_train += 1
                            elif set_num == 1:
                                peptide_valid_list.append(peptide)
                                output_handle = output_handle_valid
                                counter_valid += 1
                            else:
                                peptide_test_list.append(peptide)
                                output_handle = output_handle_test
                                counter_test += 1
                        output_handle.write(line)
                        line = input_handle.readline()

    input_handle.close()
    output_handle_train.close()
    output_handle_valid.close()
    output_handle_test.close()

    print("counter = {0:d}".format(counter))
    print("counter_train = {0:d}".format(counter_train))
    print("counter_valid = {0:d}".format(counter_valid))
    print("counter_test = {0:d}".format(counter_test))
    print("counter_unique = {0:d}".format(counter_unique))


