import csv
import re


def denovo_setup():
    print("Function denovo_setup was called.")
    # TODO: Write function to automatically download all necessary models, knapsack files, test data.

'''
Function for formatting MGF file is taken from PostNovo

Title: PostNovo
Author: SE Miller / semiller10
Date: 14.Aug. 2019
Version: 1.0.9-alpha
Link: https://github.com/semiller10/postnovo/blob/988b728fad96815fbe94a2bd4dc79ec3b417f099/userargs.py#L1558
'''
def reformat_MGF(mgf_in, mgf_out):
    print("Function reformat_MGF was called.")
    print(mgf_in)
    print(mgf_out)
    new_index = 1
    new_scan = 1
    with open(mgf_in) as in_f, open(mgf_out, 'w') as out_f, open(mgf_out.replace(".mgf", "_deepnovo.mgf"),
                                                                 'w') as out_f2:
        for line in in_f:
            if line == 'BEGIN IONS\n':
                ms2_peak_lines = []
            elif 'TITLE=' == line[:6]:
                split_line = line.split('TITLE=Run: ')[1]
                run_id, split_line = split_line.split(', Index: ')
                old_index, split_line = split_line.split(', Scan: ')
                old_scan = split_line.rstrip('\n')
            elif 'PEPMASS=' == line[:8]:
                # Remove intensity data: DeepNovo only looks for the mass and not intensity.
                if ' ' in line:
                    pepmass_line = line.split(' ')[0] + '\n'
                else:
                    pepmass_line = line
            elif 'CHARGE=' == line[:7]:
                charge_line = line
            elif 'RTINSECONDS=' == line[:12]:
                rt_line = line
            elif 'SEQ=' == line[:4]:
                seq_line = line
            elif line == 'END IONS\n':
                # Avoid peptides without MS2 peaks.
                if len(ms2_peak_lines) > 0:
                    out_f.write('BEGIN IONS\n')
                    out_f.write(
                        'TITLE=Run: ' + run_id + ', Index: ' + str(new_index) + \
                        ', Old index: ' + old_index + ', Old scan: ' + old_scan + '\n')
                    out_f.write(pepmass_line)
                    out_f.write(charge_line)
                    out_f.write('SCANS=' + str(new_scan) + '\n')
                    out_f.write(rt_line)
                    out_f.write(''.join(ms2_peak_lines))
                    out_f.write(line)

                    out_f2.write('BEGIN IONS\n')
                    out_f2.write(
                        'TITLE=Run: ' + run_id + ', Index: ' + str(new_index) + \
                        ', Old index: ' + old_index + ', Old scan: ' + old_scan + '\n')
                    out_f2.write(pepmass_line)
                    out_f2.write(charge_line)
                    out_f2.write('SCANS=' + str(new_scan) + '\n')
                    out_f2.write(rt_line)
                    out_f2.write("SEQ=AAAAAAA\n")
                    out_f2.write(''.join(ms2_peak_lines))
                    out_f2.write(line)
                    new_index += 1
                    new_scan += 1
            else:
                ms2_peak_lines.append(line)
    return


def transfer_mgf(old_mgf_file_name, output_feature_file_name, spectrum_fw):
    with open(old_mgf_file_name, 'r') as fr:
        with open(output_feature_file_name, 'w') as fw:
            writer = csv.writer(fw, delimiter=',')
            header = ["spec_group_id", "m/z", "z", "rt_mean", "seq", "scans", "profile", "feature area"]
            writer.writerow(header)
            flag = False
            for line in fr:
                if "BEGIN ION" in line:
                    flag = True
                    spectrum_fw.write(line)
                elif not flag:
                    spectrum_fw.write(line)
                elif line.startswith("TITLE="):
                    spectrum_fw.write(line)
                elif line.startswith("PEPMASS="):
                    mz = re.split("=|\r|\n", line)[1]
                    spectrum_fw.write(line)
                elif line.startswith("CHARGE="):
                    z = re.split("=|\r|\n|\+", line)[1]
                    spectrum_fw.write("CHARGE=" + z + '\n')
                elif line.startswith("SCANS="):
                    scan = re.split("=|\r|\n", line)[1]
                    spectrum_fw.write(line)
                elif line.startswith("RTINSECONDS="):
                    rt_mean = re.split("=|\r|\n", line)[1]
                    spectrum_fw.write(line)
                elif line.startswith("SEQ="):
                    seq = re.split("=|\r|\n", line)[1]
                elif line.startswith("END IONS"):
                    feature = Feature(spec_id=scan, mz=mz, z=z, rt_mean=rt_mean, seq=seq, scan=scan)
                    writer.writerow(feature.to_list())
                    flag = False
                    del scan
                    del mz
                    del z
                    del rt_mean
                    del seq
                    spectrum_fw.write(line)
                else:
                    spectrum_fw.write(line)
