# change the old deepnovo data to new format
import csv
import re
from dataclasses import dataclass

#species_name = "yeast"

#folder_name = "MSV000081382/cross.9high_80k.exclude_{}/".format(species_name)
#denovo_folder_name = "MSV000081382/high.{}.PXD003868/".format(species_name)

folder_name = "/home/dbeslic/master/DeepLearning_TrainingData/03_NIST_HCD/"
#denovo_folder_name = "human_test/"


train_mgf_file = folder_name + 'NIST_transformed_train.mgf'
valid_mgf_file = folder_name + 'NIST_transformed_valid.mgf'
test_mgf_file = folder_name + 'NIST_transformed_test.mgf'

#denovo_mgf_file = denovo_folder_name + 'peaks.db.mgf'


output_mgf_file = folder_name + 'spectrum.mgf'
output_train_feature_file = folder_name + 'features.train.csv'
output_valid_feature_file = folder_name + 'features.valid.csv'
output_test_feature_file = folder_name + 'features.test.csv'

#denovo_output_feature_file = denovo_folder_name + 'features.csv'
#denovo_spectrum_fw = open(denovo_folder_name + "spectrum.mgf", 'w')

spectrum_fw = open(output_mgf_file, 'w')


@dataclass
class Feature:
    spec_id: str
    mz: str
    z: str
    rt_mean: str
    seq: str
    scan: str

    def to_list(self):
        return [self.spec_id, self.mz, self.z, self.rt_mean, self.seq, self.scan, "0.0:1.0", "1.0"]


def transfer_mgf(old_mgf_file_name, output_feature_file_name, spectrum_fw=spectrum_fw):
    with open(old_mgf_file_name, 'r') as fr:
        with open(output_feature_file_name, 'w') as fw:
            writer = csv.writer(fw, delimiter=',')
            header = ["spec_group_id","m/z","z","rt_mean","seq","scans","profile","feature area"]
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


transfer_mgf(train_mgf_file, output_train_feature_file)
transfer_mgf(valid_mgf_file, output_valid_feature_file)
transfer_mgf(test_mgf_file, output_test_feature_file)

#transfer_mgf(denovo_mgf_file, denovo_output_feature_file, denovo_spectrum_fw)


#denovo_spectrum_fw.close()
spectrum_fw.close()
