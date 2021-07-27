''' Parse the training data for Deep Learning tools SMSNet, DeepNovo and PointNovo '''
import random
from pyteomics import mgf, mass
import csv
import re
from dataclasses import dataclass

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

def transfer_mgf(old_mgf_file_name, output_feature_file_name, spectrum_fw):
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

'''
Take NISTHCD and transform correctly for PointNovo Training
'''
def readNISTHCDmgf(fn):
    with open(fn, 'r') as fr:
        with open(fn.replace(".mgf","_transformed.mgf"), 'w') as spectrum_new:
            i = 0
            all_seqs = []
            for line in fr:
                if True:
                    if "BEGIN IONS" in line:
                        i = i + 1
                        spectrum_new.write(line)
                    elif "Title" in line:
                        #print(line)
                        true_seq = line.rpartition("=")[-1].replace("M(O)", "M(+15.99)").replace("C","C(+57.02)").replace("(P)","P")
                        all_seqs.append(true_seq)
                        spectrum_new.write("TITLE= Run: "+true_seq.replace("\n","") + ", Index: " +str(i)+"\n")
                    elif "PEPMASS" in line:
                        pepmass = line
                        #spectrum_new.write(line)
                    elif "CHARGE" in line:
                        charge = line
                        scans =  "SCANS="+str(i)+"\n"
                        #spectrum_new.write(line)
                        #spectrum_new.write("SCANS="+str(i)+"\n")
                    elif "HCD" in line:
                        spectrum_new.write(pepmass)
                        spectrum_new.write(charge)
                        spectrum_new.write(scans)
                        spectrum_new.write("RTINSECONDS=555.55\n")
                        spectrum_new.write("SEQ="+true_seq)
                    elif "END IONS" in line:
                        spectrum_new.write(line)
                    else:
                        #spectrum_new.write(line)
                        spectrum_new.write(line.rpartition("\t")[0]+"\n")

'''
Reads massive mgf and transforms it for training with PointNovo  
'''
def readMassIVEmgf(fn):
    mgf_out = open(fn.replace(".mgf", "_transformed.mgf"), "w")
    sps = mgf.read(fn, convert_arrays=1, read_charges=False,
                   dtype='float32', use_index=False)
    for sp in sps:
        MODS = ["+43.006", "+42.011", "-17.027"]
        param = sp['params']
        #check that none of mods are in the file
        if any(x in param["seq"] for x in MODS):
            None
        else:
        #if "+" in param['seq']: print(param['seq'])
            mgf_out.write("BEGIN IONS\n")
            mgf_out.write("TITLE=" + param['filename'] + "\n")
            mgf_out.write("PEPMASS=" + str(param['pepmass'][0]) + "\n")
            mgf_out.write("CHARGE=" + str(param['charge']) + "\n")
            mgf_out.write("SCANS=" + param['scans'] + "\n")
            mgf_out.write("RTINSECONDS=555.55\n")
            seq = param['seq'].replace("M+15.995", "M(+15.99)").replace("C","C(+57.02)").replace("Q+0.984","Q(+.98)").replace("N+0.984","N(+.98)")
            #if "+" in seq: print(seq)
            mgf_out.write("SEQ=" + seq + "\n")
            for mz, intensity in zip(sp['m/z array'], sp['intensity array']):
                mgf_out.write(str(mz) + " " + str(intensity) + "\n")
            mgf_out.write("END IONS\n")

def readNISTmab_mgf(fn):
    #mgf_out = open(fn.replace(".mgf", "_transformed.mgf"), "w")
    sps = mgf.read(fn, convert_arrays=1, read_charges=False,
                   dtype='float32', use_index=False)
    with open(fn, 'r') as fr:
        with open(fn.replace(".mgf","_transformed.mgf"), 'w') as spectrum_new:
            i = 0
            all_seqs = []
            for line in fr:
                #print(line)
                if "Name:" in line:
                    i +=1
                    #print(line.split("/")[0].split("Name: ")[1])
                    sp_title = "Run"+line.split("/")[0].split("Name: ")[1]+".Index"+ str(i)
                    sp_seq = line.split("/")[0].split("Name: ")[1].replace("C","C(+57.02)")
                    None
                elif "Comment:" in line:
                    #print(line)
                    sp_charge = line.split("Charge=")[1].split(" Parent=")[0]
                    sp_pepmass = line.split("d Full ms2 ")[1].split("@hcd")[0]
                    None
                elif "MW" in line:
                    #print(line)
                    None
                elif "Num peaks" in line:
                    #print(line)
                    #spectrum_new.write("END IONS\n\n")
                    spectrum_new.write("BEGIN IONS\n")
                    spectrum_new.write("TITLE=" + str(sp_title) + "\n")
                    spectrum_new.write("PEPMASS=" + str(sp_pepmass) + "\n")
                    spectrum_new.write("CHARGE=" + str(sp_charge) + "\n")
                    spectrum_new.write("SCANS=" + str(i) + "\n")
                    spectrum_new.write("RTINSECONDS=555.55\n")
                    spectrum_new.write("SEQ=" + str(sp_seq) + "\n")
                    None
                elif line.startswith("\n"):
                    spectrum_new.write("END IONS\n\n")
                    None
                else:
                    #print(line.split("\t"))
                    if len(line.split("\t")) > 1:
                        spectrum_new.write(str(round(float(line.split("\t")[0]),2)) + " " + str(round(float(line.split("\t")[1])))+"\n")
                    #print(line.split("\t")[0])
                    #print(line.split("\t")[1])
                    None


def readINSILICO_mgf(fn):
    mgf_out = open(fn.replace(".mgf", "_transformed12K.mgf"), "w")
    sps = mgf.read(fn, convert_arrays=1, read_charges=False,
                   dtype='float32', use_index=False)
    i = 0
    for sp in sps:
        MODS = ["+43.006", "+42.011", "-17.027"]
        param = sp['params']
        i += 1
        # check that none of mods are in the file
        if any(x in param["peptide"] for x in MODS):
            None
        elif (i < 12000):
            # if "+" in param['seq']: print(param['seq'])
            mgf_out.write("BEGIN IONS\n")
            mgf_out.write("TITLE=" + param['title']+ ". Index: " +  str(i) +"\n")
            mgf_out.write("PEPMASS=" + str(param['pepmass'][0]) + "\n")
            mgf_out.write("CHARGE=" + str(param['charge']) + "\n")
            mgf_out.write("SCANS=" + str(i) + "\n")
            mgf_out.write("RTINSECONDS=555.55\n")
            seq = param['peptide'].replace("M+15.995", "M(+15.99)").replace("C", "C(+57.02)").replace("Q+0.984",
                                                                                                  "Q(+.98)").replace(
                "N+0.984", "N(+.98)")
            # if "+" in seq: print(seq)
            mgf_out.write("SEQ=" + seq + "\n")
            for mz, intensity in zip(sp['m/z array'], sp['intensity array']):
                mgf_out.write(str(mz) + " " + str(intensity) + "\n")
            mgf_out.write("END IONS\n")


def splitForTraining2(fn):
    file1 = open(fn, "r")
    valid_f = open(fn.replace(".mgf","_valid.mgf"), "w")
    test_f = open(fn.replace(".mgf", "_test.mgf"), "w")
    training_f = open(fn.replace(".mgf", "_train.mgf"), "w")
    sps = mgf.read(file1, convert_arrays=1, read_charges=False,
                    dtype='float32', use_index=False)
    seq_all= []
    i = 0

    list_of_spectras = []
    for sp in sps:
        list_of_spectras.append(sp)
    print(list_of_spectras[0:2])
    random.shuffle(list_of_spectras)
    print(list_of_spectras[0:2])


    for sp in list_of_spectras:
        param = sp['params']
        if 'seq' in param:
            seq_all.append(param['seq'])
            i = i + 1

    # take all double values out
    # important because valid, test, train should not share any peptides with each other
    seq_all = list(dict.fromkeys(seq_all))
    seq_all_length = int(len(seq_all)/10) #10%
    #seq_all_length = int(len(seq_all)/100) #1%
    #seq_all_length = 10000 #set to 20000 smaller set

    random.seed(385)
    valid_and_test_list = (random.sample(seq_all, seq_all_length*2))
    test_list = (random.sample(valid_and_test_list, seq_all_length))
    test_list = set(test_list)
    valid_list = [x for x in valid_and_test_list if x not in test_list]
    valid_list = set(valid_list)

    file1.close()
    file1 = open(fn, "r")
    sps = mgf.read(file1, convert_arrays=1, read_charges=False,
                   dtype='float32', use_index=False)
    for sp in list_of_spectras:
        param = sp['params']
        if 'seq' in param and len(sp['m/z array'])>0:
            # transform valid_list to set to speed up comparison?
            if param['seq'] in valid_list:
                valid_f.write("BEGIN IONS\n")
                valid_f.write("TITLE="+param['title']+"\n")
                valid_f.write("PEPMASS="+str(param['pepmass'][0])+"\n")
                valid_f.write("CHARGE="+str(param['charge'])+"\n")
                valid_f.write("SCANS="+param['scans']+"\n")
                valid_f.write("RTINSECONDS="+str(param['rtinseconds'])+"\n")
                valid_f.write("SEQ="+param['seq']+"\n")
                for mz, intensity in zip(sp['m/z array'], sp['intensity array']):
                    valid_f.write(str(mz) + " " + str(intensity) + "\n")
                valid_f.write("END IONS\n")
            elif param['seq'] in test_list:
                test_f.write("BEGIN IONS\n")
                test_f.write("TITLE=" + param['title'] + "\n")
                test_f.write("PEPMASS=" + str(param['pepmass'][0]) + "\n")
                test_f.write("CHARGE=" + str(param['charge']) + "\n")
                test_f.write("SCANS=" + param['scans'] + "\n")
                test_f.write("RTINSECONDS=" + str(param['rtinseconds']) + "\n")
                test_f.write("SEQ=" + param['seq'] + "\n")
                for mz, intensity in zip(sp['m/z array'], sp['intensity array']):
                    test_f.write(str(mz) + " " + str(intensity) + "\n")
                test_f.write("END IONS\n")
            else:
                training_f.write("BEGIN IONS\n")
                training_f.write("TITLE=" + param['title'] + "\n")
                training_f.write("PEPMASS=" + str(param['pepmass'][0]) + "\n")
                training_f.write("CHARGE=" + str(param['charge']) + "\n")
                training_f.write("SCANS=" + param['scans'] + "\n")
                training_f.write("RTINSECONDS=" + str(param['rtinseconds']) + "\n")
                training_f.write("SEQ=" + param['seq'] + "\n")
                for mz, intensity in zip(sp['m/z array'], sp['intensity array']):
                    training_f.write(str(mz) + " " + str(intensity) + "\n")
                training_f.write("END IONS\n")

    valid_f.close()
    test_f.close()
    training_f.close()

    '''print(param)

        if 'seq' in param:
            pep = param['seq']
        else:
            pep = param['title']
        if 'pepmass' in param:
            mass = param['pepmass'][0]
        else:
            mass = float(param['parent'])

        mz = sp['m/z array']
        it = sp['intensity array']

        for mass, inten in zip(mz,it):
            mgf_file.write(str(mz) + ' ' + str(intensity) + "\n")
        mgf_file.write("END IONS" + "\n")

        db.append({'pep': pep, 'charge': c, 'it': it})

    #print(db)'''

def trainingSplitToSMSNet(fn):
    csv_out = open(fn.replace(".mgf", "0.csv"), "w")
    csv_tgt_out = open(fn.replace(".mgf", "0_tgt.csv"), "w")
    sps = mgf.read(fn, convert_arrays=1, read_charges=False,
                   dtype='float32', use_index=False)

    csv_index = 0
    spectra_index = 0


    no_double_seqs = set()
    list_of_spectras = []
    # append it to list so we can use random.shuffle
    '''for sp in sps:
        list_of_spectras.append(sp)
    print(list_of_spectras[0:2])
    random.shuffle(list_of_spectras)
    print(list_of_spectras[0:2])'''

    #random.shuffle(sps)



    for sp in sps:
        param = sp['params']
        #if param['seq'] in no_double_seqs:
        #    continue
        #no_double_seqs.add(param['seq'])
        if spectra_index >= 50000: # only
            csv_out.close()
            csv_tgt_out.close()
            spectra_index = 0
            csv_index += 1
            csv_out = open(fn.replace(".mgf", str(csv_index)+".csv"), "w")
            csv_tgt_out = open(fn.replace(".mgf", str(csv_index)+"_tgt.csv"), "w")
        if len(sp['m/z array']) > 0:
            param['seq'] = param['seq'].replace("M(+15.99)", "m").replace("C(+57.02)", "C").replace("Q(+.98)", "q").replace("N(+.98)", "n")
            csv_out.write(param['seq']+"|"+str(param['charge']).replace("+", "")+"|"+str(param['pepmass'][0])+"|")
            for mz, intensity in zip(sp['m/z array'], sp['intensity array']):
                csv_out.write(str(mz) + "," + str(intensity)+",")
            csv_out.write("\n")
            csv_tgt_out.write(' '.join(param['seq'])+"\n")
            spectra_index += 1



#readNISTHCDmgf("/home/dbeslic/master/DeepLearning_TrainingData/03_NIST_HCD/NIST.mgf")
#splitForTraining2("/home/dbeslic/master/DeepLearning_TrainingData/03_NIST_HCD/NIST_transformed.mgf")

#readMassIVEmgf("/home/dbeslic/master/DeepLearning_TrainingData/05_MassIVE_HCD/MassIVE.mgf")
#splitForTraining2("/home/dbeslic/master/DeepLearning_TrainingData/05_MassIVE_HCD/MassIVE_transformed.mgf")

#readNISTmab_mgf("/home/dbeslic/master/DeepLearning_TrainingData/02_NISTmAb/ONLY_NistMab/NISTmAb_v20190711.mgf")
#splitForTraining2("/home/dbeslic/master/DeepLearning_TrainingData/02_NISTmAb/ONLY_NistMab/NISTmAb_v20190711_transformed.mgf")

#readINSILICO_mgf("/home/dbeslic/master/DeepLearning_TrainingData/01_simulatedSet/proteaseguru_predfull.mgf")
#splitForTraining2("/home/dbeslic/master/DeepLearning_TrainingData/02_NISTmAb/NISTmAb_12Ksimulated_transformed.mgf")

trainingSplitToSMSNet("/home/dbeslic/master/DeepLearning_TrainingData/06_MassIVE_HCD_smallerValidation/MassIVE_transformed_train.mgf")
trainingSplitToSMSNet("/home/dbeslic/master/DeepLearning_TrainingData/06_MassIVE_HCD_smallerValidation/MassIVE_transformed_valid.mgf")
trainingSplitToSMSNet("/home/dbeslic/master/DeepLearning_TrainingData/06_MassIVE_HCD_smallerValidation/MassIVE_transformed_test.mgf")

# For PointNovo Transform

folder_name = "/home/dbeslic/master/DeepLearning_TrainingData/02_NISTmAb/ONLY_NistMab/"
train_mgf_file = folder_name + 'NISTmAb_v20190711_transformed_train.mgf'
valid_mgf_file = folder_name + 'NISTmAb_v20190711_transformed_valid.mgf'
test_mgf_file = folder_name + 'NISTmAb_v20190711_transformed_test.mgf'
output_mgf_file = folder_name + 'spectrum.mgf'
output_train_feature_file = folder_name + 'features.train.csv'
output_valid_feature_file = folder_name + 'features.valid.csv'
output_test_feature_file = folder_name + 'features.test.csv'
spectrum_fw = open(output_mgf_file, 'w')
transfer_mgf(train_mgf_file, output_train_feature_file, spectrum_fw)
transfer_mgf(valid_mgf_file, output_valid_feature_file, spectrum_fw)
transfer_mgf(test_mgf_file, output_test_feature_file,spectrum_fw)
spectrum_fw.close()

