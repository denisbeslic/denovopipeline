# -*- coding: future_fstrings -*-

import os
import shutil
from time import time
import platform
import logging
import subprocess
import shlex

logger = logging.getLogger(__name__)

def log_subprocess_output(pipe):
    for line in iter(pipe.readline, b''): # b'\n'-separated lines
        logging.info('got line from subprocess: %r', line)

def run_denovogui(mgf_in, dir_out, params):
    logger.info("run_denovogui called. Sequencing with Novor started.")
    start_time1 = time()
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    subprocess.call(['java', '-cp', 'resources/DeNovoGUI-1.16.6/DeNovoGUI-1.16.6.jar',
                     'com.compomics.denovogui.cmd.DeNovoCLI', '-spectrum_files', mgf_in,
                     '-output_folder', dir_out + '/DeNovoCLI', '-id_params',
                     params, '-directag', '0', '-pepnovo', '0', '-novor', '1'], stdout=subprocess.DEVNULL)
    denovogui_time = time() - start_time1
    minute = int(denovogui_time // 60)
    sec = round(denovogui_time - minute * 60, 1)
    logger.info(f"Sequencing with Novor via DeNovoGUI finished. Total time: {minute} minutes and {sec} seconds.")


def run_smsnet(mgf_in, dir_out, smsnet_model):
    logger.info("run_smsnet called. Sequencing with SMSNet started.")
    cwd = os.getcwd()
    start_time1 = time()
    mgf_in = os.path.abspath(mgf_in)
    dir_out = os.path.join(os.path.abspath(dir_out),"SMSNet")
    if not os.path.exists(f"{dir_out}"):
        os.makedirs(f"{dir_out}")
    model = os.path.abspath(smsnet_model)
    os.chdir("resources/SMSNet")
    os.system(f"python run.py --model_dir {model} --inference_input_file {mgf_in} --inference_output_file {dir_out}")
    os.chdir(cwd)
    smsnet_time = time() - start_time1
    minute = int(smsnet_time // 60)
    sec = round(smsnet_time - minute * 60, 1)
    logger.info(f"Sequencing with SMSNet finished. Total time: {minute} minutes and {sec} seconds.")


def run_deepnovo(mgf_in, dir_out, deepnovo_model):
    logger.info("run_deepnovo called. Sequencing with DeepNovo started.")
    cwd = os.getcwd()
    start_time1 = time()
    mgf_in = os.path.abspath(mgf_in)

    # manipulate config file to change to correct input

    configfile = "resources/DeepNovo_Antibody/data_utils.py"
    with open(configfile, 'r') as file:
        data = file.readlines()
    for i in range(len(data)):
        if data[i].startswith("decode_test_file"):
            data[i] = f"decode_test_file = \"{mgf_in}\"\n"
            data[i - 1] = f"input_file_test = \"{mgf_in}\"\n"
    with open(configfile, 'w') as file:
        file.writelines(data)

    os.chdir("resources/DeepNovo_Antibody")
    traindir = os.path.abspath(deepnovo_model)
    os.system(f"python main.py --train_dir {traindir} --decode --beam_search --beam_size 10")
    os.chdir(cwd)
    if not os.path.exists(f"{dir_out}/DeepNovo/"):
        os.makedirs(f"{dir_out}/DeepNovo/")
    resultdecode = f"{traindir}/decode_output.tab"
    deepnovo_output = mgf_in.rpartition('/')[2].replace('.mgf', '')
    shutil.copyfile(resultdecode, f"{dir_out}/DeepNovo/{deepnovo_output}.tab")

    deepnovo_time = time() - start_time1
    minute = int(deepnovo_time // 60)
    sec = round(deepnovo_time - minute * 60, 1)
    logger.info(f"Sequencing with DeepNovo finished. Total Time: {minute} Minutes and {sec} Seconds.")


def run_pointnovo(mgf_in, dir_out, pointnovo_model):
    logger.info("run_pointnovo called. Sequencing with PointNovo started.")
    import denovo_pointnovo
    cwd = os.getcwd()
    start_time1 = time()
    mgf_in = os.path.abspath(mgf_in)
    dir_in = mgf_in.rpartition("/")[0] + "/"
    # creates feature file for pointnovo
    denovo_output_feature_file = f"{dir_in}features.csv"
    denovo_spectrum_fw = open(f"{dir_in}spectrum.mgf", 'w')
    spectrum_fw = open(f"{dir_in}spectrum.mgf", 'w')
    denovo_pointnovo.transfer_mgf(mgf_in, denovo_output_feature_file, denovo_spectrum_fw)
    denovo_spectrum_fw.close()
    spectrum_fw.close()

    if not os.path.exists(f"{dir_out}/PointNovo/"):
        os.makedirs(f"{dir_out}/PointNovo/")
    predict_out = os.path.abspath(f"{dir_out}/PointNovo/features.csv.deepnovo_denovo")
    configfile = "resources/PointNovo/config.py"
    with open(configfile, 'r') as file:
        data = file.readlines()

    mgf_in = mgf_in.replace("reformatted_deepnovo", "reformatted")
    for i in range(len(data)):
        if data[i].startswith("denovo_input_spectrum_file "):
            data[i] = f"denovo_input_spectrum_file = \"{mgf_in}\"\n"
            data[i + 1] = f"denovo_input_feature_file  = \"{denovo_output_feature_file}\"\n"
            data[i + 2] = f"denovo_output_file = \"{predict_out}\"\n"

    with open(configfile, 'w') as file:
        file.writelines(data)

    os.chdir("resources/PointNovo")
    os.system("make denovo")
    os.chdir(cwd)
    pointnovo_time = time() - start_time1
    minute = int(pointnovo_time // 60)
    sec = round(pointnovo_time - minute * 60, 1)
    logger.info(f"Sequencing with PointNovo finished. Total time: {minute} minutes and {sec} seconds.")


def denovo_seq(mgf_in, resultdir, denovogui, smsnet, deepnovo, pnovo3, pointnovo, params, smsnet_model, deepnovo_model,
               pointnovo_model):
    logger.info("Function denovo_seq was called.")

    if denovogui == 1:
        run_denovogui(mgf_in, resultdir + "/DeNovoCLI/", params)

    if smsnet == 1:
        run_smsnet(mgf_in, resultdir, smsnet_model)

    if deepnovo == 1:
        run_deepnovo(mgf_in, resultdir, deepnovo_model)

    if pointnovo == 1:
        run_pointnovo(mgf_in, resultdir, pointnovo_model)

    if pnovo3 == 1:
        logger.info("pNovo 3 is not part of the pipeline and needs to be executed separately.")
        if platform.system() == "Windows":
            path_pNovo = os.path.abspath("resources/pNovo_v3.1.3/pNovo3_Search.exe")
            os.system(path_pNovo)
        else:
            logger.error("pNovo 3 does only run on Windows and not on your OS: " + platform.system())

    if all(v is None for v in [pnovo3, deepnovo, smsnet, denovogui, pointnovo]):
        logger.error("No Tool was activated for de novo sequencing")
