import numpy as np
import pandas as pd
import statistics
import math
import logging

from config import vocab_reverse_nomods

logger = logging.getLogger(__name__)


def lcs(s1, s2):
    """Length of Longest Common Substring between two strings"""
    matrix = [["" for x in range(len(s2))] for x in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = s1[i]
                else:
                    matrix[i][j] = matrix[i - 1][j - 1] + s1[i]
            else:
                matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1], key=len)
    cs = matrix[-1][-1]
    return len(cs)


def process_novor(novor_path):
    """parse de nov results from novor to dataframe

        :param
            novor_path: path to the result file of Novor
        :return
            novor_df: dataframe with ScanNum, Score, Peptide, AAScore of Novor
    """
    try:
        with open(novor_path) as f:

            # jump over 17 rows which are the header of a Novor result file

            novor_df = pd.read_csv(novor_path, sep=",", header=17)

            # change single_score from "14-52-12.." to "14 52 12.."

            single_score = novor_df[' aaScore'].tolist()
            for i in range(len(single_score)):
                single_score[i] = single_score[i].replace(" ", "").replace("-", " ")
            novor_df[' aaScore'] = single_score
            novor_df = novor_df[[' scanNum', ' score', ' peptide', ' aaScore']]
            novor_df = novor_df.set_index(' scanNum')
            novor_df.columns = ['Novor Score', 'Novor Peptide', 'Novor aaScore']
            # Replace the specific annotation of Novor for Modifications
            novor_peptide = novor_df['Novor Peptide'].tolist()
            novor_df['Novor Peptide'] = [i.replace('M(0)', 'm').replace('Q(2)', 'q').replace('N(1)', 'n').replace(' ',
                                        '').replace('C(3)', 'C') for i in novor_peptide]
            return novor_df
    except IOError:
        logger.error(f"Novor results not accessible. Make sure they are placed in {novor_path}")
        return pd.DataFrame()


def process_pepnovo(pepnovo_path):
    """parse de novo results from pepnovo to dataframe

            :param
                pepnovo_path: path to the result file of Novor
            :return
                pepnovo_df: dataframe with ScanNum, Score, Peptide, AAScore of Novor
        """
    try:
        with open(pepnovo_path) as f:
            pepnovo = f.readlines()
            pepnovo_peptide = []
            pepnovo_ID = []
            pepnovo_index = []
            pepnovo_score = []

            for i in range(len(pepnovo)):
                # Header Line of PepNovo start with #

                if pepnovo[i].startswith("#"):
                    pepnovo_ID.append(pepnovo[i - 1])
                    pepnovo_index.append(int(pepnovo[i - 1].split(" ")[2]))
                    if pepnovo[i].startswith("#Index"):
                        pepnovo_score.append(pepnovo[i + 1].split("	")[1])
                        pepnovo_peptide.append(
                            pepnovo[i + 1].split("	")[7].replace("\n", "").replace("M+16", "m").replace("+1", ""))

                    # If it doesnt start with "#Index" no solutions could have been found
                    else:
                        pepnovo_score.append(np.nan)
                        pepnovo_peptide.append(np.nan)
            pepnovo_df = pd.DataFrame(
                {'Index': pepnovo_index, 'PepNovo Score': pepnovo_score, "PepNovo Peptide": pepnovo_peptide})
            pepnovo_df = pepnovo_df.set_index('Index')
            pepnovo_score = pepnovo_df['PepNovo Score']
            pepnovo_df['PepNovo Score'] = [4 * (float(i) + 10) for i in pepnovo_score]
            return pepnovo_df
    except IOError:
        logger.error(f"PepNovo results not accessible. Make sure they are placed in {pepnovo_path}")
        return pd.DataFrame()


def process_smsnet(smsnet_path):
    """parse de novo results from SMSNet to dataframe

            :param
                smsnet_path: path to the result file of Novor
            :return
                smsnet_df: dataframe with Score, Peptide, AAScore of SMSNet
        """
    try:
        with open(smsnet_path) as f, open(
                smsnet_path + '_prob') as g:  # change _rescore and _prob to switch between rescoring and real
            smsnet_peptide = pd.Series([line.rstrip() for line in f])
            peptide_list = [x.replace(" ", "").replace("I", "L") for x in
                            smsnet_peptide]
            smsnet_peptide = pd.DataFrame(peptide_list)
            aa_score = g.readlines()
            aa_score = [i.strip().split(' ') for i in aa_score]
            score_sum = []
            for i in range(len(aa_score)):
                if not aa_score[i] == ['']:
                    for j in range(len(aa_score[i])):
                        aa_score[i][j] = float(np.exp(float(aa_score[i][j])) * 100)
                else:
                    aa_score[i] = [0]
            for i in range(len(aa_score)):
                if not aa_score[i] == [0]:
                    score_sum.append(statistics.mean(aa_score[i]))
                else:
                    score_sum.append(0)
            df = pd.DataFrame({'aaScore': aa_score, 'Peptide Score': score_sum})
            smsnet_df = pd.concat([smsnet_peptide, df], axis=1)
            smsnet_df.columns = ['SMSNet Peptide', 'SMSNet aaScore', 'SMSNet Score']
            clist = ['SMSNet Score', 'SMSNet Peptide', 'SMSNet aaScore']
            smsnet_df = smsnet_df[clist]
            smsnet_df.index = range(1, len(smsnet_df) + 1)

            smsnet_aascore = smsnet_df['SMSNet aaScore'].tolist()

            # remove peptide predictions which look like "<s><s><ink>SSSSLASSS"

            smsnet_df['SMSNet aaScore'] = [str(i).replace(' ', '').replace(',', ' ').replace('[', '').replace(']', '')
                                           for i in
                                           smsnet_aascore]
            smsnet_peptide = smsnet_df['SMSNet Peptide'].tolist()
            for i, pep in enumerate(smsnet_peptide):
                if not type(pep) == float:
                    if ">" in pep:
                        smsnet_peptide[i] = np.nan

            # in case you chose the phosphorylation model for de novo mode

            """smsnet_df['SMSNet Peptide'] = [
                str(i).replace('t', 'T').replace('s', 'S').replace('y', 'Y') for i in smsnet_peptide]"""
            return smsnet_df
    except IOError:
        logger.error(f"SMSNet results not accessible. Make sure they are placed in {smsnet_path}")
        return pd.DataFrame()


def process_deepnovo(deepnovo_path):
    """parse de novo results from DeepNoo to dataframe

            :param
                deepnovo_path: path to the result file of DeepNovo
            :return
                deepnovo_df: dataframe with Score, Peptide, AAScore of DeepNovo
    """
    try:
        with open(deepnovo_path) as f:
            deepnovo_df = pd.read_csv(deepnovo_path, sep="	", header=0)

            deepnovo_df = deepnovo_df[['scan', 'output_score', 'output_seq', 'aa_score']]

            # bug from deepnovo (v.PNAS) which causes result to be wrong aligned

            '''
            scan    output_score    output_seq  aa_scpre
            1       NaN             NaN         NaN
            NaN     24              S,E,L       12, 41, 12
            2       NaN ...
            '''
            deepnovo_scan = deepnovo_df['scan'].tolist()

            # Fill up NaN scan number with the one before

            for i, a in enumerate(deepnovo_scan):
                if math.isnan(a):
                    deepnovo_scan[i] = deepnovo_scan[i - 1]
            deepnovo_df['scan'] = deepnovo_scan

            # remove every second row to keep only rows without NaN

            deepnovo_df = deepnovo_df[deepnovo_df.index % 2 != 0]
            deepnovo_df = deepnovo_df.set_index('scan')

            # change peptide from P,E,P,T,I,D to PEPTID

            deepnovo_peptide = deepnovo_df['output_seq'].tolist()
            for i in range(len(deepnovo_peptide)):
                deepnovo_peptide[i] = str(deepnovo_peptide[i])
                deepnovo_peptide[i] = deepnovo_peptide[i].replace(",", "").replace("I", "L").replace("Cmod",
                    "C").replace("Mmod", "m").replace("Nmod", "n").replace("Qmod", "q")
            deepnovo_df['output_seq'] = deepnovo_peptide
            deepnovo_df.columns = ['DeepNovo Score', 'DeepNovo Peptide', 'DeepNovo aaScore']

            # scale peptide score from 0 to 100

            deepnovo_score = deepnovo_df['DeepNovo Score']
            deepnovo_df['DeepNovo Score'] = [np.exp(i) * 100 for i in deepnovo_score]

            # scale amino acid score from 0 to 100

            deepnovo_aascore = deepnovo_df['DeepNovo aaScore'].tolist()
            deepnovo_aascore = [str(i).replace(",", " ") for i in deepnovo_aascore]
            deepnovo_aascore = [i.split() for i in deepnovo_aascore]
            deepnovo_aascore = [[str(np.exp(float(j)) * 100) for j in i] for i in deepnovo_aascore]
            deepnovo_df['DeepNovo aaScore'] = [" ".join(i) for i in deepnovo_aascore]

            return deepnovo_df
    except IOError:
        logger.error(f"DeepNovo results not accessible. Make sure they are placed in {deepnovo_path}")
        return pd.DataFrame()


def process_pointnovo(pointnovo_path):
    """parse de novo results from PointNovo to dataframe
            :param
                pointnovo_path: path to the result file of PointNovo
            :return
                pointnovo_df: dataframe with ScanNum, Score, Peptide, AAScore of PointNovo
    """
    try:
        with open(pointnovo_path) as f:
            pointnovo_df = pd.read_csv(pointnovo_path, sep="\t", header=0)
            pointnovo_df = pointnovo_df.set_index('feature_id')
            pointnovo_df = pointnovo_df[['predicted_score', 'predicted_sequence', 'predicted_position_score']]

            # change peptide from P,E,P,T,I,D to PEPTID

            pointnovo_peptide = pointnovo_df['predicted_sequence'].tolist()
            for i in range(len(pointnovo_peptide)):
                pointnovo_peptide[i] = str(pointnovo_peptide[i])
                pointnovo_peptide[i] = pointnovo_peptide[i].replace(",", "").replace("I", "L").replace("N(Deamidation)",
                "n").replace("Q(Deamidation)", "q").replace("C(Carbamidomethylation)", "C").replace("M(Oxidation)", "m")
            pointnovo_df['predicted_sequence'] = pointnovo_peptide
            pointnovo_df.columns = ['PointNovo Score', 'PointNovo Peptide', 'PointNovo aaScore']

            # scale peptide score from 0 to 100

            pointnovo_score = pointnovo_df['PointNovo Score']
            pointnovo_df['PointNovo Score'] = [np.exp(i) * 100 for i in pointnovo_score]

            # scale AAscore from 0 to 100

            pointnovo_aascore = pointnovo_df['PointNovo aaScore'].tolist()
            pointnovo_aascore = [str(i).replace(",", " ") for i in pointnovo_aascore]
            pointnovo_aascore = [i.split() for i in pointnovo_aascore]
            pointnovo_aascore = [[str(np.exp(float(j)) * 100) for j in i] for i in pointnovo_aascore]
            pointnovo_df['PointNovo aaScore'] = [" ".join(i) for i in pointnovo_aascore]
            return pointnovo_df
    except IOError:
        logger.error(f"PointNovo results not accessible. Make sure they are placed in {pointnovo_path}")
        return pd.DataFrame()


def process_pnovo(pnovo_path):
    """parse de novo results from pNovo3 to dataframe

            :param
                pnovo_path: path to the result file of PNovo3
            :return
                pnovo_df: dataframe with Score, Peptide, AAScore of pNovo3
        """
    try:
        with open(pnovo_path) as f:
            pnovo_df = pd.read_csv(pnovo_path, sep="	", header=None)
            pnovo_index = pnovo_df[0].tolist()
            for i in range(len(pnovo_index)):
                pnovo_index[i] = pnovo_index[i].split("Index:")[1]
                pnovo_index[i] = int(pnovo_index[i].split(",")[0])
            pnovo_df[0] = pnovo_index

            # see pNovo 3 User Guide: http://pfind.ict.ac.cn/software/pNovo/pNovo%203%20User%20Guide.pdf

            pnovo_df = pnovo_df[[0, 4, 1, 5]]
            pnovo_df.columns = ['pNovo Index', 'pNovo Score', 'pNovo Peptide', 'pNovo aaScore']

            pnovo_peptide = pnovo_df['pNovo Peptide'].tolist()

            # depending on the enabled mods for pNovo 3, a b c etc. can stand for different AAs
            # see param/pNovo.param

            pnovo_peptide = [str(i).replace('a', 'n').replace('b', 'q').replace('c', 'm').replace('B', 'q') for i in
                             pnovo_peptide]

            # using mods sometimes a unmodified amino acid following the modified mod is lowercase f.e PEPTaiDE

            for index, peptide in enumerate(pnovo_peptide):
                for aminoacid in peptide:
                    if aminoacid in set(vocab_reverse_nomods):
                        peptide = peptide.replace(aminoacid, aminoacid.upper())
                        pnovo_peptide[index] = peptide
            pnovo_df['pNovo Peptide'] = pnovo_peptide

            pnovo_df = pnovo_df.set_index('pNovo Index')
            pnovo_df = pnovo_df.sort_index(axis=0)
            pnovo_aascore = pnovo_df['pNovo aaScore'].tolist()
            pnovo_df['pNovo aaScore'] = [str(i).replace(',', ' ') for i in pnovo_aascore]
            return pnovo_df
    except IOError:
        logger.error(f"pNovo3 results not accessible. Make sure they are placed in {pnovo_path}")
        return pd.DataFrame()


def access_mgf_file(mgf_path):
    """accesses mgf file to get ID and charge of each scan
            :param
                mgf_path: path to the mgf file
            :return
                spectrum_name: list of spectrum title lines including the scan number of each scan
                charge_sp: list of charge states of each scan
    """
    spectrum_name = []
    charge_sp = []
    try:
        with open(mgf_path) as z:
            lines = z.readlines()
            for line in lines:
                if line.startswith("TITLE"):
                    spectrum_name.append(line.replace("\n", "").replace("TITLE=", ""))
                if line.startswith("CHARGE="):
                    charge_sp.append(line.replace("CHARGE=", "").replace("+", "").replace("\n", ""))
        return spectrum_name, charge_sp
    except IOError:
        logger.error(f"MGF file is not accessible. Make sure it is placed in {mgf_path}")


def process_peptideshaker(dbreport_path):
    try:
        with open(dbreport_path) as a:
            dbreport_df = pd.read_csv(dbreport_path, sep="\t")
            validated_list = dbreport_df['Validation'] != 'Not Validated'
            dbreport_df = dbreport_df[validated_list]
            spectrum_title = dbreport_df['Spectrum Title']
            dbreport_df['Index'] = [int(i.split("Index: ")[1].split(",")[0]) for i in spectrum_title]
            # Replace Isoleucin with Leucin in Database Search
            db_peptide = dbreport_df['Modified Sequence']
            dbreport_df['Modified Sequence'] = [(i.replace("I", "L").replace("NH2-", "").replace("-COOH", "").replace(
                "C<cmm>", "C").replace("M<ox>", "m").replace("pyro-", "")).replace("N<deam>", "n").replace("Q<deam>",
                                                                                                           "q") for i in
                                                db_peptide]
            dbreport_df = dbreport_df[['Index', 'Modified Sequence', 'Validation', 'Spectrum Title']]
            dbreport_df = dbreport_df.set_index('Index').sort_index()
            return dbreport_df
    except IOError:
        logger.error(f"Database Search Report from PeptideShaker is not accessible. Make sure it is in {dbreport_path}")
        return pd.DataFrame()


def denovo_summary(mgf_in, resultdir, dbreport):
    """generates summary file showing results and scores of all tools with database results

        :param
            mgf_in: path to processed mgf file
            resultdir: path to result directory where the summary file will be exported
            dbreport: path to the database report file from PeptideShaker
        :returns
            exports summary dataframe to csv to the result directory
    """

    logger.debug("Function denovo_summary was called.")
    mgf_in_path = mgf_in
    mgf_in = mgf_in.rpartition("/")
    mgf_in = mgf_in[-1].rpartition(".mgf")[0]

    # Novor
    novor_path = resultdir + 'DeNovoCLI/' + mgf_in + '.novor.csv'
    novor_df = process_novor(novor_path)
    # PepNovo
    pepnovo_path = resultdir + 'DeNovoCLI/' + mgf_in + '.mgf.out'
    pepnovo_df = process_pepnovo(pepnovo_path)
    # SMSNet
    smsnet_path = resultdir + "smsnet/" + mgf_in
    smsnet_df = process_smsnet(smsnet_path)
    # DeepNovo
    deepnovo_path = resultdir + "DeepNovo/" + mgf_in + "_deepnovo.tab"
    deepnovo_df = process_deepnovo(deepnovo_path)
    # PointNovo
    pointnovo_path = resultdir + "PointNovo/features.csv.deepnovo_denovo"
    pointnovo_df = process_pointnovo(pointnovo_path)
    # pNovo3
    pnovo_path = resultdir + "pNovo3/result/results.res"
    pnovo_df = process_pnovo(pnovo_path)
    # input MGF file
    spectrum_name, charge_sp = access_mgf_file(mgf_in_path)

    tools_list = [novor_df, pepnovo_df, smsnet_df, deepnovo_df, pointnovo_df, pnovo_df]
    summary_df = pd.concat(tools_list, axis=1)
    summary_df.insert(0, 'Spectrum Name', spectrum_name)
    summary_df.insert(1, 'Charge', charge_sp)

    # Extended PSM Report of Peptide Shaker
    dbreport_df = process_peptideshaker(dbreport)

    # Concat the de novo summary with the PeptideShaker Report
    summary_df = pd.concat([summary_df, dbreport_df], axis=1)
    summary_df.index.name = "Index"
    summary_df.to_csv(resultdir + 'summary.csv', index=True)
    logger.info("Export of summary.csv successful!")