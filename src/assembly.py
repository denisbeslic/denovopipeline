# -*- coding: future_fstrings -*-

from json import tool
import csv
import numpy as np
import os
import pandas as pd
import subprocess
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
from sklearn import metrics
#import tikzplotlib
from fast_diff_match_patch import diff
import logging

from config import vocab, _match_AA_novor, arePermutation, tools_list
from createsummary import lcs

pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# TODO: Add threshold mode: Are we looking at the confidence score, peptide length, noise factor, or missing cleavages?
def precision_recall_with_threshold(peptides_truth, peptides_predicted, peptides_predicted_confidence, threshold):
    """
    Calculate precision and recall for the given confidence score threshold
    Parameters
    ----------
    peptides_truth : list 
        List of confidence scores for correct amino acids predictions
    peptides_predicted : list
        List of confidence scores for all amino acids prediction
    num_original_aa : list
        Number of amino acids in the predicted peptide sequences
    threshold : float
        confidence score threshold
           
    Returns
    -------
    aa_precision: float
        Number of correct aa predictions divided by all predicted aa
    aa_recall: float
        Number of correct aa predictions divided by all ground truth aa   
    peptide_recall: float
        Number of correct peptide preiditions divided by number of ground truth peptides  
    """  
    length_of_predictedAA = 0
    length_of_realAA = 0
    number_peptides = 0
    sum_peptidematch = 0
    sum_AAmatches = 0
    for i, (predicted_peptide, true_peptide) in enumerate(zip(peptides_predicted, peptides_truth)):
        length_of_realAA += len(true_peptide)
        number_peptides += 1
        if (type(predicted_peptide) is str and type(true_peptide) is str and peptides_predicted_confidence[i] >= threshold):
            length_of_predictedAA += len(predicted_peptide)
            predicted_AA_id = [vocab[x] for x in predicted_peptide]
            target_AA_id = [vocab[x] for x in true_peptide]
            recall_AA = _match_AA_novor(target_AA_id, predicted_AA_id)
            sum_AAmatches += recall_AA
            if recall_AA == len(true_peptide):
                sum_peptidematch += 1
        else:
            sum_AAmatches += 0
    return length_of_predictedAA, length_of_realAA, number_peptides, sum_peptidematch, sum_AAmatches


def precision_recall_with_threshold_missingCleavage(peptides_truth, peptides_predicted, peptides_predicted_confidence, threshold):
    """
    Calculate precision and recall for the given confidence score threshold
    Parameters
    ----------
    peptides_truth : list 
        List of confidence scores for correct amino acids predictions
    peptides_predicted : list
        List of confidence scores for all amino acids prediction
    num_original_aa : list
        Number of amino acids in the predicted peptide sequences
    threshold : float
        confidence score threshold
           
    Returns
    -------
    aa_precision: float
        Number of correct aa predictions divided by all predicted aa
    aa_recall: float
        Number of correct aa predictions divided by all ground truth aa   
    peptide_recall: float
        Number of correct peptide preiditions divided by number of ground truth peptides  
    """  
    length_of_predictedAA = 0
    length_of_realAA = 0
    number_peptides = 0
    sum_peptidematch = 0
    sum_AAmatches = 0
    for i, (predicted_peptide, true_peptide) in enumerate(zip(peptides_predicted, peptides_truth)):
        length_of_realAA += len(true_peptide)
        number_peptides += 1
        if (type(predicted_peptide) is str and type(true_peptide) is str): #and peptides_predicted_confidence[i] == threshold):
            length_of_predictedAA += len(predicted_peptide)
            predicted_AA_id = [vocab[x] for x in predicted_peptide]
            target_AA_id = [vocab[x] for x in true_peptide]
            recall_AA = _match_AA_novor(target_AA_id, predicted_AA_id)
            sum_AAmatches += recall_AA
            if recall_AA == len(true_peptide):
                sum_peptidematch += 1
        else:
            sum_AAmatches += 0
    return length_of_predictedAA, length_of_realAA, number_peptides, sum_peptidematch, sum_AAmatches




def process_summaryfile(summary_csv):
    """process summary dataframe for further analysis
            :param
                summary_csv: path of summary.csv file
            :return
                summary_df: preprocessed dataframe of summary csv-file
    """
    try:
        summary_df = pd.read_csv(summary_csv, header=0)
        summary_df['Area'] = 1  # is nedded for ALPS
        name_spectrum = summary_df["Spectrum Name"]
        summary_df["Spectrum Name"] = [i.replace(",", ";") for i in name_spectrum]
        return summary_df
    except IOError:
        logger.error(f"Summary File is not accessible. Make sure it's in {summary_csv}")


def process_ALPS(summary_df, resultdir, kmer_ALPS, contigs_ALPS, quality_cutoff_ALPS, Confidence_OVER_50_AA_Prec = []):
    """process summary file with ALPS to assemble full sequence
            :param
                summary_df: dataframe of different tools with score, peptide and aascore
            :return
                executed ALPS.jar and generates fasta files with top contigs
    """
    if Confidence_OVER_50_AA_Prec == []:
        Confidence_OVER_50_AA_Prec = [quality_cutoff_ALPS] * 5

    for i, tool in enumerate(tools_list):
        alps_df = summary_df[["Spectrum Name", tool + " Peptide", tool + " aaScore", tool + " Score", "Area"]]
        peptides_mods = alps_df[tool + " Peptide"]
        aaScores_mods = alps_df[tool + " aaScore"]

        # check if score and peptide is same length else put 0

        for inx, (peptide, score) in enumerate(zip(peptides_mods, aaScores_mods)):
            if type(peptide) == str:
                if type(score) != str:
                    score = "0"
                a_list = score.split()
                map_object = map(float, a_list)
                list_of_integers = list(map_object)

                if len(peptide) != len(list_of_integers):
                    peptides_mods[inx] = np.nan
                    aaScores_mods[inx] = np.nan

        alps_df[tool + " Peptide"] = [
            peptide.replace("m", "M(+15.99)").replace("q", "Q(+.98)").replace("n", "N(+.98)") if type(
                peptide) == str else None
            for peptide in peptides_mods]

        # will drop empty Peptides
        alps_df.dropna(subset=[tool + " Peptide"], inplace=True)
        for kmer in [6,7]:
            alps_df.to_csv(resultdir + tool + '_totalscore.csv', index=False)
            subprocess.run(('java', '-jar', 'resources/ALPS.jar', resultdir + tool + '_totalscore.csv', str(kmer), str(contigs_ALPS), '>>',
                            resultdir + 'assembly.log'), stdout=subprocess.DEVNULL)

            # Quality Cut-Off for assembly at given de novo Peptide Score
            quality_cutoff_ALPS = Confidence_OVER_50_AA_Prec[i]
            alps_df = alps_df[alps_df[tool + " Score"] > quality_cutoff_ALPS]
            alps_df.to_csv(resultdir + tool + f'_totalscore_cutoff_{str(quality_cutoff_ALPS)}.csv', index=False)
            subprocess.run(
                ('java', '-jar', 'resources/ALPS.jar', resultdir + tool + f'_totalscore_cutoff_{str(quality_cutoff_ALPS)}.csv', str(kmer), str(contigs_ALPS), '>>',
                resultdir + 'assembly.log'), stdout=subprocess.DEVNULL)


def generate_stats(summary_df, resultdir):
    """
    Generate stats (AA-Recall, AA-Precision, Peptide Recall) based on database search for all tools
            :param
                summary_df: dataframe of different tools with score, peptide and aascore
                resultdir: path to result directory
            :return
                generates .txt file with error stats and several figures
    """
    # take out peptides that are not verified by database
    confident = summary_df['Validation'] == 'Confident'  # | (summary_df['Validation'] == 'Doubtful')
    summary_df = summary_df[confident]
    logger.info(r"The Database Report File has classified " + str(
        len(summary_df.index)) + " spectras as confident (FDR < 1%).")
    
    AUC = []
    tools_stats = []
    total_peptide_recall = []
    total_AA_recall = []
    total_AA_precision = []

    Confidence_Values = []
    AA_Recall_Values = []
    AA_Precision_Values = []
    Tool_Values = []

    Confidence_OVER_50_AA_Prec = []

    for tools in tools_list:
        logger.debug(f"Calculating precision-recall for {tools}")
        tool_AArecall = []
        tool_accuracy = []
        tool_AAprecision = []
        tool_scorecutoff = []
        score_cutoff = 100 
        while (score_cutoff > -1):
            true_list = summary_df['Modified Sequence'].tolist()
            to_test = summary_df[tools + ' Peptide'].tolist()
            to_test_score = summary_df[tools + ' Score'].tolist()
            length_of_predictedAA, length_of_realAA, number_peptides, sum_peptidematch, sum_AAmatches = precision_recall_with_threshold(true_list, to_test, to_test_score, score_cutoff)
            if length_of_predictedAA != 0:
                tool_accuracy.append(str(sum_peptidematch * 100 / number_peptides))
                tool_AAprecision.append(str(sum_AAmatches * 100 / length_of_predictedAA))
                tool_AArecall.append(str(sum_AAmatches * 100 / length_of_realAA))
                tool_scorecutoff.append(score_cutoff)
            score_cutoff = score_cutoff - 5
        tool_AAprecision = [float(i) for i in tool_AAprecision]
        tool_AArecall = [float(i) for i in tool_AArecall]
        tool_accuracy = [float(i) for i in tool_accuracy]
        tool_scorecutoff = [int(i) for i in tool_scorecutoff]
        AUC.append(metrics.auc(tool_AArecall, tool_accuracy) / 10000)

        for score, AA_precision  in zip(tool_scorecutoff[::-1], tool_AAprecision[::-1]):
            if AA_precision >= 50:
                Confidence_OVER_50_AA_Prec.append(score)
                break
            else:
                Confidence_OVER_50_AA_Prec.append(50)
        
        total_peptide_recall.append(tool_accuracy[-1])
        total_AA_recall.append(tool_AArecall[-1])
        total_AA_precision.append(tool_AAprecision[-1])

        AA_Recall_Values.append(tool_AArecall)
        AA_Precision_Values.append(tool_AAprecision)
        Confidence_Values.append(tool_scorecutoff)
        Tool_Values.append([tools]*len(tool_scorecutoff))

    #print(Confidence_OVER_50_AA_Prec)
    # TOTAL PEPTIDE RECALL
    df_total_peptide_recall = pd.DataFrame(list(zip(tools_list, total_peptide_recall)),
               columns =['Tool', 'Total Peptide Recall'])
    df_total_peptide_recall.to_csv(resultdir+"TotalPeptideRecall.csv", index=False)

    # TOTAL AA RECALL
    df_total_AA_recall = pd.DataFrame(list(zip(tools_list, total_AA_recall)),
               columns =['Tool', 'Total AA Recall'])
    df_total_AA_recall.to_csv(resultdir+"TotalAARecall.csv", index=False)

    # TOTAL AA PRECISION
    df_total_AA_precision = pd.DataFrame(list(zip(tools_list, total_AA_precision)),
               columns =['Tool', 'Total AA Precision'])
    df_total_AA_precision.to_csv(resultdir+"TotalAAPrecision.csv", index=False)

    # AUC VALUE
    df_AUC = pd.DataFrame(list(zip(tools_list, AUC)),
               columns =['Tool', 'AUC'])
    df_AUC.to_csv(resultdir+"AUC.csv", index=False)

    # PRECISION RECALL AA CURVE
    AA_Recall_Values = [item for sublist in AA_Recall_Values for item in sublist]
    AA_Precision_Values = [item for sublist in AA_Precision_Values for item in sublist]
    Tool_Values = [item for sublist in Tool_Values for item in sublist]
    Confidence_Values = [item for sublist in Confidence_Values for item in sublist]
    df_PRcurve = pd.DataFrame(list(zip(Tool_Values, AA_Recall_Values, AA_Precision_Values, Confidence_Values)),
               columns =['Tool', 'AA Recall', 'AA Precision', 'Confidence level'])
    df_PRcurve.to_csv(resultdir+"PRcurve.csv", index=False)


    total_peptide_recall = []
    total_AA_recall = []
    total_AA_precision = []

    df_AA_recall_combined = pd.DataFrame(tools_list)
    df_peptide_recall_combined = pd.DataFrame(tools_list)

    # Loop for evaluation of two tools
    logger.debug("Calculating recall stats for pairs of tools")
    for tool_one in tools_list:
        for tool_two in tools_list:
            logger.debug(f"Calculating stats between {tool_one} and {tool_two}")
            tool_AArecall = []
            tool_accuracy = []
            tool_AAprecision = []
            tool_scorecutoff = []

            true_list = summary_df['Modified Sequence'].tolist()
            to_test_first = summary_df[tool_one + ' Peptide'].tolist()
            to_test_second = summary_df[tool_two + ' Peptide'].tolist()

            length_of_predictedAA = 0
            length_of_realAA = 0
            number_peptides = 0
            sum_peptidematch = 0
            sum_AAmatches = 0

            for i, (pred_peptide_first, pred_peptide_second, true_peptide) in enumerate(zip(to_test_first, to_test_second, true_list)):
                length_of_realAA += len(true_peptide)
                number_peptides += 1
                if ((type(pred_peptide_first) is str or type(pred_peptide_second) is str) and type(true_peptide) is str):
                    if type(pred_peptide_first) is str:
                        predicted_AA_id_first = [vocab[x] for x in pred_peptide_first]
                    else:
                        predicted_AA_id_first = ""
                    if type(pred_peptide_second) is str:
                        predicted_AA_id_second = [vocab[x] for x in pred_peptide_second] # TODO: What if empty?
                    else:
                        predicted_AA_id_second = ""
                    target_AA_id = [vocab[x] for x in true_peptide]
                    values=[_match_AA_novor(target_AA_id, predicted_AA_id_first), _match_AA_novor(target_AA_id, predicted_AA_id_second)]
                    recall_AA = max(_match_AA_novor(target_AA_id, predicted_AA_id_first), _match_AA_novor(target_AA_id, predicted_AA_id_second))
                    index_max = max(range(len(values)), key=values.__getitem__)
                    if index_max == 0 and type(pred_peptide_first) is str:
                        length_of_predictedAA += len(pred_peptide_first)
                    else:
                        length_of_predictedAA += len(pred_peptide_second)
                    sum_AAmatches += recall_AA
                    if recall_AA == len(true_peptide):
                        sum_peptidematch += 1
                else:
                    sum_AAmatches += 0
            if length_of_predictedAA != 0:
                tool_accuracy.append(str(sum_peptidematch * 100 / number_peptides))
                tool_AAprecision.append(str(sum_AAmatches * 100 / length_of_predictedAA))
                tool_AArecall.append(str(sum_AAmatches * 100 / length_of_realAA))
            tool_AAprecision = [float(i) for i in tool_AAprecision]
            tool_AArecall = [float(i) for i in tool_AArecall]
            tool_accuracy = [float(i) for i in tool_accuracy]
            total_peptide_recall.append(tool_accuracy[-1])
            total_AA_recall.append(tool_AArecall[-1])
            total_AA_precision.append(tool_AAprecision[-1])
            
    k=0
    for i in tools_list:
        df_AA_recall_combined[i] = total_AA_recall[k:k+len(tools_list)]
        df_peptide_recall_combined[i] = total_peptide_recall[k:k+len(tools_list)]
        k = k + len(tools_list) 

    df_AA_recall_combined.to_csv(resultdir+"CombinedAARecall.csv", index=False)
    df_peptide_recall_combined.to_csv(resultdir+"CombinedPeptideRecall.csv", index=False)


    # Number of spectra with at least one missing cleavage site
    AtLeastOneMissing = len(summary_df[summary_df['Number of missing cleavages'] > 0])
    logger.info(f"{AtLeastOneMissing*100/len(summary_df)}% of confident peptide spectrum matches are missing at least 1 fragment ions.")
    
    # Number of spectra without missing cleavage at first position
    list_of_positions = summary_df['Position of present cleavages'].tolist()
    missingcleavage_at_first_position = 0
    for i in list_of_positions:
        if(i[1] != "1"):
            missingcleavage_at_first_position += 1
    logger.info(f"{missingcleavage_at_first_position*100/len(summary_df)}% of confident peptide spectrum matches are missing the first cleavage site.")

    # PRINT % of peaks which are noise
    total_amount_noisepeaks = sum(summary_df["Number of noise peaks"].tolist())
    total_amount_fragmentpeaks = sum(summary_df["Number of fragment peaks"].tolist())
    logger.info(f"{total_amount_noisepeaks*100/(total_amount_noisepeaks+total_amount_fragmentpeaks)}% of all peaks are considered noise")



    ######################
    # Get Recall vs Number of Cleavage Sites missing
    # from 0 to 10
    tools_stats = []
    total_peptide_recall = []
    total_AA_recall = []
    total_AA_precision = []
    Confidence_Values = []
    AA_Recall_Values = []
    AA_Precision_Values = []
    Peptide_Recall_Values = []
    Tool_Values = []
    for tools in tools_list:
        logger.debug(f"Calculating precision-recall (missing cleavages) for {tools}")
        tool_AArecall = []
        tool_accuracy = []
        tool_AAprecision = []
        tool_scorecutoff = []
        missing_cleavages_cutoff = 8 
        while (missing_cleavages_cutoff > -1):
            smaller_df = summary_df[summary_df['Number of missing cleavages'] == missing_cleavages_cutoff]
            true_list = smaller_df['Modified Sequence'].tolist()
            to_test = smaller_df[tools + ' Peptide'].tolist()
            to_test_score = smaller_df['Number of missing cleavages'].tolist() # Replace confidecne score with number of misssing cleavages
            length_of_predictedAA, length_of_realAA, number_peptides, sum_peptidematch, sum_AAmatches = precision_recall_with_threshold_missingCleavage(true_list, to_test, to_test_score, missing_cleavages_cutoff)
            if length_of_predictedAA != 0:
                tool_accuracy.append(str(sum_peptidematch * 100 / number_peptides))
                tool_AAprecision.append(str(sum_AAmatches * 100 / length_of_predictedAA))
                tool_AArecall.append(str(sum_AAmatches * 100 / length_of_realAA))
                tool_scorecutoff.append(missing_cleavages_cutoff)
            missing_cleavages_cutoff = missing_cleavages_cutoff - 1
        tool_AAprecision = [float(i) for i in tool_AAprecision]
        tool_AArecall = [float(i) for i in tool_AArecall]
        tool_accuracy = [float(i) for i in tool_accuracy]
        tool_scorecutoff = [int(i) for i in tool_scorecutoff]
        AA_Recall_Values.append(tool_AArecall)
        AA_Precision_Values.append(tool_AAprecision)
        Peptide_Recall_Values.append(tool_accuracy)
        Confidence_Values.append(tool_scorecutoff)
        Tool_Values.append([tools]*len(tool_scorecutoff))

    # PRECISION RECALL AA CURVE
    AA_Recall_Values = [item for sublist in AA_Recall_Values for item in sublist]
    AA_Precision_Values = [item for sublist in AA_Precision_Values for item in sublist]
    Peptide_Recall_Values = [item for sublist in Peptide_Recall_Values for item in sublist]
    Tool_Values = [item for sublist in Tool_Values for item in sublist]
    Confidence_Value = [item for sublist in Confidence_Values for item in sublist]
    df_PRcurve = pd.DataFrame(list(zip(Tool_Values, AA_Recall_Values, Peptide_Recall_Values, Confidence_Value)),
               columns =['Tool', 'AA Recall', 'Peptide Recall', 'Missing Cleavage Sites'])
    df_PRcurve.to_csv(resultdir+"MissingCleavagesVSRecall.csv", index=False)


    #############################################################
    # Get Recall vs Noise Factor
    # start at maximum of 3 
    # or minimum of 30

    tools_stats = []
    total_peptide_recall = []
    total_AA_recall = []
    total_AA_precision = []
    Confidence_Values = []
    AA_Recall_Values = []
    AA_Precision_Values = []
    Peptide_Recall_Values = []
    Tool_Values = []
    for tools in tools_list:
        logger.debug(f"Calculating precision-recall (noise factor) for {tools}")
        tool_AArecall = []
        tool_accuracy = []
        tool_AAprecision = []
        tool_scorecutoff = []
        noise_factor_cutoff = 23 
        while (noise_factor_cutoff > 0):
            smaller_df = summary_df[np.isclose(summary_df['Noise factor'], noise_factor_cutoff, atol=1)]
            true_list = smaller_df['Modified Sequence'].tolist()
            to_test = smaller_df[tools + ' Peptide'].tolist()
            to_test_score = smaller_df['Noise factor'].tolist() # Replace confidecne score with number of misssing cleavages
            length_of_predictedAA, length_of_realAA, number_peptides, sum_peptidematch, sum_AAmatches = precision_recall_with_threshold_missingCleavage(true_list, to_test, to_test_score, noise_factor_cutoff)
            if length_of_predictedAA != 0:
                tool_accuracy.append(str(sum_peptidematch * 100 / number_peptides))
                tool_AAprecision.append(str(sum_AAmatches * 100 / length_of_predictedAA))
                tool_AArecall.append(str(sum_AAmatches * 100 / length_of_realAA))
                tool_scorecutoff.append(noise_factor_cutoff)
            noise_factor_cutoff = noise_factor_cutoff - 2
        tool_AAprecision = [float(i) for i in tool_AAprecision]
        tool_AArecall = [float(i) for i in tool_AArecall]
        tool_accuracy = [float(i) for i in tool_accuracy]
        tool_scorecutoff = [int(i) for i in tool_scorecutoff]
        AA_Recall_Values.append(tool_AArecall)
        AA_Precision_Values.append(tool_AAprecision)
        Peptide_Recall_Values.append(tool_accuracy)
        Confidence_Values.append(tool_scorecutoff)
        Tool_Values.append([tools]*len(tool_scorecutoff))

    # PRECISION RECALL AA CURVE
    AA_Recall_Values = [item for sublist in AA_Recall_Values for item in sublist]
    AA_Precision_Values = [item for sublist in AA_Precision_Values for item in sublist]
    Peptide_Recall_Values = [item for sublist in Peptide_Recall_Values for item in sublist]
    Tool_Values = [item for sublist in Tool_Values for item in sublist]
    Confidence_Value = [item for sublist in Confidence_Values for item in sublist]
    df_PRcurve = pd.DataFrame(list(zip(Tool_Values, AA_Recall_Values, Peptide_Recall_Values, Confidence_Value)),
               columns =['Tool', 'AA Recall', 'Peptide Recall', 'Noise Factor'])
    df_PRcurve.to_csv(resultdir+"NoiseFactorVSRecall.csv", index=False)

    ######################################################
    # Get Recall vs Noise Factor & Cleavage Site increasing
    #####################################################
    #TODO: solve problem
    tools_stats = []
    total_peptide_recall = []
    total_AA_recall = []
    total_AA_precision = []
    Confidence_Values1 = []
    Confidence_Values2 = []
    AA_Recall_Values = []
    AA_Precision_Values = []
    Peptide_Recall_Values = []
    Tool_Values = []
    for tools in tools_list:
        logger.debug(f"Calculating precision-recall (noise factor & missing cleavages) for {tools}")
        tool_AArecall = []
        tool_accuracy = []
        tool_AAprecision = []
        tool_scorecutoff_noise = []
        tool_scorecutoff_cleavages = []
        noise_factor_cutoff = 23
        missing_cleavages_cutoff = 8 
        while (missing_cleavages_cutoff > -1):
            noise_factor_cutoff = 23
            while (noise_factor_cutoff > 0):
                smaller_df = summary_df[np.isclose(summary_df['Noise factor'], noise_factor_cutoff, atol=1)]
                smaller_df = smaller_df[smaller_df['Number of missing cleavages'] == missing_cleavages_cutoff]
                true_list = smaller_df['Modified Sequence'].tolist()
                to_test = smaller_df[tools + ' Peptide'].tolist()
                to_test_score = smaller_df['Noise factor'].tolist() # Replace confidecne score with number of misssing cleavages
                length_of_predictedAA, length_of_realAA, number_peptides, sum_peptidematch, sum_AAmatches = precision_recall_with_threshold_missingCleavage(true_list, to_test, to_test_score, noise_factor_cutoff)
                if length_of_predictedAA != 0:
                    tool_accuracy.append(str(sum_peptidematch * 100 / number_peptides))
                    tool_AAprecision.append(str(sum_AAmatches * 100 / length_of_predictedAA))
                    tool_AArecall.append(str(sum_AAmatches * 100 / length_of_realAA))
                    tool_scorecutoff_noise.append(noise_factor_cutoff)
                    tool_scorecutoff_cleavages.append(missing_cleavages_cutoff)
                noise_factor_cutoff = noise_factor_cutoff - 2
            missing_cleavages_cutoff = missing_cleavages_cutoff - 1
            
            tool_AAprecision = [float(i) for i in tool_AAprecision]
            tool_AArecall = [float(i) for i in tool_AArecall]
            tool_accuracy = [float(i) for i in tool_accuracy]
            tool_scorecutoff_noise = [int(i) for i in tool_scorecutoff_noise]
            tool_scorecutoff_cleavages = [int(i) for i in tool_scorecutoff_cleavages]

        AA_Recall_Values.append(tool_AArecall)
        AA_Precision_Values.append(tool_AAprecision)
        Peptide_Recall_Values.append(tool_accuracy)
        Confidence_Values1.append(tool_scorecutoff_noise)
        Confidence_Values2.append(tool_scorecutoff_cleavages)
        Tool_Values.append([tools]*len(tool_scorecutoff_noise))
        

    # PRECISION RECALL AA CURVE
    AA_Recall_Values = [item for sublist in AA_Recall_Values for item in sublist]
    AA_Precision_Values = [item for sublist in AA_Precision_Values for item in sublist]
    Peptide_Recall_Values = [item for sublist in Peptide_Recall_Values for item in sublist]
    Tool_Values = [item for sublist in Tool_Values for item in sublist]
    Confidence_Values1 = [item for sublist in Confidence_Values1 for item in sublist]
    Confidence_Values2 = [item for sublist in Confidence_Values2 for item in sublist]
    df_PRcurve = pd.DataFrame(list(zip(Tool_Values, AA_Recall_Values, Peptide_Recall_Values, Confidence_Values1, Confidence_Values2)),
               columns =['Tool', 'AA Recall', 'Peptide Recall', 'Noise Factor', 'Missing Cleavages'])
    df_PRcurve.to_csv(resultdir+"NoiseFactorANDMissingcleavagesVSRecall.csv", index=False)




    ###################################
    ## LENGTH CUTOFF
    ###################################

    total_number_peptides = []
    AA_Recall_Values = []
    AA_Precision_Values = []
    Peptide_Recall_Values = []
    Total_Number_Of_Peptides_Values = []
    Predicted_Number_Of_Peptides_Values = []
    Number_Of_Peptides_WithNoMissingCleavages_Values = []
    Confidence_Values = []
    Tool_Values = []
    Tool_Values = []

    for tools in tools_list:
        logger.debug(tools)
        tool_AArecall = []
        tool_accuracy = []
        tool_AAprecision = []
        tool_predictedpeptides = []
        tool_scorecutoff = []
        tool_totalnumberpeptides = []
        tool_totalnumberpredictedpeptides = []
        tool_nomissingcleavages = []

        # get recall-precision relationship by adjusting threshold for tools score
        score_cutoff = 5
        while score_cutoff < 22:
            # cutoff_df = summary_df[summary_df[tools+' Score'] >= score_cutoff]
            true_list = summary_df['Modified Sequence'].tolist()
            to_test = summary_df[tools + ' Peptide'].tolist()
            to_test_score = summary_df[tools + ' Score'].tolist()
            number_of_cleavages = summary_df['Number of missing cleavages'].tolist()
            length_of_predictedAA = 0
            length_of_realAA = 0
            number_peptides = 0
            number_peptides_nomissing_cleavage = 0
            sum_peptidematch = 0
            sum_AAmatches = 0
            number_predicted_peptides = 0

            for i, (a, b, cleavages) in enumerate(zip(to_test, true_list, number_of_cleavages)):
                length_of_realAA += len(b)
                if score_cutoff == 21:
                    if (len(b) >= score_cutoff - 1):
                        number_peptides += 1
                        if (cleavages == 0):
                            number_peptides_nomissing_cleavage += 1
                    if (type(a) is str and type(b) is str and len(b) >= score_cutoff - 1):
                        length_of_predictedAA += len(a)
                        predicted_AA_id = [vocab[x] for x in a]
                        target_AA_id = [vocab[x] for x in b]
                        recall_AA = _match_AA_novor(target_AA_id, predicted_AA_id)
                        number_predicted_peptides += 1
                        sum_AAmatches += recall_AA

                        if recall_AA == len(b):
                            sum_peptidematch += 1
                    else:
                        sum_AAmatches += 0
                else:
                    if (len(b) == score_cutoff):
                        number_peptides += 1
                        if (cleavages == 0):
                            number_peptides_nomissing_cleavage += 1
                    if (type(a) is str and type(b) is str and len(b) == score_cutoff):
                        length_of_predictedAA += len(a)
                        predicted_AA_id = [vocab[x] for x in a]
                        target_AA_id = [vocab[x] for x in b]
                        recall_AA = _match_AA_novor(target_AA_id, predicted_AA_id)
                        number_predicted_peptides += 1
                        sum_AAmatches += recall_AA

                        if recall_AA == len(b):
                            sum_peptidematch += 1
                    else:
                        sum_AAmatches += 0
            if (number_peptides != 0):
                tool_accuracy.append(str(sum_peptidematch * 100 / number_peptides))
            else:
                tool_accuracy.append(str(0))
            if (length_of_predictedAA != 0):
                tool_AAprecision.append(str(sum_AAmatches * 100 / length_of_predictedAA))
                tool_AArecall.append(str(sum_AAmatches * 100 / length_of_realAA))
            else:
                tool_AAprecision.append(str(0))
                tool_AArecall.append(str(0))
            tool_totalnumberpredictedpeptides.append(number_predicted_peptides)
            tool_predictedpeptides.append(sum_peptidematch)
            tool_totalnumberpeptides.append(number_peptides)
            tool_scorecutoff.append(score_cutoff)
            tool_nomissingcleavages.append(number_peptides_nomissing_cleavage)
            score_cutoff = score_cutoff + 1

        tool_AAprecision = [float(i) for i in tool_AAprecision]
        tool_AArecall = [float(i) for i in tool_AArecall]
        tool_accuracy = [float(i) for i in tool_accuracy]
        tool_scorecutoff = [int(i) for i in tool_scorecutoff]
        tool_predictedpeptides = [int(i) for i in tool_predictedpeptides]
        tool_totalnumberpeptides = [int(i) for i in tool_totalnumberpeptides]
        tool_falsepositiverate = [100 - float(i) for i in tool_AAprecision]
        tools_stats.append((tool_AAprecision, tool_AArecall, tool_falsepositiverate, tool_accuracy, tool_scorecutoff))
        
        
        AA_Recall_Values.append(tool_AArecall)
        AA_Precision_Values.append(tool_AAprecision)
        Peptide_Recall_Values.append(tool_accuracy)
        Total_Number_Of_Peptides_Values.append(tool_totalnumberpeptides)
        Predicted_Number_Of_Peptides_Values.append(tool_predictedpeptides)
        Number_Of_Peptides_WithNoMissingCleavages_Values.append(tool_nomissingcleavages)
        Confidence_Values.append(tool_scorecutoff)
        Tool_Values.append([tools]*len(tool_scorecutoff))

    
    # LENGTH
    AA_Recall_Values = [item for sublist in AA_Recall_Values for item in sublist]
    AA_Precision_Values = [item for sublist in AA_Precision_Values for item in sublist]
    Tool_Values = [item for sublist in Tool_Values for item in sublist]
    Confidence_Values = [item for sublist in Confidence_Values for item in sublist]
    Total_Number_Of_Peptides_Values = [item for sublist in Total_Number_Of_Peptides_Values for item in sublist]
    Predicted_Number_Of_Peptides_Values = [item for sublist in Predicted_Number_Of_Peptides_Values for item in sublist]
    Number_Of_Peptides_WithNoMissingCleavages_Values = [item for sublist in Number_Of_Peptides_WithNoMissingCleavages_Values for item in sublist]

    df_lengthValues = pd.DataFrame(list(zip(Tool_Values, AA_Recall_Values, AA_Precision_Values, Confidence_Values, Total_Number_Of_Peptides_Values,
        Number_Of_Peptides_WithNoMissingCleavages_Values, Predicted_Number_Of_Peptides_Values)),
               columns =['Tool', 'AA Recall', 'AA Precision', 'Length', 'Total number of peptides', 
               'Total number of peptides without missing cleavage', 'Number of predicted peptides'])
    df_lengthValues.to_csv(resultdir+"LengthVsRecall.csv", index=False)

    #################################################################################################
    #                                         Error stats                                            #
    #################################################################################################

    logger.debug("Error Evaluation.")
    with open(resultdir + "error_eval.txt", "w+") as text_file:
        None
    for tools in tools_list:
        score_cutoff = 50
        while (score_cutoff > -1):
            true_list = summary_df['Modified Sequence'].tolist()
            to_test = summary_df[tools + ' Peptide'].tolist()
            to_test_score = summary_df[tools + ' Score'].tolist()

            longest_mismatch_sum = 0
            permutations_first3 = 0
            permutations_last3 = 0
            permutations_last_and_first3 = 0
            amount_1AA_replacements = 0
            amount_1AA_replacements_firstposition = 0
            amount_1AA_replacements_twoposition = 0
            amount_1AA_replacements_lastposition = 0
            amount_2AA_replacements = 0
            amount_3AA_replacements = 0
            amount_4AA_replacements = 0
            amount_5AA_replacements = 0
            amount_6AA_replacements = 0
            amount_moreThan6AA_replacements = 0
            unknown_error = 0
            total_errors = 0
            number_of_predictions = 0

            Which_2AA_list = []

            for i, (pred_peptide, true_peptide) in enumerate(zip(to_test, true_list)):
                if type(pred_peptide) is str and type(true_peptide) is str and to_test_score[i] >= score_cutoff:
                    changes = diff(pred_peptide, true_peptide, timelimit=0, checklines=False)
                    longest_mismatch_neg = 0
                    longest_mismatch_pos = 0
                    longest_mismatch = 0
                    longest_mismatch_exactposition = -1
                    length_seq = 0
                    number_of_predictions += 1
                    for z, (op, length) in enumerate(changes):
                        length_seq += length
                        # if op == "-": print("next", length, "characters are deleted")
                        # if op == "=": print("next", length, "characters are in common")
                        # if op == "+": print("next", length, "characters are inserted")
                        if op == "+" and length > longest_mismatch_pos:
                            longest_mismatch_pos = length
                        if op == "-" and length > longest_mismatch_neg:
                            longest_mismatch_neg = length
                        if (op == "+" or op == "-") and length > longest_mismatch:
                            longest_mismatch = length
                            longest_mismatch_exactposition = length_seq - length

                    if longest_mismatch > 0:
                        total_errors += 1
                    if longest_mismatch == 0:
                        pass
                    elif (arePermutation(pred_peptide[0:3], true_peptide[0:3]) or arePermutation(pred_peptide[-3:],
                                                                                                 true_peptide[
                                                                                                 -3:])) and longest_mismatch < 6:
                        if arePermutation(pred_peptide[-3:], true_peptide[-3:]) and arePermutation(pred_peptide[0:3],
                                                                                                   true_peptide[0:3]):
                            permutations_last_and_first3 += 1
                        elif arePermutation(pred_peptide[0:3], true_peptide[0:3]):
                            permutations_first3 += 1
                        elif arePermutation(pred_peptide[-3:], true_peptide[-3:]):
                            permutations_last3 += 1
                    elif (longest_mismatch_pos == 1 and 1 >= longest_mismatch_neg <= 2) or (
                            1 >= longest_mismatch_pos <= 2 and longest_mismatch_neg == 1):
                        amount_1AA_replacements += 1
                        if pred_peptide[0] != true_peptide[0]:
                            amount_1AA_replacements_firstposition += 1
                        elif pred_peptide[0:1] != true_peptide[0:1]:
                            amount_1AA_replacements_twoposition += 1
                        if pred_peptide[-1] != true_peptide[-1]:
                            amount_1AA_replacements_lastposition += 1
                    elif longest_mismatch_pos == 2 and longest_mismatch_neg == 2:
                        amount_2AA_replacements += 1
                    elif longest_mismatch_pos == 3 and longest_mismatch_neg == 3:
                        amount_3AA_replacements += 1
                    elif longest_mismatch_pos == 4 and longest_mismatch_neg == 4:
                        amount_4AA_replacements += 1
                    elif longest_mismatch_pos == 5 and longest_mismatch_neg == 5:
                        amount_5AA_replacements += 1
                    elif longest_mismatch_pos == 6 and longest_mismatch_neg == 6:
                        amount_6AA_replacements += 1
                    elif longest_mismatch > 6:
                        amount_moreThan6AA_replacements += 1
                    else:
                        unknown_error += 1

            if total_errors == 0:
                total_errors = 1

            with open(resultdir + "error_eval.txt", "a+") as text_file:
                text_file.write("\n\nError Evaluation for " + str(tools))
                text_file.write("\nScore Cutoff: " + str(score_cutoff))
                text_file.write("\nNumber of total errors: " + str(total_errors))
                text_file.write("\nNumber of predictions: " + str(number_of_predictions))
                text_file.write("\nAmount of permutations at first three positions: in total numbers " + str(
                    permutations_first3) + " and in % " + str(permutations_first3 * 100 / total_errors))
                text_file.write("\nAmount of permutations at last three positions: in total numbers " + str(
                    permutations_last3) + " and in % " + str(permutations_last3 * 100 / total_errors))
                text_file.write(
                    "\nAmount of permutations at last three and first three posistions: in total numbers " + str(
                        permutations_last_and_first3) + " and in % " + str(
                        permutations_last_and_first3 * 100 / total_errors))
                text_file.write("\nNumber where 1 AA was replaced by 1 or 2 AA: " + str(
                    amount_1AA_replacements) + " and in % " + str(
                    amount_1AA_replacements * 100 / total_errors))
                text_file.write("\nNumber where 1 AA was replaced by 1 or 2 AA in the first position: " + str(
                    amount_1AA_replacements_firstposition) + " and in % " + str(
                    amount_1AA_replacements_firstposition * 100 / total_errors))
                text_file.write("\nNumber where 1 AA was replaced by 1 or 2 AA in the first two positions: " + str(
                    amount_1AA_replacements_twoposition) + " and in % " + str(
                    amount_1AA_replacements_twoposition * 100 / total_errors))
                text_file.write("\nNumber where 1 AA was replaced by 1 or 2 AA in the last position: " + str(
                    amount_1AA_replacements_lastposition) + " and in % " + str(
                    amount_1AA_replacements_lastposition * 100 / total_errors))
                text_file.write(
                    "\nNumber where 2 AA was replaced by 2 AA: " + str(amount_2AA_replacements) + " and in % " + str(
                        amount_2AA_replacements * 100 / total_errors))
                text_file.write(
                    "\nNumber where 3 AA was replaced by 3 AA: " + str(amount_3AA_replacements) + " and in % " + str(
                        amount_3AA_replacements * 100 / total_errors))
                text_file.write(
                    "\nNumber where 4 AA was replaced by 4 AA: " + str(amount_4AA_replacements) + " and in % " + str(
                        amount_4AA_replacements * 100 / total_errors))
                text_file.write(
                    "\nNumber where 5 AA was replaced by 5 AA: " + str(amount_5AA_replacements) + " and in % " + str(
                        amount_5AA_replacements * 100 / total_errors))
                text_file.write(
                    "\nNumber where 6 AA was replaced by 6 AA: " + str(amount_6AA_replacements) + " and in % " + str(
                        amount_6AA_replacements * 100 / total_errors))
                text_file.write("\nNumber where more than 6 AA Errors: " + str(
                    amount_moreThan6AA_replacements) + " and in % " + str(
                    amount_moreThan6AA_replacements * 100 / total_errors))
                text_file.write(
                    "\nOther Error: " + str(unknown_error) + " and in % " + str(unknown_error * 100 / total_errors))
                text_file.write("\n--------------------")
            score_cutoff = score_cutoff - 50

    summary_df_nomissingCleavages = summary_df[summary_df['Number of missing cleavages'] == 0]
    with open(resultdir + "error_eval_nomissingcleavages.txt", "w+") as text_file:
        None
    for tools in tools_list:
        score_cutoff = 0
        while (score_cutoff > -1):
            true_list = summary_df_nomissingCleavages['Modified Sequence'].tolist()
            to_test = summary_df_nomissingCleavages[tools + ' Peptide'].tolist()
            to_test_score = summary_df_nomissingCleavages[tools + ' Score'].tolist()

            longest_mismatch_sum = 0
            permutations_first3 = 0
            permutations_last3 = 0
            permutations_last_and_first3 = 0
            amount_1AA_replacements = 0
            amount_1AA_replacements_firstposition = 0
            amount_1AA_replacements_twoposition = 0
            amount_1AA_replacements_lastposition = 0
            amount_2AA_replacements = 0
            amount_3AA_replacements = 0
            amount_4AA_replacements = 0
            amount_5AA_replacements = 0
            amount_6AA_replacements = 0
            amount_moreThan6AA_replacements = 0
            unknown_error = 0
            total_errors = 0

            Which_2AA_list = []

            for i, (pred_peptide, true_peptide) in enumerate(zip(to_test, true_list)):
                if type(pred_peptide) is str and type(true_peptide) is str and to_test_score[i] >= score_cutoff:
                    changes = diff(pred_peptide, true_peptide, timelimit=0, checklines=False)
                    longest_mismatch_neg = 0
                    longest_mismatch_pos = 0
                    longest_mismatch = 0
                    longest_mismatch_exactposition = -1
                    length_seq = 0
                    for z, (op, length) in enumerate(changes):
                        length_seq += length
                        # if op == "-": print("next", length, "characters are deleted")
                        # if op == "=": print("next", length, "characters are in common")
                        # if op == "+": print("next", length, "characters are inserted")
                        if op == "+" and length > longest_mismatch_pos:
                            longest_mismatch_pos = length
                        if op == "-" and length > longest_mismatch_neg:
                            longest_mismatch_neg = length
                        if (op == "+" or op == "-") and length > longest_mismatch:
                            longest_mismatch = length
                            longest_mismatch_exactposition = length_seq - length

                    if longest_mismatch > 0:
                        total_errors += 1
                    if longest_mismatch == 0:
                        pass
                    elif (arePermutation(pred_peptide[0:3], true_peptide[0:3]) or arePermutation(pred_peptide[-3:],
                                                                                                 true_peptide[
                                                                                                 -3:])) and longest_mismatch < 6:
                        if arePermutation(pred_peptide[-3:], true_peptide[-3:]) and arePermutation(pred_peptide[0:3],
                                                                                                   true_peptide[0:3]):
                            permutations_last_and_first3 += 1
                        elif arePermutation(pred_peptide[0:3], true_peptide[0:3]):
                            permutations_first3 += 1
                        elif arePermutation(pred_peptide[-3:], true_peptide[-3:]):
                            permutations_last3 += 1
                    elif (longest_mismatch_pos == 1 and 1 >= longest_mismatch_neg <= 2) or (
                            1 >= longest_mismatch_pos <= 2 and longest_mismatch_neg == 1):
                        amount_1AA_replacements += 1
                        if pred_peptide[0] != true_peptide[0]:
                            amount_1AA_replacements_firstposition += 1
                        elif pred_peptide[0:1] != true_peptide[0:1]:
                            amount_1AA_replacements_twoposition += 1
                        if pred_peptide[-1] != true_peptide[-1]:
                            amount_1AA_replacements_lastposition += 1
                    elif longest_mismatch_pos == 2 and longest_mismatch_neg == 2:
                        amount_2AA_replacements += 1
                    elif longest_mismatch_pos == 3 and longest_mismatch_neg == 3:
                        amount_3AA_replacements += 1
                    elif longest_mismatch_pos == 4 and longest_mismatch_neg == 4:
                        amount_4AA_replacements += 1
                    elif longest_mismatch_pos == 5 and longest_mismatch_neg == 5:
                        amount_5AA_replacements += 1
                    elif longest_mismatch_pos == 6 and longest_mismatch_neg == 6:
                        amount_6AA_replacements += 1
                    elif longest_mismatch > 6:
                        amount_moreThan6AA_replacements += 1
                    else:
                        unknown_error += 1

            score_cutoff = score_cutoff - 50
            # TODO: The following should be exported as CSV!

            if total_errors == 0:
                total_errors = 1

            with open(resultdir + "error_eval_nomissingcleavages.txt", "a+") as text_file:
                text_file.write("\n\nError Evaluation for " + str(tools))
                text_file.write("\nScore Cutoff: " + str(score_cutoff + 50))
                text_file.write("\nNumber of total errors: " + str(total_errors))
                text_file.write("\nNumber of predictions: " + str(len(to_test)))
                text_file.write("\nAmount of permutations at first three positions: in total numbers " + str(
                    permutations_first3) + " and in % " + str(permutations_first3 * 100 / total_errors))
                text_file.write("\nAmount of permutations at last three positions: in total numbers " + str(
                    permutations_last3) + " and in % " + str(permutations_last3 * 100 / total_errors))
                text_file.write(
                    "\nAmount of permutations at last three and first three posistions: in total numbers " + str(
                        permutations_last_and_first3) + " and in % " + str(
                        permutations_last_and_first3 * 100 / total_errors))
                text_file.write("\nNumber where 1 AA was replaced by 1 or 2 AA: " + str(
                    amount_1AA_replacements) + " and in % " + str(
                    amount_1AA_replacements * 100 / total_errors))
                text_file.write("\nNumber where 1 AA was replaced by 1 or 2 AA in the first position: " + str(
                    amount_1AA_replacements_firstposition) + " and in % " + str(
                    amount_1AA_replacements_firstposition * 100 / total_errors))
                text_file.write("\nNumber where 1 AA was replaced by 1 or 2 AA in the first two positions: " + str(
                    amount_1AA_replacements_twoposition) + " and in % " + str(
                    amount_1AA_replacements_twoposition * 100 / total_errors))
                text_file.write("\nNumber where 1 AA was replaced by 1 or 2 AA in the last position: " + str(
                    amount_1AA_replacements_lastposition) + " and in % " + str(
                    amount_1AA_replacements_lastposition * 100 / total_errors))
                text_file.write(
                    "\nNumber where 2 AA was replaced by 2 AA: " + str(amount_2AA_replacements) + " and in % " + str(
                        amount_2AA_replacements * 100 / total_errors))
                text_file.write(
                    "\nNumber where 3 AA was replaced by 3 AA: " + str(amount_3AA_replacements) + " and in % " + str(
                        amount_3AA_replacements * 100 / total_errors))
                text_file.write(
                    "\nNumber where 4 AA was replaced by 4 AA: " + str(amount_4AA_replacements) + " and in % " + str(
                        amount_4AA_replacements * 100 / total_errors))
                text_file.write(
                    "\nNumber where 5 AA was replaced by 5 AA: " + str(amount_5AA_replacements) + " and in % " + str(
                        amount_5AA_replacements * 100 / total_errors))
                text_file.write(
                    "\nNumber where 6 AA was replaced by 6 AA: " + str(amount_6AA_replacements) + " and in % " + str(
                        amount_6AA_replacements * 100 / total_errors))
                text_file.write("\nNumber where more than 6 AA Errors: " + str(
                    amount_moreThan6AA_replacements) + " and in % " + str(
                    amount_moreThan6AA_replacements * 100 / total_errors))
                text_file.write(
                    "\nOther Error: " + str(unknown_error) + " and in % " + str(unknown_error * 100 / total_errors))
                text_file.write("\n--------------------")
    logger.debug("Error Evaluation finished.")
    return Confidence_OVER_50_AA_Prec



def convert_For_ALPS(summary_csv, kmer_ALPS, contigs_ALPS, quality_cutoff_ALPS, create_stats_results):
    logger.info("Converting to ALPS started.")
    resultdir = summary_csv.rpartition('/')[0] + '/ALPS_Assembly/'
    try:
        os.makedirs(resultdir)
    except FileExistsError:
        pass
    
    summary_df = process_summaryfile(summary_csv)
    Confidence_OVER_50_AA_Prec = []
    if create_stats_results == True:
        logger.info("Evaluation started.")
        Confidence_OVER_50_AA_Prec = generate_stats(summary_df, resultdir)
        logger.info(f"Evaluation finished. You can find the results in {resultdir}.")
    logger.info("Assembly with ALPS started.")
    process_ALPS(summary_df, resultdir, kmer_ALPS, contigs_ALPS, quality_cutoff_ALPS, Confidence_OVER_50_AA_Prec)
    logger.info("Assembly with ALPS finished.")
