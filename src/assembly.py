# -*- coding: future_fstrings -*-

from json import tool
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

from config import vocab, _match_AA_novor, arePermutation, tools_list, figure_colors
from createsummary import lcs

pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


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


def process_ALPS(summary_df, resultdir, kmer_ALPS, contigs_ALPS, quality_cutoff_ALPS):
    """process summary file with ALPS to assemble full sequence
            :param
                summary_df: dataframe of different tools with score, peptide and aascore
            :return
                executed ALPS.jar and generates fasta files with top contigs
    """
    for i in tools_list:
        alps_df = summary_df[["Spectrum Name", i + " Peptide", i + " aaScore", i + " Score", "Area"]]
        peptides_mods = alps_df[i + " Peptide"]
        aaScores_mods = alps_df[i + " aaScore"]

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

        alps_df[i + " Peptide"] = [
            peptide.replace("m", "M(+15.99)").replace("q", "Q(+.98)").replace("n", "N(+.98)") if type(
                peptide) == str else None
            for peptide in peptides_mods]

        # will drop empty Peptides

        alps_df.dropna(subset=[i + " Peptide"], inplace=True)
        alps_df.to_csv(resultdir + i + '_totalscore.csv', index=False)
        subprocess.run(('java', '-jar', 'resources/ALPS.jar', resultdir + i + '_totalscore.csv', str(kmer_ALPS), str(contigs_ALPS), '>>',
                        resultdir + 'assembly.log'), stdout=subprocess.DEVNULL)

        # Quality Cut-Off for assembly at given de novo Peptide Score

        alps_df = alps_df[alps_df[i + " Score"] > quality_cutoff_ALPS]
        alps_df.to_csv(resultdir + i + f'_totalscore_cutoff_{str(quality_cutoff_ALPS)}.csv', index=False)
        subprocess.run(
            ('java', '-jar', 'resources/ALPS.jar', resultdir + i + f'_totalscore_cutoff_{str(quality_cutoff_ALPS)}.csv', str(kmer_ALPS), str(contigs_ALPS), '>>',
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

    #with open(resultdir + "stats_summary.txt", "w") as text_file:
    #    text_file.write("### STATS ###")

    
    # TODO: https://pyteomics.readthedocs.io/en/latest/examples/example_annotation.html
    # use mgf read
    # Compute stats for high noise
    # FIRST get median noise value for all spectra
    # number of missing fragment ions
    # and so on



    ## TODO Step 1
    # Calculate Peptide recall 
    
    AUC = []
    tools_stats = []
    for tools in tools_list:
        logger.info(f"Calculating precision-recall for {tools}")
        tool_AArecall = []
        tool_accuracy = []
        tool_AAprecision = []
        tool_LCS = []
        tool_scorecutoff = []
        # get recall-precision relationship by adjusting threshold for tools score
        score_cutoff = 100
        # TODO: Write 
        while (score_cutoff > -1):
            true_list = summary_df['Modified Sequence'].tolist()
            to_test = summary_df[tools + ' Peptide'].tolist()
            to_test_score = summary_df[tools + ' Score'].tolist()

            length_of_predictedAA = 0
            length_of_realAA = 0
            number_peptides = 0
            sum_peptidematch = 0
            sum_AAmatches = 0

            for i, (pred_peptide, true_peptide) in enumerate(zip(to_test, true_list)):
                length_of_realAA += len(true_peptide)
                number_peptides += 1
                if (type(pred_peptide) is str and type(true_peptide) is str and to_test_score[i] >= score_cutoff):
                    length_of_predictedAA += len(pred_peptide)
                    predicted_AA_id = [vocab[x] for x in pred_peptide]
                    target_AA_id = [vocab[x] for x in true_peptide]
                    recall_AA = _match_AA_novor(target_AA_id, predicted_AA_id)
                    sum_AAmatches += recall_AA
                    if recall_AA == len(true_peptide):
                        sum_peptidematch += 1
                else:
                    sum_AAmatches += 0
            if length_of_predictedAA != 0:
                tool_accuracy.append(str(sum_peptidematch * 100 / number_peptides))
                tool_AAprecision.append(str(sum_AAmatches * 100 / length_of_predictedAA))
                tool_AArecall.append(str(sum_AAmatches * 100 / length_of_realAA))
                tool_scorecutoff.append(score_cutoff)
            score_cutoff = score_cutoff - 5
        with open(resultdir + "stats_summary.txt", "a+") as text_file:
            text_file.write("\n\nScore Cutoff Results for " + str(tools))
            text_file.write("\nPeptide Accuracy in %: " + str(tool_accuracy))
            text_file.write("\nAA Precision in %: " + str(tool_AAprecision))
            text_file.write("\nAA Recall in %: " + str(tool_AArecall))
            text_file.write("\n--------------------")
        tool_AAprecision = [float(i) for i in tool_AAprecision]
        tool_AArecall = [float(i) for i in tool_AArecall]
        tool_accuracy = [float(i) for i in tool_accuracy]
        tool_scorecutoff = [int(i) for i in tool_scorecutoff]
        tool_scorecutoff.reverse()
        #AUC.append(metrics.auc([0] + tool_AArecall, [100] + tool_accuracy) / 10000)
        #print(tools)
        #print(tool_AArecall)
        #print(tool_accuracy)
        #print(tool_AAprecision)
        AUC.append(metrics.auc(tool_AArecall, tool_accuracy) / 10000)
        tool_falsepositiverate = [100 - float(i) for i in tool_AAprecision]
        tools_stats.append(
            (tool_AAprecision, tool_AArecall, tool_falsepositiverate, tool_accuracy, tool_scorecutoff, tool_LCS))

    # put in CSV 01 
    # TODO: Look up dataframe to CSV
    # Tool Confidence Score  Value

    # TOTAL AA RECALL

    # TOTAL AA PRECISION

    # AUC


    # TOTAL PEPTIDE RECALL
    # Tool DatasetName Value



    '''   
    ###
    ### Recall Precision Curve
    ###
    AA_recall_histoData = []
    AA_prec_histoData = []
    AA_LCS_histoData = []
    with open(resultdir + "stats_summary.txt", "a+") as text_file:
        text_file.write("\n Tab separated Table with Amino Acid Recall and Precision \n")
    plt.style.use('seaborn-colorblind')
    for i, a in enumerate(tools_list):
        plt.plot(tools_stats[i][1], tools_stats[i][0], 'o', linestyle='dashed', markersize=4, label=tools_list[i],
                 color=figure_colors[i])
        with open(resultdir + "stats_summary.txt", "a+") as text_file:
            text_file.write(str(round(tools_stats[i][1][-1], 2)) + "\t")
            AA_recall_histoData.append(round(tools_stats[i][1][-1], 2))
            AA_LCS_histoData.append(round(tools_stats[i][5][-1], 2))
    with open(resultdir + "stats_summary.txt", "a+") as text_file:
        text_file.write("\n")
    for i, a in enumerate(tools_list):
        with open(resultdir + "stats_summary.txt", "a+") as text_file:
            text_file.write(str(round(tools_stats[i][0][-1], 2)) + "\t")
            AA_prec_histoData.append(round(tools_stats[i][0][-1], 2))
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, prop={'size': 7})
    plt.xlabel('AA Recall (in %)')
    plt.ylabel('AA Precision (in %)')
    plt.tight_layout()
    plt.savefig(resultdir + "recall_precision.jpg", dpi=1200)
    tikzplotlib.save(resultdir + "recall_precision.tex")
    plt.close()

    ###
    ### AUC Histogram
    ###
    plt.style.use('seaborn-colorblind')
    plt.figure(figsize=[15, 10])
    with open(resultdir + "stats_summary.txt", "a+") as text_file:
        text_file.write("\n Tab separated Table with AUC AAlevel \n")
    with open(resultdir + "stats_summary.txt", "a+") as text_file:
        for area in AUC:
            text_file.write(str(round(area, 4)) + "\t")
    plt.bar(tools_list, AUC, width=0.5, alpha=0.7, color=figure_colors)
    plt.grid(axis='y', alpha=0.75)
    plt.ylabel('AUC', fontsize=18)
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    for i in range(len(tools_list)):
        plt.text(i, round(AUC[i], 10), round(AUC[i], 3), ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(resultdir + "AUC_histogram.jpg", dpi=1200)
    tikzplotlib.save(resultdir + "AUC_histogram.tex")
    plt.close()

    with open(resultdir + "stats_summary.txt", "a+") as text_file:
        text_file.write("\n Tab separated Table with Peptide Recall\n")

    for i, a in enumerate(tools_list):
        tools_stats[i][3].reverse()
    peptide_histo_data = []
    ###
    ### Score vs Peptide Accuracy Curve
    ###
    plt.style.use('seaborn-colorblind')
    for i, a in enumerate(tools_list):
        plt.plot(tools_stats[i][4], tools_stats[i][3], 'o', linestyle='dashed', markersize=4, label=tools_list[i],
                 color=figure_colors[i])
        with open(resultdir + "stats_summary.txt", "a+") as text_file:
            text_file.write(str(round(tools_stats[i][3][0], 2)) + "\t")
            peptide_histo_data.append(round(tools_stats[i][3][0], 2))
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.xlabel('Confidence Score')
    plt.ylabel('Peptide Recall (in %)')
    plt.tight_layout()
    plt.savefig(resultdir + "peptideacc_vs_score.jpg", dpi=1200)
    tikzplotlib.save(resultdir + "peptideacc_vs_score.tex")
    plt.close()

    for i, a in enumerate(tools_list):
        tools_stats[i][0].reverse()

    ###
    ### Score vs AA Accuracy
    ###
    plt.style.use('seaborn-colorblind')
    for i, a in enumerate(tools_list):
        plt.plot(tools_stats[i][4], tools_stats[i][0], 'o', linestyle='dashed', markersize=4, label=tools_list[i],
                 color=figure_colors[i])
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.xlabel('Confidence Score')
    plt.ylabel('AA Accuracy (in %)')
    plt.tight_layout()
    plt.savefig(resultdir + "AAprec_vs_score.jpg", dpi=1200)
    tikzplotlib.save(resultdir + "AAprec_vs_score.tex")
    plt.close()

    for i, a in enumerate(tools_list):
        tools_stats[i][4].reverse()

    ###
    ### Score vs AA Recall
    ###
    plt.style.use('seaborn-colorblind')
    for i, a in enumerate(tools_list):
        plt.plot(tools_stats[i][4], tools_stats[i][1], 'o', linestyle='dashed', markersize=4, label=tools_list[i],
                 color=figure_colors[i])
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.xlabel('Confidence Score')
    plt.ylabel('AA Recall (in %)')
    plt.tight_layout()
    plt.savefig(resultdir + "AArecall_vs_score.jpg", dpi=1200)
    tikzplotlib.save(resultdir + "AArecall_vs_score.tex")
    plt.close()

    with open(resultdir + "stats_summary.txt", "a+") as text_file:
        text_file.write("\n Tab separated Table with Peptide Recall\n")

    ###
    ### Histogram Peptide Recall
    ###
    plt.bar(tools_list, peptide_histo_data, width=0.5, alpha=0.7, color=figure_colors)
    plt.grid(axis='y', alpha=0.75)
    plt.ylabel('Peptide Recall (%)', fontsize=8)
    plt.xticks(fontsize=8, rotation=45)
    plt.yticks(fontsize=8)
    for i in range(len(tools_list)):
        plt.text(i, round(peptide_histo_data[i], 3), round(peptide_histo_data[i], 3), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(resultdir + "peptide_recall_histogram.jpg", dpi=1200)
    tikzplotlib.save(resultdir + "peptide_recall_histogram.tex")
    plt.close()

    ###
    ### Histogram Amino Acid Recall
    ###
    plt.bar(tools_list, AA_recall_histoData, width=0.5, alpha=0.7, color=figure_colors)
    plt.grid(axis='y', alpha=0.75)
    plt.ylabel('Amino Acid Recall (%)', fontsize=8)
    plt.xticks(fontsize=8, rotation=45)
    plt.yticks(fontsize=8)
    for i in range(len(tools_list)):
        plt.text(i, round(AA_recall_histoData[i], 10), round(AA_recall_histoData[i], 3), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(resultdir + "aminoacid_recall_histogram.jpg", dpi=1200)
    tikzplotlib.save(resultdir + "aminoacid_recall_histogram.tex")
    plt.close()

    ###
    ### Histogram Amino Acid Precision
    ###
    plt.bar(tools_list, AA_prec_histoData, width=0.5, alpha=0.7, color=figure_colors)
    plt.grid(axis='y', alpha=0.75)
    plt.ylabel('Amino Acid Precision (%)', fontsize=8)
    plt.xticks(fontsize=8, rotation=45)
    plt.yticks(fontsize=8)
    for i in range(len(tools_list)):
        plt.text(i, round(AA_prec_histoData[i], 10), round(AA_prec_histoData[i], 3), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(resultdir + "aminoacid_precision_histogram.jpg", dpi=1200)
    tikzplotlib.save(resultdir + "aminoacid_precision_histogram.tex")
    plt.close()'''


    ## LENGTH CUTOFF

    AUC = []
    tools_stats = []
    length_stats = []
    for tools in tools_list:
        logger.debug(tools)
        tool_AArecall = []
        tool_accuracy = []
        tool_AAprecision = []
        tool_predictedpeptides = []
        tool_scorecutoff = []
        tool_totalnumberpeptides = []
        tool_totalnumberpredictedpeptides = []

        # get recall-precision relationship by adjusting threshold for tools score
        score_cutoff = 5
        while score_cutoff < 22:
            # cutoff_df = summary_df[summary_df[tools+' Score'] >= score_cutoff]
            true_list = summary_df['Modified Sequence'].tolist()
            to_test = summary_df[tools + ' Peptide'].tolist()
            to_test_score = summary_df[tools + ' Score'].tolist()
            length_of_predictedAA = 0
            length_of_realAA = 0
            number_peptides = 0
            sum_peptidematch = 0
            sum_AAmatches = 0
            number_predicted_peptides = 0

            for i, (a, b) in enumerate(zip(to_test, true_list)):
                length_of_realAA += len(b)
                if score_cutoff == 21:
                    if (len(b) >= score_cutoff - 1):
                        number_peptides += 1
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
            if (length_of_predictedAA != 0):
                tool_AAprecision.append(str(sum_AAmatches * 100 / length_of_predictedAA))
                tool_AArecall.append(str(sum_AAmatches * 100 / length_of_realAA))
            tool_totalnumberpredictedpeptides.append(number_predicted_peptides)
            tool_predictedpeptides.append(sum_peptidematch)
            tool_totalnumberpeptides.append(number_peptides)
            tool_scorecutoff.append(score_cutoff)
            score_cutoff = score_cutoff + 1

        with open(resultdir + "stats_summary.txt", "a+") as text_file:
            text_file.write("\n\nLength Cutoff Results for " + str(tools))
            text_file.write("\nLength cutoff for " + str(tool_scorecutoff))
            text_file.write("\nNumber of correctly predicted peptides " + str(tool_predictedpeptides))
            text_file.write("\nNumber of total predicted peptides " + str(tool_totalnumberpredictedpeptides))
            text_file.write("\nNumber of total real peptides" + str(tool_totalnumberpeptides))
            text_file.write("\nPeptide Accuracy in %: " + str(tool_accuracy))
            text_file.write("\nAA Precision in %: " + str(tool_AAprecision))
            text_file.write("\nAA Recall in %: " + str(tool_AArecall))
            text_file.write("\n--------------------")

        tool_AAprecision = [float(i) for i in tool_AAprecision]
        tool_AArecall = [float(i) for i in tool_AArecall]
        tool_accuracy = [float(i) for i in tool_accuracy]
        tool_scorecutoff = [int(i) for i in tool_scorecutoff]
        tool_predictedpeptides = [int(i) for i in tool_predictedpeptides]
        tool_totalnumberpeptides = [int(i) for i in tool_totalnumberpeptides]
        tool_falsepositiverate = [100 - float(i) for i in tool_AAprecision]
        tools_stats.append((tool_AAprecision, tool_AArecall, tool_falsepositiverate, tool_accuracy, tool_scorecutoff))
        length_stats.append((tool_predictedpeptides, tool_totalnumberpeptides))

    #################################################################################################
    #                                         Error stats                                            #
    #################################################################################################

    # TODO: Error stats for High-Quality (low noise) spectra ...

    logger.info("Start calculating error statistics.")
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
            if total_errors == 0:
                total_errors = 1
            with open(resultdir + "stats_summary.txt", "a+") as text_file:
                text_file.write("\n\nError Stats for " + str(tools))
                text_file.write("\n Score Cutoff: " + str(score_cutoff + 50))
                text_file.write("\nNumber of total errors: " + str(total_errors))
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
    logger.info("Calculation of Errors finished.")


def convert_For_ALPS(summary_csv, kmer_ALPS, contigs_ALPS, quality_cutoff_ALPS, create_stats_results):
    logger.info("Converting to ALPS started.")
    resultdir = summary_csv.rpartition('/')[0] + '/ALPS_Assembly/'
    try:
        os.makedirs(resultdir)
    except FileExistsError:
        pass
    
    summary_df = process_summaryfile(summary_csv)
    #process_ALPS(summary_df, resultdir, kmer_ALPS, contigs_ALPS, quality_cutoff_ALPS)
    logger.info("Assembly with ALPS finished.")
    if create_stats_results == True:
        generate_stats(summary_df, resultdir)
