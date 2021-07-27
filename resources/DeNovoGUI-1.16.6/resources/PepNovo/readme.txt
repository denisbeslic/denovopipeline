PepNovo+ v3.1 (beta) de novo sequencing for MS/MS. All rights
reserved to the Regents of the University of California, 2009.

Programmed by Ari Frank, please send any questions, comments or bug reports to arf@cs.ucsd.edu.

PepNovo is a de novo sequencing algorithm for MS/MS spectra. PepNovo accepts MS/MS spectra in the following formats: dta,mgf,mzxml. This version of PepNovo is optimized for ion-trap mass spectromtetry that uses CID fragmentation (charges 1-3, dominant b/y ladders).  The only model supplied with this version is CID_IT_TRYP (CID ion-trap for mostly tryptic peptide), however data from other instruments can be processed by manipulating the fragment and PM tolerances as explained below. When the required de novo sequences are short (peptide tags 3-6 amino acids long), PepNovo generates them according to the method described in Frank, et al. (JPR 2005).

This readme file covers the following topics:
---------------------------------------------
1. Installation

2. Running PepNovo (arguments, file types, PTMs, output, etc.)
    2a command line arguments
    2b examples of commands
    2c output format
    2d models
    2e post-translational modifications (PTMs)
    2f high-resolution data

3. Fragmentation prediction using PepNovo.

4. Quality filtering and Precursor mass correction using PepNovo (see also readme_filter.txt)

5. Rescoring InsPecT/tag generation for InsPecT (see readme_mod_inspect.txt)

6. Generating queries for MS-Blast (see readme_msblast.txt)

7. PepNovo citations.


1. Installation
---------------
Windows: there is an executable file (PepNovo.change_to_exe). Rename it PepNovo.exe and it is good to go.

Linux: There is a Makefile in the src directory. Change to that directory and type "make". Then copy the file PepNovo_bin to the parent directory. Also, you might need to run dos2unix to convert the model text files to Unix format. This should be done in two stages:
dos2unix Models/*.*
dos2unix Models/*/*

2. Running PepNovo
------------------

2a Arguments
------------
PepNovo runs via command line arguments (run PepNovo without arguments to get full list):

-file <full path to input file> to specify a single input file (mgf,dta,mzXML)

   OR

-list <full path to txt file> to give a list of input files (this is the preferred method for large amounts of files since the models are not reread for each input file).

-model <model name> (currently only CID_IT_TRYP is available)

Optional arguments:

-prm        - only print spectrum graph nodes with scores

-prm_norm   - prints spectrum graph scores after normalization and removal of negative scores.

-correct_pm - finds optimal precursor mass and charge values.

-use_spectrum_charge - does not correct charge.

-use_spectrum_mz     - does not correct the precursor m/z value that appears in the file.

-no_quality_filter   - does not remove low quality spectra.

-fragment_tolerance < 0-0.75 > - the fragment tolerance (each model has a default setting)

-pm_tolerance       < 0-5.0 > - the precursor mass tolerance (each model has a default setting)

-PTMs   <PTM string>    - separated by a colons (no spaces) e.g., M+16:S+80:N+1

-digest <NON_SPECIFIC,TRYPSIN> - default TRYPSIN

-num_solutions < number > - default 20

-tag_length < 3-6> - returns peptide sequence of the specified length (only lengths 3-6 are allowed).

-model_dir  < path > - directory where model files are kept (default ./Models)

-output_aa_probs         - calculates the probabilities of individual amino acids.

-output_cumulative_probs - calculates the cumulative probabilities (that at least one sequence up to rank X is correct).


2b Examples of commands to run PepNovo:
---------------------------------------

If running under Windows, you can use the supplied executable (be sure to rename the file from PepNovo.change_to_exe to PepNovo.exe)

>PepNovo.exe –list c:\Data\MSMS\paths_of_lots_of_spectra.txt –model CID_IT_TRYP –PTMs C+57:M+16 –digest TRYPSIN

This command runs PepNovo on all the spectra files in “paths_of_lots_of_spectra.txt”  assumes that peptides were digested with trypsin and that the cystine are carbomethylated and that the methionine can be oxidized. The output is the defaults output of 20 sequences.

>PepNovo.exe –file c:\Data\MSMS\my_great_spectra.mgf –model CID_IT_TRYP  C+57:M+16 –digest NON_SPECIFIC –tag_length 3 –num_solutions 50

Runs pepnovo on a single mgf file and generates 50 tags of length 3 for each spectrum (assumes that the digest was not with trypsin).

To capture PepNovo's output in a file simply use command line redirection. For example:

PepNovo.exe –file c:\Data\MSMS\my_great_spectra.mgf –model CID_IT_TRYP  C+57:M+16 > my_output.txt

This sends the output into the text file "my_output.txt" in the current directory.


2c PepNovo output:
------------------

The output gives the following tab delimited fields for each MS/MS spectrum:
Idx – the sequence/tag rank (starts at 0)
RnkScr - the ranking score (the major score that is used)
PnvScr – the PepNovo score of the sequence (see Anal Chem 2005, and JPR 2006 for more details on the score).
N-Gap - the mass gap from the N-terminal to the start of the de novo sequence.
C-Gap - the mass gap from the C-terminal to the end of the de novo sequence.
Sequence – the predicted amino acid sequence.

2d Models:
----------
Model files should be placed in a directory called \Models which should reside in the same directory as the PepNovo executable. Model files should not be altered except for some of the values in the “_config.txt” file (such as the tolerance or parent tolerance).

Additional models can be created, for instance for MALDI or QSTAR mass spectrometers, or for other proteolytic enzymes. Training these models requires sets of identified spectra (preferably at least 10000 peptides per charge). Please contact the author if you have such data sets and wish to create additional PepNovo models.

Currently there is only one model trained on tryptic peptides:

- CID_IT_TRYP - a model for spectra of tryptic peptides acquired on low precision LTQ mass spectrometers (fragment tolerance 0.4 Da, precursor mass tolerance 2.5 Da). The previous version of PepNovo (also available from peptide.ucsd.edu) is designed for this type of data and at times might different results.


2e Post-Translational Modifications:
------------------------------------

PepNovo can be run with post-translational modifications (PTMs), though running PepNovo with many types of PTMs can degrade the performance (especially if the MS/MS fragments are not measured with high accuracy). Adding PTMs to the search is done with the command flag –PTMs followed by a colon separated list of the PTMs (e.g., -PTMs M+16:C+57:N+1). The allowed PTMs are given in the PepNovo_PTMs.txt file in the Models directory. User defined PTMs can be added to this file (following the syntax of the existing PTMs).

2f High-resolution data:
------------------------

If the MS/MS spectra come from high-resolution instruments (e.g., FT-ICR or OrbiTrap) the sequencing performance can be improved by manipulating the tolerances. For instance if the spectra have fragment tolerances of 0.01, this can be set with the flag: -fragment_tolerance 0.01 . Likewise, if the precursor mass is determined within 0.02 Da, you should use the flag -pm_tolerance 0.02. If you "trust" the spectrum's charge assignment or precursor m/z you can force PepNovo to use the values written in the spectrum file with the flags -use_spectrum_charge and -use_spectrum_mz, respectively.


3. Predicting peptide fragmentation using PepNovo
-------------------------------------------------

PepNovo can be used to predict fragmentation of peptides using a new ranking based model.
To run in this mode use a command line as follows:
PepNovo.exe -model CID_IT_TRYP -PTMs C+57:M+16  -predict_fragmentation input.txt -num_peaks 25

The input file should be a text file with peptides and charges. For example:
FGLSVLR 2
AMNGDLK 2
GGTRQSDLR 3
.
.
.

The output will be in the following format:
>> FGLSLVR      2
Rank    Ion     m/z     Score
1       y:4      474.30 3.492
2       y:3      387.27 2.549
3       y:6      644.41 2.117
.
.
.
13      b-H2O:5  500.29 0.248
14      b-H2O:3  300.17 0.128
15      b-NH3:5  501.27 0.064

Where score is proportional to the expected intensity of the peak.



4. Quality filtering and parent mass correction
-----------------------------------------------

PepNovo now has the capability to perform filtering/charge selection and parent mass correction. Currently the only model supporting this capability is CID_IT_TRYP. For more details see "readme_filter.txt".

A useful option:

-min_filter_prob <xx=0-1.0> - filter out spectra from denovo/tag/prm run with a quality probability less than x (e.g., x=0.1)


5. Creating Tags for InsPecT and Rescoring InsPecT results:
-----------------------------------------------------------

See the file readme_mod_inspect.txt for details.


6. Generating queries for MS-Blast
----------------------------------

See the file readme_msblast.txt


7. Citations:
-------------

1) The original PepNovo paper:
Frank, A. and Pevzner, P. "PepNovo: De Novo Peptide Sequencing via Probabilistic Network Modeling", Analytical Chemistry 77:964-973, 2005.

2) Paper describing PepNovo's tag generation:
Frank, A., Tanner, S., Bafna, V. and Pevzner, P. "Peptide sequence tags for fast database search in mass-spectrometry", J. Proteome Res. 2005 Jul-Aug;4(4):1287-95.

3) PepNovo used with high-precision data:
Frank, A.M., Savitski, M.M., Nielsen, L.M., Zubarev, R.A., Pevzner, P.A. "De Novo Peptide Sequencing and Identification with Precision Mass Spectrometry", J. Proteome Res. 6:114-123, 2007.

4) PepNovo+'s novel fragment prediction models:
Frank, A.M. Predicting Intensity Ranks of Peptide Fragment Ions, J. Proteome Research 2009, 8, 2226-2240.

5) PepNovo+'s novel scoring models:
Frank, A.M. A Ranking-Based Scoring Function for Peptide-Spectrum Matches. J.Proteome Research 2009, 8, 2241-2252.
