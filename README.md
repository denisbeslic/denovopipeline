# Denovopipeline

Denovopipeline uses multiple de novo sequencing algorithms ([pNovo3](http://pfind.ict.ac.cn/software/pNovo/index.html), [SMSNet](https://github.com/cmb-chula/SMSNet/tree/master#readme), [Novor](https://github.com/compomics/denovogui), [DeepNovo](https://github.com/nh2tran/DeepNovo), and [PointNovo](https://github.com/volpato30/PointNovo) for identification and assembly of peptide sequences by tandem mass spectrometry.

## How to use

### Download pre-trained models
To download the pre-trained models for PointNovo, SMSNet and DeepNovo use following link:
https://drive.google.com/drive/folders/1LFmez1yq7eXNTNs7IWhYy9vQpLzD8rLI?usp=sharing

Move each corressonding model to the resources/ directory of each de novo sequencing tool.
Move `knapsack.npy` to `resources/PointNovo` and `resources/DeepNovo`. 

### Format raw data

De novo sequencing requires mgf (Mascot generic format) files.

You can use Proteowizard `msconvert` to convert your .raw/.mzxml/.mzml files to .mgf format. Proteowizard can be simply installed using [conda](https://anaconda.org/bioconda/proteowizard).

`msconvert preformatted_spectra.raw --mgf --filter "peakPicking vendor" --filter "zeroSamples removeExtra"--filter "titleMaker Run: <RunId>, Index: <Index>, Scan: <ScanNumber>"`

Your .mgf file should now look like this
```
START IONS
TITLE= Run: antibody_tryptic_1, Index: 745, Scan: 756
RTINSECONDS=199.46399
PEPMASS=593.2639
CHARGE=2+
145.9389801 163.0
145.9406433 490.0
145.9423218 762.0
...
END IONS
```

### Reformat mgf file
Another reformatting operation is necessary, because certain tools ignore the old indexing and do not work with predefined
IDs. Spectrum indices and scan IDs are changed to integers from 1 to N. Information on old IDs is preserved in the TITLE line.

`python main.py reformatMGF --input YOURDATA.mgf --output YOURDATA_reformatted.mgf`

This will produce two .mgf files. One called YOURDATA_reformatted_deepnovo.mgf for DeepNovo and PointNovo and another one called YOURDATA_reformatted.mgf for all other tools. The file for DeepNovo includes the 'SEQ=' line, which is necessary for DeepNovo to run.

Your final *_reformatted.mgf file should now look like this.
```
START IONS
TITLE=Run: Light-Chain-Trypsin-1, Index: 1, Old index: 3175, Old scan: 3176
PEPMASS=404.12344
CHARGE=2+
SCANS=1
RTINSECONDS=852.852
SEQ=AAAAAA
145.9389801 163.0
...
END IONS
```
It is very important that your files use the same formatting. Otherwise, the de novo sequencing and summary step will not work correctly.


### Build Conda Environments

Since DeepNovo, SMSNet and Python have different requirements regarding their Python version and other dependencies, we recommend
using [conda](https://docs.anaconda.com/anaconda/install/) to build virtual environments.
```
conda env create -n deepnovo -f envs/requirements_deepnovo.yml
conda env create -n smsnet -f envs/requirements_smsnet.yml
conda env create -n pointnovo -f envs/requirements_pointnovo.yml
conda env create -n denovopipeline -f envs/requirements_denovopipeline.yml
```

### Run de novo sequencing tools

Novor will be executed by using DeNovoCLI from DeNovoGUI. It is necessary to provide a parameter file. We recommend using the [instructions from DeNovoCLI](https://github.com/compomics/denovogui/wiki/IdentificationParametersCLI).

DeepNovo, SMSNet and PointNovo need to be run separately, because they use different dependencies. We provide some pre-trained models, but it is recommended to train models yourself. You can change the model each DeepLearning Tool is using by the command line arguments `--smsnet_model`, `--deepnovo_model` and `--pointnovo_model`

Important: PointNovo and DeepNovo require the *_reformatted_deepnovo.mgf, while SMSNet uses the *reformatted.mgf as input for the prediction. 

pNovo3 can only run on Windows and does not work within the pipeline. You can run it separately by following the [instructions on its website](http://pfind.ict.ac.cn/software/pNovo/index.html) and put its final output in the results directory.

Use following commands
```
conda activate smsnet
python src/main.py denovo --input example_dataset/YOURDATA_reformatted.mgf --output example_dataset/results --denovogui 1 --smsnet 1


conda activate deepnovo
python src/main.py denovo --input example_dataset/YOURDATA_reformatted_deepnovo.mgf --output example_dataset/results --deepnovo 1


conda activate pointnovo
python src/main.py denovo --input example_dataset/YOURDATA_reformatted_deepnovo.mgf --output example_dataset/results --pointnovo 1
```

### Postprocessing

After having used all desired utilized all denovo tools, use `denovo_summary` to generate the summary file. 
You need to specify the directory where all the de novo results are stored and provide your initial reformatted mgf file to correctly assign the predictions to each spectrum. Additionally, you can specify a feature file from Peptide Shaker to compare your de novo sequencing results with Database results. We recommend using DeNovoGUI and PeptideShaker and exporting the "Default PSM Report with non-validated matches".

``` 
conda activate denovopipeline
python src/main.py summary --input example_dataset/YOURDATA_reformatted.mgf --results example_dataset/results/ --db example_dataset/results/Default\ PSM\ Report\ with\ non-validated\ matches.txt
```

The summary file will be generated in your results directory and include Spectrum Title, Peptide Prediction, Peptide Score, Single Amino Acid score for each tool.


### Assembly results

To finally assembly the sequence, use convertForALPS

```
conda activate denovopipeline
python src/main.py convertForALPS --input example_dataset/results/summary.csv
```
The command will split up the summary file and generate contigs for each tool in results/ALPS_Assembly. Additionally, it will also generate CSVs with information about the Peptide Recall, AA Recall, AA Precision 

