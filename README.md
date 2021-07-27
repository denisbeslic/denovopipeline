# denovopipeline
denovopipeline uses multiple de novo sequencing algorithms ([pNovo3](http://pfind.ict.ac.cn/software/pNovo/index.html), [SMSNet](https://github.com/cmb-chula/SMSNet/tree/master#readme), [Novor](https://github.com/compomics/denovogui), [DeepNovo](https://github.com/nh2tran/DeepNovo), [PointNovo](https://github.com/volpato30/PointNovo), [PepNovo+](https://pubmed.ncbi.nlm.nih.gov/15858974/)) identification and assembly of peptide sequences by tandem mass spectrometry.

## How to use

### Download public data
To download the test data, pre-trained models for PointNovo, SMSNet and DeepNovo use following link:
https://1drv.ms/f/s!AuC2G9KYrL1KgZJuI0hxrv5QznTU5w

Move the files to their corresponding directories without renaming.

`smsnet_phospho/` to `resources/SMSNet`

`train/` to `resources/PointNovo`

`train.doremon.resolution_50.epoch_20.da_4500.ab.training.mouse/` to `resources/DeepNovo`

`knapsack.npy` to `resources/PointNovo` and `resources/DeepNovo`


### Format raw data

The de novo sequencing step requires mgf (Mascot generic format) files.

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
[...]
END IONS
```

### Reformat mgf file
Another reformatting operation is necessary, because certain tools ignore the old indexing and do not work with predefined
IDs.
Spectrum indices and scan IDs are changed to integers from 1 to N. Information on old IDs is preserved in the TITLE line.

`python main.py reformatMGF --input YOURDATA.mgf --output YOURDATA_reformatted.mgf`

It will produce two .mgf files. One called YOURDATA_reformatted_deepnovo.mgf for DeepNovo and PointNovo and another one called YOURDATA_reformatted.mgf for all other tools.
The file for DeepNovo includes the 'SEQ=' line, which is necessary for DeepNovo to run.

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
[...]
END IONS
```
It is very important that your files use the same formatting. Otherwise, the de novo sequencing step will not work correctly.


### Build Conda Environments

Because DeepNovo and SMSNet have different requirements regarding their Python version and other dependencies, we recommend
using conda to build virtual environments.

See https://docs.anaconda.com/anaconda/install/ for installation instructions.

After installation create environments for DeepNovo and SMSNet by:

```
conda create -n deepnovo python=2.7
conda create -n smsnet python=3.5.2 --channel conda-forge
conda create -n pointnovo python=3.6
```
Use the requirements.txt in the main folder to install all dependencies.

```
conda activate deepnovo
pip install -r requirements_deepnovo.txt
conda activate smsnet
pip install -r requirements_smsnet.txt
conda activate pointnovo
pip install -r requirements_pointnovo.txt
```

### Run de novo sequencing tools

Novor and PepNovo+ will be executed by using DeNovoCLI from DeNovoGUI. It is necessary to provide a parameter file. We recommend using the [instructions from DeNovoCLI](https://github.com/compomics/denovogui/wiki/IdentificationParametersCLI).

PointNovo, SMSNet and DeepNovo need to be run separately, because they use different dependencies. We provide some pre-trained models, but it is recommended to train models yourself. You can change the model each DeepLearning Tool is using by using the command line arguments `--smsnet_model`, `--deepnovo_model` and `--pointnovo_model`

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

After having used all desired denovo tools, use `denovo_summary` to generate the summary file. 
You need to specify the directory where all the de novo results are stored and provide your initial reformatted mgf file to correctly assign the predictions to each spectrum.

``` 
python src/main.py summary  summary --input example_dataset/YOURDATA_reformatted.mgf --results example_dataset/results/
```
The summary file will be generated in your results directory and include Spectrum Title, Peptide Prediction, Peptide Score, Single Amino Acid score for each tool.
It will also generate a "BEST" column, which compares the scores the tools and chooses the one with the highest score for each spectrum.


### Assembly results

To finally assembly the sequence, use convertForALPS

```
python src/main.py convertForALPS --input example_dataset/results/summary.csv
```
The command will split up the summary file and generate contigs for each tool in results/ALPS_Assembly.



