# PointNovo

The DeepNovo branch contains a pytorch re-implementation of [DeepNovo](https://github.com/nh2tran/DeepNovo)

The PointNovo branch contains the implementation of our proposed PointNovo model. The software is tested on Ubuntu 1604/1804.

## Dependency
python >= 3.6

pytorch >= 1.0

dataclasses, biopython, pyteomics, cython

For database search you also need to install [percolator](http://percolator.ms/).

## data files

The ABRF DDA spectrums file could be downloaded [here](https://drive.google.com/drive/folders/1sS9fTUjcwQukUVCXLzAUufbpR0UjJfSc?usp=sharing).
The PXD008844 and PXD010559 spectra for training, validation and testing and the EThcD NIST antibody sequence data could be found [here](https://1drv.ms/u/s!AvnYi33QHIzqwyaJdF89AneoTVUY?e=BJCHqZ).


And the 9 species data (published by the DeepNovo paper) could be downloaded [here](ftp://massive.ucsd.edu/MSV000081382/peak/DeepNovo/HighResolution/).

It is worth noting that
 in our implementation we represent training samples in a slightly different format (i.e. peptide stored in a csv file and spectrums stored in mgf files).
 We also include a script for converting the file format (data_format_converter.py in PointNovo branch).
 
## knapsack files
Like DeepNovo, in PointNovo we also use the knapsack algorithm to further limit the search space. This means when performing de novo sequencing,
the program needs to either read or create a knapsack matrix based on the selected PTMs (one time computation). Pre-built knapsack matrix files could be found [here](https://1drv.ms/u/s!AvnYi33QHIzqwyaJdF89AneoTVUY?e=BJCHqZ):

You can use symbolic links to choose which knapsack file to use. i.e.

~~~
ln -s fix_C_var_NMQ_knapsack.npy knapsack.npy
~~~

## usage
### first build cython modules

~~~
make build
~~~

### train mode:

~~~
make train
~~~

On a RTX 2080 Ti GPU it takes around 0.3 seconds to train a batch of 16 annotated spectra. By default the trained model will be saved under ./train directory


### denovo mode:

~~~
make denovo
~~~

On a RTX 2080 Ti GPU it takes around 0.4 second to train a batch of 16 annotated spectra

### evaluate denovo result:

~~~
make test
~~~

This script is borrowed from the original DeepNovo implementation. It will generate the metrics defined by the paper.

### database search mode:

~~~
make db
~~~




