How to train scoring models  for PepNovo
There are two types of models for PepNovo:
1.	Basic scoring models - involve aspects like fragments,  tolerance, spectrum quality, parent mass correction and PRM scoring
2.	Advanced models - these are ranking models for predicting peptide fragmentation and rescoring de novo and database search results.
Training the basic models is relatively simple and will be explained below. The advanced models are complicated to train; they require much more training data and manual steps. These steps will not be explained. If you really need to train these models then you'll have to do a lot of digging in the code (the author will be happy to help a bit, but you'll be doing much of the work).

To train the basic models you need to prepare the training data:
1.	Annotated spectra, preferably more than a few thousand (the  more the merrier).  These spectra should be placed in MGF file with a line like SEQ=AHKDLKNR  after the "BEGIN IONS" of each spectrum. Put the paths to all the training files in a text file (one path per line) and call it something like "good_training.txt"
2.	The spectra quality models require that you supply them with a set of  bad spectra too. This can be done by taking a dataset and searching it against a large decoy. Any spectrum whose top-scoring peptide was to the decoy and whose score is in the 0-30% of decoy scores is most likely a bad spectrum. Collect a bunch of these spectra and put them in MGF files. Put the paths to all the bad spectra in a file called "bad_training.txt"

Now you are ready to go. The training involves six steps, and can be done in one command (if all goes well).

The syntax of  the command is something like:
./PepNovo_bin -train_model -train_tolerance 0.5 -PTMs C+57:M+16 -list good_training.txt -neg_spec_list crap.txt -start_train_idx 0 -end_train_idx 7 -model NEW_MODEL_NAME
Where:
-train_tolerance 0.5 - means you want to train a new model where you think the fragment tolerance should be 0.5 Da (this is just a rough estimate)
-PTMs C+57:M+16 - tells about all the PTMs that are in the training data spectra (these PTMs must be in the PepNovo_PTMs.txt file in the models directory)

-list good_training.txt - all the good training samples
-neg_spec_list bad_training.txt - all the negative examples (only needed for the spectral quality models)
-model NEW - the name we want to give the model and its files (in this case NEW)

Yu can specify specific start and end training stages with the flags -strat_train_idx and -end_train_idx (you might not want or need to train all model  types).
The relevant training steps described in this document are:
0          Partitioning according to size and charge (depending on amount of training data)
1          Choosing set of fragment ion types
2           Precursor ion and fragment ion mass tolerances
3           Sequence Quality Score models (SQS)
4           Precursor Mass Correction models (PMCR)
5            Breakage score models (PRM node scores)
6	PRM score normalizers
7            Edge score models
Note that depending on the amount of training data, some stages make take a few days to train (especially stage 5). In this case, you can parallelize the process by training only specific size/charge by adding the flags -specific _size and -specific_charge. For example if charge 1 has 3 sizes (0,1, and 2) you can train each of those models using separate processes running in parallel with the following commands:
./PepNovo_bin -train_model -train_tolerance 0.5 -PTMs C+57:M+16 -list good_training.txt -start_train_idx 5 -end_train_idx 5 -model NEW_MODEL_NAME -specific_charge 1 -specific_size 0
...
./PepNovo_bin -train_model -train_tolerance 0.5 -PTMs C+57:M+16 -list good_training.txt -start_train_idx 5 -end_train_idx 5 -model NEW_MODEL_NAME -specific_charge 1 -specific_size 2
Note that the number of sizes for each charge is reported at the end of stage 0. 



After the training of these stages is complete you should have all the model files in the directory "Models". To use your new model  use its name with the "-model"  flag when you invoke PepNovo.

If there are any questions or problems please contact Ari Frank (arf@cs.ucsd.edu).


