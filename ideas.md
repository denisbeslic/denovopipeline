# Future Projects / Ideas for de novo sequencing pipeline

## 1. Implementation of Rescoring Function 
   Develop Rescorer which re-ranks peptides from different tools & find the best one. Improving Precision. Maybe as a Hybrid-Tool together with Database Search. SVM. Learning to Rank.Random-Forest?
   
Would take different features of single tools into consideration: 
- AAscore, consensus sequences, Peptide Score, RTtime, Precursor Mass, etc.
- using these as features to find best common subsequence
https://github.com/semiller10/postnovo/blob/988b728fad96815fbe94a2bd4dc79ec3b417f099/classifier.py#L1482

See [PostNovo Publication](https://pubmed.ncbi.nlm.nih.gov/30277077/)
### Use of Wavelet / Fourier Transform instead of Cosine
Implement weighted cosine & use formulas from this paper, which should improve identification by 1-3%
http://europepmc.org/article/PMC/3136582#S14


## 2. Development of advanced Protein Assembler
Currently, [ALPS](https://www.nature.com/articles/srep31730) and [MetaSPS](https://pubmed.ncbi.nlm.nih.gov/22798278/) seem like the only available software for protein/antibody assembly for MS data. There is also [PASS](https://github.com/warrenlr/PASS), but it does not take single AA scores into consideration when assembling sequences. 
Implementation of an advanced assembler could improve results. Get some ideas from genomic assemblers? 
Assemble de novo or together with database results?
   

## 3. Fully automate Pipeline
As for now, the pipeline needs to be run by single commands. Future Implementation could improve this by completly automating
the whole process from formatting, preprocessing, de novo sequnecing to assembly and post-processing. (Through Snakemake? GUI?)

The pipeline was built using single commands, because the Deep Learning Tools (DeepNovo, SMSNet, PointNovo, etc.) use different dependecies
and python versions (2.7, 3.5.2, 3.6). One task could be to update all tools, so they use the same dependecies. Would also improve the runtime.

Certain attempts have already been made with the current [DeepNovo Version](https://github.com/nh2tran/DeepNovo/pull/8) (not implemented in our pipeline, because it does not produce single AA scores, unlike its older version [DeepNovo-PNAS](https://github.com/nh2tran/DeepNovo/tree/PNAS/Antibody)).
Implement DeepNovo-DIA (tensorflow=1.2), which can be used in DDA mode instead of DeepNovo-PNAS (tensorflow=0.10)

## 4. Use of genetic information 
Use of VDJ-gene regions to improve Antibody coverage.
See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5392168/
