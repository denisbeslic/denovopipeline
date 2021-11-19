# Copyright 2019 Sira Sriswasdi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os, re, gzip, math, time
from multiprocessing.pool import ThreadPool, Pool

num_thread = 10      ## number of CPU thread to use
denovo_mass_tol = 50 ## mass error for filtering de novo results, in ppm
pep_mass_tol = 20    ## mass error for comparing candidate peptide mass to theoretical mass, in ppm

input_path = '/data/users/kppt/denovo/'   ## location where de novo report file is located 
db_path = '/data/users/kppt/protein_db/'  ## location where protein database is located

db_name = 'human_all_isoform'             ## name of protein database (.fasta format)

exp_name = 'PXD009227_EGF_stimulate'      ## name of de novo report file: [exp_name]_[mode_tag]_[fdr_tag].tsv)
mode_tag = 'p-mod'
fdr_tag = 'fdr5'

if mode_tag == 'p-mod':
    mod_aas = 'MSTY'
elif mode_tag == 'm-mod':
    mod_aas = 'M'

mod_aas_lower = mod_aas.lower()
    
mass_tag_cutoff = 3
pep_mass_tol_abs = pep_mass_tol / 1000000
mass_tag_re = re.compile('(\([^\)]*\))')

aa_list = 'GASPVTCLINDQKEMHFRYWmstyUXZB'
aa_mass_list = [57.02146,71.03711,87.03203,
                97.05276,99.06841,101.04768,
                103.00919 + 57.02146,113.08406,113.08406, ## C are C mod
                114.04293,115.02694,128.05858,
                128.09496,129.04259,131.04049,
                137.05891,147.06841,156.10111,
                163.06333,186.07931,131.04049 + 15.99491, ## m = M(ox)
                87.03203 + 79.96633,101.04768 + 79.96633,163.06333 + 79.96633, ## s = S(ph), t = T(ph), y = Y(ph)
                99999, 99999, 99999, 99999] ## very large masses for ambiguous amino acids

aa_mass = {}
for i in range(len(aa_list)):
    aa_mass[aa_list[i]] = aa_mass_list[i]
    
aa_mod_delta_mass = {}
for i in range(len(aa_list)):
    if aa_list[i].lower() in aa_list:
        aa_mod_delta_mass[aa_list[i].upper()] = aa_mass[aa_list[i].lower()] - aa_mass[aa_list[i].upper()]

proton = 1.007276
water = 18.010565

min_aa_mass = min(aa_mass_list)
max_aa_mass = max(aa_mass_list)

all_results = []
unique_peptides = set()

with open(os.path.join(input_path, exp_name + '_' + mode_tag + '_' + fdr_tag + '.tsv'), 'rt') as fin:
    header = fin.readline().rstrip('\n').split('\t') ## header
    mass_tol_col = header.index('MassError(ppm)') + 1
    peptide_col = header.index('Prediction') + 1
    
    current_id = 0
    
    for line in fin.readlines():
        content = [current_id]
        content.extend(line.rstrip('\n').split('\t'))
        
        if abs(float(content[mass_tol_col])) <= denovo_mass_tol: ## filter by mass tolerance
            all_results.append(content)
            unique_peptides.add(content[peptide_col])
            
        current_id += 1
    
all_proteins = {}
protein_info = {}

with gzip.open(os.path.join(db_path, db_name + '.fasta.gz'), 'rt') as fin:
    line = fin.readline()
    
    while line:
        if line.startswith('>'):
            temp = line.split('|')
            uniprot_id = temp[1]
            primary_uniprot_id = uniprot_id.split('-')[0]

            if not 'OS=' in temp[2]:
                protein_name = temp[2]
                species_name = 'Unknown'
            else:
                temp = temp[2].split(' OS=')
                protein_name = temp[0]
                
                temp = temp[1].split('=')[0].split()
                species_name = ' '.join(temp[:-1])

            line = fin.readline()
            seq = ''

            while line and not line.startswith('>'):
                seq += line.strip()
                line = fin.readline()
            
            #if not 'U' in seq and not 'X' in seq and not 'Z' in seq and not 'B' in seq:
            if not species_name in all_proteins:
                all_proteins[species_name] = {}

            if not primary_uniprot_id in all_proteins[species_name]:
                all_proteins[species_name][primary_uniprot_id] = [[uniprot_id, seq, seq.replace('L', 'I')]]
            else:
                all_proteins[species_name][primary_uniprot_id].append([uniprot_id, seq, seq.replace('L', 'I')])

            if not uniprot_id in protein_info:
                protein_info[uniprot_id] = [species_name, primary_uniprot_id, protein_name]
        else:
            line = fin.readline()
        
def seq_to_mass(sequence):
    return [aa_mass[x] for x in sequence]

###################################
all_proteins_in_masses = {}

for sp in all_proteins:
    all_proteins_in_masses[sp] = {}
    
    for pri_id in all_proteins[sp]:
        all_proteins_in_masses[sp][pri_id] = []
        
        for entry in all_proteins[sp][pri_id]:
            all_proteins_in_masses[sp][pri_id].append(seq_to_mass(entry[1]))

def get_signature(peptide): ## the regular expression will ignore the first mass-tag
    content = re.split(mass_tag_re, peptide)
    
    if content[0] == '':
        content = content[1:]
    
    sig = []
    sig_case = []
    sig_rev = []
    sig_rev_case = []
    seed = ['', -1]
    seed_case = ['', -1]
    tag = []
    sig_flag = []
  
    for i in range(len(content)):
        if content[i].startswith('('):
            mass = float(content[i][1:-1])
            max_length = math.ceil(mass / min_aa_mass)
            min_length = math.ceil(mass / max_aa_mass)
            
            temp = [mass * (1 - pep_mass_tol_abs), mass * (1 + pep_mass_tol_abs)]
            sig.append(temp)
            sig_case.append(temp)
            sig_rev.append(temp)
            sig_rev_case.append(temp)
            sig_flag.append(False)
        elif not content[i] == '':
            sig.append(content[i].upper())
            sig_case.append(content[i])
            sig_rev.append(content[i][::-1].upper())
            sig_rev_case.append(content[i][::-1])
            sig_flag.append(True)
            
            if len(content[i]) > len(seed_case[0]):
                seed_case[0] = content[i]
                seed_case[1] = i
            
            if len(content[i]) >= mass_tag_cutoff:
                tag.append(content[i].upper())

    seed[0] = seed_case[0].upper()
    seed[1] = seed_case[1]
                
    max_prefix_len = 0
    
    for i in range(seed[1]):
        if isinstance(sig[i], str):
            max_prefix_len += len(sig[i])
        else:
            max_prefix_len += math.ceil(sig[i][1] / min_aa_mass)
            
    max_suffix_len = 0
    
    for i in range(seed[1] + 1, len(sig)):
        if isinstance(sig[i], str):
            max_suffix_len += len(sig[i])
        else:
            max_suffix_len += math.ceil(sig[i][1] / min_aa_mass)
            
    #print(sig, sig_rev, tag, seed, max_prefix_len, max_suffix_len)          
    return sig, sig_case, sig_rev, sig_rev_case, sig_flag, tag, seed, seed_case, max_prefix_len, max_suffix_len

## use lower case from pep but L from prot
def merge_seq_info(pep, prot):
    template = list(prot)

    for i in range(len(pep)):
        if pep[i] in mod_aas_lower:
            template[i] = pep[i]
    
    return ''.join(template)

## compare candidate protein section against peptide signatures (mass or seq)
## pep_sig_flag is True for string, False for mass tag
def search_hybrid(pep_sig, pep_sig_case, pep_sig_flag, prot_seq, prot_seq_noIL, prot_mass, current_prefix, mass_offset, current_pep_pos, current_prot_pos):
#     print('comparing:', pep_sig, current_pep_pos, 'and', prot_seq, current_prot_pos, current_prefix, mass_offset)
    if current_pep_pos == len(pep_sig): ## matched until the end
        return [current_prefix]
    elif current_prot_pos < len(prot_seq): ## there are some protein section left
        if pep_sig_flag[current_pep_pos]: ## string matching
            if prot_seq_noIL[current_prot_pos:].startswith(pep_sig[current_pep_pos]): ## matched
                return search_hybrid(pep_sig, pep_sig_case, pep_sig_flag, prot_seq, prot_seq_noIL, prot_mass, \
                                     current_prefix + merge_seq_info(pep_sig_case[current_pep_pos], prot_seq[current_prot_pos:(current_prot_pos + len(pep_sig[current_pep_pos]))]), \
                                     0, current_pep_pos + 1, current_prot_pos + len(pep_sig[current_pep_pos]))
            else: ## mismatched
                return None
        else: ## mass matching
            current_mass = mass_offset
            current_index = current_prot_pos
            mod_flag = prot_seq[current_index] in mod_aas
            
            while current_mass < pep_sig[current_pep_pos][0] and current_index < len(prot_seq) - 1 and not mod_flag: ## keep adding more mass
                current_mass += prot_mass[current_index]
                current_index += 1
                mod_flag = prot_seq[current_index] in mod_aas
            
            if current_mass >= pep_sig[current_pep_pos][0]: ## exceeded the lower bound of mass
                if current_mass <= pep_sig[current_pep_pos][1]: ## the right amount of mass was achieved
                    return search_hybrid(pep_sig, pep_sig_case, pep_sig_flag, prot_seq, prot_seq_noIL, prot_mass, \
                                         current_prefix + prot_seq[current_prot_pos:current_index], 0, \
                                         current_pep_pos + 1, current_index)
                
            elif current_index == len(prot_seq) - 1: ## arrived at the end of protein section, but the mass is still too low
                current_mass += prot_mass[current_index]
                
                if current_mass >= pep_sig[current_pep_pos][0] and current_mass <= pep_sig[current_pep_pos][1]: ## the right amount of mass was achieved
                    return search_hybrid(pep_sig, pep_sig_case, pep_sig_flag, '', '', [], \
                                         current_prefix + prot_seq[current_prot_pos:(current_index + 1)], 0, \
                                         current_pep_pos + 1, current_index + 1)
                
                if mod_flag: ## the next amino acid can be modified
                    current_mass += aa_mod_delta_mass[prot_seq[current_index]] ## try adding delta mass
                    
                    if current_mass >= pep_sig[current_pep_pos][0] and current_mass <= pep_sig[current_pep_pos][1]: ## the right amount of mass was achieved
                        return search_hybrid(pep_sig, pep_sig_case, pep_sig_flag, '', '', [], \
                                             current_prefix + prot_seq[current_prot_pos:(current_index + 1)], 0, \
                                             current_pep_pos + 1, current_index + 1)

            else: ## must have reached a modifiable position that is not at the end of protein section, mass is also still too low
                current_mass += prot_mass[current_index]
                
                future_nomod = search_hybrid(pep_sig, pep_sig_case, pep_sig_flag, prot_seq, prot_seq_noIL, prot_mass, \
                                             current_prefix + prot_seq[current_prot_pos:(current_index + 1)], current_mass, \
                                             current_pep_pos, current_index + 1)
                
                future_mod = search_hybrid(pep_sig, pep_sig_case, pep_sig_flag, prot_seq, prot_seq_noIL, prot_mass, \
                                           current_prefix + prot_seq[current_prot_pos:current_index] + prot_seq[current_index].lower(), \
                                           current_mass + aa_mod_delta_mass[prot_seq[current_index]], \
                                           current_pep_pos, current_index + 1)
                
                if future_nomod is None:
                    return future_mod
                elif future_mod is None:
                    return future_nomod
                else:
                    future_nomod.extend(future_mod)
                    return future_nomod
                
    return None ## return None for any situation not caught above
    
def search_main(peptide):
    return search(peptide, all_proteins, all_proteins_in_masses)

def search(peptide, proteins, masses):
    hits = []
    pep_sig, pep_sig_case, pep_sig_rev, pep_sig_rev_case, pep_sig_flag, pep_tag, pep_seed, pep_seed_case, pep_max_prefix_len, pep_max_suffix_len = get_signature(peptide)
                
    for sp in proteins:
        for pri_id in proteins[sp]:
            for i in range(len(proteins[sp][pri_id])):
                prot_info = proteins[sp][pri_id][i]
                prot_mass = masses[sp][pri_id][i]
                
                matched_tag_flag = True
                
                for tag in pep_tag:
                    if not tag in prot_info[2]: ## compare against no-IL version
                        matched_tag_flag = False
                        break

                if matched_tag_flag: ## all tags can be found
                    start = prot_info[2].find(pep_seed[0], 0)
                    
                    while start > -1: ## continue while 'seed' can be found                    
                        updated_seed = merge_seq_info(pep_seed_case[0], prot_info[1][start:(start + len(pep_seed[0]))])
    
                        if pep_seed[1] < len(pep_sig):
                            L = len(pep_seed[0])
                            forward_hit = search_hybrid(pep_sig[(pep_seed[1] + 1):], pep_sig_case[(pep_seed[1] + 1):], \
                                                        pep_sig_flag[(pep_seed[1] + 1):], \
                                                        prot_info[1][(start + L):(start + L + pep_max_suffix_len)], \
                                                        prot_info[2][(start + L):(start + L + pep_max_suffix_len)], \
                                                        prot_mass[(start + L):(start + L + pep_max_suffix_len)], '', 0, 0, 0)
                        else:
                            forward_hit = ['']

                        if not forward_hit is None:
                            if pep_seed[1] > 0:
                                L = start - pep_max_prefix_len - 1
                                reverse_hit = search_hybrid(pep_sig_rev[(pep_seed[1] - 1)::-1], pep_sig_rev_case[(pep_seed[1] - 1)::-1],
                                                            pep_sig_flag[(pep_seed[1] - 1)::-1], \
                                                            prot_info[1][(start - 1):L:-1], prot_info[2][(start - 1):L:-1], \
                                                            prot_mass[(start - 1):L:-1], '', 0, 0, 0)
                            else:
                                reverse_hit = ['']

                            if not reverse_hit is None: ## success                                              
                                for prefix_rev in reverse_hit:
                                    prefix = prefix_rev[::-1]
                                    actual_start = start + 1 - len(prefix)

                                    for suffix in forward_hit:
                                        hits.append([sp, pri_id, prot_info[0], str(actual_start), prefix + updated_seed + suffix])

                        start = prot_info[2].find(pep_seed[0], start + 1)

    return [peptide, hits]

pool = Pool(processes = num_thread)
#begin = time.time()
map_results = pool.map(search_main, unique_peptides)

pool.close()
pool.join()

#print(time.time() - begin)

with open(os.path.join(input_path, exp_name + '_' + mode_tag + '_' + fdr_tag + '_against_' + db_name + '.tsv'), 'w') as fout:
    for res in map_results:
        for entry in res[1]:
            fout.write(res[0] + '\t' + '\t'.join(entry) + '\n')
