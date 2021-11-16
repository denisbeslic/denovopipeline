import torch
import torch.nn as nn
import numpy as np
from db_searcher import accession_id_delim, PeptideCandidate
from data_reader import DBSearchDataset, DBSearchData
from model import DeepNovoPointNet, device
from writer import PercolatorWriter, PSM
import torch.nn.functional as F
import config
import logging

logger = logging.getLogger(__name__)


def wrap_infer_data(value_net_infer_data: DBSearchData):
    """

    :param value_net_infer_data:
        peak_location
    :return:
    """
    peak_location = torch.from_numpy(value_net_infer_data.peak_location).to(device)
    peak_intensity = torch.from_numpy(value_net_infer_data.peak_intensity).to(device)
    forward_id_target = torch.from_numpy(value_net_infer_data.forward_id_target).to(device)
    backward_id_target = torch.from_numpy(value_net_infer_data.backward_id_target).to(device)
    forward_ion_location_index = torch.from_numpy(value_net_infer_data.forward_ion_location_index).to(device)
    backward_ion_location_index = torch.from_numpy(value_net_infer_data.backward_ion_location_index).to(device)
    ppm = torch.from_numpy(value_net_infer_data.ppm).to(device)
    num_var_mod = torch.from_numpy(value_net_infer_data.num_var_mod).to(device)
    charge = torch.from_numpy(value_net_infer_data.charge).to(device)
    precursor_mass = torch.tensor(value_net_infer_data.precursor_mass).to(device)
    # expand tensors so that they match the requirement for multi-gpu api
    num_candidates = forward_id_target.size(0)
    peak_location = peak_location.unsqueeze(0).expand(num_candidates, -1)
    peak_intensity = peak_intensity.unsqueeze(0).expand(num_candidates, -1)
    precursor_mass = precursor_mass.unsqueeze(0).expand(num_candidates)
    return peak_location, peak_intensity, forward_id_target, backward_id_target, forward_ion_location_index, \
           backward_ion_location_index, ppm, num_var_mod, charge, precursor_mass


class PSMRank(object):
    def __init__(self, data_reader: torch.utils.data.DataLoader,
                 forward_model: DeepNovoPointNet,
                 backward_model: DeepNovoPointNet,
                 writer: PercolatorWriter,
                 num_spectra: int):
        self.data_reader = data_reader
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} gpus!")
            forward_model = nn.DataParallel(forward_model)
            forward_model.to(device)
            backward_model = nn.DataParallel(backward_model)
            backward_model.to(device)
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.forward_model.eval()
        self.backward_model.eval()
        self.writer = writer
        self.num_spectra = num_spectra

    def search(self):
        logger.info(f"searching result for {self.num_spectra} spectrums")

        with torch.no_grad():
            for i, value_net_infer_data in enumerate(self.data_reader):
                # data loader should return a list.
                value_net_infer_data: DBSearchData = value_net_infer_data[0]
                if i % 100 == 0:
                    logger.info(f"searching for {i}th/{self.num_spectra} spectra")
                if value_net_infer_data is None:
                    # no candidate peptides found
                    continue
                peaks_location, peaks_intensity, forward_id_target, backward_id_target, forward_ion_location_index, \
                backward_ion_location_index, ppm, num_var_mod, charge, precursor_mass = wrap_infer_data(
                    value_net_infer_data)

                i = -1
                # when candidates too much, split into multiple batches or will have cudnn error
                peptide_log_prob = []
                aa_length = []
                for i in range(forward_id_target.size(0) // config.inference_value_max_batch_size):
                    index_range = range(i * config.inference_value_max_batch_size,
                                        (i + 1) * config.inference_value_max_batch_size)
                    temp_forward_id_target = forward_id_target[index_range]
                    temp_forward = self.forward_model(forward_ion_location_index[index_range],
                                                      peaks_location[index_range],
                                                      peaks_intensity[index_range]
                                                      )  # (num_candidate, T, 26)
                    forward_masking_matrix = ~(temp_forward_id_target == config.PAD_ID)
                    forward_masking_matrix = forward_masking_matrix.float()  # (num_candidate, T)
                    length = torch.sum(forward_masking_matrix, dim=1)

                    forward_logprob = torch.gather(F.log_softmax(temp_forward, dim=-1),
                                                   dim=2,
                                                   index=temp_forward_id_target.unsqueeze(-1)).squeeze(2)
                    forward_logprob = forward_logprob * forward_masking_matrix
                    forward_logprob = torch.sum(forward_logprob, dim=1)  # (num_candidate,)

                    temp_backward_id_target = backward_id_target[index_range]
                    temp_backward = self.backward_model(backward_ion_location_index[index_range],
                                                        peaks_location[index_range],
                                                        peaks_intensity[index_range])  # (num_candidate, T, 26)
                    backward_masking_matrix = ~(temp_backward_id_target == config.PAD_ID)
                    backward_masking_matrix = backward_masking_matrix.float()
                    backward_logprob = torch.gather(F.log_softmax(temp_backward, dim=-1),
                                                    dim=2,
                                                    index=temp_backward_id_target.unsqueeze(-1)).squeeze(2)
                    backward_logprob = torch.sum(backward_logprob * backward_masking_matrix, dim=1)  # (num_candidate,)

                    peptide_log_prob.append(torch.max(forward_logprob, backward_logprob))
                    aa_length.append(length)

                if forward_id_target.size(0) - (i + 1) * config.inference_value_max_batch_size > 0:
                    temp_forward = self.forward_model(
                        forward_ion_location_index[(i + 1) * config.inference_value_max_batch_size:],
                        peaks_location[(i + 1) * config.inference_value_max_batch_size:],
                        peaks_intensity[(i + 1) * config.inference_value_max_batch_size:]
                        )  # (num_candidate, T, 26)
                    temp_forward_id_target = forward_id_target[(i + 1) * config.inference_value_max_batch_size:]
                    forward_masking_matrix = ~(temp_forward_id_target == 0)
                    forward_masking_matrix = forward_masking_matrix.float()  # (num_candidate, T)
                    length = torch.sum(forward_masking_matrix, dim=1)

                    forward_logprob = torch.gather(F.log_softmax(temp_forward, dim=-1),
                                                   dim=2,
                                                   index=temp_forward_id_target.unsqueeze(-1)).squeeze(2)
                    forward_logprob = torch.sum(forward_logprob * forward_masking_matrix, dim=1)  # (num_candidate,)

                    temp_backward = self.backward_model(
                        backward_ion_location_index[
                        (i + 1) * config.inference_value_max_batch_size:],
                        peaks_location[(i + 1) * config.inference_value_max_batch_size:],
                        peaks_intensity[(i + 1) * config.inference_value_max_batch_size:]
                        )  # (num_candidate, T, 26)
                    temp_backward_id_target = backward_id_target[
                                              (i + 1) * config.inference_value_max_batch_size:]
                    backward_masking_matrix = ~(temp_backward_id_target == 0)
                    backward_masking_matrix = backward_masking_matrix.float()
                    backward_logprob = torch.gather(F.log_softmax(temp_backward, dim=-1),
                                                    dim=2,
                                                    index=temp_backward_id_target.unsqueeze(-1)).squeeze(2)
                    backward_logprob = torch.sum(backward_logprob * backward_masking_matrix, dim=1)  # (num_candidate,)

                    peptide_log_prob.append(torch.max(forward_logprob, backward_logprob))
                    aa_length.append(length)

                peptide_log_prob = torch.cat(peptide_log_prob, dim=0)
                aa_length = torch.cat(aa_length, dim=0)

                length_normalized_score = peptide_log_prob / aa_length
                log_length_normalized_score = peptide_log_prob / torch.log(aa_length)

                log_length_normalized_score = log_length_normalized_score.cpu().numpy()
                length_normalized_score = length_normalized_score.cpu().numpy()

                # psm_scores = length_normalized_score

                def get_normalization_stats(psm_scores):
                    # normalizing scores so that they are comparable across spectrums.
                    k = min(config.normalizing_std_n, len(psm_scores))
                    mean_n = config.normalizing_mean_n
                    if k < config.normalizing_mean_n:
                        logger.warn(
                            f"encounter a feature with less than {config.normalizing_mean_n} candidates")
                        mean_n = k
                    top_k_ind = np.argpartition(psm_scores, -k)[-k:]
                    top_k_scores = psm_scores[top_k_ind]
                    sigma = np.std(top_k_scores) + 1e-6
                    sorted_ind = np.argsort(top_k_scores)[::-1]
                    mu = np.mean(top_k_scores[sorted_ind[:mean_n]])
                    return mu, sigma

                length_normalize_mu, length_normalize_sigma = get_normalization_stats(length_normalized_score)
                log_length_normalize_mu, log_length_normalize_sigma = get_normalization_stats(
                    log_length_normalized_score)

                from_fasta_indices = []
                from_fasta_pc_list = []
                for i, pc in enumerate(value_net_infer_data.peptide_candidates):
                    if not pc.for_dist:
                        from_fasta_indices.append(i)
                        from_fasta_pc_list.append(pc)
                # select best candidate from the PCs from the fasta file
                valid_indices = np.array(from_fasta_indices)
                psm_scores = length_normalized_score[valid_indices]
                nn = config.num_psm_per_scan_for_percolator
                if len(psm_scores) < nn:
                    logger.warning(f"do not have {nn} psm for percolator")
                    nn = len(psm_scores)
                top_n_psm_indices = np.argpartition(psm_scores, -nn)[-nn:]

                for i, index in enumerate(top_n_psm_indices):
                    candidate: PeptideCandidate = from_fasta_pc_list[index]
                    charge = int(value_net_infer_data.dda_feature.z)
                    length_score = float(length_normalized_score[valid_indices][index])
                    log_length_score = float(log_length_normalized_score[valid_indices][index])
                    exp_mass = value_net_infer_data.dda_feature.mass
                    calc_mass = candidate.mass
                    current_psm = PSM(feature_id=value_net_infer_data.dda_feature.feature_id,
                                      scan=value_net_infer_data.dda_feature.scan,
                                      num_id=i,
                                      exp_mass=exp_mass,
                                      calc_mass=calc_mass,
                                      charge=charge,
                                      peptide_str=''.join(candidate.seq),
                                      accession_id=candidate.accession_id,
                                      is_decoy=self.accession_id_is_decoy(candidate.accession_id),
                                      length_score=length_score,
                                      log_length_score=log_length_score,
                                      length_normalized_score=float(
                                          (length_score - length_normalize_mu) / length_normalize_sigma),
                                      log_length_normalized_score=float(
                                          (log_length_score - log_length_normalize_mu) / log_length_normalize_sigma),
                                      ppm=candidate.ppm,
                                      peptide_length=len(candidate.seq),
                                      num_var_mod=candidate.num_var_mod
                                      )
                    self.writer.write(current_psm)

    @staticmethod
    def accession_id_is_decoy(accession_id: str) -> bool:
        ids = accession_id.split(accession_id_delim)
        return all(["decoy" in p_id.lower() for p_id in ids])

    @staticmethod
    def identification_at_fdr(psm_list, fdr=0.01):
        """
        sort the input psm_list based on decreasing order of psm score.
        :param psm_list:
        :param fdr:
        :return:
            number of identified peptide at fdr threshold
        """
        target_count, decoy_count = 0, 0
        # sort by normalized score
        psm_list.sort(key=lambda x: x.normalized_score, reverse=True)
        for psm in psm_list:
            if not PSMRank.accession_id_is_decoy(psm.accession_id):
                target_count += 1
            else:
                decoy_count += 1
            if (decoy_count / (target_count + 1e-7)) > float(fdr):
                break
        logger.info(f"{target_count} psm identified at fdr {fdr}")
        return target_count
