import os
import torch
import time
import logging
import config
from typing import List
import numpy as np
from dataclasses import dataclass
from model import Direction, InferenceModelWrapper, device, torch_get_ion_index
from data_reader import BatchDenovoData
from writer import BeamSearchedSequence, DenovoWriter, DDAFeature

logger = logging.getLogger(__name__)


@dataclass
class BeamSearchStartPoint:
    prefix_mass: float
    suffix_mass: float
    mass_tolerance: float
    direction: Direction


@dataclass
class DenovoResult:
    dda_feature: DDAFeature
    best_beam_search_sequence: BeamSearchedSequence


class KnapsackSearcher(object):
    def __init__(self, MZ_MAX, knapsack_file):
        self.knapsack_file = knapsack_file
        self.MZ_MAX = MZ_MAX
        self.knapsack_aa_resolution = config.KNAPSACK_AA_RESOLUTION
        if os.path.isfile(knapsack_file):
            logging.info("KnapsackSearcher.__init__(): load knapsack matrix")
            self.knapsack_matrix = np.load(knapsack_file)
        else:
            logging.info("KnapsackSearcher.__init__(): build knapsack matrix from scratch")
            self.knapsack_matrix = self._build_knapsack()

    def _build_knapsack(self):
        max_mass = self.MZ_MAX - config.mass_N_terminus - config.mass_C_terminus
        max_mass_round = int(round(max_mass * self.knapsack_aa_resolution))
        max_mass_upperbound = max_mass_round + self.knapsack_aa_resolution
        knapsack_matrix = np.zeros(shape=(config.vocab_size, max_mass_upperbound), dtype=bool)
        for aa_id in range(3, config.vocab_size):
            mass_aa = int(round(config.mass_ID[aa_id] * self.knapsack_aa_resolution))

            for col in range(max_mass_upperbound):
                current_mass = col + 1
                if current_mass < mass_aa:
                    knapsack_matrix[aa_id, col] = False
                elif current_mass == mass_aa:
                    knapsack_matrix[aa_id, col] = True
                elif current_mass > mass_aa:
                    sub_mass = current_mass - mass_aa
                    sub_col = sub_mass - 1
                    if np.sum(knapsack_matrix[:, sub_col]) > 0:
                        knapsack_matrix[aa_id, col] = True
                        knapsack_matrix[:, col] = np.logical_or(knapsack_matrix[:, col], knapsack_matrix[:, sub_col])
                    else:
                        knapsack_matrix[aa_id, col] = False
        np.save(self.knapsack_file, knapsack_matrix)
        return knapsack_matrix

    def search_knapsack(self, mass, knapsack_tolerance):
        mass_round = int(round(mass * self.knapsack_aa_resolution))
        mass_upperbound = mass_round + knapsack_tolerance
        mass_lowerbound = mass_round - knapsack_tolerance
        if mass_upperbound < config.mass_AA_min_round:
            return []
        mass_lowerbound_col = mass_lowerbound - 1
        mass_upperbound_col = mass_upperbound - 1
        candidate_aa_id = np.flatnonzero(np.any(self.knapsack_matrix[:, mass_lowerbound_col:(mass_upperbound_col + 1)],
                                                axis=1))
        return candidate_aa_id.tolist()


@dataclass
class SearchPath:
    aa_id_list: list
    aa_seq_mass: float
    score_list: list
    score_sum: float
    lstm_state: tuple  # state tupe store in search path is of shape [num_lstm_layers, num_units]
    direction: Direction


@dataclass
class SearchEntry:
    feature_index: int
    current_path_list: list  # list of search paths
    spectrum_state: tuple  # tuple of (peak_location, peak_intensity)


class IonCNNDenovo(object):
    def __init__(self, MZ_MAX, knapsack_file, beam_size):
        self.MZ_MAX = MZ_MAX  # legacy, not used here
        self.beam_size = beam_size
        self.knapsack_searcher = KnapsackSearcher(MZ_MAX, knapsack_file)

    def _beam_search(self, model_wrapper: InferenceModelWrapper,
                     batch_denovo_data: BatchDenovoData, start_point_batch: list) -> list:
        """

        :param model_wrapper:
        :param batch_denovo_data:
        :param start_point_batch:
        :return:
        """
        num_features = len(batch_denovo_data.original_dda_feature_list)
        top_path_batch = [[] for _ in range(num_features)]

        direction_cint_map = {Direction.forward: 0, Direction.backward: 1}

        direction = start_point_batch[0].direction
        if direction == Direction.forward:
            get_start_mass = lambda x: x.prefix_mass
            first_label = config.GO_ID
            last_label = config.EOS_ID
        elif direction == Direction.backward:
            get_start_mass = lambda x: x.suffix_mass
            first_label = config.EOS_ID
            last_label = config.GO_ID
        else:
            raise ValueError('direction neither forward nor backward')

        # step 1: extract original spectrum

        batch_peak_location = batch_denovo_data.peak_location.to(device)
        batch_peak_intensity = batch_denovo_data.peak_intensity.to(device)
        batch_spectrum_representation = batch_denovo_data.spectrum_representation.to(device)

        initial_hidden_state_tuple = model_wrapper.initial_hidden_state(batch_spectrum_representation) if \
            config.use_lstm else None

        # initialize activate search list
        active_search_list = []
        for feature_index in range(num_features):
            # all feature in the same batch should be from same direction
            assert direction == start_point_batch[feature_index].direction

            spectrum_state = (batch_peak_location[feature_index], batch_peak_intensity[feature_index])

            if config.use_lstm:
                lstm_state = (initial_hidden_state_tuple[0][:, feature_index, :],
                              initial_hidden_state_tuple[1][:, feature_index, :]
                              )
            else:
                lstm_state = None

            path = SearchPath(
                aa_id_list=[first_label],
                aa_seq_mass=get_start_mass(start_point_batch[feature_index]),
                score_list=[0.0],
                score_sum=0.0,
                lstm_state=lstm_state,
                direction=direction,
            )
            search_entry = SearchEntry(
                feature_index=feature_index,
                current_path_list=[path],
                spectrum_state=spectrum_state,
            )
            active_search_list.append(search_entry)

        # repeat STEP 2, 3, 4 until the active_search_list is empty.
        while True:
            # STEP 2: gather data from active search entries and group into blocks.

            # model input
            block_aa_id_input = []
            block_ion_location_param = []
            block_peak_location = []
            block_peak_intensity = []
            block_lstm_h = []
            block_lstm_c = []
            # data stored in path
            block_aa_id_list = []
            block_aa_seq_mass = []
            block_score_list = []
            block_score_sum = []
            block_knapsack_candidates = []

            # store the number of paths of each search entry in the big blocks
            #     to retrieve the info of each search entry later in STEP 4.
            search_entry_size = [0] * len(active_search_list)

            for entry_index, search_entry in enumerate(active_search_list):
                feature_index = search_entry.feature_index
                current_path_list = search_entry.current_path_list
                precursor_mass = batch_denovo_data.original_dda_feature_list[feature_index].mass
                peak_mass_tolerance = start_point_batch[feature_index].mass_tolerance

                for path in current_path_list:
                    aa_id_list = path.aa_id_list
                    aa_id = aa_id_list[-1]
                    score_sum = path.score_sum
                    aa_seq_mass = path.aa_seq_mass
                    score_list = path.score_list
                    original_spectrum_tuple = search_entry.spectrum_state
                    lstm_state_tuple = path.lstm_state

                    if aa_id == last_label:
                        if abs(aa_seq_mass - precursor_mass) <= peak_mass_tolerance:
                            seq = aa_id_list[1:-1]
                            trunc_score_list = score_list[1:-1]
                            if direction == Direction.backward:
                                seq = seq[::-1]
                                trunc_score_list = trunc_score_list[::-1]

                            top_path_batch[feature_index].append(
                                BeamSearchedSequence(sequence=seq,
                                                     position_score=trunc_score_list,
                                                     score=path.score_sum / len(seq))
                            )
                        continue
                    if not block_ion_location_param:
                        block_ion_location_param = [[precursor_mass], [aa_seq_mass], direction_cint_map[direction]]
                    else:
                        block_ion_location_param[0].append(precursor_mass)
                        block_ion_location_param[1].append(aa_seq_mass)

                    residual_mass = precursor_mass - aa_seq_mass - config.mass_ID[last_label]
                    knapsack_tolerance = int(round(peak_mass_tolerance * config.KNAPSACK_AA_RESOLUTION))
                    knapsack_candidates = self.knapsack_searcher.search_knapsack(residual_mass, knapsack_tolerance)
                    if not knapsack_candidates:
                        # if not possible aa, force it to stop.
                        knapsack_candidates.append(last_label)

                    block_aa_id_input.append(aa_id)
                    # get hidden state block
                    block_peak_location.append(original_spectrum_tuple[0])
                    block_peak_intensity.append(original_spectrum_tuple[1])
                    if config.use_lstm:
                        block_lstm_h.append(lstm_state_tuple[0])
                        block_lstm_c.append(lstm_state_tuple[1])

                    block_aa_id_list.append(aa_id_list)
                    block_aa_seq_mass.append(aa_seq_mass)
                    block_score_list.append(score_list)
                    block_score_sum.append(score_sum)
                    block_knapsack_candidates.append(knapsack_candidates)
                    # record the size of each search entry in the blocks
                    search_entry_size[entry_index] += 1

            # step 3 run model on data blocks to predict next AA.
            #     output is stored in current_log_prob
            # assert block_aa_id_list, 'IonCNNDenovo._beam_search(): aa_id_list is empty.'
            if not block_ion_location_param:
                # all search entry finished in the previous step
                break

            block_ion_location = torch_get_ion_index(*block_ion_location_param)
            block_peak_location = torch.stack(block_peak_location, dim=0).contiguous()
            block_peak_intensity = torch.stack(block_peak_intensity, dim=0).contiguous()
            if config.use_lstm:
                block_lstm_h = torch.stack(block_lstm_h, dim=1).contiguous()
                block_lstm_c = torch.stack(block_lstm_c, dim=1).contiguous()
                block_state_tuple = (block_lstm_h, block_lstm_c)
                block_aa_id_input = torch.from_numpy(np.array(block_aa_id_input, dtype=np.int64)).unsqueeze(1).to(
                    device)
            else:
                block_state_tuple = None
                block_aa_id_input = None

            current_log_prob, new_state_tuple = model_wrapper.step(block_ion_location,
                                                                   block_peak_location,
                                                                   block_peak_intensity,
                                                                   block_aa_id_input,
                                                                   block_state_tuple,
                                                                   direction)
            # transfer log_prob back to cpu
            current_log_prob = current_log_prob.cpu().numpy()  # [block_size, num_tokens]

            # STEP 4: retrieve data from blocks to update the active_search_list
            #     with knapsack dynamic programming and beam search.
            block_index = 0
            for entry_index, search_entry in enumerate(active_search_list):
                new_path_list = []
                direction = search_entry.current_path_list[0].direction
                for index in range(block_index, block_index + search_entry_size[entry_index]):
                    for aa_id in block_knapsack_candidates[index]:
                        if aa_id > 2:
                            # do not add score of GO, EOS, PAD
                            new_score_list = block_score_list[index] + [current_log_prob[index][aa_id]]
                            new_score_sum = block_score_sum[index] + current_log_prob[index][aa_id]
                        else:
                            new_score_list = block_score_list[index] + [0.0]
                            new_score_sum = block_score_sum[index] + 0.0

                        if config.use_lstm:
                            new_path_state_tuple = (new_state_tuple[0][:, index, :], new_state_tuple[1][:, index, :])
                        else:
                            new_path_state_tuple = None

                        new_path = SearchPath(
                            aa_id_list=block_aa_id_list[index] + [aa_id],
                            aa_seq_mass=block_aa_seq_mass[index] + config.mass_ID[aa_id],
                            score_list=new_score_list,
                            score_sum=new_score_sum,
                            lstm_state=new_path_state_tuple,
                            direction=direction
                        )
                        new_path_list.append(new_path)
                if len(new_path_list) > self.beam_size:
                    new_path_score = np.array([x.score_sum for x in new_path_list])
                    top_k_index = np.argpartition(-new_path_score, self.beam_size)[:self.beam_size]
                    search_entry.current_path_list = [new_path_list[ii] for ii in top_k_index]
                else:
                    search_entry.current_path_list = new_path_list

                block_index += search_entry_size[entry_index]

            active_search_list = [x for x in active_search_list if x.current_path_list]

            if not active_search_list:
                break
        return top_path_batch

    @staticmethod
    def _get_start_point(batch_denovo_data: BatchDenovoData) -> tuple:
        mass_GO = config.mass_ID[config.GO_ID]
        forward_start_point_lists = [BeamSearchStartPoint(prefix_mass=mass_GO,
                                                          suffix_mass=dda_feature.mass - mass_GO,
                                                          mass_tolerance=config.PRECURSOR_MASS_PRECISION_TOLERANCE,
                                                          direction=Direction.forward)
                                     for dda_feature in batch_denovo_data.original_dda_feature_list]

        mass_EOS = config.mass_ID[config.EOS_ID]
        backward_start_point_lists = [BeamSearchStartPoint(prefix_mass=dda_feature.mass - mass_EOS,
                                                           suffix_mass=mass_EOS,
                                                           mass_tolerance=config.PRECURSOR_MASS_PRECISION_TOLERANCE,
                                                           direction=Direction.backward)
                                      for dda_feature in batch_denovo_data.original_dda_feature_list]
        return forward_start_point_lists, backward_start_point_lists

    @staticmethod
    def _select_path(batch_denovo_data: BatchDenovoData, top_candidate_batch: list) -> list:
        """
        for each feature, select the best denovo sequence given by DeepNovo model
        :param batch_denovo_data:
        :param top_candidate_batch: defined in _search_denovo_batch
        :return:
        list of DenovoResult
        """
        feature_batch_size = len(batch_denovo_data.original_dda_feature_list)

        refine_batch = [[] for x in range(feature_batch_size)]
        for feature_index in range(feature_batch_size):
            precursor_mass = batch_denovo_data.original_dda_feature_list[feature_index].mass
            candidate_list = top_candidate_batch[feature_index]
            for beam_search_sequence in candidate_list:
                sequence = beam_search_sequence.sequence
                sequence_mass = sum(config.mass_ID[x] for x in sequence)
                sequence_mass += config.mass_ID[config.GO_ID] + config.mass_ID[
                    config.EOS_ID]
                if abs(sequence_mass - precursor_mass) <= config.PRECURSOR_MASS_PRECISION_TOLERANCE:
                    logger.debug(f"sequence {sequence} of feature "
                                 f"{batch_denovo_data.original_dda_feature_list[feature_index].feature_id} refined")
                    refine_batch[feature_index].append(beam_search_sequence)
        predicted_batch = []
        for feature_index in range(feature_batch_size):
            candidate_list = refine_batch[feature_index]
            if not candidate_list:
                best_beam_search_sequence = BeamSearchedSequence(
                    sequence=[],
                    position_score=[],
                    score=-float('inf')
                )
            else:
                # sort candidate sequence by average position score
                best_beam_search_sequence = max(candidate_list, key=lambda x: x.score)

            denovo_result = DenovoResult(
                dda_feature=batch_denovo_data.original_dda_feature_list[feature_index],
                best_beam_search_sequence=best_beam_search_sequence
            )
            predicted_batch.append(denovo_result)
        return predicted_batch

    def _search_denovo_batch(self, batch_denovo_data: BatchDenovoData, model_wrapper: InferenceModelWrapper) -> list:
        start_time = time.time()
        feature_batch_size = len(batch_denovo_data.original_dda_feature_list)
        start_points_tuple = self._get_start_point(batch_denovo_data)
        top_candidate_batch = [[] for x in range(feature_batch_size)]

        for start_points in start_points_tuple:
            beam_search_result_batch = self._beam_search(model_wrapper, batch_denovo_data, start_points)
            for feature_index in range(feature_batch_size):
                top_candidate_batch[feature_index].extend(beam_search_result_batch[feature_index])
        predicted_batch = self._select_path(batch_denovo_data, top_candidate_batch)
        test_time = time.time() - start_time
        logger.info("beam_search(): batch time {}s".format(test_time))
        return predicted_batch

    def search_denovo(self, model_wrapper: InferenceModelWrapper,
                      beam_search_reader: torch.utils.data.DataLoader,
                      denovo_writer: DenovoWriter) -> List[DenovoResult]:
        logger.info("start beam search denovo")
        predicted_denovo_list = []
        total_batch_num = int(len(beam_search_reader.dataset) / config.batch_size)
        for index, batch_denovo_data in enumerate(beam_search_reader):
            logger.info("Read {}th/{} batches".format(index, total_batch_num))
            predicted_batch = self._search_denovo_batch(batch_denovo_data, model_wrapper)
            predicted_denovo_list += predicted_batch
            for denovo_result in predicted_batch:
                denovo_writer.write(denovo_result.dda_feature, denovo_result.best_beam_search_sequence)

        return predicted_denovo_list

