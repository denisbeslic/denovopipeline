import torch
import subprocess
import cProfile
import logging
import logging.config
import config
from train_func import train, build_model, validation, perplexity
from data_reader import DeepNovoDenovoDataset, collate_func, DeepNovoTrainDataset, DBSearchDataset, denovo_collate_func
from db_searcher import DataBaseSearcher
from psm_ranker import PSMRank
from model import InferenceModelWrapper
from denovo import IonCNNDenovo
import time
from writer import DenovoWriter, PercolatorWriter
import deepnovo_worker_test
from deepnovo_dia_script_select import find_score_cutoff

logger = logging.getLogger(__name__)


def main():
    if config.FLAGS.train:
        logger.info("training mode")
        train()
    elif config.FLAGS.search_denovo:
        logger.info("denovo mode")
        data_reader = DeepNovoDenovoDataset(feature_filename=config.denovo_input_feature_file,
                                            spectrum_filename=config.denovo_input_spectrum_file)
        denovo_data_loader = torch.utils.data.DataLoader(dataset=data_reader, batch_size=config.batch_size,
                                                         shuffle=False,
                                                         num_workers=config.num_workers,
                                                         collate_fn=denovo_collate_func)
        denovo_worker = IonCNNDenovo(config.MZ_MAX,
                                     config.knapsack_file,
                                     beam_size=config.FLAGS.beam_size)
        forward_deepnovo, backward_deepnovo, init_net = build_model(training=False)
        model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo, init_net)
        writer = DenovoWriter(config.denovo_output_file)
        start_time = time.time()
        with torch.no_grad():
            denovo_worker.search_denovo(model_wrapper, denovo_data_loader, writer)
            # cProfile.runctx("denovo_worker.search_denovo(model_wrapper, denovo_data_loader, writer)", globals(), locals())
        logger.info(f"de novo {len(data_reader)} spectra takes {time.time() - start_time} seconds")
    elif config.FLAGS.valid:
        valid_set = DeepNovoTrainDataset(config.input_feature_file_valid,
                                         config.input_spectrum_file_valid)
        valid_data_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                                        batch_size=config.batch_size,
                                                        shuffle=False,
                                                        num_workers=config.num_workers,
                                                        collate_fn=collate_func)
        forward_deepnovo, backward_deepnovo, init_net = build_model(training=False)
        forward_deepnovo.eval()
        backward_deepnovo.eval()
        validation_loss = validation(forward_deepnovo, backward_deepnovo, init_net, valid_data_loader)
        logger.info(f"validation perplexity: {perplexity(validation_loss)}")

    elif config.FLAGS.test:
        logger.info("test mode")
        worker_test = deepnovo_worker_test.WorkerTest()
        worker_test.test_accuracy()

        # show 95 accuracy score threshold
        accuracy_cutoff = 0.95
        accuracy_file = config.accuracy_file
        score_cutoff = find_score_cutoff(accuracy_file, accuracy_cutoff)

    elif config.FLAGS.search_db:
        logger.info("data base search mode")
        start_time = time.time()
        db_searcher = DataBaseSearcher(config.db_fasta_file)
        dataset = DBSearchDataset(config.search_db_input_feature_file,
                                  config.search_db_input_spectrum_file,
                                  db_searcher)
        num_spectra = len(dataset)

        def simple_collate_func(train_data_list):
            return train_data_list

        data_reader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=config.num_db_searcher_worker,
                                                  collate_fn=simple_collate_func)

        forward_deepnovo, backward_deepnovo, init_net = build_model(training=False)
        forward_deepnovo.eval()
        backward_deepnovo.eval()

        writer = PercolatorWriter(config.db_output_file)
        psm_ranker = PSMRank(data_reader, forward_deepnovo, backward_deepnovo, writer, num_spectra)
        psm_ranker.search()
        writer.close()
        # call percolator
        with open(f"{config.db_output_file}" + '.psms', "w") as fw:
            subprocess.run(["percolator", "-X", "/tmp/pout.xml", f"{config.db_output_file}"],
                           stdout=fw)

    else:
        raise RuntimeError("unspecified mode")


if __name__ == '__main__':
    log_file_name = 'PointNovo.log'
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(d)
    main()
