# -*- coding: future_fstrings -*-

import user_args
import logging
import logging.config
from datetime import datetime
import os
import sys

logger = logging.getLogger(__name__)

def main():
    test_argv = None
    logger.info("DeNovoSeq Pipeline started.")
    user_args.setup(test_argv)
    sys.exit(0)

if __name__ == '__main__':
    date = datetime.now().strftime("%Y-%m-%d_%I-%M-%S")
    log_file_name = (f"log/denovopipeline_{date}.log")
    if not os.path.exists("log/"):
        os.makedirs("log/")
    d = {
        'version': 1,
        'disable_existing_loggers': False,
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
