import user_args
import logging
import logging.config
import sys

logger = logging.getLogger(__name__)
#https://stackoverflow.com/questions/42097052/can-i-import-pythons-3-6s-formatted-string-literals-f-strings-into-older-3-x

def main():
    test_argv = None
    logger.info("DeNovoSeq Pipeline started.")
    user_args.setup(test_argv)
    sys.exit(0)

if __name__ == '__main__':
    log_file_name = 'IGNAS.log'
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
