import sys
import logging

from config import Config

class Logger():

    def __init__(self, name, logging_level=logging.DEBUG, filename="logger.log"):
        config = Configuration()
        logger = logging.getLogger(name)
        logger.setLevel(logging_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging_level)
        ch.setFormatter(formatter)
        file_handler = logging.FileHandler(
            filename=os.path.join(config.basedir, filename)
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(file_handler)
        self._logger = logger
