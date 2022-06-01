"""Logger class file.

Authors:
    Renato Nobre <renato.avellarnobre@studenti.unimi.it>

Changes:
    31/05/2022  Python Edition
"""

from config import GeneralConfig


class Logger:
    """Logger class file for standard output and file logging.

    Parameters:
        name (str): Name of the logger.
            Usually pass __name__.

    Attributes:
        name (str): Name of the logger.
        verbose_flag (bool): Verbose flag.
    """

    def __init__(self, name):
        self.name = name
        self.verbose_flag = GeneralConfig.VERBOSE

    def _log(self, text, type=""):
        """Logger template function.

        Args:
            text (str): Message to log
            type (str, optional): Type of message.
                Defaults to "".
        """
        print("["+ type + self.name + "] " + text)

    def verbose(self, text):
        """Verbose specialized logger.

        Note:
            Only log if `VERBOSE` is set on config.py.

        Args:
            text (str): Message to log
        """
        if self.verbose_flag:
            self._log(text, "INFO: ")

    def warn(self, text):
        """Warning Specialized logger.

        Args:
            text (str): Message to log
        """
        self._log(text, "WARNING: ")

    def error(self, text):
        """Error specialized logger.

        Args:
            text (str): Message to log
        """
        self._log(text, "ERROR: ")

    def info(self, text):
        """Info specialized logger.

        Args:
            text (str): Message to log
        """
        self._log(text, "")
