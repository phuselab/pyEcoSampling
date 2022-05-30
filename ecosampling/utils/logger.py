from config import GeneralConfig

class Logger:

    def __init__(self, name):
        self.name = name
        self.verbose_flag = GeneralConfig.VERBOSE

    def _log(self, text, type=""):
        print("["+ type + self.name + "] " + text)

    def verbose(self, text):
        if self.verbose_flag:
            self._log(text, "INFO: ")

    def warn(self, text):
        self._log(text, "WARNING: ")

    def error(self, text):
        self._log(text, "ERROR: ")

    def info(self, text):
        self._log(text, "")
