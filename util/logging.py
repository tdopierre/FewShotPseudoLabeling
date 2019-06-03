import logging
import sys


class Logger:
    def __init__(self, name, logs_path=None, level=logging.INFO,
                 log_format='%(asctime)s [%(levelname)s] %(message)s'):
        if name in logging.Logger.manager.loggerDict:
            self._init_from_existant(name=name, level=level)
        else:
            self._init_from_scratch(logs_path, name, level, log_format)
        self._init_logging_functions()

    def _init_from_existant(self, name, level=None):
        self.logger = logging.getLogger(name)
        if level:
            self.level = level
            self.logger.setLevel(level)
            # self.level = self.logger.level

    def _init_from_scratch(self, logs_path, name, level=logging.INFO,
                           log_format='%(asctime)s [%(levelname)s] %(message)s'):
        self.logs_path = logs_path
        self.level = level
        self.format = log_format

        # Create logger and sets default level to DEBUG
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Adds a handler which logs to STDOUT
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _init_logging_functions(self):
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.warn = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

    def log(self, msg):
        self.logger.log(level=self.level, msg=msg)
