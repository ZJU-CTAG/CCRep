import logging

LOG_VERBOSE = 1
LOG_DEBUG = 2
LOG_INFO = 3
LOG_WARNING = 4
LOG_ERROR = 5
LOG_CRITICAL = 6


class Logger(object):
    ch = logging.StreamHandler()

    def __init__(self, level=LOG_DEBUG):
        self.Logger = logging.getLogger("mylogger")
        self.Logger.setLevel(5)
        self.Enabled = True
        self.LogLevel = level

    def set_enabled(self, enabled):
        self.Enabled = enabled

    def set_level(self, level):
        self.LogLevel = level

    def debug(self, module, message):
        if not self.Enabled or self.LogLevel > LOG_DEBUG:
            return
        self.font_color('\033[0;34m%s\033[0m', module)
        self.Logger.debug(message)

    def info(self, module, message):
        if not self.Enabled or self.LogLevel > LOG_INFO:
            return
        self.font_color('\033[0;32m%s\033[0m', module)
        self.Logger.info(message)

    def warning(self, module, message):
        if not self.Enabled or self.LogLevel > LOG_WARNING:
            return
        self.font_color('\033[0;33m%s\033[0m', module)
        self.Logger.warning(message)

    def error(self, module, message):
        if not self.Enabled or self.LogLevel > LOG_ERROR:
            return
        self.font_color('\033[0;31m%s\033[0m', module)
        self.Logger.error(message)

    def critical(self, module, message):
        if not self.Enabled or self.LogLevel > LOG_CRITICAL:
            return
        self.font_color('\033[0;35m%s\033[0m', module)
        self.Logger.critical(message)

    def font_color(self, color, module):
        formatter = logging.Formatter(color % f'%(asctime)s - %(levelname)s - [{module}] %(message)s')
        self.ch.setFormatter(formatter)
        self.Logger.addHandler(self.ch)


if __name__ == "__main__":
    logger = Logger(2)
    logger.info(1, "12345")
    logger.debug(1, "12345")
    logger.warning(1, "12345")
    logger.error(1, "12345")
    logger.critical(1, '1234')