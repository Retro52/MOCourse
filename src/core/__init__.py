log_level = None


class _COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LogLevel:  # simplest form there is
    Trace = "Trace", 0
    Debug = "Debug", 1
    Warning = "Warning", 2
    Error = "Error", 3
    Important = "Info", 4


def set_log_level(_log_level):
    """
    Updates global log level
    :param _log_level: new log level
    :return: void
    """
    global log_level
    log_level = _log_level


def get_log_level():
    global log_level

    if log_level is None:
        return LogLevel.Debug

    return log_level
