import datetime


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


log_level = LogLevel.Trace


def set_log_level(_log_level):
    """
    Updates global log level
    :param _log_level: new log level
    :return: void
    """
    global log_level
    log_level = _log_level


def program_log(_log_level, _message: str):
    """
    Writes log message to the console
    :param _log_level:
    :param _message:
    :return:
    """
    global log_level
    if _log_level[1] < log_level[1]:
        return

    _color = _COLORS.OKCYAN
    if _log_level == LogLevel.Debug:
        _color = _COLORS.OKBLUE
    if _log_level == LogLevel.Warning:
        _color = _COLORS.WARNING
    if _log_level == LogLevel.Error:
        _color = _COLORS.FAIL

    print(_color, "LOG | {:15} | {:8}: {}".format(
        datetime.datetime.now().strftime("%m/%d/%Y | %H:%M:%S:%f"), _log_level[0], _message), _COLORS.ENDC)


# module test
# program_log(LogLevel.Trace, "Trace")
# program_log(LogLevel.Debug, "Debug")
# program_log(LogLevel.Warning, "Warning")
# program_log(LogLevel.Error, "Error")
