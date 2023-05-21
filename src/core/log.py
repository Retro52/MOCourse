import datetime
from src.core import get_log_level
from src.core import LogLevel, _COLORS


def program_log(_log_level, _message: str):
    """
    Writes log message to the console
    :param _log_level:
    :param _message:
    :return:
    """

    if _log_level[1] < get_log_level()[1]:
        return

    _color = _COLORS.OKCYAN
    if _log_level == LogLevel.Debug:
        _color = _COLORS.OKBLUE
    if _log_level == LogLevel.Warning:
        _color = _COLORS.WARNING
    if _log_level == LogLevel.Error:
        _color = _COLORS.FAIL
    if _log_level == LogLevel.Important:
        _color = _COLORS.OKGREEN

    print(_color, "LOG | {:15} | {:8}: {}".format(
        datetime.datetime.now().strftime("%m/%d/%Y | %H:%M:%S:%f"), _log_level[0], _message), _COLORS.ENDC)


# module test
# program_log(LogLevel.Trace, "Trace")
# program_log(LogLevel.Debug, "Debug")
# program_log(LogLevel.Warning, "Warning")
# program_log(LogLevel.Error, "Error")
