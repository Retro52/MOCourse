import numpy as np
from src.core import log


def _clipping(l_list, f_list, interval):
    if f_list[0] > f_list[1]:
        interval[0] = l_list[0]
    elif f_list[0] < f_list[1]:
        interval[1] = l_list[1]
    else:
        interval = l_list
    return interval


def golden_selection(_f, _interval_original, _x: np.ndarray, _s: np.ndarray, _eps=1e-2) -> float:
    _iter = 0
    _ITER_MAX = 10000
    _inter = _interval_original.copy()
    while True:
        _a = _inter[0]
        _b = _inter[1]
        _L = _b - _a

        # checking whether it is time to return
        if abs(_b - _a) < _eps or _iter > _ITER_MAX:
            if _iter > _ITER_MAX:
                log.program_log(log.LogLevel.Warning, "golden_selection: return because _iter > _ITER_MAX")

            log.program_log(log.LogLevel.Trace, f"Call to the golden_selection returned: {(_a + _b) / 2}")
            return (_a + _b) / 2

        # calculating offsets
        _l_1 = _a + 0.382 * _L
        _l_2 = _a + 0.618 * _L

        # evaluating function values in his interval
        _f_1 = _f(_x + _l_1 * _s)
        _f_2 = _f(_x + _l_2 * _s)

        # updating interval
        _inter = _clipping([_l_1, _l_2], [_f_1, _f_2], _inter)

        # updating iterations count
        _iter += 1


def f(_x):
    return 3 * ((_x[0] - 14) ** 2) - _x[0] * _x[1] + 4 * (_x[1] ** 2)


# test: return value should be 18.20156757591134
# print(golden_selection(f, [8.826, 32.362], np.array([-20.8, -20.8]), np.array([0, 1]), 1e-2))
