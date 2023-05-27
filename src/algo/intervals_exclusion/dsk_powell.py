import numpy as np
from src.core import log


def dsk_powell(_f, _inter, _x: np.ndarray, _s: np.ndarray, _eps: float = 1e-2):
    _iter = 1
    _l_0 = _inter[0]
    _l_1 = (_inter[0] + _inter[1]) / 2
    _l_2 = _inter[1]

    _f_0 = _f(_x + _l_0 * _s)
    _f_1 = _f(_x + _l_1 * _s)
    _f_2 = _f(_x + _l_2 * _s)

    if (2 * (_f_0 - 2 * _f_1 + _f_2)) == 0:
        return _l_1

    _l = _l_1 + (_l_1 - _l_0) * (_f_0 - _f_2) / (2 * (_f_0 - 2 * _f_1 + _f_2))

    while True:

        if abs(_f_1 - _f(_x + _l * _s)) < _eps and abs(_l_1 - _l) < _eps:
            return _l

        if _l < _l_1:
            _l_2 = _l_1
            _l_1 = _l
        elif _l > _l_1:
            _l_0 = _l_1
            _l_1 = _l
        else:
            return _l

        _f_list = [_f_0, _f_1, _f_2]

        _a1 = (_f_list[1] - _f_list[0]) / (_l_1 - _l_0)
        _a2 = ((_f_list[2] - _f_list[0]) / (_l_2 - _l_0) - _a1) / (_l_2 - _l_1)
        _l = (_l_0 + _l_1)/2 - _a1/(2*_a2)
        _f_min = min(_f_list)

        if _f_list[0] == _f_min:
            _l_1 = _l_0
        elif _f_list[1] != _f_min:
            _l_1 = _l_2

        _f_0 = _f(_x + _l_0 * _s)
        _f_1 = _f(_x + _l_1 * _s)
        _f_2 = _f(_x + _l_2 * _s)

        _iter += 1


# def f(_x):
#     return 3 * ((_x[0] - 14) ** 2) - _x[0] * _x[1] + 4 * (_x[1] ** 2)
#
#
# # test: return value should be 18.19101301253714
# print(dsk_powell(f, [8.826, 32.362], np.array([-20.8, -20.8]), np.array([0, 1]), 1e-2))
