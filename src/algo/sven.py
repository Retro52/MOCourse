import numpy as np
from src.core import log


def sven_algorithm(_f: callable, _x: np.ndarray, _s: np.ndarray, _l: float = 0.0, _q: float = 0.1):
    """
    :param _f: callable: target function to evaluate the growth lambda
    :param _x: np.ndarray: start point to start the search from
    :param _s: search direction
    :param _l: start lambda, 0 by default
    :param _q: growth coefficient in delta formula:
                                     ‖_x‖
                _l += _delta = _q * —————
                                     ‖_s‖
    :return: Expanded uncertainty interval for lambda value as an array of two points, sorted by ascending
    """
    _values = []

    _f_x = _f(_x + _l * _s)
    _delta = _q * np.linalg.norm(_x) / np.linalg.norm(_s)
    _direction = 1 if _f(_x + (_l + _delta) * _s) < _f(_x + _l * _s) else -1 if _f(_x + (_l - _delta) * _s) < _f(_x + _l * _s) else 0

    _iter = 0

    while True:
        # saving points history
        _values.append(_l)

        # making one more step forward
        _l_new = _l + _direction * _delta * (2 ** _iter)
        _f_x_new = _f(_x + _l_new * _s)

        if _f_x_new < _f_x:
            _iter += 1
            _l = _l_new
            _f_x = _f_x_new

        else:
            # determining the middle point
            _middle_point = (_l + _l_new) / 2
            _f_middle_point = _f(_x + _middle_point * _s)
            """
             y
             ^
             | _f_x_new
             |     |              _f_x
             |     |       _f_m    |
             |     |        |      |
             |     |        |      |
             +————————————————————————————————————————————————>x
                 _x_new   _m_p   _x
                 if _f_m < _f_x that means that _f_m is the smallest _f appeared on search, hence we just return _x_new,
                 which is always the newest and the rightest/leftest point out there
            """
            if _f_middle_point < _f_x:
                _res = np.array([_l_new, _l])
            else:
                _res = np.array([_middle_point, _values[-2 if len(_values) > 2 else -1]])

            _res.sort()

            log.program_log(log.LogLevel.Trace, f"Call to the sven_algorithm returned: {_res}")
            return _res

