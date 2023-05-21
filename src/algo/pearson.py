import numpy as np
from src.core import log
from src.algo import sven
from src.algo.intervals_exclusion import golden_selection


def pearson2(_f, _df, _x0,
             _max_iterations: int = 100000,
             _eps: float = 1e-4,
             _sv_q: float = 0.1,
             _gs_e: float = 1e-2,
             _df_h: float = 1e-2,
             _use_gs: bool = True):
    _x_p = np.array(_x0)
    _x = _x_p

    A = np.eye(len(_x0))
    _g_p = _df(_x_p, _f, _df_h)
    _s = -1 * np.matmul(A, _g_p)
    _lam = 0.0
    _history = [_x]

    _iter = 0

    while True:
        _inter = sven.sven_algorithm(_f, _x, _s, _l=0.0, _q=_sv_q)

        # determine whether it is necessary to use golden-selection algorithm, or the other one
        if _use_gs:
            _lam = golden_selection.golden_selection(_f, _inter, _x, _s, _eps=_gs_e)
        else:
            pass

        _iter += 1
        _x = _x_p + _lam * _s
        _history.append(_x)

        _g = _df(_x, _f, _df_h)

        _dx = _x - _x_p
        _dg = _g - _g_p

        # denominator = np.dot(_dg.T, A).dot(_dg)
        denominator = np.dot(_dg.T, _dg)
        if np.abs(denominator) == 0 or np.linalg.norm(_dx) < _eps or np.linalg.norm(_s) == 0 or _iter > _max_iterations:
            if _iter > _max_iterations:
                log.program_log(log.LogLevel.Warning, f"pearson2: returned because _iter > _max_iterations")
            break

        A += np.outer(_dx - np.dot(A, _dg), np.dot(A, _dg).T) / denominator

        _x_p = _x
        _g_p = _g
        _s = -1 * np.matmul(A, _g_p)

    return _x, _history
