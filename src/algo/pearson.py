import numpy as np
from src.core import log
from src.algo import sven
from src.algo.intervals_exclusion import dsk_powell
from src.algo.intervals_exclusion import golden_selection


def pearson2(_f, _df, _x0: np.ndarray,
             _max_iterations: int = 100000,
             _eps: float = 1e-4,
             _sv_q: float = 0.1,
             _gs_e: float = 1e-2,
             _df_h: float = 1e-2,
             _use_gs: bool = True,
             _restarts: int = 0):
    _x_p = np.array(_x0)
    _x = _x_p

    _A = np.eye(len(_x_p))
    _g_p = _df(_x_p, _f, _df_h)
    _s = -1 * np.matmul(_A, _g_p)
    _lam = 0.0
    _history = [_x]

    _iter = 0
    _restarts_count = 0

    while True:
        _inter = sven.sven_algorithm(_f, _x, _s, _l=0.0, _q=_sv_q)

        # determine whether it is necessary to use golden-selection algorithm, or the other one
        if not _use_gs:
            _lam = dsk_powell.dsk_powell(_f, _inter, _x, _s, _eps=_gs_e)
        else:
            _lam = golden_selection.golden_selection(_f, _inter, _x, _s, _eps=_gs_e)

        _iter += 1
        _x = _x_p + _lam * _s
        _history.append(_x)

        _g = _df(_x, _f, _df_h)

        _dx = _x - _x_p
        _dg = _g - _g_p

        denominator = np.dot(_dg.T, _A).dot(_dg)
        # denominator = np.dot(_dg.T, _dg)

        if np.abs(denominator) == 0 or np.linalg.norm(_s) == 0 or _iter > _max_iterations:
            if _restarts_count < _restarts:
                log.program_log(log.LogLevel.Important, f"pearson2: "
                                                        f"Restarting search: Restart count: {_restarts_count + 1}")

                # the mos recent point is highly likely to be closer to the answer than _x0 was
                _x_p = _x
                _A = np.eye(len(_x_p))
                _g_p = _df(_x_p, _f, _df_h)
                _s = -1 * np.matmul(_A, _g_p)
                _lam = 0.0

                _restarts_count += 1

                continue

            log.program_log(log.LogLevel.Warning, f"pearson2: returned because denominator is 0, "
                                                  f"or S = 0 or iter > max iterations:"
                                                  f"{np.abs(denominator) == 0:5} | {np.linalg.norm(_s) == 0:5} | {_iter > _max_iterations:5}")
            break

        if (np.linalg.norm(_dx) / np.linalg.norm(_x_p)) <= _eps and abs(_f(_x) - _f(_x_p)) <= _eps:
            break

        _A += np.outer(_dx - np.dot(_A, _dg), np.dot(_A, _dg).T) / denominator

        _x_p = _x
        _g_p = _g
        _s = -1 * np.matmul(_A, _g_p)

    log.program_log(log.LogLevel.Debug, f"pearson2: Search is over:\n"
                                        f"{'Function min point':25} | {'Iterations':10} | {'Derivative h':12} | {'Search eps':12} | {'Eps':9} | {'Sven q':9} \n"
                                        f"{str(_x):25} | {_iter:10} | {_df_h:5.10F} | {_gs_e:5.10F} | {_eps:5.7F} | {_sv_q:5.7F} \n")
    return _x, _history
