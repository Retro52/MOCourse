import core
import numpy as np
from core import log
from algo import pearson
from functools import partial
import matplotlib.pyplot as plt
from algo.intervals_exclusion import dsk_powell
from algo.intervals_exclusion import golden_selection
from helpers import pyplot_wrapper as plotter

core.set_log_level(log.LogLevel.Trace)

function_values = {}
function_call_counter = 0


def f(x: np.ndarray):
    global function_values
    global function_call_counter

    key = x.tobytes()
    if key in function_values.keys():
        return function_values[key]
    else:
        function_call_counter += 1
        value = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        function_values[value.tobytes()] = value
        return value


def fee(_x, _radius: float):
    return _radius - (_x[0] ** 2) - (_x[1] ** 2)


def fee_sq(_x, _r: float, _radius: float):
    if (fee(_x, _radius)) >= 0:
        _r = 0

    return f(_x) + _r * (fee(_x, _radius) ** 2)


def fee_complex(_x, _r, _radius_1, _radius_2):
    if -1 * fee(_x, _radius_1) >= 0 and (fee(_x, _radius_2)) >= 0:
        _r = 0

    return f(_x) + _r * (fee(_x, _radius_1) ** 2) + _r * ((-1 * fee(_x, _radius_2)) ** 2)


def df(_x, _f, _h=1e-2):
    _x_dx_p = np.array([_x[0] + _h, _x[1]])
    _x_dx_n = np.array([_x[0] - _h, _x[1]])
    _x_dx = (_f(_x_dx_p) - _f(_x_dx_n)) / (2 * _h)

    _y_dx_p = np.array([_x[0], _x[1] + _h])
    _y_dx_n = np.array([_x[0], _x[1] - _h])
    _y_dx = (_f(_y_dx_p) - _f(_y_dx_n)) / (2 * _h)
    return np.array([_x_dx, _y_dx])


def df_left(_x, _f, _h=1e-2):
    _x_dx_p = np.array([_x[0] - _h, _x[1]])
    _x_dx = (_f(_x_dx_p) / _h)

    _y_dx_p = np.array([_x[0], _x[1] - _h])
    _y_dx = (_f(_y_dx_p) / _h)
    return np.array([_x_dx, _y_dx])


def df_right(_x, _f, _h=1e-2):
    _x_dx_p = np.array([_x[0] + _h, _x[1]])
    _x_dx = (_f(_x_dx_p) / _h)

    _y_dx_p = np.array([_x[0], _x[1] + _h])
    _y_dx = (_f(_y_dx_p) / _h)
    return np.array([_x_dx, _y_dx])


def test_by_h_step(_f, _df, _x0, _h_steps: list, *args, **kwargs):
    results = []
    histories = []
    deviations = []
    function_calls = []
    global function_call_counter

    for _h in _h_steps:
        _result, _history = pearson.pearson2(_f, _df, _x0, _df_h=_h, *args, **kwargs)

        results.append(_result)
        histories.append(_history)
        function_calls.append(function_call_counter)
        deviations.append(np.linalg.norm(_result - np.array([1.0, 1.0])))

        function_call_counter = 0

    log.program_log(log.LogLevel.Error, f"Test finished:\n"
                                        f"{str(deviations):100} |\n"
                                        f"{str(function_calls):100}\n|"
                                        f"{str(results):100}")
    plotter.plot_results(results, histories, deviations, function_calls, _h_steps, _f, "Derivatives step", "h")


def test_by_q_step(_f, _df, _x0, _q_steps: list, *args, **kwargs):
    results = []
    histories = []
    deviations = []
    function_calls = []
    global function_call_counter

    for _q in _q_steps:
        _result, _history = pearson.pearson2(_f, _df, _x0, _sv_q=_q, *args, **kwargs)

        results.append(_result)
        histories.append(_history)
        function_calls.append(function_call_counter)
        deviations.append(np.linalg.norm(_result - np.array([1.0, 1.0])))

        function_call_counter = 0

    log.program_log(log.LogLevel.Error, f"Test finished:\n"
                                        f"{str(deviations):100} |\n"
                                        f"{str(function_calls):100}\n|"
                                        f"{str(results):100}")
    plotter.plot_results(results, histories, deviations, function_calls, _q_steps, _f, "Sven coefficient", "q")


def test_by_derivatives_dir(_f, _df, _df_l, _df_r, _x0, *args, **kwargs):
    results = []
    histories = []
    deviations = []
    function_calls = []
    global function_call_counter

    _labels = ["center", "left", "right"]
    _derivatives = [_df, _df_l, _df_r]

    for _i, _df in enumerate(_derivatives):
        _result, _history = pearson.pearson2(_f, _df, _x0, *args, **kwargs)

        results.append(_result)
        histories.append(_history)
        function_calls.append(function_call_counter)
        deviations.append(np.linalg.norm(_result - np.array([1.0, 1.0])))

        function_call_counter = 0

    log.program_log(log.LogLevel.Error, f"Test finished: {deviations} | {function_calls} | {results} ")
    plotter.plot_results(results, histories, deviations, function_calls, _labels, _f, "Derivatives scheme", "scheme")


def test_by_search_precision_type(_f, _df, _x0, *args, **kwargs):
    results = []
    histories = []
    deviations = []
    function_calls = []
    global function_call_counter

    _labels = ["Golden intersection", "DSK-P"]
    _options = [golden_selection.golden_selection, dsk_powell.dsk_powell]

    for _i, _algo in enumerate(_options):
        _result, _history = pearson.pearson2(_f, _df, _x0, _search_algo=_algo, *args, **kwargs)

        results.append(_result)
        histories.append(_history)
        function_calls.append(function_call_counter)
        deviations.append(np.linalg.norm(_result - np.array([1.0, 1.0])))

        function_call_counter = 0

    log.program_log(log.LogLevel.Error, f"Test finished: {deviations} | {function_calls} | {results} ")
    plotter.plot_results(results, histories, deviations, function_calls, _labels, _f, "Search algorithm", "Algorithm")


def test_by_break_condition(_f, _df, _x0, *args, **kwargs):
    results = []
    histories = []
    deviations = []
    function_calls = []
    global function_call_counter

    _labels = ["Default", "Gradient"]
    _options = [True, False]

    for _i, _option in enumerate(_options):
        _result, _history = pearson.pearson2(_f, _df, _x0, _test_option=_option, *args, **kwargs)

        results.append(_result)
        histories.append(_history)
        function_calls.append(function_call_counter)
        deviations.append(np.linalg.norm(_result - np.array([1.0, 1.0])))

        function_call_counter = 0

    log.program_log(log.LogLevel.Error, f"Test finished: {deviations} | {function_calls} | {results} ")
    plotter.plot_results(results, histories, deviations, function_calls, _labels, _f, "Break condition", "Condition")


def test_by_search_precision_step(_f, _df, _x0, _eps_steps: list, *args, **kwargs):
    results = []
    histories = []
    deviations = []
    function_calls = []
    global function_call_counter

    for _eps in _eps_steps:
        _result, _history = pearson.pearson2(_f, _df, _x0, _gs_e=_eps, *args, **kwargs)

        results.append(_result)
        histories.append(_history)
        function_calls.append(function_call_counter)
        deviations.append(np.linalg.norm(_result - np.array([1.0, 1.0])))

        function_call_counter = 0

    log.program_log(log.LogLevel.Error, f"Test finished: {deviations} | {function_calls} | {results} ")
    plotter.plot_results(results, histories, deviations, function_calls, _eps_steps, _f, "Search precision", "eps")


def test_by_restart_count(_f, _df, _x0, _restarts_list: list, *args, **kwargs):
    results = []
    histories = []
    deviations = []
    function_calls = []
    global function_call_counter

    for _restart_count in _restarts_list:
        _result, _history = pearson.pearson2(_f, _df, _x0, _restarts=_restart_count, *args, **kwargs)

        results.append(_result)
        histories.append(_history)
        function_calls.append(function_call_counter)
        deviations.append(np.linalg.norm(_result - np.array([1.0, 1.0])))

        function_call_counter = 0

    log.program_log(log.LogLevel.Error, f"Test finished:\n"
                                        f"{str(deviations):100} |\n"
                                        f"{str(function_calls):100}\n|"
                                        f"{str(results):100}")
    plotter.plot_results(results, histories, deviations, function_calls, _restarts_list, _f, "Forced restarts count",
                         "Restarts")


def test_by_fee(_f, _radius, _pure_f, _df, _x0, _r_values: list, *args, **kwargs):
    _result = _x0
    results = [_x0]
    histories = []
    deviations = []
    function_calls = []
    global function_call_counter

    for _r in _r_values:
        _result, _history = pearson.pearson2(partial(_f, _r=_r, _radius=_radius), _df, _result, *args, **kwargs)

        results.append(_result)
        histories.append(_history)
        function_calls.append(function_call_counter)
        deviations.append(np.linalg.norm(_result - np.array([1.0, 1.0])))

        function_call_counter = 0

    log.program_log(log.LogLevel.Error, f"Test finished:\n"
                                        f"{str(deviations):100} |\n"
                                        f"{str(function_calls):100}\n|"
                                        f"{str(results):100}")
    plotter.plot_fee_results(_pure_f, results, _radius)


def test_by_fee_complex(_f, _radius_1, _radius_2, _pure_f, _df, _x0, _r_values: list, *args, **kwargs):
    results = [_x0]
    histories = []
    deviations = []
    function_calls = []
    global function_call_counter

    for _r in _r_values:
        _result, _history = pearson.pearson2(partial(_f, _r=_r, _radius_1=_radius_1, _radius_2=_radius_2), _df, _x0, *args, **kwargs)

        results.append(_result)
        histories.append(_history)
        function_calls.append(function_call_counter)
        deviations.append(np.linalg.norm(_result - np.array([1.0, 1.0])))

        function_call_counter = 0

    log.program_log(log.LogLevel.Error, f"Test finished:\n"
                                        f"{str(deviations):100} |\n"
                                        f"{str(function_calls):100}\n|"
                                        f"{str(results):100}")
    plotter.plot_fee_results_complex(_pure_f, results, _radius_1, _radius_2)


def main():
    # Example usage
    # x0 = [1.3, 1.3]  # Initial guess, consistent throughout all the tests
    x0 = [1.7, 1.7]  # Initial guess, consistent throughout all the tests
    # x0 = [-1.2, 0.]  # Initial guess, consistent throughout all the tests
    # x0 = [3.0, 3.0]  # Initial guess, consistent throughout all the tests
    h0 = 1e-3
    sv0 = 1e-3
    search_pr = 1e-3
    pearson_pr = 1e-13
    search_algo = golden_selection.golden_selection
    _restarts = 100
    _radius_inner = 1.0  # 0 <= 1 - x_1 ** 2 - x_2 ** 2

    _radius = 1.0  # x_1 ** 2 + x_2 ** 2 - 4 >= 0
    _radius_outer = 4.0  # x_1 ** 2 + x_2 ** 2 - 4 >= 0


    h_values = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    e_values = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    q_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    # r_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0]
    r_values = [1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6, 1e+7, 1e+8]

    # log.program_log(log.LogLevel.Important, f"Start testing by derivative step size: {h_values}")
    # test_by_h_step(f, df, x0, h_values, _sv_q=sv0, _gs_e=search_pr, _eps=pearson_pr, _search_algo=search_algo, _restarts=_restarts)
    h0 = 1

    # log.program_log(log.LogLevel.Important, f"Start testing derivative scheme")
    # test_by_derivatives_dir(f, df, df_left, df_right, x0, _df_h=h0, _sv_q=sv0, _gs_e=search_pr, _eps=pearson_pr, _search_algo=search_algo, _restarts=_restarts)

    # log.program_log(log.LogLevel.Important, f"Start testing for best Sven algo coef")
    # test_by_q_step(f, df, x0, q_values, _df_h=h0, _gs_e=search_pr, _eps=pearson_pr, _search_algo=search_algo, _restarts=_restarts)
    sv0 = 0.01


    # log.program_log(log.LogLevel.Important, f"Start testing by linear search precision: {e_values}")
    # test_by_search_precision_step(f, df, x0, e_values, _df_h=h0, _sv_q=sv0, _eps=pearson_pr, _search_algo=search_algo, _restarts=100)
    search_pr = 0.01

    # log.program_log(log.LogLevel.Important, f"Searching for best lambda finder")
    # test_by_search_precision_type(f, df, x0, _df_h=h0, _sv_q=sv0, _gs_e=search_pr, _eps=pearson_pr, _restarts=_restarts)
    #
    # log.program_log(log.LogLevel.Important, f"Start testing for break condition")
    # test_by_break_condition(f, df, x0, _df_h=h0, _sv_q=sv0, _gs_e=search_pr, _eps=pearson_pr, _restarts=_restarts)
    #
    # log.program_log(log.LogLevel.Important, f"Start testing for restarts count")
    # test_by_restart_count(f, df, x0, r_values, _df_h=h0, _sv_q=sv0, _gs_e=search_pr, _eps=pearson_pr)

    log.program_log(log.LogLevel.Important, f"Start testing by fee value")
    test_by_fee(fee_sq, _radius, f, df, x0, r_values, _df_h=h0, _sv_q=sv0, _gs_e=search_pr, _eps=pearson_pr)
    # test_by_fee_complex(fee_complex, _radius_inner, _radius_outer, f, df, x0, r_values, _df_h=h0, _sv_q=sv0, _gs_e=search_pr, _eps=pearson_pr)
    # print(r_values)

    plt.show()


if __name__ == '__main__':
    main()
