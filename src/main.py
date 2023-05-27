import core
from core import log
core.set_log_level(log.LogLevel.Trace)

from algo import pearson

import numpy as np
import matplotlib.pyplot as plt

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
    plot_results(results, histories, deviations, function_calls, _h_steps, "Derivatives step", "h")


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
    plot_results(results, histories, deviations, function_calls, _labels, "Derivatives scheme", "scheme")


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
    plot_results(results, histories, deviations, function_calls, _eps_steps, "Search precision", "eps")


def plot_results(_results: list,
                 _histories: list,
                 _deviations: list,
                 _function_calls: list,
                 _values: list,
                 _title: str = "",
                 _label: str = ""):

    # Plot the search paths
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title(_title)

    lines = []  # List to store Line3D objects
    _min_x = 0.0
    _min_y = 0.0
    _max_x = 0.0
    _max_y = 0.0

    for i, _history in enumerate(_histories):
        _history = np.array(_history)
        _x_values = _history[:, 0]
        _y_values = _history[:, 1]

        _min_x = min(_min_x, min(_x_values))
        _min_y = min(_min_y, min(_y_values))

        _max_x = max(_max_x, max(_x_values))
        _max_y = max(_max_y, max(_y_values))

        _z_values = f(_history.T)
        line = ax1.plot3D(_x_values, _y_values, _z_values, '-o', label=f'{_label} = {_values[i]}')
        lines.extend(line)  # Add the Line3D object to the list

    # Creating a mesh grid for the function evaluation
    x = np.linspace(_min_x, _max_x, 100)
    y = np.linspace(_min_y, _max_y, 100)

    X, Y = np.meshgrid(x, y)
    Z = f(np.array([X, Y]))

    # Plotting the surface of the function
    ax1.plot_surface(X, Y, Z, cmap='hsv', alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title(f'Points History and Function Surface: {_title}')

    ax1.legend(lines, [line.get_label() for line in lines])

    # Plot the function calls
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.bar(range(len(_values)), _function_calls)
    ax2.set_xticks(range(len(_values)))
    ax2.set_xticklabels(_values)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Function Calls')
    ax2.set_title(f'Number of Function Calls: {_title}')

    # Plot the result deviation
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.bar(range(len(_values)), _deviations)
    ax3.set_xticks(range(len(_values)))
    ax3.set_xticklabels(_values)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Deviation')
    ax3.set_title(f'Results deviation from perfect one: {_title}')


def main():
    # Example usage
    x0 = [2, 2]  # Initial guess, consistent throughout all the tests, except the last one
    h0 = 0.001
    sv0 = 1e-3
    search_pr = 1e-3
    pearson_pr = 1e-7
    _restarts = 100

    h_values = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    e_values = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    # log.program_log(log.LogLevel.Important, f"Start testing by derivative step size: {h_values}")
    # test_by_h_step(f, df, x0, h_values, _sv_q=sv0, _gs_e=search_pr, _eps=pearson_pr, _restarts=100)
    log.program_log(log.LogLevel.Important, f"Start testing derivative scheme")
    test_by_derivatives_dir(f, df, df_left, df_right, x0, _df_h=h0, _sv_q=sv0, _gs_e=search_pr, _eps=pearson_pr, _restarts=_restarts)
    # log.program_log(log.LogLevel.Important, f"Start testing by linear search precision: {e_values}")
    # test_by_search_precision_step(f, df, x0, e_values, _df_h=h0, _sv_q=sv0, _eps=pearson_pr)

    plt.show()


if __name__ == '__main__':
    main()
