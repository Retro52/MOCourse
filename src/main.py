import core
from core import log
from algo import pearson

import numpy as np
import matplotlib.pyplot as plt

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


def df(_x, _f, _h=1e-2):
    _x_dx_p = np.array([_x[0] + _h, _x[1]])
    _x_dx_n = np.array([_x[0] - _h, _x[1]])
    _x_dx = (_f(_x_dx_p) - _f(_x_dx_n)) / (2 * _h)

    _y_dx_p = np.array([_x[0], _x[1] + _h])
    _y_dx_n = np.array([_x[0], _x[1] - _h])
    _y_dx = (_f(_y_dx_p) - _f(_y_dx_n)) / (2 * _h)
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

    # Plot the search paths
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Search Paths')

    lines = []  # List to store Line3D objects
    _min_x = 0.0
    _min_y = 0.0
    _max_x = 0.0
    _max_y = 0.0

    for i, _history in enumerate(histories):
        _history = np.array(_history)
        _x_values = _history[:, 0]
        _y_values = _history[:, 1]

        _min_x = min(_min_x, min(_x_values))
        _min_y = min(_min_y, min(_y_values))

        _max_x = max(_max_x, max(_x_values))
        _max_y = max(_max_y, max(_y_values))

        _z_values = f(_history.T)
        line = ax1.plot3D(_x_values, _y_values, _z_values, '-o', label=f'h = {_h_steps[i]}')
        lines.extend(line)  # Add the Line3D object to the list

    # Creating a mesh grid for the function evaluation
    x = np.linspace(_min_x, _max_x, 100)
    y = np.linspace(_min_y, _max_y, 100)

    X, Y = np.meshgrid(x, y)
    Z = f(np.array([X, Y]))

    # Plotting the surface of the function
    ax1.plot_surface(X, Y, Z, cmap='hsv', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Points History and Function Surface')

    ax1.legend(lines, [line.get_label() for line in lines])

    # Plot the function calls
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.bar(range(len(_h_steps)), function_calls)
    ax2.set_xticks(range(len(_h_steps)))
    ax2.set_xticklabels(_h_steps)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Function Calls')
    ax2.set_title('Number of Function Calls')

    # Plot the result deviation
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.bar(range(len(_h_steps)), deviations)
    ax3.set_xticks(range(len(_h_steps)))
    ax3.set_xticklabels(_h_steps)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Deviation')
    ax3.set_title('Results deviation from perfect one')


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

    # Plot the search paths
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Search Paths')

    lines = []  # List to store Line3D objects
    _min_x = 0.0
    _min_y = 0.0
    _max_x = 0.0
    _max_y = 0.0

    for i, _history in enumerate(histories):
        _history = np.array(_history)
        _x_values = _history[:, 0]
        _y_values = _history[:, 1]

        _min_x = min(_min_x, min(_x_values))
        _min_y = min(_min_y, min(_y_values))

        _max_x = max(_max_x, max(_x_values))
        _max_y = max(_max_y, max(_y_values))

        _z_values = f(_history.T)
        line = ax1.plot3D(_x_values, _y_values, _z_values, '-o', label=f'h = {_eps_steps[i]}')
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
    ax1.set_title('Points History and Function Surface')

    ax1.legend(lines, [line.get_label() for line in lines])

    # Plot the function calls
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.bar(range(len(_eps_steps)), function_calls)
    ax2.set_xticks(range(len(_eps_steps)))
    ax2.set_xticklabels(_eps_steps)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Function Calls')
    ax2.set_title('Number of Function Calls')

    # Plot the result deviation
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.bar(range(len(_eps_steps)), deviations)
    ax3.set_xticks(range(len(_eps_steps)))
    ax3.set_xticklabels(_eps_steps)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Deviation')
    ax3.set_title('Results deviation from perfect one')


# # Example usage
x0 = [2, 2]  # Initial guess
h_values = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
e_values = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-6]
test_by_h_step(f, df, x0, h_values, _sv_q=1e-3, _gs_e=1e-3, _eps=1e-5)
test_by_search_precision_step(f, df, x0, e_values, _sv_q=1e-3, _df_h=1e-3, _eps=1e-5)

plt.show()

# result, history = broyden2(f, df, x0,
#                            _df_h=0.001,
#                            _sv_q=1e-3,
#                            _gs_e=1e-3,
#                            _eps=1e-5)
#
# print("Optimized solution:", result)
# print("Minimum value:", f(result))
# print("Function calls:", function_call_counter)
# print("Search path len:", len(history))
#
# # Plotting the points history
# history = np.array(history)
#
# x_values = history[:, 0]
# y_values = history[:, 1]
#
# z_values = [f(x) for x in history]
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot3D(x_values, y_values, z_values, '-bo')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_title('Points History')
#
# # Creating a meshgrid for the function evaluation
# print(min(x_values), max(x_values))
# print(min(y_values), max(y_values))
# x = np.linspace(min(x_values), max(x_values), 100)
# y = np.linspace(min(y_values), max(y_values), 100)
# X, Y = np.meshgrid(x, y)
# Z = f(np.array([X, Y]))
#
# # Plotting the surface of the function
# ax.plot_surface(X, Y, Z, cmap='hsv', alpha=0.8)
#
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_title('Points History and Function Surface')
# ax.legend()
#
# plt.show()
