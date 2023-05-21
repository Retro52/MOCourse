import numpy as np
from core import log
import matplotlib.pyplot as plt

log.set_log_level(log.LogLevel.Debug)

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


def broyden2(_f, _df, _x0,
             _max_iterations: int = 100000,
             _eps: float = 1e-6,
             _sv_q: float = 0.1,
             _gs_e: float = 1e-2,
             _df_h: float = 1e-2):
    _x_p = np.array(_x0)
    _x = _x_p

    A = np.eye(len(_x0))
    _g_p = _df(_x_p, _f, _df_h)
    _s = -1 * np.matmul(A, _g_p)
    _lam = 0.0
    _history = [_x]

    a = 0
    _false_count = 0

    while True:

        from algo import sven
        from algo.intervals_exclusion import golden_selection

        _inter = sven.sven_algorithm(_f, _x, _s, _l=0.0, _q=_sv_q)
        _lam = golden_selection.golden_selection(_f, _inter, _x, _s, _eps=_gs_e)

        a += 1
        _x = _x_p + _lam * _s
        _history.append(_x)

        _g = _df(_x, _f, _df_h)

        _dx = _x - _x_p
        _dg = _g - _g_p

        # denominator = np.dot(_dg.T, A).dot(_dg)
        denominator = np.dot(_dg.T, _dg)
        if np.abs(denominator) == 0 or np.linalg.norm(_dx) < _eps or np.linalg.norm(_s) == 0 or a > _max_iterations:
            print(np.abs(denominator) == 0, np.linalg.norm(_dx) < _eps, a > _max_iterations)
            break

        A += np.outer(_dx - np.dot(A, _dg), np.dot(A, _dg).T) / denominator

        _x_p = _x
        _g_p = _g
        _s = -1 * np.matmul(A, _g_p)

    print(a, _false_count, _false_count * 100 / a)
    return _x, _history


# Example usage
x0 = [5, 5]  # Initial guess
result, history = broyden2(f, df, x0,
                           _df_h=0.001,
                           _sv_q=1e-7,
                           _gs_e=1e-7,
                           _eps=1e-10)

print("Optimized solution:", result)
print("Minimum value:", f(result))
print("Function calls:", function_call_counter)
print("Search path len:", len(history))

# Plotting the points history
history = np.array(history)

x_values = history[:, 0]
y_values = history[:, 1]

z_values = [f(x) for x in history]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot3D(x_values, y_values, z_values, '-bo')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Points History')

# Creating a meshgrid for the function evaluation
print(min(x_values), max(x_values))
print(min(y_values), max(y_values))
x = np.linspace(min(x_values), max(x_values), 100)
y = np.linspace(min(y_values), max(y_values), 100)
X, Y = np.meshgrid(x, y)
Z = f(np.array([X, Y]))

# Plotting the surface of the function
ax.plot_surface(X, Y, Z, cmap='hsv', alpha=0.8)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Points History and Function Surface')
ax.legend()

plt.show()
