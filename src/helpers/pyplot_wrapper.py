import numpy as np
import matplotlib.pyplot as plt


def plot_results(_results: list,
                 _histories: list,
                 _deviations: list,
                 _function_calls: list,
                 _values: list,
                 _f: callable,
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

        _z_values = _f(_history.T)
        line = ax1.plot3D(_x_values, _y_values, _z_values, '-o', label=f'{_label} = {_values[i]}')
        lines.extend(line)  # Add the Line3D object to the list

    # Creating a mesh grid for the function evaluation
    x = np.linspace(_min_x, _max_x, 100)
    y = np.linspace(_min_y, _max_y, 100)

    _X, _Y = np.meshgrid(x, y)
    _Z = _f(np.array([_X, _Y]))

    # Plotting the surface of the function
    ax1.plot_surface(_X, _Y, _Z, cmap='hsv', alpha=0.3)
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


def plot_fee_results(_f: callable,
                     _history,
                     _radius: float = 1.0,
                     _center: np.ndarray = np.array([0, 0])):

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    _min_x = float('inf')
    _min_y = float('inf')
    _max_x = float('-inf')
    _max_y = float('-inf')

    _history = np.array(_history)
    _x_values = _history[:, 0]
    _y_values = _history[:, 1]

    _min_x = min(_min_x, min(_x_values), -np.sqrt(_radius))
    _min_y = min(_min_y, min(_y_values), -np.sqrt(_radius))

    _max_x = max(_max_x, max(_x_values), +np.sqrt(_radius))
    _max_y = max(_max_y, max(_y_values), +np.sqrt(_radius))

    _z_values = _f(_history.T)

    # Generate data points
    _x = np.linspace(_min_x, _max_x, 100)
    _y = np.linspace(_min_y, _max_y, 100)
    _X, _Y = np.meshgrid(_x, _y)
    _Z = _f(np.array([_X, _Y]))

    # Create a 2D contour plot
    contour = ax1.contourf(_X, _Y, _Z, cmap='cool')
    fig1.colorbar(contour, ax=ax1)  # Add a colorbar to the plot

    circle = plt.Circle((_center[0], _center[1]), np.sqrt(_radius), color='black', fill=False, linewidth=2)
    ax1.add_patch(circle)

    # Extract x and y coordinates from history
    ax1.plot(_x_values, _y_values, 'ro-', label='History')
    for i, point in enumerate(_history):
        plt.text(point[0], point[1], str(i + 1), fontsize=15, color='black', ha='left', va='top')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Contour Plot of Function')
