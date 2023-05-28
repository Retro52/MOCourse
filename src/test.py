import matplotlib.pyplot as plt
import numpy as np


# Define your function _f(x, y) here
def _f(x: np.ndarray):
    value = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return value



plt.show()
