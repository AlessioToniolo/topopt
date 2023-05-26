import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Constants
small = 1e-9
k = 3.1  # Young's modulus for PLA plastic
vol = 10 * 10
vol_ratio = 1.7
arr_size = 10


def clean_arr(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                matrix[i][j] = small


def hookes_law(matrix):
    stress = np.zeros((arr_size, arr_size))

    for i in range(arr_size):
        for j in range(arr_size):
            # Applies stress to the top of the matrix (starting from row 0)
            stress[i][j] = k * matrix[i*j] * i

    return stress


def objective_func(matrix):
    return np.sum(hookes_law(matrix))


def constraint(densities):
    return vol_ratio * np.sum(densities) - vol


def optimize_densities(matrix):
    n = matrix.shape[0] * matrix.shape[1]  # 100

    initial_guess = matrix.flatten()

    bounds = [(0, 1)] * n
    # Use 'ineq' for inequality constraint
    constraints = {'type': 'ineq', 'fun': constraint}

    result = minimize(objective_func, initial_guess,
                      bounds=bounds, constraints=constraints)

    optimized_densities = result.x.reshape(matrix.shape)

    return optimized_densities


def simp_method(matrix):
    modified_matrix = np.zeros_like(matrix)
    threshold = 0.5  # Adjust the threshold value (0.5 for a 50% cutoff)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= threshold:
                modified_matrix[i, j] = 1

    return modified_matrix


# arr = np.random.random((arr_size, arr_size))
arr = np.ones((arr_size, arr_size))
clean_arr(arr)
optimized_densities = optimize_densities(arr)
result_matrix = simp_method(optimized_densities)
plt.style.use('classic')
plt.contour(result_matrix)
plt.colorbar()
plt.show()
