import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Constants
small = 1e-9

# Variable Parameters
k = 2  # Young's modulus for PLA plastic
arr_size = 10  # 10x10 matrix

# Constraints
vol = arr_size**2
vol_ratio = 0.7

# If generating a matrix using random numbers, eplace 0s with the "small" value in order to avoid numerical complexities


def clean_arr(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                matrix[i][j] = small


# Note that while hookes law uses a negative constant of porportionality for force, in order to optimize stress,
# young's modulus is kept positive


def hookes_law(matrix):
    stress = np.zeros((arr_size, arr_size))

    for i in range(arr_size):
        for j in range(arr_size):
            # Applies stress to the top of the matrix (starting from row 0)
            stress[i][j] = k * matrix[[i * j]] * i

    return stress


# Similar to linear programming in math, the objective function is the function that is minimized.
# In this case, the objective function is the sum of the stress matrix, which we are trying to minimize


def objective_func(matrix):
    return np.sum(hookes_law(matrix))


# Function to calculate the constraint equation


def constraint(densities):
    used_volume = np.sum(densities)

    return (used_volume / vol) - vol_ratio


# Gradient-based approach to decrease material densities (using scipy to take care of the math)


def optimize_densities(matrix):
    initial_guess = matrix.flatten()

    bounds = [(0, 1)] * vol
    constraints = [{"type": "eq", "fun": constraint}]

    result = minimize(
        objective_func, initial_guess, bounds=bounds, constraints=constraints
    )

    optimized_densities = result.x.reshape(matrix.shape)

    return optimized_densities


# Solid Isotropic Material with Penalisation (SIMP) method
# This essentially converts the optimized densities into a binary matrix (which is easier to plot and visualize)


def simp_method(matrix):
    modified_matrix = np.zeros_like(matrix)
    threshold = 0.5  # Adjust the threshold value (0.5 for a 50% cutoff)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= threshold:
                modified_matrix[i, j] = 1

    return modified_matrix


# Running the program
# arr = np.random.random((arr_size, arr_size))
arr = np.ones((arr_size, arr_size))
clean_arr(arr)
optimized_densities = optimize_densities(arr)
result_matrix = simp_method(optimized_densities)

# Plotting the results
plt.style.use("classic")
plt.contourf(result_matrix)
plt.colorbar()
plt.show()
