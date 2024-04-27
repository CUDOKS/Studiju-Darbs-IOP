import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the function to be minimized
def f(x):
    x1, x2, x3 = x
    return 2*x1**4 - x1*x2**2 + 2*x2**2*x3**2 - 2*x3**3 + 10*x1 - 2*x2 + np.exp(x3) - np.log(x1**2 + x2**2 + 1)

# Initial guess for the parameters
x_initial = [2, 1, 2]

# Perform optimization using Nelder-Mead (Simplex) method
step_size = 0.1
epsilon = 0.01
initial_simplex = np.array([x_initial])
iter_counter = 1

for i in range(len(x_initial)):
    point = x_initial.copy()
    point[i] += step_size
    initial_simplex = np.append(initial_simplex, [point], axis=0)
    print(initial_simplex)
    iter_counter = 1 + iter_counter

result = minimize(f, x_initial, method='Nelder-Mead', options={'xatol': epsilon, 'initial_simplex': initial_simplex})
# Plot the convergence of the objective function
print(result)
print(iter_counter)