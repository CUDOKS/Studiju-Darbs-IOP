import numpy as np
from scipy.optimize import minimize

# Define the function to be minimized
def f(x):
    x1, x2, x3 = x
    return 2*x1**4 - x1*x2**2 + 2*x2**2*x3**2 - 2*x3**3 + 10*x1 - 2*x2 + np.exp(x3) - np.log(x1**2 + x2**2 + 1)

# Define the initial guess
x0 = [0.0, 0.0, 0.0]

# Minimize the function
result = minimize(f, x0)

# Print the optimum
print("Optimum point:", result.x)
print("Minimum value:", result.fun)
