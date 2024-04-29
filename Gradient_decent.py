import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importing this for 3D plotting
from scipy.optimize import minimize

def f(x):
    x1, x2, x3 = x
    return 2*x1**4 - x1*x2**2 + 2*x2**2*x3**2 - 2*x3**3 + 10*x1 - 2*x2 + np.exp(x3) - np.log(x1**2 + x2**2 + 1)

def grad_f(x):
    x1, x2, x3 = x
    grad_f1 = 8*x1**3 - x2**2 + 10/(x1**2 + x2**2 + 1)
    grad_f2 = -2*x1*x2 + 4*x2*x3**2 - 2/(x1**2 + x2**2 + 1)
    grad_f3 = 4*x2**2*x3 - 6*x3**2 + np.exp(x3)
    return np.array([grad_f1, grad_f2, grad_f3])

def gradient_descent(grad, initial_guess, learning_rate, epsilon, max_iterations):
    x = np.array(initial_guess)
    x_values = [x]
    for i in range(max_iterations):
        grad_val = grad(x)
        if np.linalg.norm(grad_val) <= epsilon:
            break
        x = x - learning_rate * grad_val
        print(f"Iterācija {i+1}:")
        print("Optimizētais punkts:", x)
        print("Funkcijas vērtība:", f(x))
        x_values.append(x)

    return np.array(x_values)

initial_guess = [0.5, 0.5, 0.5] # sakum punkts
learning_rate = 0.001 # t value
epsilon = 1
max_iterations = 1000

x_values = gradient_descent(grad_f, initial_guess, learning_rate, epsilon, max_iterations)
np.savetxt('optimization_trajectory.csv', x_values, delimiter=',')
# Plotting the function
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
x, y = np.meshgrid(x, y)
z = f([x, y, 0])
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.6)

# Plotting the optimizer's path
x_values = np.array(x_values)
ax.plot(x_values[:, 0], x_values[:, 1], f(x_values.T), marker='o', color='r', linestyle='-')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')

plt.show()
