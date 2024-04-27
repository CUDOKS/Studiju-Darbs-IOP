import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def f(x):
    x1, x2, x3 = x
    return 2*x1**4 - x1*x2**2 + 2*x2**2*x3**2 - 2*x3**3 + 10*x1 - 2*x2 + np.exp(x3) - np.log(x1**2 + x2**2 + 1)

def grad_f(x):
    x1, x2, x3 = x
    grad_f1= 8*x1**3 - x2**2 + 10/(x1**2 + x2**2 + 1)
    grad_f2 = -2*x1*x2 + 4*x2*x3**2 - 2/(x1**2 + x2**2 + 1)
    grad_f3 = 4*x2**2*x3 - 6*x3**2 + np.exp(x3)
    return np.array([grad_f1, grad_f2, grad_f3])

def gradient_descent(grad, initial_guess, learning_rate, epsilon, max_iterations):
    x = np.array(initial_guess)
    for i in range(max_iterations):
        grad_val = grad(x)
        if np.linalg.norm(grad_val) <= epsilon:
            break
        x = x - learning_rate * grad_val
        print(f"Iterācija {i+1}:")
        print("Optimizētais punkts:", x)
        print("Funkcijas vērtība:", f(x))
    return x, i+1

initial_guess = [0.0,0.0,0.0]
learning_rate = 0.001
epsilon =  1
max_iterations = 1000

xmin, num_iterations = gradient_descent(grad_f, initial_guess, learning_rate, epsilon, max_iterations)
xmin, num_iterations

for i in range(num_iterations):

    print()

x_optimized = xmin[0], xmin[1], xmin[2]
f_optimized = f(x_optimized)    

print("Optimizētais punkts:", x_optimized)
print("Funkcijas vērtība:", f_optimized)