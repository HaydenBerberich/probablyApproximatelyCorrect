import numpy as np
import matplotlib.pyplot as plt

# Set parameters for the experiment
N_trials = 10000  # Number of trials
N_test = 10000  # Number of test points
x_range = (-1, 1)  # Range of x values

# Target function
def f(x):
    return x**2

# Generate test points
x_test = np.linspace(x_range[0], x_range[1], N_test)
f_test = f(x_test)

# Store slopes (a) and intercepts (b) for each trial
slopes = []
intercepts = []

# Run trials
for _ in range(N_trials):
    # Randomly sample x1, x2 from U[-1, 1]
    x1, x2 = np.random.uniform(x_range[0], x_range[1], 2)
    
    # Compute y1, y2 using the target function
    y1, y2 = f(x1), f(x2)
    
    # Compute slope (a) and intercept (b) of the line passing through (x1, y1) and (x2, y2)
    a = x1 + x2
    b = -x1 * x2
    
    # Store the results
    slopes.append(a)
    intercepts.append(b)

# Compute average hypothesis g_bar(x)
a_bar = np.mean(slopes)
b_bar = np.mean(intercepts)
g_bar = a_bar * x_test + b_bar

# Compute E_out, Bias, and Variance
# E_out: Expected squared error between g(x) and f(x)
E_out = 0
bias = 0
variance = 0

for a, b in zip(slopes, intercepts):
    # Compute g(x) for this trial
    g = a * x_test + b
    
    # Compute squared error contributions
    E_out += np.mean((g - f_test) ** 2)
    bias += np.mean((g_bar - f_test) ** 2)
    variance += np.mean((g - g_bar) ** 2)

# Normalize the results
E_out /= N_trials
bias /= N_trials
variance /= N_trials

# Verify that E_out ≈ Bias + Variance
error_check = bias + variance

# Plot f(x) and g_bar(x)
plt.figure(figsize=(10, 6))
plt.plot(x_test, f_test, label='f(x) = x^2', color='blue', linewidth=2)
plt.plot(x_test, g_bar, label='Average g(x)', color='red', linestyle='--', linewidth=2)
plt.title("Target Function f(x) and Average Hypothesis g_bar(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

print(f"E_out: {E_out}, bias: {bias}, variance: {variance}, error_check: {error_check}")