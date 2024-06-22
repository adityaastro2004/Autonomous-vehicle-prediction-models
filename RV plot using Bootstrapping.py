import numpy as np
import matplotlib.pyplot as plt

v = 10
L = 2.0
dt = 0.1
steps = 1000
mean = 0
std_dev = np.radians(4)
num_paths = 1
theta_next = 0
x, y = 0, 0


def bicycle_model(x, y, theta, v, L, dt):
    global theta_next
    theta_next += theta
    x_next = v * np.cos(theta) * dt
    y_next = v * np.sin(theta) * dt
    return x_next, y_next, theta_next, theta


# Generate the initial path data
all_x_values = []
all_y_values = []
all_theta_values = []

x, y, theta_next = 0, 0, 0

for i in range(steps):
    theta = np.random.uniform(-np.pi / 3, np.pi / 3)
    x, y, theta_next, theta = bicycle_model(x, y, theta, v, L, dt)
    all_x_values.append(x)
    all_y_values.append(y)
    all_theta_values.append(theta)

bootstrap_samples = 1000
bootstrap_x_values = []
bootstrap_y_values = []
bootstrap_theta_values = []

for i in range(bootstrap_samples):
    resample_indices = np.random.choice(range(steps), steps, replace=True)
    resample_x = [all_x_values[i] for i in resample_indices]
    resample_y = [all_y_values[i] for i in resample_indices]
    resample_theta = [all_theta_values[i] for i in resample_indices]

    bootstrap_x_values.extend(resample_x)
    bootstrap_y_values.extend(resample_y)
    bootstrap_theta_values.extend(resample_theta)


plt.figure(figsize=(14, 6))

plt.subplot(2, 2, 1)
plt.hist(bootstrap_x_values, bins=80, color='blue', edgecolor='black')
plt.xlabel('x')
plt.ylabel('Frequency')
plt.title('Histogram of x values (Bootstrapped)')

plt.subplot(2, 2, 2)
plt.hist(bootstrap_y_values, bins=80, color='green', edgecolor='black')
plt.xlabel('y')
plt.ylabel('Frequency')
plt.title('Histogram of y values (Bootstrapped)')

plt.subplot(2, 2, 3)
plt.hist(bootstrap_theta_values, bins=80, color='yellow', edgecolor='black')
plt.xlabel('theta')
plt.ylabel('Frequency')
plt.title('Histogram of theta values (Bootstrapped)')

plt.tight_layout()
plt.show()
