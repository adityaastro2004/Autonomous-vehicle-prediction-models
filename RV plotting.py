import numpy as np
import matplotlib.pyplot as plt

v = 10
L = 2.0
dt = 0.1
steps = 100000
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


all_x_values = []
all_y_values = []
all_theta_values = []

for i in range(steps):
    theta = np.random.uniform(-np.pi / 3, np.pi / 3)
    x, y, theta_next, theta = bicycle_model(x, y, theta, v, L, dt)
    all_x_values.append(x)
    all_y_values.append(y)
    all_theta_values.append(theta)

plt.figure(figsize=(14, 6))

plt.subplot(2, 2, 1)
plt.hist(all_x_values, bins=80, color='blue', edgecolor='black')
plt.xlabel('x')
plt.ylabel('Frequency')
plt.title('Histogram of x values')

plt.subplot(2, 2, 2)
plt.hist(all_y_values, bins=80, color='green', edgecolor='black')
plt.xlabel('y')
plt.ylabel('Frequency')
plt.title('Histogram of y values')

plt.subplot(2, 2, 3)
plt.hist(all_theta_values, bins=80, color='yellow', edgecolor='black')
plt.xlabel('theta')
plt.ylabel('Frequency')
plt.title('Histogram of theta values')

plt.tight_layout()
plt.show()
