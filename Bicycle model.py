import numpy as np
import matplotlib.pyplot as plt

# theta --> heading angle,  delta --> steering angle

v = np.random.randint(5, 15)
L = 2.0  # wheelbase
dt = 0.1
steps = 5000
mean = 0
std_dev = np.radians(4)
num_paths = 5


def bicycle_model(x, y, theta, v, delta, L, dt):
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + (v / L) * np.tan(delta) * dt
    return x_next, y_next, theta_next


plt.figure(figsize=(10, 6))

for path in range(num_paths):
    x, y, theta = 0, np.random.randint(-100, 100), np.random.uniform(-1, 1)
    x_arr = [x]
    y_arr = [y]

    for i in range(steps):
        delta = np.random.normal(mean, std_dev)
        x, y, theta = bicycle_model(x, y, theta, v, delta, L, dt)
        x_arr.append(x)
        y_arr.append(y)

    plt.plot(x_arr, y_arr, label=f'Path {path + 1}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Bicycle Model for 5 paths')
plt.legend()
plt.grid(True)
plt.show()
