import numpy as np
import matplotlib.pyplot as plt

# theta --> heading angle, delta --> steering angle

v = np.random.randint(5, 15)
L = 2.0  # wheelbase
dt = 1.5
steps = 3
mean = 0
std_dev = np.radians(4)
num_paths = 1

# Define the max and min allowed delta (steering angle) in radians
max_delta = np.radians(40)
min_delta = -np.radians(40)


def bicycle_model(x, y, theta, v, delta, L, dt):
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + (v / L) * np.tan(delta) * dt
    return x_next, y_next, theta_next


plt.figure(figsize=(12, 8))

# Store the predicted and actual deltas for MSE calculation
predicted_deltas = []
actual_deltas = []

for path in range(num_paths):
    x, y, theta = 0, np.random.randint(-100, 100), np.random.uniform(-1, 1)
    x_arr = [x]
    y_arr = [y]

    for i in range(steps):
        delta = np.random.normal(mean, std_dev)
        actual_deltas.append(delta)  # Store actual delta
        x, y, theta = bicycle_model(x, y, theta, v, delta, L, dt)
        x_arr.append(x)
        y_arr.append(y)

        # Calculate vectors for allowed range (max and min delta) and predicted delta
        next_delta = np.random.normal(mean, std_dev)
        next_delta = np.clip(next_delta, min_delta, max_delta)
        predicted_deltas.append(next_delta)  # Store predicted delta

        # Increase the length of the vectors
        scale_factor = 2.0
        x_max = x + scale_factor * np.cos(theta + max_delta)
        y_max = y + scale_factor * np.sin(theta + max_delta)
        x_min = x + scale_factor * np.cos(theta + min_delta)
        y_min = y + scale_factor * np.sin(theta + min_delta)
        x_pred = x + scale_factor * np.cos(theta + next_delta)
        y_pred = y + scale_factor * np.sin(theta + next_delta)

        plt.quiver(x, y, x_max - x, y_max - y, color='g', angles='xy', scale_units='xy', scale=1/2,
                   label='Max Δ' if i == 0 else "")
        plt.quiver(x, y, x_min - x, y_min - y, color='b', angles='xy', scale_units='xy', scale=1/2,
                   label='Min Δ' if i == 0 else "")
        plt.quiver(x, y, x_pred - x, y_pred - y, color='r', angles='xy', scale_units='xy', scale=1/2,
                   label='Predicted Δ' if i == 0 else "")

        prob_density = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((next_delta - mean) / std_dev) ** 2)
        scaled_prob_density = prob_density / (1 / (std_dev * np.sqrt(2 * np.pi)))
        plt.text(x, y, f'Scaled P(Δ)={scaled_prob_density:.4f}', fontsize=10, color='black')

    plt.plot(x_arr, y_arr, label=f'Path {path + 1}', marker='o')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Bicycle Model with Delta Vectors and Probabilities')
plt.legend(loc='best')
plt.grid(True)

# Adjust the limits of the plot to ensure enough space for vectors
plt.xlim(-15, max(x_arr) + 15)
plt.ylim(min(y_arr) - 15, max(y_arr) + 15)

plt.show()

# Calculate the Mean Squared Error (MSE)
mse = np.mean((np.array(predicted_deltas) - np.array(actual_deltas)) ** 2)
print(f"Mean Squared Error of Predicted Deltas: {mse:.4f}")
