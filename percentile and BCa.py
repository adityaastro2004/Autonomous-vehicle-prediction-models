from PercentileMethod import data

import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm


def bootstrap_means(data, n_bootstrap=1000):
    bootstrap_samples = np.random.choice(data, (n_bootstrap, len(data)), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    return bootstrap_means


def percentile_confidence_interval(data, alpha=0.05):
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return lower_bound, upper_bound


def bca_confidence_interval(data, bootstrap_means, alpha=0.05):
    theta_hat = np.mean(data)
    z0 = norm.ppf((np.sum(bootstrap_means < theta_hat)) / len(bootstrap_means))
    jackknife_means = np.array([np.mean(np.delete(data, i)) for i in range(len(data))])
    jackknife_mean = np.mean(jackknife_means)
    a = np.sum((jackknife_mean - jackknife_means) ** 3) / (6 * np.sum((jackknife_mean - jackknife_means) ** 2) ** 1.5)
    lower_percentile = 100 * norm.cdf(z0 + (z0 + norm.ppf(alpha / 2)) / (1 - a * (z0 + norm.ppf(alpha / 2))))
    upper_percentile = 100 * norm.cdf(z0 + (z0 + norm.ppf(1 - alpha / 2)) / (1 - a * (z0 + norm.ppf(1 - alpha / 2))))
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    return lower_bound, upper_bound


population = data
n_simulations = 10000
n_bootstrap = 1000
sample_size = 15
alpha = 0.05

true_mean = np.mean(population)

percentile_miss_left = 0
percentile_miss_right = 0
bca_miss_left = 0
bca_miss_right = 0

for _ in range(n_simulations):
    sample = np.random.choice(population, sample_size, replace=True)
    bootstrap_means_sample = bootstrap_means(sample, n_bootstrap)

    perc_lower, perc_upper = percentile_confidence_interval(bootstrap_means_sample, alpha)
    bca_lower, bca_upper = bca_confidence_interval(sample, bootstrap_means_sample, alpha)

    if perc_lower > true_mean:
        percentile_miss_left += 1
    elif perc_upper < true_mean:
        percentile_miss_right += 1

    if bca_lower > true_mean:
        bca_miss_left += 1
    elif bca_upper < true_mean:
        bca_miss_right += 1

percentile_miss_left /= n_simulations
percentile_miss_right /= n_simulations
bca_miss_left /= n_simulations
bca_miss_right /= n_simulations

bt_means = bootstrap_means(data, n_bootstrap)

alpha = 0.05
percentile_lower_bound, percentile_upper_bound = percentile_confidence_interval(bt_means, alpha)
bca_lower_bound, bca_upper_bound = bca_confidence_interval(data, bt_means, alpha)

# Print the confidence intervals
print(f"95% Percentile confidence interval: ({percentile_lower_bound}, {percentile_upper_bound})")
print(f"95% BCa confidence interval: ({bca_lower_bound}, {bca_upper_bound})")

# Visualization
fig = go.Figure()

fig.add_trace(go.Bar(
    x=['Percentile Miss Left', 'Percentile Miss Right', 'BCa Miss Left', 'BCa Miss Right'],
    y=[percentile_miss_left, percentile_miss_right, bca_miss_left, bca_miss_right],
    marker_color=['blue', 'blue', 'red', 'red']
))

fig.update_layout(
    title='Miss Percentages for Percentile and BCa Methods',
    xaxis_title='Miss Type',
    yaxis_title='Percentage',
    yaxis=dict(tickformat=".2%"),
)

fig.show()

print(f"Percentile Method Miss Left: {percentile_miss_left:.2%}")
print(f"Percentile Method Miss Right: {percentile_miss_right:.2%}")
print(f"BCa Method Miss Left: {bca_miss_left:.2%}")
print(f"BCa Method Miss Right: {bca_miss_right:.2%}")
