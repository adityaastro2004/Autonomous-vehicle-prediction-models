from PercentileMethod import data
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm


def bootstrap_means(data, n_bootstrap=1000):
    bootstrap_samples = np.random.choice(data, (n_bootstrap, len(data)), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    return bootstrap_means


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


n_bootstrap = 1000
bootstrap_means = bootstrap_means(data, n_bootstrap)

alpha = 0.05
lower_bound, upper_bound = bca_confidence_interval(data, bootstrap_means, alpha)

print(f"95% BCa confidence interval: ({lower_bound}, {upper_bound})")

fig = px.histogram(bootstrap_means, nbins=30, title='Bootstrap Means Histogram with 95% BCa Confidence Interval')

fig.add_vline(x=lower_bound, line=dict(color='red', dash='dash'), annotation_text=f'Lower bound ({lower_bound:.2f})',
              annotation_position="top left")
fig.add_vline(x=upper_bound, line=dict(color='green', dash='dash'), annotation_text=f'Upper bound ({upper_bound:.2f})',
              annotation_position="top right")

fig.add_trace(go.Scatter(
    x=[lower_bound, upper_bound],
    y=[0, 0],
    mode='lines+text',
    text=['', '95% CI'],
    textposition='top center',
    line=dict(color='black', width=3)
))

fig.update_layout(
    xaxis_title='Mean Value',
    yaxis_title='Frequency',
    legend_title_text='Confidence Interval',
    title_x=0.5
)

fig.show()

