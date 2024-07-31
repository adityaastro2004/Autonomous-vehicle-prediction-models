import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def read_numbers_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        numbers = content.split()
        numbers = [float(num) for num in numbers]
    return numbers


file_path = 'logNormal.txt'
data = read_numbers_from_file(file_path)
# file_path2 = 'population.txt'
# pop_data = read_numbers_from_file(file_path2)


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


n_bootstrap = 1000
bootstrap_means = bootstrap_means(data, n_bootstrap)

alpha = 0.05
lower_bound, upper_bound = percentile_confidence_interval(bootstrap_means, alpha)

if "__main__" == __name__:
    print(f"95% confidence interval from percentile: ({lower_bound}, {upper_bound})")

    fig = px.histogram(bootstrap_means, nbins=30, title='Bootstrap Means Histogram with 95% Confidence Interval Percentile')

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
