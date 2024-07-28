import numpy as np
import matplotlib.pyplot as plt


def update_distribution(prior_distribution, observed_value):
    """
    Update the prior distribution based on the observed value.

    Parameters:
    prior_distribution (dict): A dictionary where keys are possible values of x and values are their probabilities.
    observed_value: The value of x observed at t=1.

    Returns:
    dict: Updated distribution.
    """
    # Increase the probability of the observed value and renormalize
    if observed_value in prior_distribution:
        prior_distribution[observed_value] += 1
    else:
        # If observed value was not initially in the distribution, add it
        prior_distribution[observed_value] = 1

    # Normalize the distribution
    total = sum(prior_distribution.values())
    for key in prior_distribution:
        prior_distribution[key] /= total

    return prior_distribution


def plot_distribution(distribution, t):
    """
    Plot the distribution.

    Parameters:
    distribution (dict): A dictionary where keys are possible values of x and values are their probabilities.
    t (int): The current time step.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(distribution.keys(), distribution.values(), color='blue', alpha=0.7)
    plt.title(f'Distribution at t={t}')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.show()


def main():
    # Initial prior distribution at t=0
    prior_distribution = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}

    # Plot the initial distribution
    plot_distribution(prior_distribution, t=0)

    # Number of updates
    num_updates = 5

    # Loop for updating the distribution
    for t in range(1, num_updates + 1):
        # Prompt the user to enter the observed value
        observed_value = int(input(f"Enter the observed value of the parameter at t={t}: "))

        # Update the distribution based on the observed value
        prior_distribution = update_distribution(prior_distribution, observed_value)

        # Print the updated distribution
        print(f"\nUpdated Distribution after t={t}:")
        for key, value in prior_distribution.items():
            print(f"Value {key}: Probability {value:.2f}")
        print("-" * 30)

        # Plot the updated distribution
        plot_distribution(prior_distribution, t)

    # Generate new list of possible values for t=2 based on the final distribution
    possible_values = list(prior_distribution.keys())
    probabilities = list(prior_distribution.values())

    # Simulate new possible values for x at t=2
    predictions_t2 = np.random.choice(possible_values, size=1000, p=probabilities)

    # Display predictions
    print("\nPredicted values for t=2 based on the final distribution:")
    print(predictions_t2)


if __name__ == "__main__":
    main()
