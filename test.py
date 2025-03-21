
import numpy as np


def generate_epsilon_rectangular_set(P, epsilon):
    """
    Generates an ε-rectangular set of transition functions.

    Parameters:
    P : numpy array (S, A, S)
        Original transition probability matrix where P[s, a, s'] = P(s' | s, a)
    epsilon : float
        Maximum allowable deviation from original probabilities

    Returns:
    P_min : numpy array (S, A, S)
        Lower bound of transition probabilities
    P_max : numpy array (S, A, S)
        Upper bound of transition probabilities
    """
    S, A, _ = P.shape

    # Compute lower and upper bounds
    P_min = np.maximum(P - epsilon, 0)  # Ensure non-negative probabilities
    P_max = np.minimum(P + epsilon, 1)  # Ensure probabilities don't exceed 1

    # Normalize each transition probability set to sum to 1
    for s in range(S):
        for a in range(A):
            P_min[s, a, :] /= P_min[s, a, :].sum() if P_min[s, a, :].sum() > 0 else 1
            P_max[s, a, :] /= P_max[s, a, :].sum() if P_max[s, a, :].sum() > 0 else 1

    return P_min, P_max

def sample_transition_function(P, epsilon, num_samples=1):
    """
    Samples transition functions from an ε-rectangular set.

    Parameters:
    P : numpy array (S, A, S)
        Original transition probability matrix where P[s, a, s'] = P(s' | s, a)
    epsilon : float
        Maximum allowable deviation from original probabilities
    num_samples : int
        Number of sampled transition functions

    Returns:
    samples : numpy array (num_samples, S, A, S)
        Array of sampled transition probability matrices
    """
    S, A, _ = P.shape
    samples = np.zeros((num_samples, S, A, S))

    for i in range(num_samples):
        P_sample = np.copy(P)

        # Generate perturbations within the ε-range
        perturbation = np.random.uniform(-epsilon, epsilon, size=P.shape)
        P_sample += perturbation  # Apply perturbation

        # Ensure probabilities remain in valid range [0,1]
        P_sample = np.clip(P_sample, 0, 1)

        # Normalize each transition probability to sum to 1
        for s in range(S):
            for a in range(A):
                P_sample[s, a, :] /= P_sample[s, a, :].sum() if P_sample[s, a, :].sum() > 0 else 1

        samples[i] = P_sample

    return samples


# Example usage:
P = np.array([[[0.7, 0.3], [0.4, 0.6]],  # Transition probabilities for state 0
              [[0.2, 0.8], [0.5, 0.5]]])  # Transition probabilities for state 1

epsilon = 0.1  # Set the ε value
num_samples = 5  # Generate 5 perturbed transition functions
samples = sample_transition_function(P, epsilon, num_samples)

print("Sampled Transition Functions:\n", samples)
# Example usage:
P = np.array([[[0.7, 0.3], [0.4, 0.6]],  # Transition probabilities for state 0
              [[0.2, 0.8], [0.5, 0.5]]])  # Transition probabilities for state 1

epsilon = 0.1  # Set the ε value
P_min, P_max = generate_epsilon_rectangular_set(P, epsilon)

print("Lower Bound of Transition Probabilities:\n", P_min)
print("Upper Bound of Transition Probabilities:\n", P_max)