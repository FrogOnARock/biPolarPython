import numpy as np

np.random.seed(36)

"""
Setting up a vector to represent factor returns
This will include time period * number of factors * number of assets
"""

T, K, N = 1000, 3, 5

# Factor returns
F = np.random.normal(0, 1, size=(T, K))

# Exposures

B_true = np.array([
    [1.2, 0.5, -0.3, 0.0, 1.0],  # Market
    [0.8, -0.2, 0.4, 1.1, -0.5],  # Size
    [0.1, 0.6, 0.9, 0.2, 0.0],  # Value
])

# Idiosyncratic noise
epsilon = np.random.normal(0, 0.1, size=(T, N))

# Asset returns
R = F @ B_true + epsilon
print(R)
print(len(R))
print(R[0])