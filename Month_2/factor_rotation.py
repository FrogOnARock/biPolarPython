import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.stats import ortho_group


np.random.seed(36)

#generating correlated data
n, p, k = 1000, 6, 2
true_loadings = np.array([
    [0.9, 0.1],
    [0.8, 0.2],
    [0.7, 0.3],
    [0.1, 0.9],
    [0.2, 0.8],
    [0.3, 0.7],
])

factors = np.random.randn(n, k)
noise = 0.2 * np.random.randn(n, p)
X = factors @ true_loadings.T + noise

# fit factor model
fa = FactorAnalysis(n_components=k)
Z = fa.fit_transform(X)
L = fa.components_.T

# rotating factors
r = ortho_group.rvs(dim=k)
L_rot = L @ r
Z_rot = Z @ r

# variance checking
orig_var = np.var(Z @ L.T)
rot_var = np.var(Z_rot @ L_rot.T)

print("Original variance:", orig_var)
print("Rotated variance :", rot_var)
print("Difference       :", abs(orig_var - rot_var))

np.set_printoptions(precision=3, suppress=True)

print("Original loadings:\n", L)
print("\nRotated loadings:\n", L_rot)