
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

"""
Just going to create a random dataset with some high variance columns and a low variance column that
provides significant signal to make prediction.

General idea:
4 columns:
Target: Happy Girlfriend
Income: Continuous
Seconds spent on video games: Continuous
Spent time with girlfriend: Binary
"""

np.random.seed(42)
n_samples = 1000
income = np.random.normal(100000, 10000, size=n_samples)
s_video = np.random.normal(0, 100000, size=n_samples)
s_girlfriend = np.random.randint(0, 2, size=n_samples)
target = np.where(s_girlfriend == 1, np.random.binomial(1, 0.85, size=n_samples), 0)
dataset = np.stack([target, income, s_video, s_girlfriend], axis=1)


y = dataset[:, 0]
x = dataset[:, 1:]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

"""
First will just assess the performance using a random forest classifier with all 3 dimensions then I will compress utilizing
PCA and assess the performance
"""

rfc = RandomForestClassifier(n_estimators=100, max_leaf_nodes=100, random_state=42)
rfc.fit(train_x, train_y)
pred = rfc.predict(test_x)
print(f"Accuracy: {rfc.score(test_x, test_y)}")


pca = PCA(n_components=1)
pca.fit_transform(train_x)
rfc_2 = RandomForestClassifier(n_estimators=100, max_leaf_nodes=100, random_state=42)
rfc_2.fit(pca.transform(train_x), train_y)
print(f"Accuracy: {rfc_2.score(pca.transform(test_x), test_y)}")



