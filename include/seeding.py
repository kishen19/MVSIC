import numpy as np

def random_seeding(X, k):
    return X[np.random.choice(len(X), k, replace=False)]