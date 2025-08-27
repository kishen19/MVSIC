import numpy as np

def random_seeding(X, k):
    indices = np.random.choice(X.size(), k, replace=False)
    return [X.coords(i) for i in indices]