import numpy as np


def fc_matrix(n):
    return np.ones((n, n)) - np.eye(n)
