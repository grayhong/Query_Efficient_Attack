import numpy as np

def anti_sample(num_samples, n_noise):
    half_num = int(num_samples/2)
    half_samples = np.random.multivariate_normal(np.zeros(n_noise), np.eye(n_noise), (half_num))
    samples = np.append(half_samples, -half_samples, 0)
    return samples

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()