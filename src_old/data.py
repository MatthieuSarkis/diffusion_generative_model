from sklearn.datasets import make_swiss_roll
import numpy as np

def sample_batch(
    size: int, 
    noise: float = 0.5
) -> np.ndarray:

    x, _= make_swiss_roll(size, noise=noise)
    return x[:, [0, 2]] / 10.0