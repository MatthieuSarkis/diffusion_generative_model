
from torch.utils import data
import os
import numpy as np
import torch
from typing import Tuple, Union

class Dataset(data.Dataset):

    def __init__(
        self, 
        dataset_size=1000,
        image_size=128 
    ):

        super().__init__()

        self.image_size = image_size
        self.dataset, _ = generate_percolation_data(dataset_size=dataset_size, lattice_size=image_size)

    def __len__(self):
        
        return len(self.dataset)

    def __getitem__(self, index):

        return self.dataset[index]


def percolation_configuration(
    L: int, 
    p: float,
) -> np.ndarray:

    spin = (np.random.random(size=(L,L)) < p).astype(np.int8)
    return 2 * spin - 1

def generate_percolation_data(
    dataset_size, 
    lattice_size=128, 
    p_list=None, 
    split=False, 
    save_dir=None
) -> Union[Tuple[torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]]:

    X = []
    y = []

    for _ in range(dataset_size):

        if p_list is not None:
            y.append(np.random.choice(p_list))
        else:
            y.append(np.random.rand())
            
        X.append(percolation_configuration(lattice_size, y[-1]))

    X = np.array(X)
    y = np.array(y)

    X = torch.from_numpy(X).float().unsqueeze(1)
    y = torch.from_numpy(y).float().view(-1, 1)
    
    if save_dir is not None:
        torch.save(X, os.path.join(save_dir, 'images.pt'))
        torch.save(y, os.path.join(save_dir, 'labels.pt'))
    
    if split:
        X_train = X[:(3*dataset_size)//4]
        X_test = X[(3*dataset_size)//4:]
        y_train = y[:(3*dataset_size)//4]
        y_test = y[(3*dataset_size)//4:]
 
    return (X, y) if not split else (X_train, y_train, X_test, y_test)