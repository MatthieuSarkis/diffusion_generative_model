import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from src.diffusion import Diffusion
from src.neural_net import ConditionalModel
from src.ema import EMA

class Trainer():

    def __init__(
        self, 
        n_steps: int = 100
    ) -> None:

        self.diffusion = Diffusion(n_steps=n_steps)
        self.model = ConditionalModel(self.diffusion.n_steps)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.ema = EMA(module=self.model, mu=0.9)
        self.ema.register()

    def train(
        self, 
        dataset: torch.tensor, 
        batch_size: int, 
        epochs: int = 1000
    ) -> None:

        for t in range(epochs):
        
            permutation = torch.randperm(dataset.shape[0])
            for i in range(0, dataset.shape[0], batch_size):

                indices = permutation[i:i+batch_size]
                data_batch = dataset[indices]

                loss = self.diffusion.noise_estimation_loss(self.model, data_batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()

                self.ema.update()

            if (t % 100 == 0):
                print(loss)
                x_seq = self.diffusion.p_sample_loop(self.model, dataset.shape)
                fig, axs = plt.subplots(1, 10, figsize=(28, 3))
                for i in range(1, 11):
                    cur_x = x_seq[i * 10].detach()
                    axs[i-1].scatter(cur_x[:, 0], cur_x[:, 1], s=10);
                    axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*100)+'})$')
