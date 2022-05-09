import torch

from src.data import sample_batch
from src.train import Trainer

def main():

    dataset = torch.tensor(sample_batch(10**4), dtype=torch.float)
    n_steps = 100
    batch_size = 128
    epochs = 1000

    trainer = Trainer(n_steps=n_steps)
    trainer.train(dataset=dataset, batch_size=batch_size, epochs=epochs)

if __name__ == '__main__':

    main()