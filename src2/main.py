from src2.neural_net import Unet
from src2.diffusion import GaussianDiffusion 
from src2.train import Trainer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim = 64,
    channels=1,
    dim_mults = (1, 2, 4, 8)
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    channels=1,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).to(device)

trainer = Trainer(
    diffusion,
    #'path/to/your/images',
    dataset_size=1000,
    train_batch_size = 32,
    train_lr = 2e-5,
    save_and_sample_every = 1000,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainer.train()