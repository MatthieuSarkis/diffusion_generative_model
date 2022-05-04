from src2.neural_net import Unet
from src2.diffusion import GaussianDiffusion 
from src2.train import Trainer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim = 64,
    channels=1,
    dim_mults = (1, 2, 4, 8),
    device=DEVICE
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    channels=1,
    timesteps = 1000,
    loss_type = 'l1',
    device=DEVICE
)

trainer = Trainer(
    diffusion,
    dataset_size=1000,
    train_batch_size = 32,
    train_lr = 2e-5,
    ema_decay = 0.995,
    amp = True
)

trainer.train(
    train_num_steps=700000,
    gradient_accumulate_every=2,
    update_ema_every=10,
    save_and_sample_every=1000,
)