import copy
import torch
from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from torch.optim import Adam
from torchvision import utils

from src.ema import EMA
from src.data import Dataset
from src.utils import cycle, num_to_groups

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        ema_decay = 0.995,
        dataset_size=1000,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        amp = False,
        step_start_ema = 2000,
        results_folder = './results',
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)

        self.step_start_ema = step_start_ema

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size

        self.ds = Dataset(dataset_size=dataset_size, image_size=image_size)
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
        train_num_steps: int,
        gradient_accumulate_every: int = 2,
        update_ema_every: int = 10,
        save_and_sample_every: int = 1000,
    ) -> None:

        while self.step < train_num_steps:
            for i in range(gradient_accumulate_every):
                data = next(self.dl).to(self.device)

                with autocast(enabled = self.amp):
                    loss = self.model(data)
                    self.scaler.scale(loss / gradient_accumulate_every).backward()

                print(f'{self.step}: {loss.item()}')

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % save_and_sample_every == 0:
                milestone = self.step // save_and_sample_every
                batches = num_to_groups(12, self.batch_size)
                all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = 6)
                self.save(milestone)

            self.step += 1

        print('training completed')
