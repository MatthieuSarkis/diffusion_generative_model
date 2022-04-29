import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import torch.nn.functional as F
import numpy as np

def sample_batch(
    size: int, 
    noise: float = 0.5
) -> np.ndarray:

    x, _= make_swiss_roll(size, noise=noise)
    return x[:, [0, 2]] / 10.0

class ConditionalLinear(nn.Module):

    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        n_steps: int
    ) -> None:

        super(ConditionalLinear, self).__init__()
        self.output_dim = output_dim
        self.lin = nn.Linear(input_dim, output_dim)
        self.embed = nn.Embedding(n_steps, output_dim)
        self.embed.weight.data.uniform_()

    def forward(
        self, 
        x: torch.tensor, 
        y: torch.tensor
    ) -> torch.tensor:

        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.output_dim) * out
        return out

class ConditionalModel(nn.Module):

    def __init__(
        self, 
        n_steps: int
    ) -> None:

        super(ConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(2, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, 2)
    
    def forward(
        self, 
        x: torch.tensor, 
        y: torch.tensor
    ) -> torch.tensor:

        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x)

class EMA():

    def __init__(
        self, 
        mu: float = 0.999
    ) -> None:

        self.mu = mu
        self.shadow = {}

    def register(
        self, 
        module: nn.Module
    ) -> None:

        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(
        self, 
        module: nn.Module
    ) -> None:

        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(
        self, 
        module: nn.Module
    ) -> None:

        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(
        self, 
        module: nn.Module
    ) -> nn.Module:

        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):

        return self.shadow

    def load_state_dict(
        self, 
        state_dict
    ) -> dict:

        self.shadow = state_dict

class Markov():

    def __init__(
        self, 
        n_steps: int
    ) -> None:

        self.n_steps = n_steps
        self.betas = self.make_beta_schedule(schedule='sigmoid', n_timesteps=self.n_steps, start=1e-5, end=1e-2)
        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

    @staticmethod
    def extract(
        input: torch.tensor, 
        t: torch.tensor, 
        x: torch.tensor
    ) -> torch.tensor:

        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    @staticmethod
    def make_beta_schedule(
        schedule: str = 'linear', 
        n_timesteps: int = 1000, 
        start: float = 1e-5, 
        end: float = 1e-2
    ) -> torch.tensor:

        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas

    def noise_estimation_loss(
        self, 
        model: ConditionalModel, 
        x_0: torch.tensor
    ) -> torch.tensor:

        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,))
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size].long()
        a = self.extract(self.alphas_bar_sqrt, t, x_0)
        am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, x_0)
        e = torch.randn_like(x_0)
        x = x_0 * a + e * am1
        output = model(x, t)
        return (e - output).square().mean()

    def p_sample(
        self, 
        model: ConditionalModel, 
        x: torch.tensor, 
        t: int
    ) -> torch.tensor:

        t = torch.tensor([t])
        eps_factor = ((1 - self.extract(self.alphas, t, x)) / self.extract(self.one_minus_alphas_bar_sqrt, t, x))
        eps_theta = model(x, t)
        mean = (1 / self.extract(self.alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
        z = torch.randn_like(x)
        sigma_t = self.extract(self.betas, t, x).sqrt()
        sample = mean + sigma_t * z
        return (sample)

    def p_sample_loop(
        self, 
        model: ConditionalModel,
        shape: torch.Size
    ) -> torch.tensor:

        cur_x = torch.randn(shape)
        x_seq = [cur_x]
        for i in reversed(range(self.n_steps)):
            cur_x = self.p_sample(model, cur_x, i)
            x_seq.append(cur_x)
        return x_seq

class Trainer():

    def __init__(
        self, 
        n_steps: int = 100
    ) -> None:

        self.markov = Markov(n_steps=n_steps)
        self.model = ConditionalModel(self.markov.n_steps)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.ema = EMA(0.9)
        self.ema.register(self.model)

    def train(
        self, 
        data: np.ndarray, 
        batch_size: int, 
        epochs: int = 1000
    ) -> None:

        dataset = torch.tensor(data.T).float()

        for t in range(epochs):
        
            permutation = torch.randperm(dataset.size()[0])
            for i in range(0, dataset.size()[0], batch_size):

                indices = permutation[i:i+batch_size]
                batch_x = dataset[indices]

                loss = self.markov.noise_estimation_loss(self.model, batch_x)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()

                self.ema.update(self.model)

            if (t % 100 == 0):
                print(loss)
                x_seq = self.markov.p_sample_loop(self.model, dataset.shape)
                fig, axs = plt.subplots(1, 10, figsize=(28, 3))
                for i in range(1, 11):
                    cur_x = x_seq[i * 10].detach()
                    axs[i-1].scatter(cur_x[:, 0], cur_x[:, 1], s=10);
                    axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*100)+'})$')

def main():
    data = sample_batch(10**4).T
    n_steps = 100
    batch_size = 128
    epochs = 1000

    trainer = Trainer(n_steps=n_steps)
    trainer.train(data=data, batch_size=batch_size, epochs=epochs)

main()