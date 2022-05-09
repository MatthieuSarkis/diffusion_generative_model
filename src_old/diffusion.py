import torch

from src.neural_net import ConditionalModel

class Diffusion():

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

    @staticmethod
    def cosine_beta_schedule(n_timesteps, s = 0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = n_timesteps + 1
        x = torch.linspace(0, n_timesteps, steps)
        alphas_cumprod = torch.cos(((x / n_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

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

    @torch.no_grad()
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

    @torch.no_grad()
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