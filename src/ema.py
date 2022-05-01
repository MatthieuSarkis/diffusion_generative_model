import torch.nn as nn

class EMA():

    def __init__(
        self, 
        module: nn.Module,
        mu: float = 0.999
    ) -> None:

        self.mu = mu
        self.shadow = {}
        self.module = module

    def register(self) -> None:

        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:

        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self) -> None:

        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self) -> nn.Module:

        module_copy = type(self.module)(self.module.config).to(self.module.config.device)
        module_copy.load_state_dict(self.module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self) -> dict:

        return self.shadow

    def load_state_dict(
        self, 
        state_dict
    ) -> dict:

        self.shadow = state_dict