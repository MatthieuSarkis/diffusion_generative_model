import torch
import torch.nn as nn
import torch.nn.functional as F

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