import torch
import torch.nn.functional as F


class SurrogateMLP(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)

    def functional_forward(self, x, params):
        x = F.relu(F.linear(x, params["fc1.weight"], params["fc1.bias"]))
        x = F.relu(F.linear(x, params["fc2.weight"], params["fc2.bias"]))
        return F.linear(x, params["fc3.weight"], params["fc3.bias"]).squeeze(-1)
