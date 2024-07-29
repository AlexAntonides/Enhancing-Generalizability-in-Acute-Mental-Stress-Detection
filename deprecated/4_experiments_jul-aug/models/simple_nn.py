from torch import nn
from lightning import LightningModule

class Model(LightningModule):
    def __init__(self, n_in=35):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            # nn.Softmax(dim=1)
            nn.Sigmoid()
        )

    def forward(self, x):
        embedding = self.layers(x)
        return embedding.squeeze()