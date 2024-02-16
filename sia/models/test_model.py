from torch import nn
from lightning import LightningModule

class Model(LightningModule):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Softmax(-1),
        )

    def forward(self, x):
        embedding = self.layers(x)
        return embedding.squeeze()