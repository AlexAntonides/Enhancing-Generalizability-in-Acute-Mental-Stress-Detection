from torch import nn
from lightning import LightningModule

class Model(LightningModule):
    """
        Something I found on https://github.com/Edouard99/Stress_Detection_ECG/
    """
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(1000,512,bias=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(512,128,bias=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(128,64,bias=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(64,16,bias=True),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(16,4,bias=True),
            nn.BatchNorm1d(4),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(4,1,bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedding = self.layers(x)
        return embedding.squeeze()