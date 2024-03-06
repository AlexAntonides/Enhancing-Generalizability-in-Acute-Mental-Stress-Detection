from torch import nn
from lightning import LightningModule

class Model(LightningModule):
    """
        https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
    """
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(32, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, (5, 5), padding=2, activation='relu'),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(1000, activation='relu'),
            nn.Linear(1)
        )

    def forward(self, x):
        embedding = self.layers(x)
        return embedding.squeeze()