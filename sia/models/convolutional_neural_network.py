from torch import nn
from lightning import LightningModule

class Model(LightningModule):
    """
        https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
    """
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=self.hparams.batch_size, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 28 * 28, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        embedding = self.layers(x)
        return embedding.squeeze()