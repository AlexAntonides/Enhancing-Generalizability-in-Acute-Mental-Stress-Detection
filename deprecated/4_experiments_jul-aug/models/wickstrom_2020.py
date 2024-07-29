import torch
from torch import nn
from lightning import LightningModule

class Model(LightningModule):
    """
        https://arxiv.org/pdf/2010.11310
    """
    def __init__(self, n_in=1, n_c=2):
        super(Model, self).__init__()

        n_hid = 128
        act = nn.ReLU()

        self.l1 = nn.Sequential(
            nn.Conv1d(n_in, n_hid, kernel_size=7, padding=3),
            nn.BatchNorm1d(n_hid),
            act
        )

        self.l2 = nn.Sequential(
            nn.Conv1d(n_hid, 2*n_hid, kernel_size=5, padding=2),
            nn.BatchNorm1d(2*n_hid),
            act
        )

        self.l3 = nn.Sequential(
            nn.Conv1d(2*n_hid, n_hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_hid),
            act
        )

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.out = nn.Linear(n_hid, n_c)

    def forward(self, x):
        l1 = self.l1(x)
        l2 = self.l2(l1)
        l3 = self.l3(l2)

        gap = self.gap(l3)
        out = self.out(gap)

        return out

    def CAM(self, x):
 
        l1 = self.l1(x)
        l2 = self.l2(l1)
        l3 = self.l3(l2)

        out = torch.matmul(l3.transpose(2, 1), self.out.weight.T)
        out = nn.functional.relu(out)
        out = out.cpu().detach().numpy()

        return out