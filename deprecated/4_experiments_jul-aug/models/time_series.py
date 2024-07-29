import torch
from torch import nn
from lightning import LightningModule

class Model(LightningModule):
    def __init__(self, hidden_size=128, num_layers=1, rnn_type='lstm'):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN Layer
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True, bidirectional=False)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size, 
                              num_layers=num_layers, batch_first=True, bidirectional=False)
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        
        self.layers = nn.Sequential(
            # nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )  

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1) # Now shape: (batch_size, sequence_length, 1)
        
        # RNN forward pass
        if self.rnn_type == 'lstm':
            _, (hidden, _) = self.rnn(x)
        else:  # GRU
            _, hidden = self.rnn(x)
        
        # Get the last hidden state
        hidden = hidden[-1]  # shape: (batch_size, hidden_size)
        
        out = self.layers(hidden)
        
        # return out.squeeze(-1)  # shape: (batch_size,)
        return out
