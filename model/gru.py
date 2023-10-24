from torch import nn, Tensor, optim
import torch
import numpy as np
import lightning.pytorch as pl
import torch.nn.functional as F
from torchmetrics.functional import accuracy, f1_score as f1
from model.classification_model import ClassificationLightningModule


class GRU(ClassificationLightningModule):

    def __init__(self, input_size=1, in_dim=3, output_size=1, hidden_sizes=64, n_hidden_layers=2, dropout_p=0.2, learning_rate=1e-3, model_id: str = ""):
        """ 
        Parameters
        ----------
        in_features : int, optional
            Number of features in the input, by default 1
        out_features : int, optional    
            Number of features in the output, by default 1
        hidden_size : int, optional
            Number of features in the hidden state, by default 64
        """
        super().__init__(input_size, output_size, in_dim, hidden_sizes, n_hidden_layers, dropout_p, learning_rate, model_id)
       
        self.gru = nn.GRU(in_dim, hidden_sizes, n_hidden_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.output_layer = nn.Linear(hidden_sizes, output_size)

    def init_hidden(self, batch_size: int):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_hidden_layers, batch_size, self.hidden_sizes).zero_()
        return hidden

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, in_features)
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, out_features)
        """
        h = self.init_hidden(x.shape[0])
        x = x.reshape(x.shape[0], -1, self.in_dim)
        x, _ = self.gru(x, h)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
