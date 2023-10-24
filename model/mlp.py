from torch import nn
import torch
from torch import nn, optim,Tensor
import torch.nn.functional as F
from torchmetrics.functional import accuracy, f1_score as f1
import lightning.pytorch as pl
from model.classification_model import ClassificationLightningModule
import numpy as np


class MLP(ClassificationLightningModule):
    def __init__(self,  input_size: int, output_size: int, in_dim: int, hidden_sizes: int, n_hidden_layers: int=4, 
                 dropout_p: float=0.1, learning_rate: float=1e-3, model_id:str=""):
        super().__init__(input_size, output_size, in_dim, hidden_sizes, n_hidden_layers, dropout_p, learning_rate, model_id)
        
        layers = nn.ModuleList([
            nn.Linear(input_size * in_dim, hidden_sizes),
            nn.BatchNorm1d(hidden_sizes),
            nn.LeakyReLU()
        ])  

        for i in range(n_hidden_layers):
            layers.extend([
                nn.Linear(hidden_sizes, hidden_sizes),
                nn.BatchNorm1d(hidden_sizes),
                nn.LeakyReLU(),
            ])
            
        layers.extend([
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_sizes, output_size),
        ])
        
        self.layers = nn.ModuleList(layers)
      

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x
  