from torch import nn
import torch
from torch import nn, optim,Tensor
import torch.nn.functional as F
from torchmetrics.functional import accuracy, f1_score as f1
import lightning.pytorch as pl
import numpy as np
from model.classification_model import ClassificationLightningModule


class MLPEmbedding(ClassificationLightningModule):
    def __init__(self, input_size, output_size, in_dim, hidden_sizes, n_hidden_layers=4, dropout_p=0.1, learning_rate=1e-3, model_id:str=""):
        super().__init__(input_size, output_size, in_dim, hidden_sizes, n_hidden_layers, dropout_p, learning_rate, model_id)
       
        embed_dim = 16
        self.learning_rate = learning_rate
        self.embedding = nn.Embedding(256, embed_dim)
        layers = nn.ModuleList([
            nn.Linear(embed_dim * in_dim * input_size, hidden_sizes),
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
        x = self.embedding(x)
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x
