from torch import nn
import torch
from torch import nn, optim,Tensor
import torch.nn.functional as F
from torchmetrics.functional import accuracy, f1_score as f1
import lightning.pytorch as pl
import numpy as np
from model.classification_model import ClassificationLightningModule


class MLP_BC(ClassificationLightningModule):
    def __init__(self, input_size, output_size, in_dim, hidden_sizes, n_hidden_layers=4, dropout_p=0.1, learning_rate=1e-3, model_id:str=""):
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
        return x.reshape(-1)
    
    def loss(self, logits, labels):
        
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    def _get_preds_loss_accuracy(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Helper function to compute predictions, loss and accuracy
        Input:
            batch: tuple of (x, y)
        Output:
            preds: predictions
            loss: loss value
            acc: accuracy
            acc_good: accuracy for good quality
            acc_bad: accuracy for bad quality
            f1score: f1 score
        """
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        acc = accuracy(logits, y, task='binary')
        acc_good = (logits[y == 1] > 0.5)
        acc_good = torch.mean(acc_good, dtype=torch.float32) if acc_good.numel() > 0 else torch.tensor(0)
        acc_bad = (logits[y == 0] <= 0.5)
        acc_bad = torch.mean(acc_bad, dtype=torch.float32) if acc_bad.numel() > 0 else torch.tensor(0)
        f1score = f1(logits, y, task='binary')
        return logits, loss, acc, acc_good, acc_bad, f1score
   