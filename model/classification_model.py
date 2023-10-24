from torch import nn
import torch
from torch import nn, optim,Tensor
import torch.nn.functional as F
from torchmetrics.functional import accuracy, f1_score as f1
import lightning.pytorch as pl
import numpy as np
from abc import abstractmethod

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class ClassificationLightningModule(pl.LightningModule):
    def __init__(self, input_size: int, output_size: int, in_dim: int, hidden_sizes: int, n_hidden_layers: int=4, 
                 dropout_p: float=0.1, learning_rate: float=1e-3, model_id:str="", warmup: int = 150,
                 max_iters: int = 10_000):
        super(ClassificationLightningModule, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.in_dim = in_dim
        self.hidden_sizes = hidden_sizes
        self.n_hidden_layers = n_hidden_layers
        self.dropout_p = dropout_p
        self.learning_rate = learning_rate


        self.val_f1_scores = []
        self.hyper_search_value = None
        self.val_acc_scores = []
        self.val_acc_score = None
        self.test_f1_scores = []
        self.test_f1_score = None
        self.test_acc_scores = []
        self.test_acc_score = None
        
        self.warmup = warmup
        self.max_iters = max_iters

        self.model_id = f"{model_id}/" if model_id != "" else ""
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Reconstruction loss, data reconstruction, perplexity
        """
        raise NotImplementedError
    

    def loss(self, logits, labels):
        """
        Loss function
        
        Args:
            logits (torch.Tensor): Logits
            labels (torch.Tensor): Labels
            
        Returns:
            torch.Tensor: Loss value
        """
        return F.cross_entropy(logits, labels)
    
    
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
        preds = F.log_softmax(logits, dim=1).argmax(dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y, task='multiclass', num_classes=2)
        acc_good = (preds[y == 1] == 1)
        acc_good = torch.mean(acc_good, dtype=torch.float32) if acc_good.numel() > 0 else torch.tensor(0)
        acc_bad = (preds[y == 0] == 0)
        acc_bad = torch.mean(acc_bad, dtype=torch.float32) if acc_bad.numel() > 0 else torch.tensor(0)
        f1score = f1(preds, y, task='binary')
        return preds, loss, acc, acc_good, acc_bad, f1score
    
    def training_step(self, batch, batch_idx):
        """
        PyTorch Lightning calls this inside the training loop
        """
        _, loss, acc, acc_good, acc_bad, f1score = self._get_preds_loss_accuracy(batch)
        if batch_idx % 50 == 0:
            self.log(f'{self.model_id}train/loss', loss.item())
            self.log(f'{self.model_id}train/acc', acc.item())
            self.log(f'{self.model_id}train/acc_good', acc_good.item())
            self.log(f'{self.model_id}train/acc_bad', acc_bad.item())
            self.log(f'{self.model_id}train/f1_score', f1score.item(), prog_bar=True)
        # # log learning rate of optimizer
        # self.log('learning_rate', self.lr_scheduler.get_last_lr()[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        PyTorch Lightning calls this inside the validation loop
        """
        _, loss, acc, acc_good, acc_bad, f1score = self._get_preds_loss_accuracy(batch)
        self.log(f'{self.model_id}val/loss', loss.item(), sync_dist=True)
        self.log(f'{self.model_id}val/acc', acc.item(), sync_dist=True)
        self.log(f'{self.model_id}val/acc_good', acc_good.item(), sync_dist=True)
        self.log(f'{self.model_id}val/acc_bad', acc_bad.item(), sync_dist=True)
        self.log(f'{self.model_id}val/f1_score', f1score.item(), sync_dist=True)
        self.val_f1_scores.append(f1score.item())
        self.val_acc_scores.append(acc.item())
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        PyTorch Lightning calls this inside the test loop
        """
        _, loss, acc, acc_good, acc_bad, f1score = self._get_preds_loss_accuracy(batch)
        self.log(f'{self.model_id}test/loss', loss.item(), sync_dist=True, on_epoch=True)
        self.log(f'{self.model_id}test/acc', acc.item(), sync_dist=True, on_epoch=True)
        self.log(f'{self.model_id}test/acc_good', acc_good.item(), sync_dist=True, on_epoch=True)
        self.log(f'{self.model_id}test/acc_bad', acc_bad.item(), sync_dist=True, on_epoch=True)
        self.log(f'{self.model_id}test/f1_score', f1score.item(), sync_dist=True, on_epoch=True)
        self.test_f1_scores.append(f1score.item())
        self.test_acc_scores.append(acc.item())
        # print(f'f1 score: {f1score.item()} acc: {acc.item()} acc_good: {acc_good.item()} acc_bad: {acc_bad.item()}')
        return loss
    
    def on_validation_epoch_end(self):
        val_score = np.array(self.val_f1_scores).mean()
        val_acc_score = np.array(self.val_acc_scores).mean()
        self.log(f'{self.model_id}val/f1_score_mean', val_score, sync_dist=True, prog_bar=True)
        self.log(f'{self.model_id}val/acc_mean', val_acc_score, sync_dist=True, prog_bar=True)
        self.hyper_search_value = val_score
        self.val_acc_score = val_acc_score
        self.val_f1_scores.clear()
        self.val_acc_scores.clear()
        
    def on_test_epoch_end(self):
        test_score = np.array(self.test_f1_scores).mean()
        self.log(f'{self.model_id}test/f1_score_mean', test_score, sync_dist=True)
        self.test_f1_score = test_score
        self.test_f1_scores.clear()
        test_acc_score = np.array(self.test_acc_scores).mean()
        self.test_acc_score = test_acc_score
        self.test_acc_scores.clear()


    
    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.learning_rate)
        return optimizer
    



    


