import torch
from torch import nn
from torch.nn import functional as F
from model.plot_helper import plot_recon
import lightning.pytorch as pl
from abc import abstractmethod
from lightning.pytorch.loggers.wandb import WandbLogger


class Autoencoder(pl.LightningModule):

    def __init__(self, wandb_logger: WandbLogger, hidden_dim: int, input_dim: int, num_embeddings: int, embedding_dim: int, 
                 n_resblocks: int, learning_rate: float, seq_len: int=200, dropout_p: float=0.1):
        """
        Initialize Autoencoder
        
        Args:
            input_dim (int): Input dimension
            num_embeddings (int): Number of embeddings
            embedding_dim (int): Embedding dimension
            n_resblocks (int): Number of residual blocks
            learning_rate (float): Learning rate
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.wandb_logger = wandb_logger
        self.learning_rate = learning_rate
        self.dropout_p = dropout_p
        self.n_resblocks = n_resblocks
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        self.last_recon = []

        # optimizer params
        self.betas = (0.9, 0.95)
        self.weight_decay = 0.1 
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Reconstruction loss, data reconstruction, perplexity
        """
        raise NotImplementedError


    def loss(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Loss function (Rconstructon loss (MSE))
        
        Args:
            preds (torch.Tensor): Predictions
            labels (torch.Tensor): Labels
            
        Returns:
            torch.Tensor: Loss
        """
        return F.mse_loss(preds, labels)
    
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            try:
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)
            except AttributeError:
                print("Skipping initialization of ", classname)

    def _forward_setp(self, x: torch.Tensor):
        embedding_loss, data_recon, perplexity = self(x)
        recon_error = F.mse_loss(data_recon, x)
        loss = recon_error + embedding_loss
        return loss, recon_error, data_recon

    def training_step(self, batch, batch_idx):
        """
        PyTorch Lightning calls this inside the training loop
        """
        loss, recon_error, data_recon = self._forward_setp(batch)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/recon_error', recon_error)

        sample_idx = torch.randint(0, len(batch), (1,))
        self.last_recon = (batch[sample_idx], data_recon[sample_idx])  
        return {'loss': loss,
                'recon_error': recon_error}

    def validation_step(self, batch, batch_idx):
        """
        PyTorch Lightning calls this inside the validation loop
        """
        loss, recon_error, data_recon = self._forward_setp(batch)
        self.log('val/loss', loss, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log('val/recon_error', recon_error, sync_dist=True, on_epoch=True)

        return {'loss': loss,
                'recon_error': recon_error,
                'data_recon': data_recon}

    def test_step(self, batch, batch_idx):
        """
        PyTorch Lightning calls this inside the test loop
        """
        loss, recon_error, data_recon = self._forward_setp(batch)
        self.log('test/loss', loss, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log('test/recon_error', recon_error, sync_dist=True, on_epoch=True)
        return {'loss': loss,
                'recon_error': recon_error,
                'data_recon': data_recon}

    def on_train_epoch_end(self) -> None:
        x_batch = self.last_recon[0].squeeze(0)
        data_recon = self.last_recon[1].squeeze(0)
        # plot_recon(self.wandb_logger, x_batch, data_recon, title=f"Training Reconstruction")
    
    def on_validation_batch_end(self, outputs, batch, batch_idx) -> None:
        if batch_idx % 50 == 0:
            sample_idx = torch.randint(0, len(batch), (1,))
            data_recon = outputs['data_recon'][sample_idx].squeeze(0)
            x_batch = batch[sample_idx].squeeze(0)
            # plot_recon(self.wandb_logger, x_batch, data_recon, title=f"Validation Reconstruction {batch_idx}")

    def on_test_batch_end(self, outputs, batch, batch_idx) -> None:
        if batch_idx % 10 == 0:
            sample_idx = torch.randint(0, len(batch), (1,))
            data_recon = outputs['data_recon'][sample_idx].squeeze(0)
            x_batch = batch[sample_idx].squeeze(0)
            plot_recon(self.wandb_logger, x_batch, data_recon, title=f"Test Reconstruction {batch_idx}")  

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.learning_rate)
        return optimizer

