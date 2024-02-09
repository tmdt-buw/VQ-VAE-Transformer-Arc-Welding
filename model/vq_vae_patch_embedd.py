import torch
from torch import nn
from model.autencoder_lightning_base import Autoencoder
from model.vector_quantizer import VectorQuantizer, ResidualVQLightning


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1).unsqueeze(1)
        x = self.proj(x) 
        return x

class PatchEmbeddingInverse(nn.Module):
    def __init__(self, patch_size, embed_dim, input_dim):
        super().__init__()
        self.patch_size = patch_size
        
        if patch_size == 25:
            kernel_size = 5
            self.proj = nn.Sequential(
                nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=kernel_size, stride=kernel_size),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose1d(embed_dim, 1, kernel_size=kernel_size, stride=kernel_size),
            )
        elif patch_size == 10:
            self.proj = nn.Sequential(
                nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose1d(embed_dim, 1, kernel_size=5, stride=5),
            )
        elif patch_size == 50:
            self.proj = nn.Sequential(
                nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=10, stride=10),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose1d(embed_dim, 1, kernel_size=5, stride=5),
            )
        else:
            raise NotImplementedError(f"Patch size not implemented: {patch_size}")

        
        self.input_dim = input_dim

    def forward(self, x):

        x = self.proj(x)

        x = x.reshape(x.shape[0], -1, self.input_dim)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dropout_p: float = 0.1, batch_norm: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(channels) if batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(channels) if batch_norm else nn.Identity(),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return x + self.block(x)


class SepCNNBlock(nn.Module):
    
    def __init__(self, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.shared_conv = nn.Conv1d(hidden_dim, embedding_dim, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        _, _, dims = x.shape
        cnn_out = []
        for i in range(dims):
            x_t = self.shared_conv(x[:,:,i].unsqueeze(2)) 
            cnn_out.append(x_t)
        x = torch.cat(cnn_out, dim=2)

        return x.permute(0, 2, 1)

class CNNBlock(nn.Module):
    def __init__(self, embed_dim: int, seperate: bool = True, kernel_size: int = 3, stride: int = 1, padding: int = 1, dropout_p: float = 0.1, batch_norm: bool = True, n_resblocks : int = 1):
        super(CNNBlock, self).__init__()
        # Single convolutional layer blocks whose weights will be shared
        self.seperate = seperate
        self.shared_conv = nn.Sequential(
            *[ResBlock(channels=embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, dropout_p=dropout_p, batch_norm=batch_norm) for _ in range(n_resblocks)],
        )
        
        
    def forward(self, x):
        _, _, dims = x.shape

        if self.seperate:
            cnn_out = []
            for i in range(dims):
                x_t = self.shared_conv(x[:,:,i].unsqueeze(2)) 
                cnn_out.append(x_t)
            x = torch.cat(cnn_out, dim=2)
        else:
            x = self.shared_conv(x)
        return x

    
class VQVAEPatch(Autoencoder):
    
    def __init__(self, hidden_dim: int, input_dim: int, num_embeddings: int, embedding_dim: int, 
                 n_resblocks: int, learning_rate: float, dropout_p: float=0.1, patch_size: int=25, seq_len: int=200, batch_norm: bool = True, beta: float=0.25, 
                 use_improved_vq: bool = False, kmeans_iters: int = 0, threshold_ema_dead_code: int = 2):
        
        super().__init__(hidden_dim=hidden_dim, input_dim=input_dim, num_embeddings=num_embeddings, 
                    embedding_dim=embedding_dim, n_resblocks=n_resblocks, learning_rate=learning_rate, seq_len=seq_len, dropout_p=dropout_p)
        self.patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=hidden_dim)
        self.encoder = nn.Sequential(
            CNNBlock(embed_dim=hidden_dim, n_resblocks=n_resblocks, dropout_p=dropout_p, batch_norm=batch_norm),
            SepCNNBlock(hidden_dim=hidden_dim, embedding_dim=embedding_dim)
        
        )

        if use_improved_vq:
            self.vector_quantization = ResidualVQLightning(
                num_quantizers=1, e_dim=embedding_dim, n_e=num_embeddings,
                kmeans_init=True, kmeans_iters=kmeans_iters, threshold_ema_dead_code=threshold_ema_dead_code

            )
        else:
            self.vector_quantization = VectorQuantizer(n_e=num_embeddings, e_dim=embedding_dim, beta=beta)


        self.decoder =  nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            CNNBlock(embed_dim=hidden_dim, seperate=False, n_resblocks=n_resblocks, dropout_p=dropout_p, batch_norm=batch_norm)
        )
        
        self.reverse_patch_embed = PatchEmbeddingInverse(patch_size=patch_size, embed_dim=hidden_dim, input_dim=input_dim)
        
        self.enc_out_len = seq_len // patch_size * input_dim
        self.patch_size = patch_size
        self.apply(self.weights_init)
        
        

    def forward(self, x):
        # print("input", x.shape)
        x = self.patch_embed(x)
        z_e = self.encoder(x)
        # print("z_e", z_e.shape)
        
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        # print("z_q", z_q.shape)
        
        x_hat = self.decoder(z_q.permute(0, 2, 1))
        x_hat = self.reverse_patch_embed(x_hat)
       
        return embedding_loss, x_hat, perplexity