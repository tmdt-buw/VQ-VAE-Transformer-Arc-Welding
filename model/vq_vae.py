from torch import nn
from model.plot_helper import plot_recon
from model.vector_quantizer import VectorQuantizer
from model.autencoder_lightning_base import Autoencoder
import lightning.pytorch as pl


class ResBlock(nn.Module):
    def __init__(self, dim, dropout_p=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return x + self.block(x)
    
class ResBlockLinear(nn.Module):
    def __init__(self, dim, dropout_p=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, dim*2),
            nn.BatchNorm1d(dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return x + self.block(x)



class VectorQuantizedVAE(Autoencoder):
    def __init__(self, wandb_logger, decoder_type: str, input_dim: int, hidden_dim: int, num_embeddings: int, embedding_dim: int, 
                 n_resblocks: int, learning_rate: float, seq_len: int=200, dropout_p: float=0.1):
        """
        A PyTorch Lightning module implementing a Vector Quantized Variational Autoencoder (VQ-VAE).

        Args:
            decoder_type (str): The type of decoder to use. Must be either "Conv" or "Linear". This indicates whether the decoder should use convolutional layers or linear layers.
            input_dim (int): The number of input features.
            hidden_dim (int): The number of hidden features.
            num_embeddings (int): The number of embeddings in the codebook.
            embedding_dim (int): The dimensionality of the embeddings in the codebook.
            n_resblocks (int): The number of residual blocks in the encoder and decoder.
            learning_rate (float): The learning rate to use for training.
            seq_len (int, optional): The length of the input sequence. Defaults to 200.
        """
        super().__init__(wandb_logger=wandb_logger, hidden_dim=hidden_dim, input_dim=input_dim, num_embeddings=num_embeddings, 
                         embedding_dim=embedding_dim, n_resblocks=n_resblocks, learning_rate=learning_rate, seq_len=seq_len, dropout_p=dropout_p)

        self.enc_out_len = self.compute_out_len(seq_len=seq_len)
        # print(f"Output length: {self.enc_out_len}")
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            *[ResBlock(hidden_dim, dropout_p=dropout_p) for _ in range(n_resblocks)],
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, embedding_dim, kernel_size=3, stride=2, padding=2),
        )

        self.vector_quantization = VectorQuantizer(
            num_embeddings, embedding_dim, 0.25)

        if decoder_type == "Conv":
            output_padding = 1 - (seq_len % 2)
            
            if seq_len % 8 == 0 or seq_len % 8 == 7:
                pad = 3
            elif seq_len % 8 == 6 or seq_len % 8 == 5:
                pad = 4
            elif seq_len % 8 == 2 or seq_len % 8 == 1:
                pad = 6
            else: pad = 5


            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(embedding_dim, hidden_dim, kernel_size=3, stride=2, padding=2, output_padding=1),
                *[ResBlock(hidden_dim, dropout_p=dropout_p) for _ in range(n_resblocks)],
                nn.GELU(),
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=7, stride=2, padding=pad, output_padding=output_padding)
            )
        elif decoder_type == "Linear":
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim * self.enc_out_len, hidden_dim),
                *[ResBlockLinear(hidden_dim, dropout_p=dropout_p) for _ in range(n_resblocks)],
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, seq_len * input_dim),
            )
        else:
            raise ValueError(f"Decoder type must be either Conv or Linear, but is {decoder_type}")
        self.decoder_type = decoder_type
        self.apply(self.weights_init)
        

    @staticmethod
    def compute_out_len(seq_len, k_1=7, k_2=5, k_3=3, s_1=2, s_2=2, s_3=2, p_1=3, p_2=2, p_3=2):
        out_len_1 = (seq_len + 2 * p_1 - k_1) // s_1 + 1
        out_len_2 = (out_len_1 + 2 * p_2 - k_2) // s_2 + 1
        return (out_len_2 + 2 * p_3 - k_3) // s_3 + 1
    
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            try:
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)
            except AttributeError:
                print("Skipping initialization of ", classname)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        z_e = self.encoder(x)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        if self.decoder_type == "Linear":
            z_q = z_q.reshape(z_q.shape[0], z_q.shape[1] * z_q.shape[2])
        x_hat = self.decoder(z_q)
        if self.decoder_type == "Conv":
            x_hat = x_hat.permute(0, 2, 1)
            assert x_hat.shape == x.permute(0,2,1).shape, f"Shape of x_hat is {x_hat.shape}, but should be {x.permute(0,2,1).shape}"
        x_hat = x_hat.reshape(-1, self.seq_len, self.input_dim)
       
        return embedding_loss, x_hat, perplexity



