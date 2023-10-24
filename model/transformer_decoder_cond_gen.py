# https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
import torch
import math
import numpy as np
from torch import nn, Tensor, optim
from torch.nn import functional as F
import lightning.pytorch as pl
from model.embedding import LatentEmbeddingCond
from model.transformer_block import Block


class MyTransformerDecoder(pl.LightningModule):

    def __init__(self, d_model: int = 64, n_classes: int = 131, seq_len: int = 100, n_blocks: int = 2, n_head: int = 6, res_dropout=0.1, att_dropout=0.0, learning_rate: float = 1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.betas = (0.9, 0.95)
        self.weight_decay = 0.1 
        self.seq_len = seq_len
        self.embedding = LatentEmbeddingCond(
            input_size=n_classes, d_model=d_model, cond_size=2)

        self.transformer = nn.ModuleDict(dict(
            drop=nn.Dropout(res_dropout),
            h=nn.ModuleList([Block(d_model=d_model, seq_len=seq_len, n_head=n_head,
                            res_dropout=res_dropout, att_dropout=att_dropout) for _ in range(n_blocks)]),
            ln_f=nn.LayerNorm(d_model),
        ))
        self.lm_head = nn.Linear(d_model, n_classes, bias=False)
        
        # initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_blocks))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.4fM" % (n_params/1e6,))
        self.save_hyperparameters()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(
                list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(
            optim_groups, lr=self.learning_rate, betas=self.betas)
        return optimizer

    def forward(self, x, cond, targets=None):

        b, t = x.size()
        x = self.embedding(x, cond)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits
    
    def _step(self, batch):
        x, cond, y = batch
        logits = self(x, cond)
        loss = self.loss(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        """
        PyTorch Lightning calls this inside the training loop
        """
        loss = self._step(batch)
        self.log(f'train/loss', loss.item(), prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        PyTorch Lightning calls this inside the validation loop
        """
        loss = self._step(batch)
        self.log(f'val/loss', loss.item(), prog_bar=True)
        return loss
        
    
    def test_step(self, batch, batch_idx):
        """
        PyTorch Lightning calls this inside the test loop
        """
        loss = self._step(batch)
        self.log(f'test/loss', loss.item())
        return loss
    
    def generate(self, x, cond, do_sample=False, top_k=None):
        with torch.no_grad(): 
            for _ in range(self.seq_len):
                
                x_cond = x if x.size(1) <= self.seq_len else x[:, -self.seq_len:]
                # print(f"{x_cond.shape=} - {x.shape=}")
                logits = self(x_cond, cond)

                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                probs = probs[:, -1]
                # print(f"{probs.shape=} - {logits.shape=}")
                if do_sample:
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    _, idx_next = torch.topk(probs, k=1, dim=-1)
                # idx_next = idx_next[:, [-1]].squeeze(-1)
                # print(f"{idx_next.shape=}")
                x = torch.cat([x, idx_next], dim=-1)
        return x

    def loss(self, logits, labels):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
    

def sample(autoreg_mode, cond_value: int, batch_size=32, top_k=None, device="cpu"):
    start_token = 128
    x = np.full((batch_size), start_token, dtype=np.int64) 
    x = x.reshape((batch_size, 1))
    
    cond = np.full((batch_size), cond_value, dtype=np.int64)
    
    x = torch.from_numpy(x)
    cond = torch.from_numpy(cond)
    # print(f"{x.shape=} - {cond.shape=}")
    x, cond = x.to(device), cond.to(device)
    # print(f"{x.device=} - {cond.device=},{model.device=}")
    pred = autoreg_mode.generate(x, cond, do_sample=True)
        # remove start end token
    pred = pred[:, 1:]
    pred = pred[:, :-1]
    pred[pred > 127] = 127
    return pred, cond

def main():
    batch_size = 3
    seq_len = 2
    d_model = 4
    num_classes = 130

    x = torch.randint(0, 130, (batch_size, seq_len))
    cond = torch.randint(0, 2, (batch_size, ))
    print(f"{x.shape=}")
    print(x)
    model = MyTransformerDecoder(
        d_model=d_model, seq_len=seq_len, n_classes=num_classes, n_head=4, n_blocks=2)
    logits = model(x, cond)
    print(f"{logits.shape=}")
    print(torch.argmax(F.softmax(logits, dim=-1), axis=-1))

    x = torch.randint(0, 130, (batch_size, 2))
    model.generate(x, cond, max_len=10)
if __name__ == '__main__':
    main()
