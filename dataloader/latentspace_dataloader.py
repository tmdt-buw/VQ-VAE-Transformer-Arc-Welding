import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
import pandas as pd

from dataloader.base_dataloader import BaseDataloader
from dataloader.asimow_dataloader import ASIMoWDataLoader, DataSplitId
import lightning.pytorch as pl

import numpy as np
from tqdm import tqdm
import wandb
import os


class LatentSpaceDataLoader(BaseDataloader):

    def __init__(self, latent_space_model: pl.LightningModule, model_name: str, val_data_ids: list[DataSplitId], 
                 test_data_ids: list[DataSplitId], cycle_seq_number: int, wandb_model_name: str, task: str = "classification", 
                 window_size: int = 50, window_offset: int = 10, shuffle_val_test: bool = True, **kwargs):
        if task == "classification" or task == "classification_ids":
            dataset_name = f"asimow_ls_{task}_{model_name}_cycle_{cycle_seq_number}_{wandb_model_name}"
        elif task == f"autoregressive_ids" or task == f"autoregressive_ids_classification":
            dataset_name = f"{task}_cycle_{cycle_seq_number}_{wandb_model_name}"
        else: 
            raise ValueError(f"task {task} not supported")
        self.val_data_ids = val_data_ids
        self.test_data_ids = test_data_ids
        self.cycle_seq_number = cycle_seq_number
        if model_name == "VQ VAE":
            model_name = "VQ-VAE"
        self.model_name = model_name

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.latent_space_model = latent_space_model.to(self.device)
        self.window_size = window_size
        self.window_offset = window_offset
        self.shuffle_val_test = shuffle_val_test
        super().__init__(dataset_name=dataset_name, task=task, **kwargs)

    def load_raw_data(self, batch_size: int= 512):
        if self.task == "autoregressive_ids":
            task_latent = "reconstruction"
        else:
            task_latent = "classification"
        asimow_dataloader = ASIMoWDataLoader(val_data_ids=self.val_data_ids, test_data_ids=self.test_data_ids,
                                             task=task_latent, cycle_seq_number=self.cycle_seq_number, batch_size=batch_size, 
                                             num_workers=1, seed=42, shuffle=False, undersample=False, window_size=self.window_size, window_offset=self.window_offset)
        train_loader, val_loader, test_loader = asimow_dataloader.get_data_loader(
            batch_size=batch_size, num_workers=1, pin_memory=False)
        return train_loader, val_loader, test_loader
    
    def get_dataset(self):
        train_ds, val_ds, test_ds = self.load_dataset()
        train_ds = self.Dataset(*train_ds)
        if self.shuffle_val_test:
            shuffle_idx = np.random.permutation(len(val_ds[0]))
            val_ds = (val_ds[0][shuffle_idx], val_ds[1][shuffle_idx])
            shuffle_idx = np.random.permutation(len(test_ds[0]))
            test_ds = (test_ds[0][shuffle_idx], test_ds[1][shuffle_idx])
        val_ds = self.Dataset(*val_ds)
        test_ds = self.Dataset(*test_ds)
        return train_ds, val_ds, test_ds

    def get_data_loader(self, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool):
        train_ds, val_ds, test_ds = self.get_dataset()

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                num_workers=num_workers, shuffle=False, pin_memory=pin_memory)
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False, pin_memory=pin_memory)
        return train_loader, val_loader, test_loader
    
    def preprocessing(self, data):
        train_loader, val_loader, test_loader = data
        
        if self.task == "classification":
            new_train_ds, new_val_ds, new_test_ds = self.preprocess_classification(train_loader, val_loader, test_loader) 
        elif self.task == "classification_ids":
            new_train_ds, new_val_ds, new_test_ds = self.preprocess_classification_one_hot(train_loader, val_loader, test_loader)
        elif self.task == "autoregressive_ids" or self.task == "autoregressive_ids_classification":
            new_train_ds, new_val_ds, new_test_ds = self.preprocess_classification_autoregressive(train_loader, val_loader, test_loader)
            pass
        else:
            raise ValueError(f"task {self.task} not supported")    
        
        return new_train_ds, new_val_ds, new_test_ds
    
    def preprocess_classification(self, train_loader, val_loader, test_loader):
        has_patch_embed = self.model_name == "VQ-VAE-Patch"
        
        if self.model_name == "VQ-VAE" or self.model_name == "vqvae" or \
                self.model_name == "VQ-VAE2" or self.model_name == "vqvae2" \
                or self.model_name == "VQ-VAE-Patch":
            new_train_ds = self.create_latent_space_dataset_VQ_VAE(
                train_loader, seq_len=self.cycle_seq_number, has_patch_embed=has_patch_embed)
            new_val_ds = self.create_latent_space_dataset_VQ_VAE(
                val_loader, seq_len=self.cycle_seq_number, has_patch_embed=has_patch_embed)
            new_test_ds = self.create_latent_space_dataset_VQ_VAE(
                test_loader, seq_len=self.cycle_seq_number, has_patch_embed=has_patch_embed)
        else:
            raise ValueError(f"Model name: {self.model_name} is not supported.")
        return new_train_ds, new_val_ds, new_test_ds
    
    def preprocess_classification_one_hot(self, train_loader, val_loader, test_loader):
        has_patch_embed = self.model_name == "VQ-VAE-Patch"
        if self.model_name == "VQ-VAE" or self.model_name == "vqvae" or self.model_name == "VQ-VAE2" \
            or self.model_name == "vqvae2" or self.model_name == "VQ-VAE-Patch":
                
            new_train_ds = self.create_latent_space_dataset_VQ_VAE_IDs(
                train_loader, seq_len=self.cycle_seq_number, has_patch_embed=has_patch_embed)
            new_val_ds = self.create_latent_space_dataset_VQ_VAE_IDs(
                val_loader, seq_len=self.cycle_seq_number, has_patch_embed=has_patch_embed)
            new_test_ds = self.create_latent_space_dataset_VQ_VAE_IDs(
                test_loader, seq_len=self.cycle_seq_number, has_patch_embed=has_patch_embed)
        else:
            raise ValueError(f"Model name: {self.model_name} is not supported.")
        return new_train_ds, new_val_ds, new_test_ds
    
    def preprocess_classification_autoregressive(self, train_loader, val_loader, test_loader):
        has_patch_embed = self.model_name == "VQ-VAE-Patch"
        if self.model_name == "VQ-VAE" or self.model_name == "vqvae" or self.model_name == "VQ-VAE2" \
            or self.model_name == "vqvae2" or self.model_name == "VQ-VAE-Patch":
                
            new_train_ds = self.create_latent_space_dataset_VQ_VAE_autoreggressive(
                train_loader, seq_len=self.cycle_seq_number, has_patch_embed=has_patch_embed)
            new_val_ds = self.create_latent_space_dataset_VQ_VAE_autoreggressive(
                val_loader, seq_len=self.cycle_seq_number, has_patch_embed=has_patch_embed)
            new_test_ds = self.create_latent_space_dataset_VQ_VAE_autoreggressive(
                test_loader, seq_len=self.cycle_seq_number, has_patch_embed=has_patch_embed)
        else:
            raise ValueError(f"Model name: {self.model_name} is not supported.")
        return new_train_ds, new_val_ds, new_test_ds


    def split_train_validation_test(self, df: pd.DataFrame):
        print("Splitting train, validation and test data is done in the asimow dataloader.")

    def data_transform(self, data):
        return data

    def get_latent_space(self, x, has_patch_embed: bool=False):
        if has_patch_embed:
            x = self.latent_space_model.patch_embed(x)
        else:
            x = x.permute(0, 2, 1)
        z_e = self.latent_space_model.encoder(x)
        loss, z_q, perplexity, min_encodings, min_encoding_indices = self.latent_space_model.vector_quantization(z_e)

        return z_q
    
    def get_latent_space_IDs(self, x, has_patch_embed: bool=False):
        if has_patch_embed:
            x = self.latent_space_model.patch_embed(x)
        else:
            x = x.permute(0, 2, 1)
        z_e = self.latent_space_model.encoder(x)
        loss, z_q, perplexity, min_encodings, min_encoding_indices = self.latent_space_model.vector_quantization(z_e)
        return min_encoding_indices
    
    @staticmethod
    def get_sampling_weights(labels):
        ratio = np.mean(labels == 0)
        sampling_weights = np.zeros_like(labels, dtype=np.float32)
        sampling_weights[labels==0] = 1 - ratio
        sampling_weights[labels==1] = ratio
        return sampling_weights

    def create_latent_space_dataset_VQ_VAE(self, loader: DataLoader, seq_len: int, has_patch_embed: bool=False):
        """
        This function is used to create a dataset for the classification based one the embedding vectors.
        
        Args:
            loader (DataLoader): The dataloader to be used.
            seq_len (int): The sequence length.
            has_patch_embed (bool, optional): Whether the model has patch embedding. Defaults to False.
            
        Returns:
            new_ds_x (np.array): The latent space dataset (batch_size, seq_len, embedding_dim*enc_out_len).
            new_ds_y (np.array): The labels of the dataset (batch_size,).
        """
        embedding_dim = self.latent_space_model.embedding_dim
        enc_out_len  =  self.latent_space_model.enc_out_len
        new_ds_x = np.empty((0, seq_len, int(embedding_dim * enc_out_len)))
        new_ds_y = np.empty((0,))
        
        self.latent_space_model.eval()
        with torch.no_grad():
            for batch_i, (x, y) in tqdm(enumerate(loader)):
                t_x = []
                for i in range(seq_len):
                    x_i = x[:, i*self.window_size:(i+1)*self.window_size, :].clone().detach()
                    x_i = x_i.to(self.device)
                    z_q = self.get_latent_space(x_i, has_patch_embed=has_patch_embed)
                    t_x.append(z_q.cpu().numpy().reshape((z_q.shape[0], -1)))
                t_x = np.array(t_x).swapaxes(0, 1)

                new_ds_x = np.append(new_ds_x, t_x, axis=0)
                new_ds_y = np.append(new_ds_y, y.cpu().numpy(), axis=0)

        return new_ds_x, new_ds_y
    
    def create_latent_space_dataset_VQ_VAE_IDs(self, loader: DataLoader, seq_len: int, has_patch_embed: bool=False, no_labels: bool=False):
        """
        This function is sampling the latent space IDs for classification.
        
        Args:
            loader (DataLoader): The dataloader to be used.
            seq_len (int): The sequence length.
            has_patch_embed (bool, optional): Whether the model has patch embedding. Defaults to False.
            no_labels (bool, optional): Whether the dataset has labels. Defaults to False.
            
        Returns:
            new_ds_x (np.array): The latent space dataset (batch_size, seq_len, embedding_dim).
            new_ds_y (np.array): The labels of the dataset (batch_size,).
        """
        enc_out_len = int(self.latent_space_model.enc_out_len)
        new_ds_x = np.empty((0, seq_len, enc_out_len), dtype=int)
        new_ds_y = np.empty((0,))
        
        self.latent_space_model.eval()
        with torch.no_grad():
            for _, batch_item in tqdm(enumerate(loader)):
                if no_labels:
                    x = batch_item
                else:
                    x, y = batch_item
                t_x = []
                for i in range(seq_len):
                    x_i = x[:, i*self.window_size:(i+1)*self.window_size, :].clone().detach()
                    x_i = x_i.to(self.device)
                    ids = self.get_latent_space_IDs(x_i, has_patch_embed)
                    t_x.append(ids.cpu().numpy().reshape((x_i.shape[0], -1)))
                t_x = np.array(t_x).swapaxes(0, 1)

                new_ds_x = np.append(new_ds_x, t_x, axis=0)
                if not no_labels:
                    new_ds_y = np.append(new_ds_y, y.cpu().numpy(), axis=0)
        if no_labels:
            new_ds_y = np.zeros(new_ds_x.shape[0])
        return new_ds_x, new_ds_y

    def create_latent_space_dataset_VQ_VAE_autoreggressive(self, loader: DataLoader, seq_len: int=1, has_patch_embed: bool=False):
        """
        This function is used to create a dataset for the autoregressive sampling.

        Args:
            loader (DataLoader): The dataloader to be used.
            seq_len (int): The sequence length.
            has_patch_embed (bool, optional): Whether the model has patch embedding. Defaults to False.
        Returns:
            new_ds_x (np.array): The latent space dataset (batch_size, seq_len, embedding_dim).
            new_ds_y (np.array): The labels of the dataset (batch_size,).
        """
        if self.task == "autoregressive_ids":
            new_ds_x, new_ds_y = self.create_latent_space_dataset_VQ_VAE_IDs(loader, seq_len=seq_len, has_patch_embed=has_patch_embed, no_labels=True)
        else:    
            new_ds_x, new_ds_y = self.create_latent_space_dataset_VQ_VAE_IDs(loader, seq_len=seq_len, has_patch_embed=has_patch_embed, no_labels=False)
        new_ds_x = new_ds_x.reshape((new_ds_x.shape[0], -1))
        
        return new_ds_x, new_ds_y


def get_metadata_and_artifact_dir(model_name: str) -> tuple[str, str]:
    """
    Download Model from wandb and return metadata and artifact_dir

    Args:
        model_name (str): artifact link from wandb - e.g. "username/project_name/artifact_name:version"

    Returns:
        tuple[dict, str]: metadata and artifact_dir
    """
    artifact_dir = f"./artifacts/{model_name.split('/')[-1]}"
    artifact = wandb.use_artifact(model_name, type='model')
    if not os.path.exists(artifact_dir):
        artifact_dir = artifact.download()

    original_filename = artifact.metadata["original_filename"]

    model_name_split = original_filename.split("-")
    if model_name_split[0] == "VQ" and model_name_split[1] == "VAE" and model_name_split[2] == "Patch":
        model_name = "VQ-VAE-Patch"
    elif model_name_split[0] == "VQ":
        model_name = f"{model_name_split[0]}-{model_name_split[1]}" 
    else:
        raise ValueError(f"Model name: {model_name} not supported.")
    artifact_dir = artifact_dir + '/model.ckpt'
    return model_name, artifact_dir


class LatentPredDataModule(pl.LightningDataModule):
 
    def __init__(self, latent_space_model, task: str, n_cycles: int, val_data_ids: list[DataSplitId], test_data_ids: list[DataSplitId], model_name: str, wandb_model_name: str, 
                 batch_size: int = 32, window_size: int = 200, window_offset: int = 0, shuffle_val_test: bool = True):
        super().__init__()
        self.n_cycles = n_cycles
        self.task = task
        self.val_ids = val_data_ids
        self.test_ids = test_data_ids
        self.model_name = model_name
        self.batch_size = batch_size
        self.latent_space_model = latent_space_model
        self.train_sampling = None
        self.wandb_model_name = wandb_model_name
        self.window_size = window_size
        self.window_offset = window_offset
        self.shuffle_val_test = shuffle_val_test

    def setup(self, stage: str):
        self.latent_dataloader = LatentSpaceDataLoader(latent_space_model=self.latent_space_model, model_name=self.model_name, task=self.task, 
                                                 cycle_seq_number=self.n_cycles, val_data_ids=self.val_ids, test_data_ids=self.test_ids, 
                                                 wandb_model_name=self.wandb_model_name, window_size=self.window_size, window_offset=self.window_offset)
        train_ds, val_ds, test_ds = self.latent_dataloader.get_dataset()
        
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        
        if self.task != "autoregressive_ids":
            self.train_sampling = self.latent_dataloader.get_sampling_weights(train_ds.labels)
        else: 
            self.train_sampling = None

    def get_sampler(self):
        if self.task != "autoregressive_ids":
            sampler = WeightedRandomSampler(self.train_sampling, num_samples=len(self.train_sampling), replacement=True)
        else:
            sampler = RandomSampler(self.train_ds)
        return sampler


    def train_dataloader(self):
        sampler = self.get_sampler()
        return DataLoader(self.train_ds, batch_size=self.batch_size, sampler=sampler, num_workers=8, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=8, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=8, pin_memory=False)