from abc import ABC, abstractmethod
import os
import logging as log
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from dataloader.utils import load_pickle_file, write_pickle_file, MyScaler, get_data_path
from torch.nn import functional as F



class MyClassificationDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        """
        Classification Dataset
        Args:
            data (np.array): Input data
            labels (np.array): Target data
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
    
    
class MyClassificationDatasetIDs(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        """
        Classification Dataset
        Args:
            data (np.array): Input data
            labels (np.array): Target data
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
        # x = torch.tensor(self.data[idx], dtype=torch.float32)
        # x = x.reshape(-1)
        # return x, torch.tensor(self.labels[idx], dtype=torch.long)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        x = x.reshape(-1)
        return x, torch.tensor(self.labels[idx], dtype=torch.long)
    
    
class MyReconstructionDataset(Dataset):
    def __init__(self, data: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Dataset for Reconstruction Task (e.g. VQ-VAE)

        Args:
            data (np.array): Input and target data
            y (Optional[np.array], optional): Not needed here but needed for compatibility with other dataloaders. Defaults to None.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)
    
class MyLatentAutoregressiveDataset(Dataset):
    def __init__(self, data: np.ndarray, y: np.ndarray | None = None):
        """
        Dataset for Autoregressive model to predict the latent space 
        To achive this we shift the data to the right and add a start token and an end token
        -> [start,2,3,4] predicts [2,3,4,end]

        Args:
            data (np.array): Input and target data
            y (Optional[np.array], optional): Not needed here but needed for compatibility with other dataloaders. Defaults to None.
        """
        max_token = int(np.max(data))
        start_token = max_token + 1
        end_token = max_token + 2
        start_vec = np.full((len(data),), start_token)
        end_vec = np.full((len(data),), end_token)
        self.num_classes = max_token + 3

        # shift right with start token
        self.data = np.concatenate([start_vec[:, None], data], axis=1)
        # add end token
        self.data_shifted = np.concatenate([data, end_vec[:, None]], axis=1)

        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = torch.tensor(self.data_shifted[idx], dtype=torch.long)

        if self.labels is not None:
            cond = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            cond = torch.zeros((1,), dtype=torch.long)
        return x, cond, y

class BaseDataloader(ABC):

    def __init__(self, dataset_name: str, task: str, seed: int = 42, data_directory_path: Optional[str] = None, shuffle: bool = True, **kwargs):
        """
        Baseclass for a Dataloader for Classification and Reconstruction Task

        Args:
            dataset_name (str): name of the dataset
            task (str): classification or reconstruction
            seed (int, optional): Random seed which is set for numpy. Defaults to 42.
            data_directory_path (Optional[str], optional): path to the /data/ directory. Defaults to None, then it gets the path from get_data_path().
            shuffle (bool): Shuffle input after train test split

        Raises:
            NotImplementedError: 
        """
        self.dataset_name = dataset_name
        self.task = task
        self.shuffle = shuffle
        self.dataset: Dataset
        if task == "classification":
            self.Dataset = MyClassificationDataset
        elif task == "classification_ids":
            self.Dataset = MyClassificationDatasetIDs
        elif task == "reconstruction":
            self.Dataset = MyReconstructionDataset
        elif task == "autoregressive_ids" or task == "autoregressive_ids_classification":
            self.Dataset = MyLatentAutoregressiveDataset
        else:
            raise NotImplementedError(f"Task {task} not implemented")

        if data_directory_path is None:
            data_directory_path = get_data_path()

        np.random.seed(seed)
        self.data_directory_path = data_directory_path
        self.dataset_path = self.data_directory_path + "/quality_prediction_data/" + self.dataset_name + "/"
        self.scaler = MyScaler()
        self.seed = seed
        self.kwargs = kwargs

        if not os.path.exists(self.dataset_path):
            log.info(f"data_path does not exists, creating new data path {self.dataset_path}")
            self.preprocess_and_save_data()
        else:
            log.info(f"Load data from {self.dataset_path}")


    def preprocess_and_save_data(self):
        """
        Preprocess the raw data and save it as a pickle file for faster loading
        1. Load raw data
        2. Preprocess data
        3. Save data as pickle file
        """
        data = self.load_raw_data()
        data = self.preprocessing(data)
        self.save_one_pickle_file(data)

    @abstractmethod
    def load_raw_data(self):
        """
        Function to load the raw data
        Raises:
            NotImplementedError: 
        """
        raise NotImplementedError

    @abstractmethod
    def preprocessing(self, data):
        """
        Preprocess the raw data
        This function can do the processing steps which are needed every task on the dataset. 
        The idea is to include time consuming steps here, so that the data is only processed once and not every time the data is loaded.

        Args:
            data (DataFrame|any): loaded raw data

        Raises:
            NotImplementedError: 
        """
        raise NotImplementedError
    
    @abstractmethod
    def data_transform(self, data):
        """
        The data_transform function is used to transform the data for the specific task after the already precprocessed data is loaded.
        Here should only processing steps be included which are needed for the specific task.
        Args:
            data (DataFrame| any): loaded data

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_data_loader(self, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get the dataloader for the specific task

        Args:
            batch_size (int): batch size
            shuffle (bool): if the training data should be shuffled
            num_workers (int): num workers for the pytorch dataloader
            pin_memory (bool): pin memory for faster loading

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]: train_data_loader, validation_data_loader, test_data_loader
        """
        raise NotImplementedError
    
    @abstractmethod
    def split_train_validation_test(self, df: pd.DataFrame):
        """
        Splits the DataFrame into train, validation and test set

        Args:
            df (pd.DataFrame): already preprocessed and transformed DataFrame

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def save_one_pickle_file(self, data: pd.DataFrame) -> None:
        """
        Save Preprocessed data as pickle file

        Args:
            data (pd.DataFrame): preprocessed data
        """
        log.info(f"Save data to {self.dataset_path}")
        write_pickle_file(data, data_path=self.dataset_path,
                          file_name="dataset.pickle")
        
    def load_dataset(self):
        """
        Load the preprocessed and transformed data

        Returns:
            pd.DataFrame: preprocessed data
        """
        log.info(f"Load data from {self.dataset_path}")
        data = load_pickle_file(data_path=self.dataset_path, file_name="dataset.pickle")
        log.info(f"Transform data {self.dataset_path}")
        data = self.data_transform(data)
        return data
