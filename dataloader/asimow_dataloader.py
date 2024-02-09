import argparse
import logging as log
import pandas as pd
import numpy as np
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, RandomSampler
from dataloader.base_dataloader import BaseDataloader
from dataloader.utils import load_pickle_file, shuffle_np, shuffle_and_undersample

import lightning.pytorch as pl
from typing import Optional


class DataSplitId:
    """
    DataSplitId is used to select a specific welding_run and experiment for test and validation.
    """

    def __init__(self, experiment: int, welding_run: int):
        self.experiment = experiment
        self.welding_run = welding_run
    
    def __repr__(self):
        return f"DataSplit({self.experiment=}, {self.welding_run=})"


class ASIMoWDataLoader(BaseDataloader):

    def __init__(self, val_data_ids: list[DataSplitId], test_data_ids: list[DataSplitId], task: str,  cycle_seq_number: int = 1, seed: int = 42, 
                 data_directory_path: str | None = None, window_size: int = 200, window_offset: int = 0, **kwargs):
        dataset_name = "asimow"
        self.val_data_ids = val_data_ids
        self.test_data_ids = test_data_ids
        self.cycle_seq_number = cycle_seq_number
        self.window_size = window_size
        self.window_offset = window_offset
        super().__init__(dataset_name, task, seed, data_directory_path, **kwargs)

    def load_raw_data(self):
        db_file_path = self.data_directory_path + "/processed_asimow_dataset.csv"
        log.info(f"Load data from {db_file_path}")
        return pd.read_csv(db_file_path)


    def preprocessing(self, data: pd.DataFrame):
        log.info(f"Extract Numpy arrays")
        data, labels, experiment, welding_run, t_wn = self.extract_vi_and_ids(data)
        log.info(f"Convert to cycle shape")
        df = self.convert_to_cycle_dataframe(data, labels, experiment, welding_run, t_wn)
        return df
    
    def data_transform(self, data):
        return data

    def split_train_validation_test(self, df: pd.DataFrame):
        """
        Split the data into train, validation and test data
        Therefore val_data_ids(experimtent, welding_run) and test_data_ids(experimtent, welding_run) are used to select the data which is used for validation and test

        """
        validation_condition = [(df.welding_run == id_split.welding_run) & (
            df.experiment == id_split.experiment) for id_split in self.val_data_ids]
        test_condition = [(df.welding_run == id_split.welding_run) & (
            df.experiment == id_split.experiment) for id_split in self.test_data_ids]

        train_df = df[~(np.any(validation_condition, axis=0)
                        | np.any(test_condition, axis=0))]

        val_df = df[np.any(validation_condition, axis=0)]

        test_df = df[np.any(test_condition, axis=0)]

        if self.task == "classification":
            train_data = self.scale_and_return_np(
                train_df[train_df.labels != -1], ds_type="train")
            val_data = self.scale_and_return_np(
                val_df[val_df.labels != -1], ds_type="val")
            test_data = self.scale_and_return_np(
                test_df[test_df.labels != -1], ds_type="test")
        elif self.task == "reconstruction":
            train_data = self.scale_and_return_np(train_df, ds_type="train")
            val_data = self.scale_and_return_np(val_df, ds_type="val")
            test_data = self.scale_and_return_np(test_df, ds_type="test")

        train_ds = self.Dataset(*train_data)
        val_ds = self.Dataset(*val_data)
        test_ds = self.Dataset(*test_data)

        return train_ds, val_ds, test_ds

    def get_dataset(self):
        """
        Function which returns the train, validation and test dataset for the LightningDataModule

        Returns:
            train_ds (PytorchDataset): train dataset
            val_ds (PytorchDataset): validation dataset
            test_ds (PytorchDataset): test dataset
        """
        df = self.load_dataset()
        train_ds, val_ds, test_ds = self.split_train_validation_test(df)

        return train_ds, val_ds, test_ds

    @staticmethod
    def get_sampling_weights(labels: np.ndarray) -> np.ndarray:
        """
        Get sampling weigt for the weighted random sampler for the classification task 

        Args:
            labels (np.ndarray): labels

        Returns:
            np.ndarray: sampling weights for the weighted random sampler (shape: (num_samples,)
        """
        ratio = np.mean(labels == 0)
        sampling_weights = np.zeros_like(labels, dtype=np.float32)
        sampling_weights[labels==0] = 1 - ratio
        sampling_weights[labels==1] = ratio
        return sampling_weights


    def get_data_loader(self, batch_size: int, num_workers: int, pin_memory: bool):
        """
        Returns the dataloader for train, validation and test data
        1. The data is loaded from the pickle files
        2. Then split into train, validation and test data based on the val_data_ids and test_data_ids
        3. The Dataloader is chosen based on the given task

        Args:
            batch_size (int): batch size
            num_workers (int): number of workers
            pin_memory (bool): pin memory
        Returns:
            train_loader (MyClassificationDataset/MyReconstructionDataset): train data loader
            val_loader (MyClassificationDataset/MyReconstructionDataset): validation data loader
            test_loader (MyClassificationDataset/MyReconstructionDataset): test data loader
        """
        train_ds, val_ds, test_ds = self.get_dataset()

        if self.task == "classification":
            train_sampling = self.get_sampling_weights(train_ds.labels)
            sampler = WeightedRandomSampler(train_sampling, len(train_sampling))
        else:
            sampler = None

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=(sampler is None), sampler=sampler, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, val_loader, test_loader
    
    def scale_and_return_np(self, df: pd.DataFrame, ds_type:str="val") -> tuple[np.ndarray, np.ndarray]:
        """
        Uses a standard scaler to scale the data and returns it as a numpy array

        Args:
            df (pd.DataFrame): dataframe with the columns "quality", "t_wn", "experiment", "welding_run", "V" and "I"
            ds_type (str, optional): train, val or test. Defaults to "val".

        Returns:
            tuple[np.ndarray, np.ndarray]: x (num_cycles, max_cycle_size, 2) and y (num_cycles,)
        
        """
        x, y = self.get_numpy_array_from_dataframe(df)
        if self.cycle_seq_number > 1:
            x, y = self.create_sequence_ds(x, y, self.cycle_seq_number)
        else:
            x = x[:, self.window_offset:self.window_offset + self.window_size,:]
        if self.scaler is not None:
            if ds_type == "train":
                self.scaler.fit(x)
            x = self.scaler.transform(x)
        if self.shuffle:
            x, y = shuffle_np(x, y)
       

        return x, y


    def create_sequence_ds(self, x: np.ndarray, y: np.ndarray, seq_len: int):
        """
        Create a dataset with a sequence of cycles.

        Args:
            x (np.ndarray): input data with the shape (num_cycles, max_cycle_size, 2)
            y (np.ndarray): labels with the shape (num_cycles,)
            seq_len (int): sequence length

        Returns:
            tuple[np.ndarray, np.ndarray]: x (num_cycles - seq_len, seq_len * max_cycle_size, 2) and y (num_cycles - seq_len,)
        """
        new_x = np.zeros(
            (x.shape[0] - seq_len, self.window_size * seq_len, x.shape[2]))
        new_y = np.zeros((y.shape[0] - seq_len))
        for i in range(x.shape[0] - seq_len):
            x_t = x[i:i+seq_len]
            x_t = x_t[:, self.window_offset:self.window_offset + self.window_size,:]            
            new_x[i] = x_t.reshape(-1, 2)
            new_y[i] = y[i+seq_len]

        return new_x, new_y

    @staticmethod
    def get_numpy_array_from_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        converts the saved dataset to a numpy array with the shape (num_cycles, max_size, 2)
        where the last dimension is the voltage and current.
        Args:
            df (pd.DataFrame): dataframe with the columns "quality", "t_wn", "experiment", "welding_run", "V" and "I"
        Returns:
            tuple[np.ndarray, np.ndarray]: x (num_cycles, max_cycle_size, 2) and y (num_cycles,)
        """
        v = np.stack(df["V"].values, axis=0)
        v = v.reshape(-1, v.shape[1], 1)
        i = np.stack(df["I"].values)
        i = i.reshape(-1, i.shape[1], 1)

        x = np.concatenate((v, i), axis=2)

        y = df["labels"].to_numpy()
        return x, y

    @staticmethod
    def convert_to_cycle_np_array(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Converts a dataframe to a numpy array with the shape (num_cycles, max_cycle_size, 2),
        where the last dimension is the voltage and current. 

        Args:
            df (pd.DataFrame): dataframe with the columns "labels", "experiment", "welding_run", "V_0",..., "V_199" and "I_0", .. "I_199"
        Returns:
            tuple[np.ndarray, pd.DataFrame]: tuple of a numpy array with voltage and current cyles and a dataframe with the ids "quality", "experiment", "welding_run"

        """

        df_ids = df[["experiment", "welding_run", "labels"]]
        v_cycles = df.iloc[:, 3:203].to_numpy()
        i_cycles = df.iloc[:, 203:].to_numpy()
        
        vi = np.concatenate([v_cycles.reshape(-1, 200, 1), i_cycles.reshape(-1, 200, 1)], axis=2)
        return vi, df_ids

    @staticmethod
    def create_tuple_list(t_wn, experiment, welding_run, labels, vi_np_array: np.ndarray):
        """
        Helper function which is needed to create a dataframe from a numpy arrays in one column
        """
        tuple_list = []
        for i in range(vi_np_array.shape[0]):
            tuple_list.append((t_wn[i], experiment[i], welding_run[i],
                              labels[i], vi_np_array[i, :, 0], vi_np_array[i, :, 1]))
        return tuple_list
    
    def extract_vi_and_ids(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts the voltage and current cycles and their corresponding IDs from a dataframe and returns them as numpy arrays.

        Args:
            df (pd.DataFrame): dataframe with the columns "quality", "t_wn", "experiment", "welding_run", "V" and "I"
            max_size (int, optional): size of one cycle. Defaults to 200.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: current and voltage array, labels, experiment, welding_run, t_wn
        """

        vi_np_array, df_ids = self.convert_to_cycle_np_array(df)
        labels = df_ids.labels.values
        experiment = df_ids.experiment.values
        welding_run = df_ids.welding_run.values
        t_wn = np.arange(0, vi_np_array.shape[0])
        return vi_np_array, labels, experiment, welding_run, t_wn
    

    def convert_to_cycle_dataframe(self, data: np.ndarray, labels: np.ndarray, experiment: np.ndarray, welding_run: np.ndarray, t_wn: np.ndarray) -> pd.DataFrame:
        """
        Converts a dataframe with the function convert_to_cycle_np_array to numpy array with each cycle as a row and the columns: 
            "t_wn", "experiment", "welding_run", "V" and "I"
        and then converts it back into a single dataframe, so that it can be saved as a pickle file.

        """
        tuple_list = self.create_tuple_list(
            t_wn, experiment, welding_run, labels, data
        )
        df = pd.DataFrame(data=tuple_list, columns=[
                          't_wn', 'experiment', 'welding_run', 'labels', 'V', 'I'])
        return df




class ASIMoWDataModule(pl.LightningDataModule):

    def __init__(self, task: str, n_cycles: int, val_data_ids, test_data_ids, batch_size: int = 32, 
                 shuffle_val_test: bool = True, window_size: int = 200, window_offset: int = 0):
        """
        Ligning Data Module for the ASIMoW dataset
        
        Args:
            task (str): classification or reconstruction
            n_cycles (int): number of cycles
            val_data_ids (list[DataSplitId]): list of DataSplitId which are used for validation
            test_data_ids (list[DataSplitId]): list of DataSplitId which are used for testing
            batch_size (int, optional): batch size. Defaults to 32.
            shuffle_val_test (bool, optional): shuffle validation and test data. Defaults to True.
            window_size (int, optional): window size for the moving average. Defaults to 50.
            window_offset (int, optional): window offset for the moving average. Defaults to 10.
        """

        super().__init__()
        self.n_cycles = n_cycles
        self.task = task
        self.val_ids = val_data_ids
        self.test_ids = test_data_ids
        self.batch_size = batch_size
        self.shuffle_val_test = shuffle_val_test
        self.window_size = window_size
        self.window_offset = window_offset

        self.train_sampling = None
        self.val_sampling = None
        self.test_sampling = None
        self.asimow_dataloader: ASIMoWDataLoader | None = None

    def get_sampling_weights(self, labels):
        ratio = np.mean(labels == 0)
        sampling_weights = np.zeros_like(labels, dtype=np.float32)
        sampling_weights[labels==0] = 1 - ratio
        sampling_weights[labels==1] = ratio
        return sampling_weights

    def setup(self, stage: str):
        self.asimow_dataloader = ASIMoWDataLoader(task=self.task, cycle_seq_number=self.n_cycles, val_data_ids=self.val_ids, test_data_ids=self.test_ids, 
            shuffle=self.shuffle_val_test, seed=42, window_size=self.window_size, window_offset=self.window_offset)
        
        train_ds, val_ds, test_ds = self.asimow_dataloader.get_dataset()

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        if self.task == "classification":
            self.train_sampling = self.asimow_dataloader.get_sampling_weights(train_ds.labels)

    def get_sampler(self,):
        if self.task == "classification":
            sampler = WeightedRandomSampler(self.train_sampling, num_samples=len(self.train_sampling), replacement=True)
        else:
            sampler = RandomSampler(self.train_ds)
        return sampler


    def train_dataloader(self):
        sampler = self.get_sampler()
        return DataLoader(self.train_ds, batch_size=self.batch_size, sampler=sampler, num_workers=8, pin_memory=False, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=8, pin_memory=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=8, pin_memory=False, drop_last=True)



def load_npy_data(config: argparse.Namespace, val_ids: list[DataSplitId], test_ids: list[DataSplitId], task: str = "classification") -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """
    Load numpy data from ASIMoW dataset
    First Load the data from the ASIMoW dataloader and convert it back to numpy array
    (This could be a bit slower but has the advantage that we can use the same dataloader)

    Args:
        config (argparse.Namespace): hyperparameters which are needed to load the data
        val_ids (list[DataSplitId]): list of DataSplitId which are used for validation
        test_ids (list[DataSplitId]): list of DataSplitId which are used for testing
        task (str, optional): classification or reconstruction. Defaults to "classification".
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: train_data, train_labels, val_data, val_labels, test_data, test_labels
    """
    print(f"Load data: {config}")



    data_module = ASIMoWDataModule(task=task, batch_size=config.batch_size, n_cycles=config.n_cycles,
                                   val_data_ids=val_ids, test_data_ids=test_ids)

    data_module.setup('fit')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    train_data = train_loader.dataset.data
    val_data = val_loader.dataset.data
    test_data = test_loader.dataset.data

    if task == "classification":
        train_labels = train_loader.dataset.labels
        val_labels = val_loader.dataset.labels
        test_labels = test_loader.dataset.labels
    else:
        train_labels = None
        val_labels = None
        test_labels = None

    return train_data, train_labels, val_data, val_labels, test_data, test_labels
