import argparse
import logging as log
import pandas as pd
import numpy as np
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, RandomSampler
from data_loader.base_dataloader import BaseDataloader
from data_loader.utils import load_pickle_file, shuffle_np, shuffle_and_undersample
from data_loader.augmentation import augment_ts

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

    def __init__(self, val_data_ids: list[DataSplitId], test_data_ids: list[DataSplitId], task: str,  cycle_seq_number: int = 1, use_augmentation: bool = False, seed: int = 42, 
                 data_directory_path: str | None = None, window_size: int = 200, window_offset: int = 0, **kwargs):
        dataset_name = "asimow-augmented" if use_augmentation else "asimow"
        self.val_data_ids = val_data_ids
        self.test_data_ids = test_data_ids
        self.cycle_seq_number = cycle_seq_number
        self.use_augmentation = use_augmentation
        self.window_size = window_size
        self.window_offset = window_offset
        super().__init__(dataset_name, task, seed, data_directory_path, **kwargs)

    def load_raw_data(self):
        db_file_path = self.data_directory_path + "/db_dataframe/"
        log.info(f"Load data from {db_file_path}")
        return load_pickle_file(data_path=db_file_path, file_name="cycles.pickle")
    
    def augment_asimow_data(self, data: np.ndarray, labels: np.ndarray, experiment: np.ndarray, welding_run: np.ndarray, t_wn: np.ndarray):
        """
        Augment the data with 8 different methods from the augmentation.py file.
        The rest of the data must be repeated 9 times (1 original + 8 augmentations) to match the augmented data.

        Args:
            data (np.ndarray): data
            labels (np.ndarray): labels
            experiment (np.ndarray): experiment
            welding_run (np.ndarray): welding_run
            t_wn (np.ndarray): t_wn

        Returns:
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray: augmented data and repeated ids
        """
        data, labels = augment_ts(data, y=labels)
        experiment = np.tile(experiment, 9)
        welding_run = np.tile(welding_run, 9)
        t_wn = np.tile(t_wn, 9)
        return data, labels, experiment, welding_run, t_wn


    def preprocessing(self, data: pd.DataFrame):
        log.info(f"Extend Labels - this could take a while")
        data = self.extend_labels_and_return_quality_data(data)
        log.info(f"Extract Numpy arrays")
        data, labels, experiment, welding_run, t_wn = self.extract_vi_and_ids(data)
        if self.use_augmentation:
            log.info(f"Augment data")
            data, labels, experiment, welding_run, t_wn = self.augment_asimow_data(data, labels, experiment, welding_run, t_wn)
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
    def get_sampling_weights(labels):
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

    def scale_shuffle_undersample_return_np(self, df: pd.DataFrame, ds_type: str = "train") -> tuple[np.ndarray, np.ndarray]:
        """
        @deprecated
        undersampling is done in the dataloader with the torch sampler
        """
        x, y = self.get_numpy_array_from_dataframe(df)
        if self.cycle_seq_number > 1:
            x, y = self.create_sequence_ds(x, y, self.cycle_seq_number)
        if ds_type == "train" and self.undersample:
            x, y = shuffle_and_undersample(x, y)
        else:
            y = y.reshape(-1)
            x, y = shuffle_np(x, y)
        if self.scaler is not None:
            if ds_type == "train":
                self.scaler.fit(x)
            x = self.scaler.transform(x)
        x = x[:, self.window_offset:self.window_offset + self.window_size,:]
        return x, y
    
    def scale_and_return_np(self, df: pd.DataFrame, ds_type:str="val") -> tuple[np.ndarray, np.ndarray]:
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

        # if self.use_augmentation and ds_type == "train":
        #     log.info("Augmenting data")
        #     x, y = augment_ts(x, y, seq_len=self.cycle_seq_number * 200)
       

        return x, y


    def create_sequence_ds(self, x: np.ndarray, y: np.ndarray, seq_len: int):
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
    def convert_to_cycle_np_array(df: pd.DataFrame, max_size: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts a dataframe to a numpy array with the shape (num_cycles, max_cycle_size, 2),
        where the last dimension is the voltage and current. 

        Args:
            df (pd.DataFrame): dataframe with the columns "quality", "t_wn", "experiment", "welding_run", "V" and "I"
            max_size (int, optional): max size of the cycles. Defaults to 200.
        Returns:
            tuple[np.ndarray, np.ndarray]: tuple of a numpy array with voltage and current cyles and a numpy array with the labels 

        """
        quality = pd.DataFrame(
            df[["quality", "t_wn", "experiment", "welding_run"]])
        quality = pd.DataFrame(quality.iloc[1:])
        quality = pd.DataFrame(quality.iloc[::4])
        quality = pd.DataFrame(quality.groupby("t_wn").first().values)

        new_cycles_v = df["V"].rolling(
            4, step=4).mean().reset_index(drop=True).dropna()
        new_cycles_i = df["I"].rolling(
            4, step=4).mean().reset_index(drop=True).dropna()

        t_wn = df.t_wn.values[::4]
        t_wn = t_wn[1:]
        df = pd.DataFrame(data=np.array(
            [t_wn, new_cycles_v.values, new_cycles_i.values]).T, columns=["t_wn", "V", "I"])
        df["t_wi"] = df.groupby("t_wn").cumcount()
        v_cycles = df.pivot(index="t_wn", columns='t_wi', values="V")
        i_cycles = df.pivot(index="t_wn", columns="t_wi", values="I")

        v_cycles = v_cycles.fillna(method="ffill", axis=1).values
        i_cycles = i_cycles.fillna(method="ffill", axis=1).values

        v_cycles = v_cycles[:, :max_size].reshape(-1, 1, max_size)
        i_cycles = i_cycles[:, :max_size].reshape(-1, 1, max_size)

        vi = np.concatenate([v_cycles, i_cycles], axis=1)
        return vi.swapaxes(1, 2), quality

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
    
    def extract_vi_and_ids(self, df: pd.DataFrame, max_size: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts the voltage and current cycles and their corresponding IDs from a dataframe and returns them as numpy arrays.

        Args:
            df (pd.DataFrame): dataframe with the columns "quality", "t_wn", "experiment", "welding_run", "V" and "I"
            max_size (int, optional): size of one cycle. Defaults to 200.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: current and voltage array, labels, experiment, welding_run, t_wn
        """

        vi_np_array, df_ids = self.convert_to_cycle_np_array(df, max_size)
        labels = df_ids.iloc[:, 0]
        experiment = df_ids.iloc[:, 1]
        welding_run = df_ids.iloc[:, 2]
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

    @staticmethod
    def extend_labels_and_return_quality_data(df: pd.DataFrame, cycle_before_quality_marker: float = 0.15):
        """
        Extend the quality labels up to the beginning of the quality marker before.
        :param df: Dataframe with quality labels
        :param cycle_before_quality_marker: How many cycles before the quality marker should the quality label be extended
        :return: Dataframe with extended quality labels only consisting of data with quality labels
        """
        for experiment_i in [1, 2, 3]:
            df_exp_i = df[df.experiment == experiment_i]
            for w_i in tqdm(df_exp_i.welding_run.unique(), desc=f"Experiment {experiment_i}"):
                df_w_i = pd.DataFrame(df_exp_i[df_exp_i.welding_run == w_i])
                cycle_max = df_w_i.cycle.max()
                new_cycle = int(cycle_max * cycle_before_quality_marker)
                # print(f"{w_i=}, {cycle_max=}, {new_cycle=}")

                for quality_marker in df_w_i.quality_marker.unique():
                    if quality_marker != -1:
                        # print("quality_marker",quality_marker)
                        if quality_marker > 1:
                            phase_before = df_w_i[df_w_i.quality_marker == (
                                quality_marker - 1)]["cycle"].max()
                        else:
                            phase_before = 0

                        phase_1_w = df_w_i[df_w_i.quality_marker ==
                                           quality_marker]["cycle"].min()
                        # print(f"{phase_1_w=}")

                        quality = df_w_i[df_w_i.cycle ==
                                         phase_1_w]["quality"][0]
                        # print("phase end before", phase_before, "--- new phase start" ,phase_1_w - new_cycle)
                        new_phase_beginning = max(
                            phase_before, phase_1_w - new_cycle)
                        # print(new_phase_beginning)
                        # print(quality_marker, phase_1_w, new_phase_beginning)
                        df_w_i.loc[(new_phase_beginning < df_w_i.cycle) & (
                            df_w_i.cycle < phase_1_w), "quality"] = quality
                        df_w_i.loc[(new_phase_beginning < df_w_i.cycle) & (
                            df_w_i.cycle < phase_1_w), "quality_marker"] = quality_marker
                        phase_1_w = df_w_i[df_w_i.quality_marker ==
                                           quality_marker]["cycle"].min()
                        # print(quality_marker, phase_1_w)
                df_exp_i.loc[df_exp_i.welding_run == w_i, :] = df_w_i
            df.loc[df.experiment == experiment_i, :] = df_exp_i
        # return df[df.quality != -1]
        return df
    



class ASIMoWDataModule(pl.LightningDataModule):

    def __init__(self, task: str, n_cycles: int, val_data_ids, test_data_ids, batch_size: int = 32, 
                 model_id: int | None = None, multi_recon: bool = False, use_augmentation: bool = False, shuffle_val_test: bool = True,
                 window_size: int = 200, window_offset: int = 0):
        """
        Ligning Data Module for the ASIMoW dataset
        
        Args:
            task (str): classification or reconstruction
            n_cycles (int): number of cycles
            val_data_ids (list[DataSplitId]): list of DataSplitId which are used for validation
            test_data_ids (list[DataSplitId]): list of DataSplitId which are used for testing
            batch_size (int, optional): batch size. Defaults to 32.
            model_id (int | None, optional): model id. Defaults to None.
            multi_recon (bool, optional): multi reconstruction. Defaults to False.
            use_augmentation (bool, optional): use augmentation. Defaults to False.
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
        self.model_id = model_id
        self.multi_recon = multi_recon
        self.use_augmentation = use_augmentation
        self.shuffle_val_test = shuffle_val_test
        self.window_size = window_size
        self.window_offset = window_offset

        self.train_sampling = None
        self.val_sampling = None
        self.test_sampling = None
        self.asimow_dataloader = None

    def get_sampling_weights(self, labels):
        ratio = np.mean(labels == 0)
        sampling_weights = np.zeros_like(labels, dtype=np.float32)
        sampling_weights[labels==0] = 1 - ratio
        sampling_weights[labels==1] = ratio
        return sampling_weights

    def setup(self, stage: str):
        self.asimow_dataloader = ASIMoWDataLoader(task=self.task, cycle_seq_number=self.n_cycles, val_data_ids=self.val_ids, test_data_ids=self.test_ids, 
            use_augmentation=self.use_augmentation, shuffle=self.shuffle_val_test, seed=42, window_size=self.window_size, 
            window_offset=self.window_offset)
        
        train_ds, val_ds, test_ds = self.asimow_dataloader.get_dataset()

        if self.multi_recon:
            train_ds = convert_ds(train_ds)
            val_ds = convert_ds(val_ds)
            test_ds = convert_ds(test_ds)

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



def load_npy_data(config: argparse.Namespace, val_ids: list[DataSplitId], test_ids: list[DataSplitId], task: str = "classification") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                                   model_id=0, val_data_ids=val_ids, test_data_ids=test_ids,
                                   use_augmentation=bool(config.use_augmentation))

    data_module.setup('fit')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    train_data = train_loader.dataset.data
    val_data = val_loader.dataset.data
    test_data = test_loader.dataset.data

    train_labels = train_loader.dataset.labels
    val_labels = val_loader.dataset.labels
    test_labels = test_loader.dataset.labels

    return train_data, train_labels, val_data, val_labels, test_data, test_labels
