import argparse
import logging as log
import os
import torch
import wandb
from dataloader.asimow_dataloader import ASIMoWDataLoader, DataSplitId, ASIMoWDataModule
from dataloader.latentspace_dataloader import LatentPredDataModule, get_metadata_and_artifact_dir
from dataloader.utils import get_val_test_ids
from model.mlp import MLP
from model.mlp_embedding import MLPEmbedding
from model.mlp_bc import MLP_BC
from model.gru import GRU
from model.vq_vae import VectorQuantizedVAE
from model.vq_vae_patch_embedd import VQVAEPatch
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Trainer


def print_training_input_shape(data_module):
    data_module.setup(stage="fit")
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    for i in range(len(batch)):
        log.info(f"Input {i} shape: {batch[i].shape} type: {batch[i].dtype}")

def get_latent_dataloader(wandb_artifact_name: str, dataset_id: int, batch_size: int, val_ids: list[DataSplitId], test_ids: list[DataSplitId], n_cycles: int, use_ids: bool=False):
    model_name, model_path = get_metadata_and_artifact_dir(wandb_artifact_name)
    latent_dim = 2
    model_name = "VQ-VAE" if model_name == "VQ VAE" else model_name

    if model_name == "VQ-VAE" or model_name == "vqvae":
        model = VectorQuantizedVAE.load_from_checkpoint(model_path)
        latent_dim = model.embedding_dim * model.enc_out_len
    elif model_name == "VQ-VAE-Patch":
        model = VQVAEPatch.load_from_checkpoint(model_path) 
        latent_dim = model.embedding_dim * model.enc_out_len
    else:
        raise ValueError(f"model name: {model_name} not supported")
    if use_ids:
        latent_dim = model.enc_out_len
    else:
        latent_dim = model.embedding_dim * model.enc_out_len
        
    wandb_model_name = wandb_artifact_name.split("-")[-1]
    task = "classification_ids" if use_ids else "classification"
    data_module = LatentPredDataModule(latent_space_model=model, model_name=f"{model_name}", val_data_ids=val_ids, test_data_ids=test_ids,
                                       n_cycles=n_cycles, task=task, batch_size=batch_size, dataset_id=dataset_id, wandb_model_name=wandb_model_name)
    return data_module, latent_dim


def main(hparams):

    wandb_logger = WandbLogger(
        log_model=True, project="asimow-predictive-quality", entity="tmdt-deep-learning")

    # read hyperparameters
    hidden_dim = hparams.hidden_dim
    learning_rate = hparams.learning_rate
    epochs = hparams.epochs
    clipping_value = hparams.clipping_value
    batch_size = hparams.batch_size
    dropout_p = hparams.dropout_p
    n_hidden_layers = hparams.n_hidden_layer
    n_cycles = hparams.n_cycles
    dataset = hparams.dataset
    model_name = hparams.model_name
    use_ids = bool(hparams.use_latent_ids)
    classification_model = model_name.split("-")[0]
    wandb_artifact_name = hparams.wandb_artifact_name
    use_augmentation = bool(hparams.use_augmentation)


    data_dict = get_val_test_ids()



    input_dim = 2 if dataset == "asimow" else 1
    dataset_id = 0


    val_ids = data_dict["val_ids"]
    test_ids = data_dict["test_ids"]

    wandb_logger.experiment.log(
        {"val_ids": str(val_ids), "test_ids": str(test_ids)})

    if dataset == "asimow" or dataset == "latent_vq_vae" or dataset == "latent_vae" \
            or dataset == "latent_vq_vae_out_of_dist" or dataset == "asimow_out_of_dist":
        wandb_logger.experiment.log({"val_ids": str(val_ids), "test_ids": str(
            test_ids), "artifact_name": wandb_artifact_name, "dataset-name": dataset})
        val_ids = [DataSplitId(experiment=item[0], welding_run=item[1])
                   for item in val_ids]
        test_ids = [DataSplitId(experiment=item[0], welding_run=item[1])
                    for item in test_ids]
        
    if dataset == "asimow" or dataset == "asimow_out_of_dist":
        data_module = ASIMoWDataModule(task="classification", batch_size=batch_size, n_cycles=n_cycles,
                                       model_id=dataset_id, val_data_ids=val_ids, test_data_ids=test_ids, use_augmentation=use_augmentation)
        if classification_model == "MLP":
            seq_len = 200 * n_cycles
            input_dim = 2
        elif classification_model == "MLP_BC":
            seq_len = 200 * n_cycles
            input_dim = 2
        elif classification_model == "GRU":
            seq_len = n_cycles
            input_dim = 200*2
        else:
            raise ValueError(f"Classification model name: {classification_model} not supported")
    elif dataset == "latent_vq_vae" or dataset == "latent_vae" or dataset == "latent_vq_vae_out_of_dist":
        data_module, latent_dim = get_latent_dataloader(
            wandb_artifact_name=wandb_artifact_name, dataset_id=dataset_id, batch_size=batch_size, val_ids=val_ids, test_ids=test_ids, 
            n_cycles=n_cycles, use_ids=use_ids)

        seq_len = n_cycles
        input_dim = latent_dim 
    else:
        raise ValueError(f"Invalid dataset name. {dataset} not supported")

    print_training_input_shape(data_module)
   
    if classification_model == "MLP":
        Model = MLP
        output_size = 2
    elif classification_model == "MLP_BC":
        Model = MLP_BC
        output_size = 1
    elif classification_model == "GRU":
        Model = GRU
        output_size = 2
    elif classification_model == "MLP_E":
        Model = MLPEmbedding
        output_size = 2
    else:
        raise ValueError("model name not supported")
    

    model = Model(input_size=seq_len, in_dim=input_dim, hidden_sizes=hidden_dim, dropout_p=dropout_p,
                      n_hidden_layers=n_hidden_layers, output_size=output_size, learning_rate=learning_rate, model_id=str(dataset_id))

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints", monitor=f"{dataset_id}/val/f1_score_mean", mode="max", filename=f"{model_name}-{dataset}-best")
    early_stop_callback = EarlyStopping(
        monitor=f"{dataset_id}/val/f1_score_mean", min_delta=0.001, patience=5, verbose=False, mode="max")

    trainer = Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        # callbacks=[],
        devices=1,
        num_nodes=1,
        gradient_clip_val=clipping_value,
        check_val_every_n_epoch=1
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
    )

    best_score = model.hyper_search_value
    best_acc_score = model.val_acc_score
    print(f"best score: {best_score}")
    print("------ Testing ------")

    trainer = Trainer(
        devices=1,
        num_nodes=1,
        logger=wandb_logger,
    )

    model = Model.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model=model, dataloaders=data_module)
    test_f1_score = model.test_f1_score
    test_acc = model.test_acc_score

    wandb_logger.experiment.log({"cross_val_f1_score": best_score, 
                                 "cross_val_acc": best_acc_score,
                                 "cross_test_f1_score": test_f1_score,
                                 "cross_test_acc": test_acc})

    wandb_logger.experiment.finish()


if __name__ == '__main__':
    wandb.require(experiment="service")
    os.environ["WANDB_DISABLE_SERVICE"] = "true"

    # define model
    model_wandb = "tmdt-deep-learning/asimow-predictive-quality/model-2cbfyxpf:v0"

    parser = argparse.ArgumentParser(description='Train Classification Model')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train', default=15)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=128)
    parser.add_argument('--hidden-dim', type=int, help='Hidden dimension', default=1024)
    parser.add_argument('--learning-rate', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--clipping-value', type=float, help='Gradient Clipping', default=0.7)
    parser.add_argument('--dropout-p', type=float, help='Dropout propability', default=0.15)
    parser.add_argument('--n-hidden-layer', type=int, help='Number of hidden layers', default=4)
    parser.add_argument('--model-name', type=str, help='Model name', default="GRU-VQ-VAE-Patch")
    parser.add_argument('--dataset', type=str, help='Dataset', default="latent_vq_vae")
    parser.add_argument('--n-cycles', type=int, help='Number of cycles', default=20)
    parser.add_argument('--use-latent-ids', type=int, help='If the dataset with latentspace IDs should be used', default=0)
    parser.add_argument('--wandb-artifact-name', type=str, help='Weights and Bias artifact name', default=model_wandb)
    args = parser.parse_args()

    FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    log.basicConfig(level=log.INFO, format=FORMAT)

    torch.set_float32_matmul_precision('medium')

    main(args)
    wandb.finish()
