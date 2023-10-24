import argparse
import logging as log
import os

import torch
import wandb
from data_loader.asimow_dataloader import DataSplitId
from data_loader.latentspace_dataloader import LatentPredDataModule, get_metadata_and_artifact_dir

from model.vq_vae import VectorQuantizedVAE
from model.vq_vae_patch_embedd import VQVAEPatch
from model.transformer_decoder import MyTransformerDecoder 

from select_val_test_set import get_val_test_ids, get_out_of_dist_dataset

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import lightning.pytorch as pl
from lightning import Trainer


def print_training_input_shape(data_module):
    data_module.setup(stage="fit")
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    for i in range(len(batch)):
        log.info(f"Input {i} shape: {batch[i].shape} type: {batch[i].dtype} max: {torch.max(batch[i])} min: {torch.min(batch[i])}")

def get_latent_dataloader(n_cycles, model_wandb, val_ids, test_ids, batch_size, dataset_id, task):
    model_name, model_path = get_metadata_and_artifact_dir(model_wandb)
    model_name = "VQ-VAE" if model_name == "VQ VAE" else model_name

    if model_name == "VQ-VAE" or model_name == "vqvae":
        model = VectorQuantizedVAE.load_from_checkpoint(model_path)
        latent_dim = model.embedding_dim * model.enc_out_len
    elif model_name == "VQ-VAE-Patch":
        model = VQVAEPatch.load_from_checkpoint(model_path) 
        latent_dim = model.embedding_dim * model.enc_out_len
    else:
        raise ValueError(f"model name: {model_name} not supported")

    wandb_model_name = model_wandb.split("-")[-1]
    
    data_module = LatentPredDataModule(latent_space_model=model, model_name=f"{model_name}", val_data_ids=val_ids, test_data_ids=test_ids,
                                        n_cycles=n_cycles, task=task, batch_size=batch_size, dataset_id=dataset_id, wandb_model_name=wandb_model_name)
    
    num_embeddings = model.num_embeddings
    patch_size = int(model.patch_size) if hasattr(model, "patch_size") else 25
    return data_module, num_embeddings, patch_size

def get_new_trainer(epochs_steps, wandb_logger, gpus=1):
    return Trainer(
        devices=gpus,
        num_nodes=1,
        max_epochs=epochs_steps,
        logger=wandb_logger, 
        callbacks=[],
        gradient_clip_val=0.8,
    )


def load_dataset(hparams, only_classify=False) -> tuple[int, int, LatentPredDataModule, LatentPredDataModule]:
    dataset_id = 0
    batch_size = hparams.batch_size
    n_cycles = hparams.n_cycles
    model_wandb = hparams.model_wandb


    data_dict = get_val_test_ids()

    val_ids = data_dict["val_ids"]
    test_ids = data_dict["test_ids"]

    val_ids = [DataSplitId(experiment=item[0], welding_run=item[1]) for item in val_ids]
    test_ids = [DataSplitId(experiment=item[0], welding_run=item[1]) for item in test_ids]

    if only_classify:
        gen_task_data_module = None 
    else:   
        gen_task_data_module, num_embeddings, patch_size = get_latent_dataloader(n_cycles, model_wandb, val_ids, test_ids, 
                                                                 batch_size, dataset_id, task="autoregressive_ids")
        print_training_input_shape(gen_task_data_module)

    class_task_data_module, num_embeddings,patch_size = get_latent_dataloader(n_cycles, model_wandb, val_ids, test_ids, 
                                                                   batch_size, dataset_id, task="autoregressive_ids_classification")
    
    
    return num_embeddings, patch_size, class_task_data_module, gen_task_data_module


def classification_finetuning(model, classification_epoch, wandb_logger, class_task_data_module, no_early_stopping=False):
    checkpoint_callback = ModelCheckpoint(dirpath='model_checkpoints/', monitor="val/cl/loss", mode="min")
    early_stop_callback = EarlyStopping(monitor="val/cl/loss", min_delta=0.001, patience=10, verbose=False, mode="min")
    model.switch_to_classification()
    

    if no_early_stopping:
        early_stop_callbacks = []
    else:
        early_stop_callbacks = [checkpoint_callback, early_stop_callback]

    trainer = Trainer(
        devices=1,
        num_nodes=1,
        max_epochs=classification_epoch,
        logger=wandb_logger, 
        callbacks=early_stop_callbacks,
        gradient_clip_val=0.8,
    )
    trainer.fit(model, class_task_data_module)

    trainer.test(model, class_task_data_module)

def main(hparams):
    wandb_logger = WandbLogger(log_model=True, project="asimow-predictive-quality", entity="tmdt-deep-learning")
    d_model = hparams.d_model
    n_heads = hparams.n_heads
    n_blocks = hparams.n_blocks
    n_cycles = hparams.n_cycles
    classification_epoch = hparams.class_epoch
    fine_tune_epochs = hparams.finetune_epochs
    classification_only = hparams.classification_only
    no_early_stopping = hparams.no_early_stopping 

    dataset = "asimow_latentspace"

    num_embeddings, patch_size, class_task_data_module, gen_task_data_module = load_dataset(hparams, only_classify=classification_only)
    
    print_training_input_shape(class_task_data_module)

    seq_len = (n_cycles * (400 // patch_size)) + 1
    num_classes = num_embeddings + 2
    
    log.info(f"{seq_len=} - {num_classes=} - {num_embeddings=} - {patch_size=}")

    if classification_only:
        model_name = hparams.model_wandb_transformer

        if model_name == "":
            model = MyTransformerDecoder(
                d_model=d_model, seq_len=seq_len, n_classes=num_classes, n_head=n_heads, n_blocks=n_blocks)
        else:
            artifact_dir = f"./artifacts/{model_name.split('/')[-1]}"
            artifact = wandb.use_artifact(model_name, type='model')
            if not os.path.exists(artifact_dir):
                artifact_dir = artifact.download()

            artifact_dir = artifact_dir + '/model.ckpt'

            model = MyTransformerDecoder.load_from_checkpoint(artifact_dir)
        classification_finetuning(model, classification_epoch, wandb_logger, class_task_data_module, no_early_stopping=no_early_stopping)
    
    else:
        model = MyTransformerDecoder(
            d_model=d_model, seq_len=seq_len, n_classes=num_classes, n_head=n_heads, n_blocks=n_blocks)

        epoch_iter = 3
        for epoch in range(epoch_iter):
            log.info("Genrerating stage")
            trainer = get_new_trainer(epochs_steps=10, wandb_logger=wandb_logger)
            model.switch_to_generate()
            trainer.fit(model, gen_task_data_module)

            if epoch == epoch_iter - 1:
                classification_finetuning(model, fine_tune_epochs, wandb_logger, class_task_data_module, no_early_stopping=no_early_stopping)
            else:
                trainer = get_new_trainer(epochs_steps=classification_epoch, wandb_logger=wandb_logger)
                log.info("Classification stage")
                model.switch_to_classification()
                trainer.fit(model, class_task_data_module)

        trainer = get_new_trainer(epochs_steps=1, wandb_logger=wandb_logger, gpus=1)
        model.switch_to_classification()
        trainer.test(model, class_task_data_module)

        model.switch_to_generate()
        trainer.test(model, gen_task_data_module)

    wandb_logger.experiment.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train-Latent-Transformer')
    parser.add_argument('--epoch_iter', type=int, help='Number of epochs iterations (15 epochs autoregressive train, 2 epochs classification', default=3)
    parser.add_argument('--dataset', type=str, help='Dataset', default="asimow")    
    parser.add_argument('--batch-size', type=int, help='Batch size', default=128)
    parser.add_argument('--n-cycles', type=int, help='Number of cycles', default=20)
    parser.add_argument('--d-model', type=int, help='Number of embeddings', default=512)
    parser.add_argument('--n-heads', type=int, help='Number of heads', default=8)
    parser.add_argument('--n-blocks', type=int, help='Number of transformer blocks', default=12)
    parser.add_argument('--model-wandb', type=str, help='Model for Discretisation', default="tmdt-deep-learning/asimow-predictive-quality/model-ye7ybqc2:v0")
    parser.add_argument('--classification-only', action=argparse.BooleanOptionalAction)
    parser.add_argument('--no-early-stopping', action=argparse.BooleanOptionalAction)
    parser.add_argument('--class-epoch', type=int, help='Number of epochs for classification', default=2)
    parser.add_argument('--finetune-epochs', type=int, help='Number of epochs for classification', default=25)
    parser.add_argument('--model-wandb-transformer', type=str, help='Transfomrer Model for classification', default="")
    args = parser.parse_args()

    FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    log.basicConfig(level=log.INFO, format=FORMAT)

    torch.set_float32_matmul_precision('medium')

    main(args)