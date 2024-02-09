import argparse
import logging as log
import os

import torch
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import lightning.pytorch as pl
from lightning import Trainer

from dataloader.asimow_dataloader import DataSplitId
from dataloader.latentspace_dataloader import LatentPredDataModule, get_metadata_and_artifact_dir
from dataloader.utils import get_val_test_ids
from model.transformer_decoder import MyTransformerDecoder 
from utils import get_latent_dataloader, print_training_input_shape, generate_funny_name
from mlflow_helper import MLFlowLogger as MLFlowLoggerHelper
from lightning.pytorch.strategies import DDPStrategy


def get_new_trainer(epochs_steps, logger, n_gpus=1):
    return Trainer(
        devices=n_gpus,
        num_nodes=1,
        max_epochs=epochs_steps,
        logger=logger, 
        callbacks=[],
        gradient_clip_val=0.8,
        strategy = DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else 'auto',
        accumulate_grad_batches=5
    )


def load_dataset(hparams, only_classify=False) -> tuple[int, int, LatentPredDataModule, LatentPredDataModule | None]:
    batch_size = hparams.batch_size
    n_cycles = hparams.n_cycles
    use_wandb = hparams.use_wandb
    vqvae_model = hparams.vqvae_model


    data_dict = get_val_test_ids()

    val_ids = data_dict["val_ids"]
    test_ids = data_dict["test_ids"]

    val_ids = [DataSplitId(experiment=item[0], welding_run=item[1]) for item in val_ids]
    test_ids = [DataSplitId(experiment=item[0], welding_run=item[1]) for item in test_ids]

    if only_classify:
        gen_task_data_module = None 
    else:   
        gen_task_data_module, _ = get_latent_dataloader(use_wandb, n_cycles, vqvae_model, val_ids, test_ids, 
                                                                 batch_size, task="autoregressive_ids")
        print_training_input_shape(gen_task_data_module)


    class_task_data_module, model_config = get_latent_dataloader(use_wandb, n_cycles, vqvae_model, val_ids, test_ids, 
                                                                   batch_size, task="autoregressive_ids_classification")
    
    num_embeddings = model_config["num_embeddings"]
    patch_size = model_config["patch_size"]

    return num_embeddings, patch_size, class_task_data_module, gen_task_data_module


def classification_finetuning(model, classification_epoch, logger, class_task_data_module, no_early_stopping=False, n_gpus=1):
    score = "val/cl/f1_score"
    mode = "max"
    # checkpoint_callback = ModelCheckpoint(dirpath='model_checkpoints/VQ-VAE-transformer/', monitor=score, mode=mode, save_last=True)
    early_stop_callback = EarlyStopping(monitor=score, min_delta=0.001, patience=5, verbose=False, mode=mode)
    model.switch_to_classification()

    if no_early_stopping:
        callbacks = []
    else:
        callbacks = early_stop_callback

    trainer = Trainer(
        devices=n_gpus,
        num_nodes=1,
        max_epochs=classification_epoch,
        logger=logger, 
        callbacks=callbacks,
        gradient_clip_val=0.8,
        strategy = DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else 'auto',
        accumulate_grad_batches=5
    )
    trainer.fit(model, class_task_data_module)
    trainer = Trainer(
        devices=1,
        num_nodes=1,
        logger=logger, 
        callbacks=callbacks,
    )
    trainer.test(model, class_task_data_module)

def main(hparams):
    d_model = hparams.d_model
    n_heads = hparams.n_heads
    n_blocks = hparams.n_blocks
    n_cycles = hparams.n_cycles
    classification_epoch = hparams.class_epoch
    fine_tune_epochs = hparams.finetune_epochs
    classification_only = hparams.classification_only
    no_early_stopping = hparams.no_early_stopping 
    use_wandb = hparams.use_wandb
    use_wandb_for_logging = hparams.use_wandb_for_logging
    epoch_iter = hparams.epoch_iter
    use_class_head_bias = hparams.use_class_head_bias
    use_class_head_dropout = hparams.use_class_head_dropout

    logging_entity = hparams.logging_entity
    logging_project = hparams.logging_project

    use_mlflow = hparams.use_mlflow
    mlflow_url = hparams.mlflow_url 

    use_all_gpus = hparams.use_all_gpus

    if use_wandb or use_wandb_for_logging:
        assert logging_entity is not None, "Wandb entity must be set"
        assert logging_project is not None, "Wandb project must be set"
        logger = WandbLogger(log_model=True, project=logging_project, entity=logging_entity)
    elif use_mlflow:
        assert logging_project is not None, "MLflow project must be set"
        assert mlflow_url is not None, "MLflow URL must be set"
        mlflow_helper = MLFlowLoggerHelper()
        logger = MLFlowLogger(experiment_name=logging_project, run_name=f"{generate_funny_name()}", tracking_uri=mlflow_url, log_model=True)
        print("MLflow logger created")
        print(f"MLflow ID: {logger.run_id}")
        print(f"MLflow URL: {mlflow_url}")
        print(f"MLflow experiment name: {logging_project}")
        logger.log_hyperparams(hparams)

    else:
        logger = CSVLogger("logs", name="vq-vae-transformer")


    num_embeddings, patch_size, class_task_data_module, gen_task_data_module = load_dataset(hparams, only_classify=classification_only)
    
    print_training_input_shape(class_task_data_module)

    seq_len = (n_cycles * (400 // patch_size)) + 1
    num_classes = num_embeddings + 2


    if use_all_gpus:
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1
    log.info(f"{n_gpus=}")
    log.info(f"{seq_len=} - {num_classes=} - {num_embeddings=} - {patch_size=}")

    if classification_only:
        model_name = hparams.model_wandb_transformer

        if model_name == "":
            model = MyTransformerDecoder(
                d_model=d_model, seq_len=seq_len, n_classes=num_classes, n_head=n_heads, n_blocks=n_blocks, class_h_bias=use_class_head_bias, class_h_dropout=use_class_head_dropout)
            
        else:
            artifact_dir = f"./artifacts/{model_name.split('/')[-1]}"
            artifact = wandb.use_artifact(model_name, type='model')
            if not os.path.exists(artifact_dir):
                artifact_dir = artifact.download()

            artifact_dir = artifact_dir + '/model.ckpt'

            model = MyTransformerDecoder.load_from_checkpoint(artifact_dir)
        classification_finetuning(model, classification_epoch, logger, class_task_data_module, no_early_stopping=no_early_stopping)
    
    else:
        model = MyTransformerDecoder(
            d_model=d_model, seq_len=seq_len, n_classes=num_classes, n_head=n_heads, n_blocks=n_blocks)

        for epoch in range(epoch_iter):
            log.info("Genrerating stage")
            trainer = get_new_trainer(epochs_steps=10, logger=logger, n_gpus=n_gpus)
            model.switch_to_generate()
            trainer.fit(model, gen_task_data_module)

            if epoch == epoch_iter - 1:
                classification_finetuning(model, fine_tune_epochs, logger, class_task_data_module, no_early_stopping=no_early_stopping, n_gpus=n_gpus)
            else:
                trainer = get_new_trainer(epochs_steps=classification_epoch, logger=logger, n_gpus=n_gpus)
                log.info("Classification stage")
                model.switch_to_classification()
                trainer.fit(model, class_task_data_module)

        trainer = get_new_trainer(epochs_steps=1, logger=logger, n_gpus=1)
        model.switch_to_classification()
        trainer.test(model, class_task_data_module)

        model.switch_to_generate()
        trainer.test(model, gen_task_data_module)

    if isinstance(logger, CSVLogger):
        pass
    elif isinstance(logger, WandbLogger):
        logger.experiment.finish()
    elif isinstance(logger, MLFlowLogger):
        logger.finalize()
    else: 
        raise ValueError("Invalid logger")
    print("Done")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train-Latent-Transformer')
    parser.add_argument('--epoch_iter', type=int, help='Number of epochs iterations (15 epochs autoregressive train, 2 epochs classification', default=3)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=16)
    parser.add_argument('--n-cycles', type=int, help='Number of cycles', default=20)
    parser.add_argument('--d-model', type=int, help='Number of embeddings', default=512)
    parser.add_argument('--n-heads', type=int, help='Number of heads', default=8)
    parser.add_argument('--n-blocks', type=int, help='Number of transformer blocks', default=6)
    parser.add_argument('--use-class-head-bias', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use-class-head-dropout', action=argparse.BooleanOptionalAction)

    parser.add_argument('--use-wandb', help='Use Weights and Bias (https://wandb.ai/) for Logging & loading the model from wandb', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use-wandb-for-logging', help='Use Weights and Bias (https://wandb.ai/) for Logging', action=argparse.BooleanOptionalAction)
    
    parser.add_argument('--use-mlflow', help='Use MLflow (https://mlflow.org/docs/latest/index.html) for Logging', action=argparse.BooleanOptionalAction)
    parser.add_argument('--mlflow-url', type=str, help='URL of the MLflow server', default='http://mlflow.tmdt.uni-wuppertal.de/')

    parser.add_argument('--logging-entity', type=str, help='Weights and Bias or MLflow entity')
    parser.add_argument('--logging-project', type=str, help='Weights and Bias or MLflow project', default="asimow-vq-vae-transformer")  
    
    parser.add_argument('--vqvae-model', type=str, help='Model URL for wandb or Path', default="model_checkpoints/VQ-VAE-Patch/vq_vae_patch_best_01.ckpt")

    parser.add_argument('--classification-only', action=argparse.BooleanOptionalAction)
    parser.add_argument('--no-early-stopping', action=argparse.BooleanOptionalAction)
    parser.add_argument('--class-epoch', type=int, help='Number of epochs for classification', default=2)
    parser.add_argument('--finetune-epochs', type=int, help='Number of epochs for classification', default=10)
    parser.add_argument('--model-wandb-transformer', type=str, help='Transfomrer Model for classification', default="")
    parser.add_argument('--use-all-gpus', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    log.basicConfig(level=log.INFO, format=FORMAT)

    torch.set_float32_matmul_precision('medium')

    main(args)