import argparse
import logging as log
import torch
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Trainer

from dataloader.asimow_dataloader import DataSplitId, ASIMoWDataModule
from dataloader.utils import get_val_test_ids
from model.mlp import MLP
from model.mlp_bc import MLP_BC
from model.gru import GRU
from utils import get_latent_dataloader, print_training_input_shape



def main(hparams):
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
    vqvae_model = hparams.vqvae_model

    use_wandb = hparams.use_wandb
    wandb_entity = hparams.wandb_entity
    wandb_project = hparams.wandb_project

    if use_wandb:
        assert wandb_entity is not None, "Wandb entity must be set"
        assert wandb_project is not None, "Wandb project must be set"
        logger = WandbLogger(log_model=True, project=wandb_project, entity=wandb_entity)
    else:
        logger = CSVLogger("logs", name="vq-vae-transformer")


    data_dict = get_val_test_ids()


    input_dim = 2 if dataset == "asimow" else 1

    val_ids = data_dict["val_ids"]
    test_ids = data_dict["test_ids"]

    if use_wandb:
        logger.experiment.log(
            {"val_ids": str(val_ids), "test_ids": str(test_ids), "artifact_name": vqvae_model, "dataset-name": dataset})
    else:
        logger.log_hyperparams(
            {"val_ids": str(val_ids), "test_ids": str(test_ids), "model_name": model_name, "artifact_name": vqvae_model})

    if dataset == "asimow" or dataset == "latent_vq_vae" or dataset == "latent_vae" \
            or dataset == "latent_vq_vae_out_of_dist" or dataset == "asimow_out_of_dist":

        val_ids = [DataSplitId(experiment=item[0], welding_run=item[1])
                   for item in val_ids]
        test_ids = [DataSplitId(experiment=item[0], welding_run=item[1])
                    for item in test_ids]
        
    if dataset == "asimow" or dataset == "asimow_out_of_dist":
        data_module = ASIMoWDataModule(task="classification", batch_size=batch_size, n_cycles=n_cycles,
                                       val_data_ids=val_ids, test_data_ids=test_ids)
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
    elif dataset == "latent_vq_vae" or dataset == "latent_vae":
        data_module, model_conf = get_latent_dataloader(use_wandb=use_wandb,
            model_path=vqvae_model, batch_size=batch_size, val_ids=val_ids, test_ids=test_ids, 
            n_cycles=n_cycles, task="classification")

        seq_len = n_cycles
        input_dim = model_conf["latent_dim"] 
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
    else:
        raise ValueError("model name not supported")
    

    model = Model(input_size=seq_len, in_dim=input_dim, hidden_sizes=hidden_dim, dropout_p=dropout_p,
                      n_hidden_layers=n_hidden_layers, output_size=output_size, learning_rate=learning_rate)

    checkpoint_callback = ModelCheckpoint(
        dirpath="model_checkpoints", monitor=f"val/f1_score_mean", mode="max", filename=f"{model_name}-{dataset}-best")
    early_stop_callback = EarlyStopping(
        monitor=f"val/f1_score_mean", min_delta=0.001, patience=5, verbose=False, mode="max")

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
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
        logger=logger,
    )

    model = Model.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model=model, dataloaders=data_module)
    test_f1_score = model.test_f1_score
    test_acc = model.test_acc_score

    logdict = {"val/mean_f1_score": best_score, 
                "val/mean_acc": best_acc_score,
                "test/mean_f1_score": test_f1_score,
                "test/mean_acc": test_acc}
    if use_wandb:
        logger.experiment.log(logdict)
        logger.experiment.finish()
    else: 
        logger.experiment.log_metrics(logdict)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Classification Model')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train', default=5)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=128)
    parser.add_argument('--hidden-dim', type=int, help='Hidden dimension', default=1024)
    parser.add_argument('--learning-rate', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--clipping-value', type=float, help='Gradient Clipping', default=0.7)
    parser.add_argument('--dropout-p', type=float, help='Dropout propability', default=0.15)
    parser.add_argument('--n-hidden-layer', type=int, help='Number of hidden layers', default=4)
    parser.add_argument('--model-name', type=str, help='Model name', default="MLP")
    parser.add_argument('--dataset', type=str, help='Dataset', default="latent_vq_vae")
    parser.add_argument('--n-cycles', type=int, help='Number of cycles', default=1)
    parser.add_argument('--use-latent-ids', type=int, help='If the dataset with latentspace IDs should be used', default=0)


    parser.add_argument('--use-wandb', help='Use Weights and Bias (https://wandb.ai/) for Logging', action=argparse.BooleanOptionalAction)
    parser.add_argument('--wandb-entity', type=str, help='Weights and Bias entity')
    parser.add_argument('--wandb-project', type=str, help='Weights and Bias project')

    parser.add_argument('--vqvae-model', type=str, help='Model URL for wandb or Path', default="model_checkpoints/VQ-VAE-Patch/VQ-VAE-Patch-asimow-best-v2.ckpt")
    args = parser.parse_args()

    FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    log.basicConfig(level=log.INFO, format=FORMAT)

    torch.set_float32_matmul_precision('medium')

    main(args)
    wandb.finish()


# python train_transformer_mtasks.py --vqvae-model="model_checkpoints/vq_vae_patch.ckpt" --use-wandb --wandb-entity="tmdt-deep-learning" --wandb-project="asimow-predictive-quality"