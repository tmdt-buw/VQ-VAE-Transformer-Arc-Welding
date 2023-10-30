import logging as log
from dataloader.asimow_dataloader import DataSplitId
from dataloader.latentspace_dataloader import LatentPredDataModule, get_metadata_and_artifact_dir
from model.vq_vae import VectorQuantizedVAE
from model.vq_vae_patch_embedd import VQVAEPatch


def print_training_input_shape(data_module):
    data_module.setup(stage="fit")
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    for i in range(len(batch)):
        log.info(f"Input {i} shape: {batch[i].shape} type: {batch[i].dtype}")


def get_latent_dataloader(use_wandb: bool, n_cycles: int, model_path: str, val_ids:list[DataSplitId], test_ids: list[DataSplitId], batch_size: int, task: str) -> tuple[LatentPredDataModule, dict[str, int]]:

    if use_wandb:
        model_id = model_path.split("-")[-1]
        model_name, model_path = get_metadata_and_artifact_dir(model_path)
    else: 
        split_path = model_path.split("/")
        model_id = split_path[-1]
        model_name = split_path[-2]

    
    model_name = "VQ-VAE" if model_name == "VQ VAE" else model_name

    if model_name == "VQ-VAE" or model_name == "vqvae":
        model = VectorQuantizedVAE.load_from_checkpoint(model_path)
    elif model_name == "VQ-VAE-Patch":
        model = VQVAEPatch.load_from_checkpoint(model_path) 
    else:
        raise ValueError(f"model name: {model_name} not supported")

    
    data_module = LatentPredDataModule(latent_space_model=model, model_name=f"{model_name}", val_data_ids=val_ids, test_data_ids=test_ids,
                                        n_cycles=n_cycles, task=task, batch_size=batch_size, model_id=model_id)
    
    num_embeddings = model.num_embeddings
    patch_size = int(model.patch_size) if hasattr(model, "patch_size") else 25
    latent_dim = num_embeddings * model.enc_out_len
    config = {"num_embeddings": num_embeddings, "patch_size": patch_size, "latent_dim": latent_dim}
    return data_module, config