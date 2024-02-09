# VQ-VAE-Transformer-Arc-Welding

## Environment
We recommand using the [devcontainer](.devcontainer) to run the code.

Otherwise, the packages can be installed using conda and the environment [file](.devcontainer/environment.yml) with the following command:
```bash
conda env create -n vqvae-transformer python=3.11 -f .devcontainer/environment.yml
conda activate vqvae-transformer

``` 


## Dataset
The dataset is available at [zenodo](https://zenodo.org/records/10017718). 
Please download the processed dataset and put it in the `data` folder.

## Model
- [VQ-VAE-Transformer](model/transformer_decoder.py)
- [VQ-VAE-Patch](model/vq_vae_patch_embedd.py)


## Training

### MLP
To train the MLP model, the following command can be used:
```bash
python train_classification_model.py --model-name="MLP"
```

### GRU
To train the GRU model, the following command can be used:
```bash
python train_classification_model.py --model-name="GRU"
```

### VQ-VAE
To train the model first the VQ-VAE must be trained. This model is trained on the reconstruction task.
```bash
python train_reconstruction_embedding.py
```

### VQ-VAE-MLP
Then the VQ-VAE-MLP can be trained. Therefore saved weights from a before trained VQ-VAE must be provided.
```bash
python train_classification_model.py --vqvae-model="model_checkpoints/VQ-VAE-Patch/vq_vae_patch_best_01.ckpt" --model-name="MLP" --dataset="latent_vq_vae"
```


### VQ-VAE-Transformer
Then the VQ-VAE-Transformer can be trained. Therefore saved weights from a before trained VQ-VAE must be provided.
```bash
python train_transformer_mtasks.py --vqvae-model="model_checkpoints/VQ-VAE-Patch/vq_vae_patch_best_01.ckpt" --finetune-epochs=10 --n-blocks=8 --n-heads=8 --epoch_iter=3
```


### MLflow
For logging in MLflow, the following the environment variable in [MLflowHelper](mlflow_helper.py) must be set:

````
# init

self.MLFLOW_SERVER_URL = "MLFLOW_SERVER_URL"
# mlflow credentials
self._user = "MLFLOW_USER"
self._password= "MLFLOW_PASSWORD"

# s3 endpoint for artifacts
self._s3_endpoint = "URL_S3_ENDPOINT"
self._aws_access_key_id = "minio"
self._aws_secret_access_key = "_aws_secret_access_key"
self._bucket_name = "_bucket_name"
```

