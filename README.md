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
To train the model first the VQ-VAE must be trained. 
```bash
python train_reconstruction_embedding.py
```

Then the VQ-VAE-Transformer can be trained.
```bash
python train_transformer_mtasks.py --vqvae-model="path to trained vq model"
```





