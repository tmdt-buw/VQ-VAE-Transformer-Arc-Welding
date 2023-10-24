# VQ-VAE-Transformer-Arc-Welding
TODO: make independent of wandb and add environment.yml

## Environment


## Dataset
The dataset is available at [zenodo](https://zenodo.org/records/10017718). 
Please download the processed dataset and put it in the `data` folder.

## Model
- [VQ-VAE-Transformer](models/vqvae_transformer.py)
- [VQ-VAE-Patch](models/vqvae_patch.py)


## Training
To train the model first the VQ-VAE must be trained. 
```bash
python train_reconstruction_embedding.py
```

Then the VQ-VAE-Transformer can be trained.
```bash
python train_reconstruction_embedding.py --vq-model="path to trained vq model"
```





