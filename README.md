# info-NAS
Info-NAS is a model for neural architecture embedding that uses input/output data
from networks as a labeled dataset. The model is a graph VAE combined with a regressor
that takes images as the input. The base VAE model is [arch2vec](https://github.com/MSU-MLSys-Lab/arch2vec).

The model is a VAE combined with a convolutional neural network regressor, and it is trained in a semi-supervised manner:
- unlabeled batches are the neural architectures
- labeled batches are input pairs (neural architecture, input image) and the output is a feature vector

Unlabeled batches are passed directly to the VAE; labeled batches also encode the neural architecture,
but they also predict the output features given the input image.
## Installing info-NAS
First, clone info-NAS:
```
git clone https://github.com/gabrielasuchopar/info-nas
```

You also need to clone some other repositories that info-NAS uses into the same repository:

```
git clone https://github.com/gabrielasuchopar/arch2vec
git clone https://github.com/gabrielasuchopar/NASBench-PyTorch.git
git clone https://github.com/gabrielasuchopar/nasbench
```

The model is written in `PyTorch`, but `TensorFlow >= 2.0` is needed for the NAS-Bench-101 dataset.
An installation script is provided for convenience:
```
bash setup_venv.sh
. ./pyt/bin/activate
```

## Download data
Download the data folder into the info-NAS directory from the following link:

https://drive.google.com/drive/folders/1buJKj4omQlAjVuh4lnMRPAQV8iUUoy31?usp=sharing

The data should be stored in `PATH-TO-INFO-NAS-ROOT/data/`

## Run training
To run the info-NAS model, execute the following command:
```
# device=cuda for gpu (default)
# device=cpu for cpu

cd ./scripts/
python train_vae.py --model_cfg ../configs/model_config.json --epochs 30 \
  --seed 1 --device $device
```
The trained checkpoints are saved to `./data/vae_checkpoints/` by default.
If you want to train the original arch2vec model alongside info-NAS, add the following option:
```
python train_vae.py --model_cfg ../configs/model_config.json --use_ref
```

The config file `./configs/model_config.json` contains training and model parameters, try to experiment with some
settings.

## Run REINFORCE search
The feature extraction and run script is located in the arch2vec repository.
The following script extracts features using a trained model and runs the
REINFORCE search 3 times:
```
# PATH_TO_CHECKPOINT=./data/vae_checkpoints/2021-.../model_orig_epoch-9.pt

cd ../arch2vec/run_scripts
bash search_run_for_checkpoint.sh $PATH_TO_CHECKPOINT.PT reinforce 3
```

The `model_orig_*.pt` checkpoints are info-NAS models without the regressor part, while 
`model_ref_*.pt` is the original arch2vec.

The features are extracted into the same directory: 
`CHCKPT_DIR/features_model_*_epoch-9.pt`.

### Performance prediction
You can train and evaluate a
performance predictor on the features, e.g. a random forest:

```
cd ../info-nas/scripts/
python run_performance_prediction.py features_model_orig_epoch-9.pt \
  --dir_name $CHCKPT_DIR --n_hashes 250 --regr_name rf --seed 1
```