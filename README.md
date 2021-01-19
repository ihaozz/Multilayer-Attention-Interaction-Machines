# Multilayer Attention Interaction Machines

This is an implementation for the paper:
`MAIM: Multilayer Attention Interaction Machines for CTR prediction`.

## Environments

+ Tensorflow (version: 1.14.0)
+ Python 3.6
+ CUDA 10.1 (For GPU)

## Input Format
* train_x: matrix with shape *(num_sample, num_field)*. train_x[s][t] is the feature value of feature field t of sample s in the dataset. The default value for categorical feature is 1.
* train_i: matrix with shape *(num_sample, num_field)*. train_i[s][t] is the feature index of feature field t of sample s in the dataset. The maximal value of train_i is the feature size.
* train_y: label of each sample in the dataset.

Please refer to `data/Dataprocess/Criteo/preprocess.py` for how to preprocess the data.

## Dataset
[Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge), 
[Avazu](https://www.kaggle.com/c/avazu-ctr-prediction), 
[KDD2012](https://www.kaggle.com/c/kddcup2012-track2)

## Usage
### Run the preprocessing.
```
cd data
mkdir Criteo
python3 ./Dataprocess/Criteo/preprocess.py
python3 ./Dataprocess/Kfold_split/stratifiedKfold.py
python3 ./Dataprocess/Criteo/scale.py
```

### Run the training.
```
python3 -m MAIM.train \
        --data_path data --data Criteo \
        --blocks 3 --heads 2  --block_shape "[64, 64, 64]" \
        --embedding_size 64 \
        --attention_size 32 \
        --is_save --has_residual \
        --save_path ./models/Criteo/b3h2e64a32_64x64x64/ \
        --field_size 39  --run_times 1 \
        --epoch 3 --batch_size 1024 \
```

## Acknowledgement
The implementation is based on [AutoInt](https://github.com/DeepGraphLearning/RecommenderSystems), thanks a lot.
