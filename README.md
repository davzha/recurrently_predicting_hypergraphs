# Recurrently Predicting Hypergraphs
This repository contains the code for reproducing the experiments described in [Recurrently Predicting Hypergraphs](https://arxiv.org/abs/2106.13919) by David Zhang, Gertjan Burghouts and Cees Snoek.

# Dependencies
Install environment with conda and pip:
```
conda create -n RPH python=3.9
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install uproot==3.12.0
pip install https://ray-wheels.s3-us-west-2.amazonaws.com/python3.9/a902f2e4ab0a9c27ece8562084aa3fc4be68eeb8/ray-1.2.0.dev0-cp39-cp39-manylinux2014_x86_64.whl
pip install numpy scipy pandas sklearn pytorch-lightning wandb
```
# Experiments
## Particle partitioning
Follow the data setup described in https://github.com/hadarser/SetToGraphPaper.
Specify the data directory in `particle_partitioning_main.py` and `particle_partitioning_baseline.py`.
Adapt the `TYPE` variable to either `slot_attention` or `set_transformer` to run the baselines.
Both scripts are meant to be run without any additional command line arguments.
All hyperparameters are specified in the python scripts directly.

## Convex hull finding
Run `python convex_hull_main.py` for RPH and `python convex_hull_baseline.py` for the baselines.
The ablations can be run by setting the hyperparameters in `convex_hull_main.py` accordingly.

## Delaunay triangulation
Run `python delaunay_main.py`.