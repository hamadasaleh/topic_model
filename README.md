# Topic modeling - Online variational Bayes for Latent Dirichlet Allocation

This is an implementation of the following paper: [Online Learning for Latent Dirichlet Allocation](https://www.di.ens.fr/~fbach/mdhnips2010.pdf).
The dataset used is the "20 newsgroup" corpus and is available [here](https://github.com/tdhopper/topic-modeling-datasets).

## Clone git project
Copy this repository to your local machine.

````
git clone <target_repo> <local_dir>
````

## Creating a conda virtual environment
````
$ conda create env_name
$ conda install --file requirements.txt
$ conda activate env_name
````

## Usage
### Training a model
````
$ python train.py --config /path/to/config.ini
````
## Configuration file
All hyperparameters and paths are centralised in a single configuration file that should be modified to taste.
Ideally, each file should correspond to a single experiment.
````
[data_paths]
train = /path/to/20news-bydate-train
test = /path/to/20news-bydate-test

[experiment]
experiment_name = experiment

[corpus_params]
freq_cutoff = 1

[model_params]
seed = 0
alpha = 0.05
eta = 0.05
K = [15, 20, 50]
batch_size_range = [1, 4, 16, 32, 64, 256]
tau_0_range = [64, 256, 1024]
kappa_range = [0.5, 0.6, 0.7, 0.8, 0.9]

[model_paths]
model_dir = /path/to/topic_models
````

## MLflow
As your experiments are running or once they have finished, you can visualise results and compare models seamlessly 
through MLflow. It is a great tool that allows for reproducibility, 
You can find a great tutorial about this tool on [Ahmed Besbes's blog](https://www.ahmedbesbes.com/case-studies/mlflow-101).

### Run in Anaconda prompt
````
$ cd /path/to/mlruns
$ mlflow ui
````

### Display different experiments in browser
http://localhost:5000



## Sources
- [Online Learning for Latent Dirichlet Allocation (Hoffman, Blei, Bach, 2010)](https://www.di.ens.fr/~fbach/mdhnips2010.pdf)
- [Latent Dirichlet Allocation (Blei, Ng, Jordan, 2003)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)

