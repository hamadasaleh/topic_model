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
[dataset_params]
dataset_name = 20news-bydate
train_path = ./20news-bydate-train
dev_path = ./20news-bydate-test

[nlp]
name = en_core_web_lg
disable = ["parser", "ner"]

[annotation]
n_process = 2
batch_size = 1000
lemmatize = False
pos_filter = ["NOUN", "ADJ"]

[mlflow_params]
exp_dir = /models/topic_models/experiments
experiment_name = topic-model

[corpus_params]
corpus_dir = /datasets/spacy_corpora
freq_cutoff = 1
bigrams_min_freq = 2
limit_train=0
limit_test=0

[model_params]
seed = 123
n_cores = 7
n_epochs = 1
update_priors = True
K = [10, 15, 20, 50]
batch_size_range = [1, 4, 16, 32, 64, 128, 256]
# downweighting of early iterations
tau_0_range = [64, 128, 256, 1024]
# learning rate of variational inference (should be in (0.5, 1])
kappa_range = [0.5, 0.6, 0.7, 0.8, 0.9]
tfidf_threshold = 0.3
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
- [Stochastic Variational Inference](https://www.jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf)
