import argparse
import configparser, json
import dill

import numpy as np

import spacy
from Corpus import Corpus
from tqdm import tqdm
from pathlib import Path

import mlflow
from mlflow import log_metric, log_param, start_run
import mlflow.pyfunc

from model.TopicModel import TopicModel
from model.TopicModelWrapper import TopicModelWrapper
from metrics.metrics_perplexity import perplexity

if __name__ == '__main__':
    # parser for command-line
    parser = argparse.ArgumentParser(description='Train a topic model')
    parser.add_argument("config", help="config file path", type=str)
    args = parser.parse_args()

    # config
    config = configparser.ConfigParser()
    config_path = args.config
    config.read(config_path)

    # reproducibility
    model_params = config['model_params']
    np.random.seed(int(model_params['seed']))

    mlflow_params = config['mlflow_params']
    exp_dir = Path(mlflow_params['exp_dir'])

    # set tracking URI where mlflow saves runs
    mlflow.set_tracking_uri("file://" + exp_dir.as_posix() + "/mlruns")
    # set experiment
    mlflow.set_experiment(mlflow_params['experiment_name'])

    # spacy language object
    corpus_params = config["corpus_params"]
    corpus_dir = Path(corpus_params["corpus_dir"])
    nlp = spacy.load(corpus_dir / "nlp.spacy")

    # custom vocab object
    vocab_path = corpus_dir / 'vocab.pkl'
    with open(vocab_path, 'rb') as f:
        custom_vocab = dill.load(file=f)

    # corpus generator
    train_corpus = Corpus(path=corpus_dir / "train.spacy", custom_vocab=custom_vocab,
                          tfidf_threshold=float(model_params['tfidf_threshold']), limit=int(corpus_params["limit_train"]))
    dev_corpus = Corpus(path=corpus_dir / "dev.spacy", custom_vocab=custom_vocab,
                        tfidf_threshold=float(model_params['tfidf_threshold']), limit=int(corpus_params["limit_test"]))

    # hyperparameters
    params = config['model_params']
    n_topics = [int(K) for K in json.loads(params['K'])]
    batch_size_range = [int(S) for S in json.loads(params['batch_size_range'])]
    kappa_range = [float(kappa) for kappa in json.loads(params['kappa_range'])]
    tau_0_range = [float(tau_0) for tau_0 in json.loads(params['tau_0_range'])]



    for K in n_topics:
        for S in batch_size_range:
            for kappa in kappa_range:
                for tau_0 in tau_0_range:

                    with start_run() as run:
                        # init custom topic model
                        topic_model = TopicModel(K=K,
                                                 W=len(custom_vocab),
                                                 alpha=float(params['alpha']),
                                                 eta=float(params['eta']),
                                                 tau_0=tau_0,
                                                 kappa=kappa,
                                                 D=custom_vocab.corpus_size,
                                                 batch_size=S)

                        # construct MLflow model using wrapper class and custom topic model
                        mlflow_topic_model = TopicModelWrapper(topic_model=topic_model)

                        # train
                        mlflow_topic_model.fit(train_corpus=train_corpus, nlp=nlp)

                        # test
                        with tqdm(desc='TEST', position=2, leave=None) as pbar:
                            dev_batches = dev_corpus(nlp, S)
                            diffs = []
                            n_sums = []
                            for n_t in dev_batches:
                                diff, n_sum = mlflow_topic_model.topic_model.predict_score_batch(n_t)
                                diffs.append(diff)
                                n_sums.append(n_sum)
                                pbar.update(len(n_t))

                        # compute perplexity
                        perp_proxy = perplexity(np.concatenate(diffs), n_sums)

                        # Log a parameter (key-value pair)
                        log_param("K", K)
                        log_param("tau_0", tau_0)
                        log_param("kappa", kappa)
                        log_param("alpha", params['alpha'])
                        log_param("eta", params["eta"])
                        log_param("W", len(custom_vocab))
                        log_param("batch_size", S)
                        log_param("tfidf threshold", params["tfidf_threshold"])
                        # log metrics
                        log_metric("Perplexity", perp_proxy)
                        #log_metric("ELBO", ELBO)

                        # log artifact: custom_vocab path
                        mlflow.log_artifact(local_path=vocab_path.as_posix())

                        # log model
                        # TODO: log artifacts and conda_env ?
                        mlflow.pyfunc.log_model(artifact_path="my_topic_model",
                                                python_model=mlflow_topic_model)

