import argparse
import configparser, json
import dill
import mlflow.exceptions

import numpy as np

import spacy
from tqdm import tqdm
from pathlib import Path

from mlflow import log_metric, log_param, start_run, set_tracking_uri, create_experiment

from Corpus import Corpus
from Doc import Doc

from model.TopicModel import TopicModel
from model.LDA.online_var_bayes import E_step
from metrics.metrics_perplexity import doc_proxy_values, perplexity_proxy

if __name__ == '__main__':
    # parser for command-line
    parser = argparse.ArgumentParser(description='Train a topic model')
    parser.add_argument("--config", help="config file path", default='./config.ini')
    args = parser.parse_args()

    # config
    config = configparser.ConfigParser()
    config_path = args.config
    config.read(config_path)

    # reproducibility
    np.random.seed(int(config['model_params']['seed']))

    exp_dir = Path(config['model_paths']['model_dir']) / config["experiment"]["experiment_name"]
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True)

    # set tracking URI where mlflow saves runs
    set_tracking_uri("file://" + str(exp_dir) + "/mlruns")
    # create an experiment
    try:
        experiment_name = config["experiment"]["experiment_name"]
        experiment_id = create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        raise FileExistsError('Experiment already exists')

    # spacy language object
    nlp = spacy.load("en_core_web_sm", exclude=["parser","ner"])

    # data
    load_path = exp_dir / 'train_corpus.pkl'
    load_corpus = False
    if load_path.exists() and load_corpus:
        with open(load_path, 'rb') as f:
            train_corpus = dill.load(f)
    else:
        train_corpus = Corpus(dir=Path(config['data_paths']['train']),
                              nlp=nlp,
                              freq_cutoff=int(config['corpus_params']['freq_cutoff']),
                              limit=-1)
        # save train_corpus
        corpus_save_path = exp_dir / 'train_corpus.pkl'
        with open(corpus_save_path, 'wb') as f:
            dill.dump(train_corpus, file=f)

    test_dir = Path(config['data_paths']['test'])
    test_corpus = Corpus(dir=test_dir,
                         nlp=train_corpus.nlp,
                         vocab=train_corpus.vocab,
                         limit=-1)

    # hyperparameters
    params = config['model_params']
    n_topics = [int(K) for K in json.loads(params['K'])]
    batch_size_range = [int(S) for S in json.loads(params['batch_size_range'])]
    kappa_range = [float(kappa) for kappa in json.loads(params['kappa_range'])]
    tau_0_range = [float(tau_0) for tau_0 in json.loads(params['tau_0_range'])]

    model_count = 0

    for K in n_topics:
        for S in batch_size_range:
            for kappa in kappa_range:
                for tau_0 in tau_0_range:

                    with start_run(experiment_id=experiment_id):
                        # model
                        topic_model = TopicModel(K=K,
                                                 W=len(train_corpus.vocab),
                                                 alpha=float(params['alpha']),
                                                 eta=float(params['eta']),
                                                 tau_0=tau_0,
                                                 kappa=kappa,
                                                 batch_size=S)

                        # train
                        topic_model.fit(train_corpus)

                        # test
                        test_batch_size = 256
                        D = len(test_corpus)
                        N = D // test_batch_size
                        N = N if D % test_batch_size == 0 else N + 1
                        batch_idx = np.array_split(np.arange(D), N)
                        E_diffs, count_sums = [], []

                        with tqdm(desc='TEST', position=2, total=D, leave=None) as pbar:
                            for t_batch in batch_idx:
                                # document info
                                n_t = test_corpus.get_batch_counts(list(t_batch))

                                # parameter optimization
                                # E
                                phi_t, gamma_t = E_step(lam=topic_model.lam, alpha=topic_model.alpha, n_t=n_t)

                                E_diff, counts_sum = doc_proxy_values(n_t, gamma_t, phi_t, topic_model.lam,
                                                                      topic_model.alpha)
                                E_diffs.extend(E_diff)
                                count_sums.extend(counts_sum)

                                pbar.update(len(t_batch))

                        # perplexity of test set
                        perp_proxy = perplexity_proxy(E_diffs, count_sums)

                        # Log a parameter (key-value pair)
                        log_param("K", K)
                        log_param("tau_0", tau_0)
                        log_param("kappa", kappa)
                        log_param("batch_size", S)
                        log_param("alpha", params['alpha'])
                        log_param("eta", params["eta"])
                        log_param("W", len(train_corpus.vocab))
                        # log metrics
                        log_metric("Perplexity", perp_proxy)
                        #log_metric("ELBO", ELBO)

                        # save topic model
                        model_dir = exp_dir / f'model_{model_count}'
                        if not model_dir.exists():
                            model_dir.mkdir(parents=True)

                        topic_model.save_model(model_dir=model_dir)
                        # save model params
                        params = {"K": K,
                                  "tau_0": tau_0,
                                  "kappa": kappa,
                                  "batch_size": S,
                                  "alpha": params['alpha'],
                                  "eta": params["eta"],
                                  "W": len(train_corpus.vocab),
                                  "perp": perp_proxy}
                        with open(model_dir / 'params.json', 'w') as f:
                            json.dump(params, f)

                        model_count += 1

