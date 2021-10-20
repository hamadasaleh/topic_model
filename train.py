import argparse
import configparser,json
import dill
import mlflow.exceptions

from datetime import datetime

import spacy
from tqdm import tqdm
from pathlib import Path
import shutil

from mlflow import log_metric, log_param, start_run, set_tracking_uri, create_experiment

from Corpus import Corpus
from Doc import Doc

from model.TopicModel import TopicModel

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

    model_dir = Path(config['model_paths']['model_dir'])
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    # set tracking URI where mlflow saves runs
    set_tracking_uri("file://" + config["model_paths"]["runs_path"])
    # create an experiment
    n = 0
    while True:
        try:
            experiment_name = config["experiment"]["experiment_name"] + " " + str(n)
            experiment_id = create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            n += 1
            continue
        break

    # spacy language object
    nlp = spacy.load("en_core_web_sm", exclude=["ner"])

    # data
    load_path = model_dir / 'train_corpus.pkl'
    load_corpus = False
    if load_path.exists() and load_corpus:
        with open(load_path, 'rb') as f:
            train_corpus = dill.load(f)
    else:
        train_corpus = Corpus(train_dir=Path(config['data_paths']['train']),
                              nlp=nlp,
                              freq_cutoff=int(config['model_params']['freq_cutoff']),
                              limit=-1)
        # save train_corpus
        corpus_save_path = model_dir / 'train_corpus.pkl'
        with open(corpus_save_path, 'wb') as f:
            dill.dump(train_corpus, file=f)

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

                    with start_run(experiment_id=experiment_id):
                        # model
                        topic_model = TopicModel(K=K,
                                                 W=len(train_corpus.vocab.tok2idx),
                                                 alpha=float(params['alpha']),
                                                 eta=float(params['eta']),
                                                 tau_0=tau_0,
                                                 kappa=kappa,
                                                 batch_size=S)

                        # train
                        ELBO = topic_model.fit(train_corpus)

                        # test
                        test_dir = Path(config['data_paths']['test'])
                        E_diffs, count_sums = [], []
                        doc_paths = [path for folder in test_dir.iterdir() for path in folder.iterdir()]
                        with tqdm(desc='TEST', position=2, total=len(doc_paths), leave=None) as pbar:
                            for n_doc, doc_path in enumerate(doc_paths):
                                test_doc = Doc(doc_path, train_corpus.nlp)
                                phi_t, gamma_t, n_t = topic_model.fit_doc_params(test_doc, train_corpus)
                                E_diff, counts_sum = doc_proxy_values(n_t, gamma_t, phi_t, topic_model.lam,
                                                                      topic_model.alpha)
                                E_diffs.append(E_diff)
                                count_sums.append(counts_sum)
                                pbar.update()

                        # perplexity of test set
                        perp_proxy = perplexity_proxy(E_diffs, count_sums)

                        # Log a parameter (key-value pair)
                        log_param("K", K)
                        log_param("tau_0", tau_0)
                        log_param("kappa", kappa)
                        log_param("batch_size", S)
                        log_param("alpha", params['alpha'])
                        log_param("eta", params["eta"])
                        log_param("W", len(train_corpus.vocab.tok2idx))
                        # log metrics
                        log_metric("Perplexity", perp_proxy)
                        log_metric("ELBO", ELBO)

                        # save topic model
                        cdt = str(datetime.now().replace(microsecond=0))
                        topic_model.save_model(model_dir=Path(str(model_dir) + cdt))
                        # copy config file
                        shutil.copy(src=config_path, dst=model_dir / 'config.ini')

