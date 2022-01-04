import argparse
import configparser
import shutil
import sys
import json
from pathlib import Path
from typing import Union, List

import spacy
from nltk import ngrams
from sklearn.feature_extraction import DictVectorizer
from spacy.tokens import Doc, DocBin

from CustomVocab import CustomVocab

from collections import Counter

from tqdm import tqdm

import dill

def annotate_raw_texts(root_dir: Union[str, Path], nlp: "Language", config: dict, train: bool = False):
    """
    Annotate, preprocess, make ngrams and count frequencies for each doc in corpus.

    :param root_dir: directory containing text files to be processed
    :param nlp:
    :param config:
    :return: spacy DocBin object
    """
    root_dir = Path(root_dir)
    raw_texts = get_texts(root_dir)
    docbin = DocBin(store_user_data=True)

    with tqdm(desc="Annotating corpus of raw texts and serializing to DocBin format", total=len(raw_texts),
              file=sys.stdout) as p_bar:
        annot_config = config["annotation"]
        corpus_params = config['corpus_params']
        corpus_freq = Counter()
        doc_pres_freq = Counter()

        for doc in nlp.pipe(raw_texts,
                            n_process=int(annot_config['n_process']),
                            batch_size=int(annot_config['batch_size'])):
            pos_filter = json.loads(annot_config['pos_filter'])
            doc_lemmas = preprocessing(doc=doc, pos_filter=pos_filter)
            counts = get_doc_counts(lemmas=doc_lemmas, bigrams_min_freq=int(corpus_params['bigrams_min_freq']))
            doc._.set("counts", counts)
            docbin.add(doc)
            corpus_freq += counts
            dft = {token: 1 for token in list(counts.keys())}
            doc_pres_freq.update(dft)
            p_bar.update()

    res = {}
    if not train:
        res['docbin'] = docbin
    else:
        res['docbin'] = docbin
        custom_vocab = CustomVocab(corpus_size=len(raw_texts),
                                   corpus_freq=corpus_freq,
                                   doc_pres_freq=doc_pres_freq,
                                   freq_cutoff=int(corpus_params['freq_cutoff']),
                                   nlp=nlp)
        res['custom_vocab'] = custom_vocab

    return res


def get_doc_counts(lemmas: list[str], bigrams_min_freq: int) -> Counter:
    counts = Counter(lemmas)
    bigrams_counts = get_ngrams_counts(lemmas, n=2, min_freq=bigrams_min_freq)
    return counts + bigrams_counts


def get_ngrams_counts(lemmas: list, n: int, min_freq: int) -> Counter:
    grams = ngrams(lemmas, n)
    grams = [" ".join(g) for g in grams]
    counts = Counter(grams)
    counts = {x: count for x, count in counts.items() if count >= min_freq}
    return Counter(counts)


def preprocessing(doc: Doc, pos_filter: List[str], lemmatize: bool = False) -> List[str]:
    """
    Preprocessing steps of a document:
    - normalize (lowercase)
    - remove non alphanumeric terms
    - remove stop words (from spaCy's list)
    - keep tokens whose pos is in pos_filter
    - lemmatization (optional)"""
    if lemmatize:
        tokens_list = [tok.lemma_.lower() for tok in doc
                      if tok.is_alpha and
                      not tok.is_stop and
                      len(tok) > 2 and
                      tok.pos_ in pos_filter]
    else:
        tokens_list = [tok.text.lower() for tok in doc
                      if tok.is_alpha and
                      not tok.is_stop and
                      len(tok) > 2 and
                      tok.pos_ in pos_filter]
    return tokens_list


def get_texts(source: Union[str, Path]) -> List[str]:
    paths = Path(source).rglob("[0-9]*")
    texts = [read_doc(p) for p in paths]
    return texts


def read_doc(path: Union[str, Path]) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        my_text = f.read()
    return my_text


if __name__ == '__main__':
    # parser for command-line
    parser = argparse.ArgumentParser(description='Annotating train and dev corpora')
    parser.add_argument("config", help="config file path", type=str, nargs=1)
    args = parser.parse_args()

    # config
    config = configparser.ConfigParser()
    config_path = args.config
    config.read(config_path)

    # nlp
    nlp_config = config["nlp"]
    nlp = spacy.load(nlp_config["name"], disable=json.loads(nlp_config["disable"]))
    if not Doc.has_extension("counts"):
        Doc.set_extension("counts", default=None)

    # ANNOTATE TRAIN CORPUS
    data_paths = config["data_paths"]
    train_dir = Path(data_paths['train'])
    train_dict = annotate_raw_texts(root_dir=train_dir, nlp=nlp, config=config, train=True)

    # save train docbin and custom vocab
    corpus_dir = Path(config['corpus_params']["corpus_dir"])
    if not corpus_dir.exists():
        corpus_dir.mkdir()

    train_dict['docbin'].to_disk(corpus_dir / "train.spacy")
    vocab_save_path = corpus_dir / 'vocab.pkl'
    with open(vocab_save_path, 'wb') as f:
        dill.dump(train_dict['custom_vocab'], file=f)

    # DEV CORPUS
    dev_dir = Path(data_paths['dev'])
    dev_dict = annotate_raw_texts(root_dir=dev_dir, nlp=nlp, config=config)
    dev_dict['docbin'].to_disk(corpus_dir / "dev.spacy")

    # Save Language object to disk
    pipe_dir = corpus_dir / "nlp.spacy"
    if pipe_dir.exists() and pipe_dir.is_dir():
        shutil.rmtree(pipe_dir)
    nlp.to_disk(pipe_dir)
