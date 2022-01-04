import warnings
from typing import Union, Iterator, Iterable, List

import spacy

import numpy as np

from pathlib import Path

from spacy import Language
from spacy.tokens import DocBin, Doc

import dill

from CustomVocab import CustomVocab

from POC.tf_idf import tfidf

FILE_TYPE = ".spacy"


def create_batches(docs, batch_size, vocab_size):
    batch = np.zeros(shape=(batch_size, vocab_size))
    i = 0
    for doc in docs:
        for key, value in doc.user_data.items():
            filtered_counts = filter_counts(value, vocab_size)
            for token, count in filtered_counts.items():
                batch[i, tok2idx[token]] = count

            i += 1
        if i == batch_size:
            yield batch
            batch = np.zeros(shape=(batch_size, vocab_size))

def filter_counts(counts_dict: dict, custom_vocab: CustomVocab = None, kind: str = 'tf-idf', tfidf_scores: dict = None,
                  tfidf_threshold: float = None) -> dict:

    if kind == 'tf-idf':
        if tfidf_scores is not None and tfidf_threshold is not None:
            filtered_counts = {token: count for token, count in counts_dict.items() if tfidf_scores[token] >= tfidf_threshold}
        else:
            raise Exception('tf-idf parameter is None')

    elif kind == 'general':
        if custom_vocab is not None:
            filtered_counts = {token: count for token, count in counts_dict.items() if token in custom_vocab.tokens}
        else:
            raise Exception('custom vocab is None')
    else:
        raise NotImplementedError
    return filtered_counts


def walk_corpus(path: Union[str, Path], file_type) -> list[Path]:
    path = Path(path)
    if not path.is_dir() and path.parts[-1].endswith(file_type):
        return [path]
    orig_path = path
    paths = [path]
    locs = []
    seen = set()
    for path in paths:
        if str(path) in seen:
            continue
        seen.add(str(path))
        if path.parts and path.parts[-1].startswith("."):
            continue
        elif path.is_dir():
            paths.extend(path.iterdir())
        elif path.parts[-1].endswith(file_type):
            locs.append(path)
    if len(locs) == 0:
        warnings.warn(f"path={orig_path}, format={file_type}")
    # It's good to sort these, in case the ordering messes up a cache.
    locs.sort()
    return locs


class Corpus:
    def __init__(self,
                 path: Union[str, Path],
                 custom_vocab: CustomVocab,
                 tfidf_threshold: float,
                 limit: int = 0,
                 shuffle: bool = False):
        self.path = path
        self.custom_vocab = custom_vocab
        self.tfidf_threshold = tfidf_threshold
        self.limit = limit
        self.shuffle = shuffle


    def __call__(self, nlp: "Language", batch_size: int) -> Iterator[np.array]:
        """Yield examples from the data.

        nlp (Language): The current nlp object.
        YIELDS (np.array): Term frequency batch.

        DOCS: https://spacy.io/api/corpus#call
        """
        ref_docs = self.read_docbin(nlp.vocab, walk_corpus(self.path, FILE_TYPE))
        if self.shuffle: # TODO: to test
            ref_docs = list(ref_docs)  # type: ignore
            random.shuffle(ref_docs)  # type: ignore

        batches = self.make_batches(ref_docs, batch_size)
        for batch in batches:
            yield batch

    def make_batches(self, ref_docs: List[Doc], batch_size: int):
        batch = np.zeros(shape=(batch_size, len(self.custom_vocab), 1))
        i = 0
        for doc in ref_docs: # TODO: deal with last docs
            for key, value in doc.user_data.items():
                if "counts" in key:
                    # get document term frequencies
                    counts_dict = value
                    # compute tf-idf scores
                    tfidf_scores = tfidf(doc_freq=counts_dict, doc_pres_freq=self.custom_vocab.doc_pres_freq,
                                         corpus_size=self.custom_vocab.corpus_size)
                    # filter document tokens based on tfidf threshold
                    filtered_counts = filter_counts(counts_dict=counts_dict, kind='tf-idf', tfidf_scores=tfidf_scores,
                                                    tfidf_threshold=self.tfidf_threshold)
                    # store in numpy array
                    for token, count in filtered_counts.items():
                        idx = self.custom_vocab.tok2idx.get(token)
                        # idx is None for OOV tokens
                        if idx is not None:
                            batch[i, idx, :] = count
                else:
                    raise Exception
                i += 1
            if i == batch_size:
                yield batch
                batch = np.zeros(shape=(batch_size, len(self.custom_vocab), 1))
                i = 0

    def read_docbin(self, vocab: "Vocab", locs: Iterable[Union[str, Path]]) -> Iterator[Doc]:
        i = 0
        for loc in locs:
            loc = Path(loc)
            if loc.parts[-1].endswith(FILE_TYPE):
                doc_bin = DocBin().from_disk(loc)
                docs = doc_bin.get_docs(vocab)
                for doc in docs:
                    if len(doc):
                        yield doc
                        i += 1
                        if self.limit >= 1 and i >= self.limit:
                            break

if __name__ == '__main__':
    corpus_dir = Path("C:/Users/hamad/Documents/datasets/corpus")
    docbin_path = corpus_dir / "train.spacy"

    nlp = spacy.load(corpus_dir / "nlp.spacy")

    custom_vocab_path = corpus_dir / 'vocab.pkl'
    with open(custom_vocab_path, 'rb') as f:
        custom_vocab = dill.load(f)

    corpus = Corpus(path=corpus_dir, custom_vocab=custom_vocab)

    # I would like for a generator to return batch of data
    # instead of single instances
    batch_size = 256
    batches = corpus(nlp, batch_size)
    for eg in batches:
        print('Serve batch to topic model for training')
