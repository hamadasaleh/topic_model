from collections import Counter
from tqdm import tqdm

import numpy as np

from Doc import Doc
from Vocab import Vocab

class Corpus:

    def __init__(self, dir, nlp,  limit, freq_cutoff=1, vocab=None):
        self.dir = dir
        if limit == -1:
            self.docs_paths = self.get_paths()
        else:
            self.docs_paths = self.get_paths()[:limit]
        self.nlp = nlp
        self.docs = self.load_corpus()

        if vocab is not None:
            self.vocab = vocab
        else:
            self.corpus_freq = self.get_corpus_freq()
            self.vocab = Vocab(self.corpus_freq, freq_cutoff, self.nlp)

    def __len__(self):
        return len(self.docs_paths)

    def __getitem__(self, index):
        return [self.docs[i] for i in index]

    def get_batch_counts(self, batch_idx):
        batch_docs = self.__getitem__(batch_idx)
        n_t = np.array([doc.get_counts_array(tok2idx=self.vocab.tok2idx) for doc in batch_docs])[:, :, None]
        return n_t

    def get_corpus_freq(self):
        counters_list = [doc.token_freq for doc in self.docs]
        corpus_freq = sum(counters_list, Counter())
        return corpus_freq

    def load_corpus(self):
        return [Doc(doc_path, self.nlp) for doc_path in tqdm(self.docs_paths, desc='Loading corpus', position=0, leave=None)]

    def get_paths(self):
        paths = []

        for subdir in self.dir.iterdir():
            paths.extend(list(subdir.iterdir()))

        return paths


