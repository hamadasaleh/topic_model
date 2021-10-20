from pathlib import Path

import spacy
from collections import Counter
from tqdm import tqdm

from Doc import Doc
from Vocab import Vocab

class Corpus:

    def __init__(self, train_dir, nlp, freq_cutoff, limit):
        self.train_dir = train_dir
        if limit == -1:
            self.docs_paths = self.get_paths()
        else:
            self.docs_paths = self.get_paths()[:limit]
        self.nlp = nlp
        self.load_corpus()

        self.corpus_freq = self.get_corpus_freq()
        self.vocab = Vocab(self.corpus_freq, freq_cutoff, self.nlp)

    def __len__(self):
        return len(self.docs_paths)

    def get_corpus_freq(self):
        counters_list = [doc.token_freq for doc in self.docs]
        corpus_freq = sum(counters_list, Counter())
        return corpus_freq

    def load_corpus(self):
        self.docs = [Doc(doc_path, self.nlp) for doc_path in tqdm(self.docs_paths, desc='Loading corpus', position=0, leave=None)]

    def get_paths(self):
        paths = []

        for subdir in self.train_dir.iterdir():
            paths.extend(list(subdir.iterdir()))

        return paths

if __name__ == '__main__':
    docs_path = Path("C:/Users/hamad/Documents/datasets/20news-bydate-train")
    nlp = spacy.load("en_core_web_sm", exclude=["ner"])
    train_corpus = Corpus(docs_path, nlp, limit=1000)
