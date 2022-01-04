from math import log
from typing import Iterable

def idf(df: int, n: int):
    idf = log((1 + n) / (1 + df)) + 1
    return idf

class CustomVocab:
    def __init__(self, corpus_size: int, corpus_freq: dict, doc_pres_freq: dict, freq_cutoff: int, nlp: "Language",
                 tf_idf: "DataFrame" = None,
                 max_vocab: int = None):
        self.corpus_size = corpus_size
        self.corpus_freq = corpus_freq
        self.doc_pres_freq = doc_pres_freq
        self.freq_cutoff = freq_cutoff
        self.stop_words = self.get_stop_words(freq_cutoff=freq_cutoff)
        self.add_stop_words(nlp=nlp, stop_words=self.stop_words)
        self.tok2idx = self.get_tok2idx()
        self.idx2tok = {idx: token for token, idx in self.tok2idx.items()}
        self.tokens = set([token for token in self.tok2idx.keys()])
        self.tf_idf = tf_idf
        self.max_vocab = max_vocab

    def __len__(self):
        return len(self.tok2idx)

    def get_tok2idx(self):
        vocab = [token for token, freq in self.corpus_freq.items() if token not in set(self.stop_words)]
        return {token: i for i, token in enumerate(vocab)}

    def get_stop_words(self, freq_cutoff: int):
        # additional stop words based on corpus frequencies
        stop_words = [token for token, freq in self.corpus_freq.items() if freq <= freq_cutoff]
        return stop_words

    @staticmethod
    def add_stop_words(nlp: "Language", stop_words: Iterable[str]):
        for word in stop_words:
            lex = nlp.vocab[word]
            lex.is_stop = True

    def set_tfidf(self, tfidf: "DataFrame"):
        if self.tf_idf is not None:
            self.tf_idf = tfidf
        else:
            raise BaseException

    def get_tfidf(self, doc_freq: dict) -> dict:
        tf_idf = {token: tf / idf(df=self.corpus_freq[token], n=self.corpus_size) for token, tf in doc_freq.items()}
        return tf_idf
