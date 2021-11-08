import spacy

import numpy as np

from preprocessing.doc_preprocessing import normalize
from collections import Counter


class Doc:

    def __init__(self, doc_path, nlp):
        self.doc_path = doc_path
        self.nlp = nlp
        self.token_freq = self.get_counts_dict()

    def get_counts_array(self, tok2idx):
        """

        :param tok2idx: (dict)
        :return:
        sparse_counts (np.array) -> document represented as an array of token counts
        """
        unique_tokens, counts = list(self.token_freq.keys()), list(self.token_freq.values())

        # remove oov tokens
        unique_tokens = [token for token in unique_tokens if token in tok2idx.keys()]
        counts = np.array([self.token_freq[token] for token in unique_tokens])

        W = len(tok2idx)
        sparse_counts = np.zeros(W)

        idx_array = np.array([tok2idx[token] for token in unique_tokens])
        sparse_counts[idx_array] = np.array(counts)
        return sparse_counts

    def get_counts_dict(self):
        tokens = self.get_normalized_tokens()
        token_freq = Counter(tokens)
        return token_freq

    def get_normalized_tokens(self):
        my_text = self.read_doc()
        tokens = normalize(my_text, self.nlp)
        return tokens

    def read_doc(self):
        with open(self.doc_path, 'r', encoding='utf-8', errors='ignore') as f:
            my_text = f.read()
        return my_text



