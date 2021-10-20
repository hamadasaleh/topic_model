class Vocab:
    def __init__(self, corpus_freq, freq_cutoff, nlp):
        self.corpus_freq = corpus_freq
        self.freq_cutoff = freq_cutoff
        self.nlp = nlp
        self.stop_words = self.get_stop_words(freq_cutoff=freq_cutoff)
        self.add_stop_words(self.stop_words)
        self.tok2idx = self.get_tok2idx()
        self.idx2tok = {idx: token for token,idx in self.tok2idx.items()}

    def get_tok2idx(self):
        vocab = [token for token, freq in self.corpus_freq.items() if token not in self.stop_words]
        return {token: i for i, token in enumerate(vocab)}

    def get_stop_words(self, freq_cutoff):
        # additional stop words based on corpus frequencies
        stop_words = [token for token, freq in self.corpus_freq.items() if freq <= freq_cutoff]
        return stop_words

    def add_stop_words(self, stop_words):
        for word in stop_words:
            lex = self.nlp.vocab[word]
            lex.is_stop = True