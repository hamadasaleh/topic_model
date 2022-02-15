import json
from pathlib import Path
from typing import Union, Tuple, List
import numpy as np

from scipy.stats import dirichlet

from model.LDA.online_var_bayes import online_var_bayes, multi_E_step
from metrics.metrics_perplexity import doc_diff, perplexity



class TopicModel:

    def __init__(self,
                 K: int,
                 W: int,
                 tau_0: float,
                 kappa: float,
                 D: int,
                 alpha: Union[float, np.array] = None,
                 eta: Union[float, np.array] = None,
                 lam: np.array = None,
                 batch_size: int = 1,
                 n_cores: int = 2,
                 update_priors: bool = False,
                 n_epochs: int = 1,
                 seed: int = 0):
        self.seed = seed
        self.K = K
        self.W = W
        if alpha is None:
            self.alpha = 1. / K
        else:
            self.alpha = alpha
        self.alpha_prior = self.alpha
        if eta is None:
            self.eta = 1. / W
        else:
            self.eta = eta
        self.eta_prior = self.eta
        self.tau_0 = tau_0
        self.kappa = kappa
        self.D = D
        if lam is not None:
            if isinstance(lam, str):
                self.lam = self.load_lambda(lam)
            elif isinstance(lam, np.ndarray):
                self.lam = lam
            # TODO:
            self.q_beta_hat = self.get_q_beta_hat()
        else:
            self.lam = lam
            self.q_beta_hat = None
        self.batch_size = batch_size
        self.n_cores = n_cores
        self.update_priors = update_priors
        self.n_epochs = n_epochs
        super().__init__()


    def fit(self, train_corpus: "Corpus", nlp: "Language"):
        batches = train_corpus(nlp, self.batch_size)
        self.lam, self.alpha, self.eta = online_var_bayes(batches=batches,
                                            D=self.D,
                                            W=self.W,
                                            K=self.K,
                                            alpha=self.alpha,
                                            eta=self.eta,
                                            tau_0=self.tau_0,
                                            kappa=self.kappa,
                                            n_cores=self.n_cores,
                                            update_priors=self.update_priors,
                                            n_epochs=self.n_epochs)
        # posterior distribution of topics
        self.q_beta_hat = self.get_q_beta_hat()

    def predict_corpus(self, corpus: "Corpus", nlp: "Language") -> Tuple[List[List], List[dirichlet]]:
        batches = corpus(nlp, self.batch_size)
        z_hat_corpus = []
        q_theta_hat_corpus = []

        for n_t in batches:
            z_hat, q_theta_hat = self.predict_doc(n_t)
            z_hat_corpus.extend(z_hat)
            q_theta_hat_corpus.extend(q_theta_hat)

        return z_hat_corpus, q_theta_hat_corpus



    def predict_doc(self, n_t: np.array) -> Tuple[List[int], List[dirichlet]] :
        """

        :param n_t: (batch_size, n_topics, vocab_size)
        :return:
        """
        # fit per document parameters
        phi_t, gamma_t = E_step(lam=self.lam, alpha=self.alpha, n_t=n_t)

        # topic assignments: e.g. this word comes from that topic
        z_hat = phi_t.argmax(axis=1).tolist()

        # topic proportions:
        # e.g. q_theta_hat[0] represents the influence of topic 0 on given document
        # if q_theta_hat[0] = 20% you could interpret it as "20% of what is discussed in the document concerns that
        # topic"
        q_theta_hat = [dirichlet(alpha=conc, seed=self.seed) for conc in gamma_t.squeeze()]
        return z_hat, q_theta_hat

    def predict_score_batch(self, n_t: np.array) -> Tuple[np.array, float]:
        phi_t, gamma_t = self.fit_doc_params(n_t)
        diff = doc_diff(n_d=n_t, gamma_d=gamma_t, phi_d=phi_t, lam=self.lam, alpha=self.alpha)
        n_sum = n_t.sum()
        return diff, n_sum

    def fit_doc_params(self, n_t: np.array):
        phi_t, gamma_t = multi_E_step(n_t=n_t, lam=self.lam, alpha=self.alpha, n_cores=self.n_cores)
        return phi_t, gamma_t

    def get_q_beta_hat(self):
        """Variational topics"""
        q_beta_hat = np.stack([dirichlet.rvs(alpha=self.lam[k], random_state=self.seed).squeeze()
                               for k in range(self.lam.shape[0])],
                              axis=0)
        return q_beta_hat

    def get_q_theta_hat(self, gamma_t: np.array) -> np.array:
        """Variational topic proportions"""
        q_theta_hat = dirichlet.rvs(alpha=gamma_t, random_state=self.seed).squeeze()
        return q_theta_hat

    def load_lambda(self, lam_path: Union[str, Path]):
        with open(lam_path, 'rb') as f:
            lam = np.load(f)
        return lam

    def save_lambda(self, model_dir: Union[str, Path]) -> None:
        save_path = model_dir / 'lam.npy'
        with open(save_path, 'wb') as f:
            np.save(f, self.lam)

    def save_params(self, params_dict: dict, model_dir: Union[str, Path]):
        with open(model_dir / "params.json", "wb") as f:
            json.dump(params_dict, f)

    def save_model(self, model_dir):
        import dill
        save_path = model_dir / 'MODEL.pkl'
        with open(save_path, 'wb') as f:
            dill.dump(self, file=f)

    def to_disk(self, model_dir: Union[str, Path]) -> None:
        model_dir = Path(model_dir)
        params_dict = self.__dict__
        json.dump(params_dict)
        self.save_lambda(model_dir)

    def from_disk(self, model_dir: Union[str, Path]) -> "TopicModel":
        model_dir = Path(model_dir)

        params_path = model_dir / "params.json"
        self.__dict__ = json.load(params_path)

        lam_path = model_dir / 'lam.npy'
        self.lam = self.load_lambda(lam_path)
        return self

