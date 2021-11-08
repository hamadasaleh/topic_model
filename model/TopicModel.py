from typing import Union
import numpy as np

from scipy.stats import dirichlet

from model.LDA.online_var_bayes import online_var_bayes, E_step




class TopicModel:

    def __init__(self,
                 K: int,
                 W: int,
                 alpha: Union[float, np.array],
                 eta: Union[float, np.array],
                 tau_0: float,
                 kappa: float,
                 lam = None,
                 batch_size: int = 1,
                 seed: int = 0):
        self.seed = seed
        self.K = K
        self.W = W
        self.alpha = alpha
        self.eta = eta
        self.tau_0 = tau_0
        self.kappa = kappa
        if lam is not None:
            self.lam = self.load_lambda(lam)
            self.q_beta_hat = self.get_q_beta_hat()
        else:
            self.lam = lam
            self.q_beta_hat = None
        self.batch_size = batch_size
        super().__init__()


    def fit(self, train_corpus):
        self.lam = online_var_bayes(train_corpus=train_corpus,
                                    K=self.K,
                                    W=self.W,
                                    alpha=self.alpha,
                                    eta=self.eta,
                                    tau_0=self.tau_0,
                                    kappa=self.kappa,
                                    batch_size=self.batch_size)
        # posterior distribution of the topics
        self.q_beta_hat = self.get_q_beta_hat()


    def fit_doc_params(self, doc, train_corpus):
        # homemade Corpus and Doc
        n_t = doc.get_counts_array(tok2idx=train_corpus.vocab.tok2idx)

        # parameter optimization
        phi_t, gamma_t = E_step(self.lam, self.alpha, n_t)

        return phi_t, gamma_t, n_t


    def predict(self, phi_t, gamma_t):
        # topic assignments
        z_hat = phi_t.argmax(axis=0)

        # topic proportions
        q_theta_hat = dirichlet.rvs(alpha=gamma_t, random_state=self.seed)

        return z_hat, q_theta_hat

    def get_q_beta_hat(self):
        try:
            q_beta_hat = np.stack([dirichlet.rvs(alpha=self.lam[k], random_state=self.seed).squeeze()
                                   for k in range(self.lam.shape[0])],
                                  axis=0)
        except:
            pass

        return q_beta_hat

    def load_lambda(self, path):
        with open(path, 'rb') as f:
            return np.load(f)

    def save_lambda(self, model_dir):
        save_path = model_dir / 'lam.npy'
        with open(save_path, 'wb') as f:
            np.save(f, self.lam)

    def save_model(self, model_dir):
        import dill
        save_path = model_dir / 'MODEL.pkl'
        with open(save_path, 'wb') as f:
            dill.dump(self, file=f)

