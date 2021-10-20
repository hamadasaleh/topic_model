import numpy as np
from scipy.special import softmax

from tqdm import tqdm

from metrics.metrics_elbo import ell

from metrics.metrics_expectations import E_log_theta, E_log_beta


def online_var_bayes(train_corpus, K, W, alpha, eta, tau_0, kappa, batch_size) -> np.array:
    """
    Online variational inference for LDA

    :param train_corpus: Corpus
    :param K: number of topics
    :param W: vocabulary size
    :param alpha: float
    :param eta: float
    :param tau_0: float
    :param kappa: float
    :return: lam np.array
    """
    lam = np.random.rand(K, W)
    D = len(train_corpus)
    phi_batch = np.zeros((batch_size, K, W))
    n_batch = np.zeros((batch_size, 1, W))
    N = D // batch_size
    L = []
    batch_count = 0

    with tqdm(total=D, position=1, leave=None, desc="TRAIN") as pbar:

        for t, doc in enumerate(train_corpus.docs):
            # document info
            n_t = doc.get_counts_array(tok2idx=train_corpus.vocab.tok2idx)

            # parameter optimization
            # E
            phi_t, gamma_t = E_step(lam=lam, alpha=alpha, n_t=n_t)

            # M
            rho_t = rho(tau_0, t, kappa)
            phi_batch[batch_count] = phi_t[None, :]
            n_batch[batch_count] = n_t[None, :]
            batch_count += 1

            if t in range(0, N*batch_size + 1, batch_size) or t == D-1:

                B = batch_size if t in range(0, N*batch_size + 1, batch_size) else D - N*batch_size
                lam_tilde = eta + (D/B) * (phi_batch * n_batch).sum(axis=0)
                lam = (1. - rho_t) * lam + rho_t * lam_tilde

                # ELBO: lower bound on log likelihood
                ell_t = ell(n_t, phi_t, gamma_t, lam, alpha, eta, D)
                L.append(ell_t)

                # reset
                batch_count = 0
                phi_batch = np.zeros((batch_size, K, W))
            else:
                continue

            pbar.update()

    ELBO = sum(L)

    return lam, ELBO


def E_step(lam, alpha, n_t, tol=1e-5):
    """
    E step - iterative update of document parameters gamma and phi until convergence

    :param lam: (K, W)
    :param alpha: float
    :param n_t: (W,)
    :param tol: float
    :return: phi_t (K, W), gamma_t (K, 1)
    """
    K = lam.shape[0]
    gamma = np.ones((K, 1))

    while True:
        phi_t = softmax(E_log_theta(gamma) + E_log_beta(lam), axis=0)
        #TESTING
        assert phi_t.shape == lam.shape

        gamma_t = alpha + phi_t @ n_t

        err = np.mean(np.abs(gamma_t - gamma))
        if err > tol:
            gamma = gamma_t
        else:
            return phi_t, gamma_t

def M_step(lam, eta, D, phi_t, n_t, rho_t):
    """
    M step - update of corpus parameter lambda given phi

    :param lam: (K, W)
    :param eta: float
    :param D: float number of docs in train set
    :param phi_t: (K, W) variational topic index distribution
    :param n_t: (W,) word count array
    :param tau_0: float
    :param t: int
    :param kappa: float
    :return: lam (K, W)
    """
    lam_tilde = eta + D * phi_t * n_t
    lam = (1. - rho_t) * lam + rho_t * lam_tilde
    return lam


def rho(tau_0: float,
        t: int,
        kappa: float):
    """
    Rho is the update weight given to lambda_tilde during the M step

    :param tau_0: slows down the early iterations of the algorithm
    :param t: current doc id
    :param kappa: (float in (0.5, 1]) controls the rate at which old values of lambda_tilde are forgotten
    :return:
    """
    return (tau_0 + t) ** - kappa
