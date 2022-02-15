from typing import Iterable, Union

import numpy as np
from scipy.special import softmax

from tqdm import tqdm

from joblib import Parallel, delayed

from metrics.metrics_expectations import E_log_theta, E_log_beta
from model.LDA.newton_raphson import update_alpha, update_eta

def online_var_bayes(batches: Iterable[np.array], D: int, W: int, K: int, alpha: float, eta: float, tau_0: float, kappa: int,
                     n_cores: int, update_priors: bool, n_epochs: int) -> np.array:
    """
    Online variational inference for LDA

    :param batches:
    :param D:
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
    t = 0

    with tqdm(total=D, position=1, leave=True, desc="TRAIN") as pbar:
        # n_t.shape = (S, W, 1)
        for n_t in batches:

            with tqdm(total=n_epochs, position=0, leave=False, desc="EPOCHS") as epoch_pbar:
                for epoch in range(n_epochs):
                    # parameter optimization
                    # E
                    phi_t, gamma_t = multi_E_step(n_t=n_t, lam=lam, alpha=alpha, n_cores=n_cores, show_pbar=True)

                    # M
                    t += n_t.shape[0] - 1
                    rho_t = rho(tau_0, t, kappa)
                    lam = M_step(lam, eta, D, phi_t, n_t, rho_t)

                    if update_priors:
                        rho_alpha = [rho(tau_0, d, kappa) for d in range(t-n_t.shape[0] + 1, t + 1)]
                        alpha = update_alpha(alpha=alpha, gamma_t=gamma_t, rho_alpha=rho_alpha)
                        eta = update_eta(eta=eta, lam=lam, rho_t=rho_t)

                    epoch_pbar.update()

            # TODO: ELBO: lower bound on log likelihood
            pbar.update(n_t.shape[0])

    return lam, alpha, eta

def multi_E_step(n_t: np.array, lam: np.array, alpha: float, n_cores: int, show_pbar: bool = False):
    """
    Multi-threaded E step
    """
    shared_exp_lam = E_log_beta(lam)[None, :, :]

    def process_chunk(chunk: np.array):
        phi_t, gamma_t = E_step(n_t=chunk, exp_lam=shared_exp_lam, alpha=alpha, show_pbar=show_pbar)
        return phi_t, gamma_t

    chunks = np.array_split(n_t, n_cores, axis=0)

    res = Parallel(n_jobs=n_cores, prefer="threads", require='sharedmem')(
        delayed(process_chunk)(chunk) for chunk in chunks)

    # unpack results
    phi_t, gamma_t = zip(*res)
    phi_t, gamma_t = np.concatenate(phi_t, axis=0), np.concatenate(gamma_t, axis=0)

    return phi_t, gamma_t

def E_step(n_t: np.array, exp_lam: np.array, alpha: Union[float, np.array], tol: float = 1e-5, show_pbar: bool = True):
    """
    E step - iterative update of document parameters gamma and phi until convergence

    :param exp_lam: (1, K, W)
    :param alpha: float
    :param n_t: (S, W, 1) token occurrences array
    :param tol: float
    :return: phi_t (S, K, W), gamma_t (S, K, 1)
    """
    K = exp_lam.shape[1]
    S = n_t.shape[0]
    gamma_save = np.empty((S, K, 1))
    gamma = np.ones_like(gamma_save)
    idx = np.arange(S)

    with tqdm(total=S, position=2, leave=None, desc="BATCH", disable=~show_pbar) as pbar:

        while True:
            phi_t = softmax(E_log_theta(gamma) + exp_lam, axis=1)

            gamma_t = alpha + phi_t @ n_t[idx, :, :]

            err = np.mean(np.abs(gamma_t - gamma), axis=1).flatten()
            cv_flag = err < tol

            if idx.size != 0:
                if all(~cv_flag):
                    gamma = gamma_t
                else:
                    # save
                    to_save = idx[cv_flag]
                    gamma_save[to_save, :, :] = gamma_t[cv_flag, :, :]

                    # to update~
                    gamma = gamma_t[~cv_flag, :, :]
                    idx = idx[~cv_flag]

                    pbar.update(n=cv_flag.sum())
            else:
                phi_save = softmax(E_log_theta(gamma_save) + exp_lam, axis=1)
                return phi_save, gamma_save


def M_step(lam: np.array, eta: Union[float, np.array], D: int, phi_t: np.array, n_t: np.array, rho_t: float):
    """
    M step - update of corpus parameter lambda given phi

    :param lam: (K, W)
    :param eta: float or (W, 1)
    :param D: float number of docs in train set
    :param phi_t: (S, K, W) variational topic index distribution
    :param n_t: (S, W, 1) word count array
    :param tau_0: float
    :param t: int
    :param kappa: float
    :return: lam (K, W)
    """
    if isinstance(eta, float):
        pass
    elif isinstance(eta, np.ndarray):
        assert eta.ndim == 2
        eta = eta.squeeze()
    lam_tilde = eta + D * (phi_t * n_t.swapaxes(1, 2)).mean(axis=0)
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





