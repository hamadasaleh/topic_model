from typing import Iterable, Union

import numpy as np
from scipy.special import softmax

from tqdm import tqdm

from metrics.metrics_expectations import E_log_theta, E_log_beta


def online_var_bayes(batches: Iterable[np.array], D: int, W: int, K: int, alpha: float, eta: float, tau_0: float, kappa: int) -> np.array:
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

    with tqdm(total=D, position=1, leave=None, desc="TRAIN") as pbar:
        for n_t in batches:
            # parameter optimization
            # E
            phi_t, gamma_t = E_step(lam=lam, alpha=alpha, n_t=n_t)

            # M
            t += n_t.shape[0]
            rho_t = rho(tau_0, t, kappa)
            lam = M_step(lam, eta, D, phi_t, n_t, rho_t)

            # TODO: ELBO: lower bound on log likelihood

            pbar.update(n_t.shape[0])

    return lam


def E_step(lam: np.array, alpha: Union[float, np.array], n_t: np.array, tol: float = 1e-5):
    """
    E step - iterative update of document parameters gamma and phi until convergence

    :param lam: (K, W)
    :param alpha: float
    :param n_t: (S, W, 1)
    :param tol: float
    :return: phi_t (S, K, W), gamma_t (S, K, 1)
    """
    K = lam.shape[0]
    S = n_t.shape[0]
    gamma_save = np.empty((S, K, 1))
    gamma = np.ones_like(gamma_save)

    idx = np.arange(S)
    exp_lam = E_log_beta(lam)[None, :, :]

    with tqdm(total=S, position=2, leave=None, desc="BATCH") as pbar:

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

def M_step(lam, eta, D, phi_t, n_t, rho_t):
    """
    M step - update of corpus parameter lambda given phi

    :param lam: (K, W)
    :param eta: float
    :param D: float number of docs in train set
    :param phi_t: (S, K, W) variational topic index distribution
    :param n_t: (S, W, 1) word count array
    :param tau_0: float
    :param t: int
    :param kappa: float
    :return: lam (K, W)
    """
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
