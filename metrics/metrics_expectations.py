from typing import Union

from scipy.special import loggamma, digamma
import numpy as np
from numpy import errstate, isneginf

def E_log_beta(lam: np.array) -> np.array:
    """
    Expectation of log topics
    :param lam: (K, W) array
    :return: (K, W) array
    """
    assert lam.ndim == 2
    return digamma(lam) - digamma(lam.sum(axis=1))[:, None]

def E_log_theta(gamma_d: np.array) -> np.array:
    """
    Expectation of log topic proportions
    :param gamma_d: (S, K, 1) array
    :return: (S, K, 1)
    """
    assert gamma_d.ndim == 3
    return digamma(gamma_d) - digamma(gamma_d.sum(axis=1, keepdims=True))

def E_log_p_beta(eta: np.array, lam: np.array) -> float:
    """
    Expectation of log of topic distribution
    :param eta: (W, 1)
    :param lam: (K, W)
    :return: (float)
    """
    assert eta.ndim == 2, lam.ndim == 2
    K, W = lam.shape
    eta = eta.squeeze()

    return K * (-loggamma(eta).sum() + loggamma(eta.sum())) + ((eta - 1.) * E_log_beta(lam)).sum()

def E_log_p_theta(alpha: np.array, gamma_d: np.array) -> np.array:
    """
    Expectation of log of topic proportion distribution
    :param alpha: (K, 1)
    :parama gamma_d: (S, K, 1) array
    :return: np.array (S, 1)
    """
    assert alpha.ndim == 2, gamma_d.ndim == 3

    return loggamma(alpha.sum()) - loggamma(alpha).sum() + ((alpha - 1.) * E_log_theta(gamma_d)).sum(axis=1)

def E_log_p_z(n_d: np.array, phi_d: np.array, gamma_d: np.array) -> np.array:
    """
    Expectation of log of topic index distribution

    :param n_d: (S, W, 1)
    :param phi_d: (S, K, W)
    :param gamma_d: (S, K, 1)
    :return: (S, 1)
    """
    assert n_d.ndim == phi_d.ndim == gamma_d.ndim == 3

    return (n_d.swapaxes(1,2) @ phi_d.swapaxes(1, 2) @ E_log_theta(gamma_d)).squeeze(-1)

def E_log_p_w(n_d: np.array, phi_d: np.array, lam: np.array) -> np.array:
    """
    Expectation of log of word distribution
    :param n_d: (S, W, 1)
    :param phi_d: (S, K, W)
    :param lam: (K, W)
    :return: (S, 1)
    """
    assert n_d.ndim == phi_d.ndim == 3
    assert lam.ndim == 2

    return (n_d * (phi_d * E_log_beta(lam)).sum(axis=1)[:, :,None]).sum(axis=1)

def E_log_q_beta(lam: np.array) -> float:
    """
    Expectation of log of variational topic
    :param lam: (K, W) array
    :return: (float)
    """
    assert lam.ndim == 2
    return (loggamma(lam.sum(axis=1)) + ((lam - 1.) * E_log_beta(lam) - loggamma(lam)).sum(axis=1)).sum()

def E_log_q_theta(gamma_d: np.array) -> np.array:
    """
    Expectation of log of variational topic proportion
    :param gamma_d: (S, K, 1) array
    :return: np.array (S, 1)
    """
    assert gamma_d.ndim == 3
    return loggamma(gamma_d.sum(axis=1)) + ((gamma_d - 1.) * E_log_theta(gamma_d) - loggamma(gamma_d)).sum(axis=1)

def E_log_q_z(n_d: np.array, phi_d: np.array) -> np.array:
    """
    Expectation of log of variational topic index distribution
    :param n_d: (S, W, 1)
    :param phi_d: (S, K, W)
    :return: (S, 1)
    """
    assert n_d.ndim == phi_d.ndim == 3

    with errstate(divide='ignore'):
        log_phi = np.log(phi_d)
        log_phi[isneginf(log_phi)] = 0

    return (n_d * (phi_d * log_phi).sum(axis=1)[:, :,None]).sum(axis=1)


