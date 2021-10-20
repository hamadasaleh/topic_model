from scipy.special import loggamma, digamma
import numpy as np
from numpy import errstate, isneginf

def E_log_beta(lam):
    """
    Expectation of log topics
    :param lam: (K, W) array
    :return: (K, W) array
    """
    return digamma(lam) - digamma(lam.sum(axis=1))[:, None]

def E_log_theta(gamma_d):
    """
    Expectation of log topic proportions
    :param gamma_d: (K, 1) array
    :return: (K, 1)
    """
    if gamma_d.ndim == 1:
        gamma_d = gamma_d[:, None]
    return digamma(gamma_d) - digamma(gamma_d.sum())

def E_log_p_beta(eta, lam):
    """
    Expectation of log of topic distribution
    :param eta: (float)
    :param lam: (K, W) array
    :return: (float)
    """
    K, W = lam.shape
    return K * (-W * loggamma(eta) + loggamma(W * eta)) + (eta - 1) * E_log_beta(lam).sum()

def E_log_p_theta(alpha, gamma_d):
    """
    Expectation of log of topic proportion distribution
    :param alpha: (float)
    :parama gamma_d: (K, 1) array
    :return: (float)
    """
    K = len(gamma_d)
    return loggamma(K * alpha) - K * loggamma(alpha) + (alpha - 1) * E_log_theta(gamma_d).sum()

def E_log_p_z(n_d, phi_d, gamma_d):
    """
    Expectation of log of topic index distribution
    :param n_d: (W, )
    :param phi_d: (W, K)
    :param gamma_d: (K, 1)
    :return: (float)
    """
    return float(n_d @ phi_d.T @ E_log_theta(gamma_d))

def E_log_p_w(n_d, phi_d, lam_d):
    """
    Expectation of log of word distribution
    :param n_d: (W, )
    :param phi_d: (W, K)
    :param lam_d: (K, W)
    :return: (float)
    """
    return n_d @ (phi_d * E_log_beta(lam_d)).sum(axis=0)

def E_log_q_beta(lam):
    """
    Expectation of log of variational topic
    :param lam: (K, W) array
    :return: (float)
    """
    return loggamma(lam.sum(axis=1)).sum() - loggamma(lam).sum() + ((lam - 1.) * E_log_beta(lam)).sum()

def E_log_q_theta(gamma_d):
    """
    Expectation of log of variational topic proportion
    :param gamma_d: (K, 1) array
    :return: (float)
    """
    return float(loggamma(gamma_d.sum()) - loggamma(gamma_d).sum() + (gamma_d - 1) @ E_log_theta(gamma_d))

def E_log_q_z(n_d, phi_d):
    """
    Expectation of log of variational topic index distribution
    :param n_d: (W, )
    :param phi_d: (K, W)
    :return: (float)
    """
    with errstate(divide='ignore'):
        log_phi = np.log(phi_d)
        log_phi[isneginf(log_phi)] = 0

    return n_d @ (phi_d * log_phi).sum(axis=0)


