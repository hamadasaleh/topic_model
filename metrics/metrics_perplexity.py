import math

from metrics.metrics_expectations import *

def perplexity_proxy(E_diffs, count_sums):
    """
    Perplexity proxy used as a measure of fit on held-out set.

    :param diffs: list of expectation differences for each test document
    :param sums: list of sums of token counts for each test document
    :return: perplexity proxy
    """
    return math.exp(- sum(E_diffs) / sum(count_sums))

def doc_proxy_values(n_d, gamma_d, phi_d, lam, alpha):
    """
    Get per document values for computing perplexity proxy

    :param n_d: (S, W, 1) token count array
    :param gamma_d: (S, K, 1) parameter for posterior over per-document topic weights
    :param phi_d: (S, K, W) parameter for posterior over per-word topic assignments
    :param lam: (K, W) parameter for posterior over the topics
    :param alpha: (float) concentration parameter for dirichlet distribution over topics
    :return: E_diff (float), counts_sum (float)
    """
    E_diff = expectations_diff(n_d, gamma_d, phi_d, lam, alpha)
    counts_sum = n_d.sum(axis=1)
    return E_diff, counts_sum

def expectations_diff(n_d, gamma_d, phi_d, lam, alpha):
    a = E_log_p_n_theta_z(n_d, gamma_d, phi_d, lam, alpha)
    b = E_log_q_theta_z(n_d, gamma_d, phi_d)
    return a - b

def E_log_p_n_theta_z(n_d, gamma_d, phi_d, lam, alpha):
    """

    :param n_d: (S, W, 1)
    :param gamma_d: (S, K, 1)
    :param phi_d: (S, K, W)
    :param lam: (K, W)
    :param alpha: float
    :return: np.array (S, )
    """
    return E_log_p_theta(alpha, gamma_d) + E_log_p_z(n_d, phi_d, gamma_d) + E_log_p_w(n_d, phi_d, lam)

def E_log_q_theta_z(n_d, gamma_d, phi_d):
    """

    :param n_d: (W, 1)
    :param gamma_d: (K, 1)
    :param phi_d: (K, W)
    :return: float
    """
    return E_log_q_theta(gamma_d) + E_log_q_z(n_d, phi_d)