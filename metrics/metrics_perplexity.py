from math import exp
from typing import List

from metrics.metrics_expectations import *

def perplexity(diff: np.array, n_sum: List[int]) -> \
        float:
    """

    :param diff: (S, 1)
    :param n_d: (S, W, 1)
    :return:
    """
    metric = exp(- diff.sum() / sum(n_sum))
    return metric

def doc_diff(n_d: np.array, gamma_d: np.array, phi_d: np.array, lam: np.array, alpha: Union[float, np.array]):
    diff = E_log_p_n_theta_z(n_d, gamma_d, phi_d, lam, alpha) - E_log_q_theta_z(n_d, gamma_d, phi_d)
    return diff

def E_log_p_n_theta_z(n_d: np.array, gamma_d: np.array, phi_d: np.array, lam: np.array, alpha: Union[float, np.array]):
    """

    :param n_d: (S, W, 1)
    :param gamma_d: (S, K, 1)
    :param phi_d: (S, K, W)
    :param lam: (K, W)
    :param alpha: float
    :return: np.array (S, 1)
    """
    return E_log_p_theta(alpha, gamma_d) + E_log_p_z(n_d, phi_d, gamma_d) + E_log_p_w(n_d, phi_d, lam)

def E_log_q_theta_z(n_d: np.array, gamma_d: np.array, phi_d: np.array) -> np.array:
    """

    :param n_d: (S, W, 1)
    :param gamma_d: (S, K, 1)
    :param phi_d: (S, K, W)
    :return: (S, 1)
    """
    return E_log_q_theta(gamma_d) + E_log_q_z(n_d, phi_d)