from typing import Union, List

import numpy as np
from scipy.special import digamma, polygamma
from metrics.metrics_expectations import E_log_theta, E_log_beta

class NegativeArgumentError(ValueError):
    pass

def update_alpha(alpha: Union[float, np.array], gamma_t: np.array, rho_alpha: List, tol: float = 1e-9):
    """

    :param alpha: float or (K, 1) prior
    :param gamma_t: (S, K, 1)
    :param rho_alpha: (S)
    :param tol:
    :return: (K, 1)
    """
    assert gamma_t.ndim == 3
    alpha_old = alpha

    for s in range(gamma_t.shape[0]):
        gamma = gamma_t[[s], :, :]
        rho_t = rho_alpha[s]

        while True:
            count = 0
            try:
                alpha_new = alpha_old - rho_t * alpha_tilde(alpha_old, gamma)
                if not (alpha_new > 0).all():
                    raise NegativeArgumentError("Linear search went out of bounds !")
                break

            except NegativeArgumentError:
                count += 1
                rho_t = rho_t / 2
                print(f'Halving gradient step: attempt {count}')

    return alpha_new


def update_eta(eta: Union[float, np.array], lam: np.array, rho_t: float):
    eta_old = eta
    while True:
        try:
            eta_new = eta_old - rho_t * eta_tilde(eta_old, lam)
            if not (eta_new > 0).all():
                raise NegativeArgumentError("Linear search went out of bounds !")
            break
        except NegativeArgumentError:
            rho_t = rho_t / 2
    return eta_new


def alpha_tilde(alpha: Union[float, np.array], gamma_d: np.array):
    """
    Inverse of the gradient of $\ell$ times the Hessian

    :param alpha: float or (K, 1)
    :param gamma_d: (1, K, 1)
    :return: (K, 1)
    """
    assert gamma_d.ndim == 3

    if isinstance(alpha, float):
        K = gamma_d.shape[1]
        alpha = alpha * np.ones((K, 1))

    g = grad_ell_alpha(alpha, gamma_d)
    h = h_alpha(alpha, gamma_d)
    c = (g * h).sum(axis=1, keepdims=True) / (z_alpha(alpha) ** - 1 + (1. / h).sum(axis=1, keepdims=True))
    res = (g - c) / h
    return res[0]


def grad_ell_alpha(alpha: Union[float, np.array], gamma_d: np.array):
    """
    Gradient of $\ell$

    :param alpha: (K, 1)
    :param gamma_d: (1, K, 1)
    :return: grad_alpha (1, K, 1)
    """
    return digamma(alpha.sum()) - digamma(alpha) + (alpha - 1.) * E_log_theta(gamma_d)


def h_alpha(alpha: Union[float, np.array], gamma_d: np.array):
    """
    Diagonal terms of the Hessian matrix of $\ell$

    :param alpha: (K, 1)
    :param gamma_d: (1, K, 1)
    :return: h (1, K, 1)
    """
    return E_log_theta(gamma_d) - polygamma(n=1, x=alpha)


def z_alpha(alpha: Union[float, np.array]):
    """
    z value for decomposing Hessian of $\ell$

    :param alpha: (K, 1)
    :return: z (float)
    """
    return float(polygamma(n=1, x=alpha.sum()))


def eta_tilde(eta: Union[float, np.array], lam: np.array):
    """
    Inverse of the gradient of L times the Hessian of L

    :param eta: float or (W, 1)
    :param lam: (K, W)
    :return: (W, 1)
    """
    K, W = lam.shape
    if isinstance(eta, float):
        eta = eta * np.ones((W, 1))

    g = grad_L_eta(eta, lam)
    h = h_eta(eta, lam)
    c = (g * h).sum() / (z_eta(eta, K) ** - 1 + (1. / h).sum())
    res = (g - c) / h
    return res


def grad_L_eta(eta: Union[float, np.array], lam: np.array):
    """
    Gradient of L

    :param eta: (W, 1)
    :param lam: (K, W)
    :return:
    """
    K, W = lam.shape
    return K * (digamma(eta.sum()) - digamma(eta)) + (eta - 1.) * E_log_beta(lam).sum(axis=0)[:, None]


def h_eta(eta: Union[float, np.array], lam: np.array):
    """
    Diagonal terms of the Hessian matrix of L

    :param eta: (W, )
    :param lam: (K, W)
    :return: h (W, 1)
    """
    K = lam.shape[0]
    return E_log_beta(lam).sum(axis=0)[:, None] - K * polygamma(n=1, x=eta.sum())


def z_eta(eta: Union[float, np.array], K: int):
    """
    z value for decomposing Hessian of L

    :param eta: (W, 1)
    :param K: (int) number of topics
    :return: z (float)
    """
    z = K * polygamma(n=1, x=eta.sum())
    return z
