import numpy as np
from scipy.special import digamma, polygamma
from metrics.metrics_expectations import E_log_theta, E_log_beta


def alpha_tilde(alpha, gamma_d):
    """
    Inverse of the gradient of $\ell$ times the Hessian

    :param alpha: (K, )
    :param gamma_d: (K, )
    :return: (K, )
    """
    if isinstance(alpha, float):
        K = gamma_d.shape[0]
        alpha = alpha * np.ones((K, 1))

    g = grad_ell_alpha(alpha, gamma_d)
    h = h_alpha(alpha, gamma_d)
    c = (g * h).sum() / (z_alpha(alpha) ** - 1 + (1. / h).sum())
    res = (g - c) / h
    return res


def grad_ell_alpha(alpha, gamma_d):
    """
    Gradient of $\ell$

    :param alpha: (K, 1)
    :param gamma_d: (K, 1)
    :return: grad_alpha (K, 1)
    """
    return digamma(alpha.sum()) - digamma(alpha) + (alpha - 1.) * E_log_theta(gamma_d)


def h_alpha(alpha, gamma_d):
    """
    Diagonal terms of the Hessian matrix of $\ell$

    :param alpha: (K, 1)
    :param gamma_d: (K, 1)
    :return: h (K, 1)
    """
    return E_log_theta(gamma_d) - polygamma(n=1, x=alpha)


def z_alpha(alpha):
    """
    z value for decomposing Hessian of $\ell$

    :param alpha: (K, 1)
    :return: z (float)
    """
    return polygamma(n=1, x=alpha.sum())


def eta_tilde(eta, lam):
    """
    Inverse of the gradient of L times the Hessian of L

    :param eta: (W, )
    :param lam: (K, W)
    :return: (W, 1)
    """
    K, W = lam.shape
    if isinstance(eta, float):
        eta = eta * np.ones((W, 1))

    g = grad_L_eta(eta, lam)
    assert all(g>0)
    h = h_eta(eta, lam)
    assert all(h<0)
    c = (g * h).sum() / (z_eta(eta, K) ** - 1 + (1. / h).sum())
    res = (g - c) / h
    return res


def grad_L_eta(eta, lam):
    """
    Gradient of L

    :param eta: (W, 1)
    :param lam: (K, W)
    :return:
    """
    K, W = lam.shape
    return K * (digamma(eta.sum()) - digamma(eta)) + (eta - 1.) * E_log_beta(lam).sum(axis=0)[:, None]


def h_eta(eta, lam):
    """
    Diagonal terms of the Hessian matrix of L

    :param eta: (W, )
    :param lam: (K, W)
    :return: h (W, 1)
    """
    K = lam.shape[0]
    return E_log_beta(lam).sum(axis=0)[:, None] - K * polygamma(n=1, x=eta.sum())


def z_eta(eta, K):
    """
    z value for decomposing Hessian of L

    :param eta: (W, 1)
    :param K: (int) number of topics
    :return: z (float)
    """
    z = K * polygamma(n=1, x=eta.sum())
    return z
