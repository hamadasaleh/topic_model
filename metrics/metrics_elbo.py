from metrics.metrics_expectations import *

def ell(n_d, phi_d, gamma_d, lam, alpha, eta, D):
    """
    Contribution of document d to the ELBO

    :param n_d: (W, )
    :param phi_d: (W, K)
    :param gamma_d: (K, 1)
    :param lam: (K, W)
    :param alpha: (float)
    :param eta: (float)
    :param D: (int)
    :return: (float)
    """
    res = E_log_p_w(n_d, phi_d, lam) + E_log_p_z(n_d, phi_d, gamma_d) - E_log_q_z(n_d, phi_d) +\
          E_log_p_theta(alpha, gamma_d) - E_log_q_theta(gamma_d) +\
          (E_log_p_beta(eta, lam) - E_log_q_beta(lam))/D
    return res

