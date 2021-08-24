import pyro.distributions as dist


class Gamma(dist.Gamma):
    r"""
    This class adds support for converting mu/sigma Gamma distribution parametrisation into alpha/beta

    :param mu: mean of Gamma distribution
    :param sigma: variance of Gamma distribution
    :param alpha: shape parameter of Gamma distribution (mu ** 2 / sigma ** 2)
    :param beta: rate parameter of Gamma distribution (mu / sigma ** 2)
    """

    def __init__(self, mu=None, sigma=None, alpha=None, beta=None, **kwargs):
        if alpha is not None and beta is not None:
            pass
        elif mu is not None and sigma is not None:
            alpha = mu ** 2 / sigma ** 2
            beta = mu / sigma ** 2
        else:
            raise ValueError("Define (mu and var) or (alpha and beta).")

        super().__init__(alpha, beta, **kwargs)
