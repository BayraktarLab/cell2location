import functools
import math
import numbers
import operator
import weakref
from typing import List

import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.utils import (_sum_rightmost, broadcast_all,
                                       lazy_property, tril_matrix_to_vec,
                                       vec_to_tril_matrix)
from torch.nn.functional import pad
from torch.nn.functional import softplus

from torch.distributions.transforms import Transform

class SoftplusTransform(Transform):
    r"""
    Transform via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    """
    domain = constraints.real_vector
    codomain = constraints.positive
    
    def __eq__(self, other):
        return isinstance(other, SoftplusTransform)

    def _call(self, x):
        return (1+x.exp()).log()

    def _inverse(self, y):
        return y.expm1().log()
      
    def log_abs_det_jacobian(self, x, y):
        return -(1+(-x).exp()).log()
