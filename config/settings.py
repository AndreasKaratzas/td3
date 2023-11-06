
import sys
sys.path.append('../')

import numpy as np


FORMAT = ('%9s', '%13s', '%9s',
          '%9s', '%12s', '%12s', 
          '%12s', '%12s', '%12s')

NAMES = [
    "length",
    "reward",
    "q_val",
    "loss_pi",
    "loss_q"
]

NAMES_DICT = {
    "length": np.inf,
    "reward": -np.inf,
    "q_val": -np.inf,
    "loss_pi": np.inf,
    "loss_q": np.inf
}

OPERATORS_STR = [
    "lower",
    "greater",
    "greater",
    "lower",
    "lower"
]

OPERATORS = [
    lambda a, b: a < b,
    lambda a, b: a > b,
    lambda a, b: a > b,
    lambda a, b: np.abs(a) < np.abs(b),
    lambda a, b: a < b
]
