
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

OPERATORS_STR = [
    "greater",
    "greater",
    "greater",
    "lower",
    "lower"
]

OPERATORS = [
    lambda a, b: a > b,
    lambda a, b: a > b,
    lambda a, b: a > b,
    lambda a, b: a < b,
    lambda a, b: a < b
]
