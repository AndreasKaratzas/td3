
import sys
sys.path.append('../')


def mse_loss(input, target):
    return ((input - target)**2).mean()


def weighted_mse_loss(input, target, weight):
    return ((weight * (input - target))**2).mean()
