
import sys
sys.path.append('../')

import re
import gymnasium as gym
import yaml
import torch
import random
import numpy as np
import torch.nn as nn

from typing import Iterable
from itertools import zip_longest
from collections import defaultdict


def str2activation(activation: str):
    if activation.lower() == 'relu':
        activation = nn.ReLU
    elif activation.lower() == 'sigmoid':
        activation = nn.Sigmoid
    elif activation.lower() == 'tanh':
        activation = nn.Tanh
    else:
        raise NotImplementedError(f"Activation function {activation} is currently not supported. "
                                  f"Try one of the following:\n\t1. Sigmoid\n\t2. ReLU\n\t3. Tanh")
    return activation


def activation2str(activation: nn.Module):
    if activation == nn.ReLU:
        activation = 'relu'
    elif activation == nn.Sigmoid:
        activation = 'sigmoid'
    elif activation == nn.Tanh:
        activation = 'tanh'
    else:
        raise NotImplementedError(f"Activation function {activation} is currently not supported. "
                                  f"Try one of the following:\n\t1. Sigmoid\n\t2. ReLU\n\t3. Tanh")
    return activation


def _seed(env: gym.Env, device: torch.device, seed: int = 0):
    # set random seeds for reproduce
    env.seed(seed)
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if 'cuda' in device.type:
        torch.cuda.manual_seed(seed)
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def polyak_update(
    params: Iterable[torch.nn.Parameter],
    target_params: Iterable[torch.nn.Parameter],
    tau: float,
) -> None:
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data,
                      alpha=tau, out=target_param.data)


def parse_configs(filepath: str):
    with open(filepath, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)


def envs():
    _game_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)

    # reading benchmark names directly from retro requires
    # importing retro here, and for some reason that crashes tensorflow
    # in ubuntu
    _game_envs['retro'] = {
        'BubbleBobble-Nes',
        'SuperMarioBros-Nes',
        'TwinBee3PokoPokoDaimaou-Nes',
        'SpaceHarrier-Nes',
        'SonicTheHedgehog-Genesis',
        'Vectorman-Genesis',
        'FinalFight-Snes',
        'SpaceInvaders-Snes',
    }

    return _game_envs


def get_env_type(env: str, game_envs: defaultdict, env_type: str = None):
    env_id = env

    if env_type is not None:
        return env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in game_envs.keys():
        env_type = env_id
        env_id = [g for g in game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(
            env_id, game_envs.keys())

    return env_type, env_id


def get_default_network(env_type: str) -> str:
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def update_args(args, settings):
    args_dict = vars(args)
    flattened_settings = dict(recursive_items(settings))
    for key, _ in recursive_items(args_dict):
        if key in flattened_settings:
            args.__dict__[key] = flattened_settings[key]
    return args


def colorstr(options, string_args):
    """Usage:
    
    >>> args = ['Andreas', 'Karatzas']
    >>> print(
    ...    f"My name is {colorstr(options=['red', 'underline'], string_args=args)} "
    ...    f"and I like {colorstr(options=['bold', 'cyan'], string_args=list(['Python']))} "
    ...    f"and {colorstr(options=['cyan'], string_args=list(['C++']))}\n")
    Parameters
    ----------
    options : [type]
        [description]
    string_args : [type]
        [description]
    Returns
    -------
    [type]
        [description]
    """
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code
    colors = {'black':          '\033[30m',  # basic colors
              'red':            '\033[31m',
              'green':          '\033[32m',
              'yellow':         '\033[33m',
              'blue':           '\033[34m',
              'magenta':        '\033[35m',
              'cyan':           '\033[36m',
              'white':          '\033[37m',
              'bright_black':   '\033[90m',  # bright colors
              'bright_red':     '\033[91m',
              'bright_green':   '\033[92m',
              'bright_yellow':  '\033[93m',
              'bright_blue':    '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan':    '\033[96m',
              'bright_white':   '\033[97m',
              'end':            '\033[0m',  # miscellaneous
              'bold':           '\033[1m',
              'underline':      '\033[4m'}
    res = []
    for substr in string_args:
        res.append(''.join(colors[x] for x in options) +
                   f'{substr}' + colors['end'])
    space_char = ''.join(colors[x] for x in options) + ' ' + colors['end']
    return space_char.join(res)


def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo
