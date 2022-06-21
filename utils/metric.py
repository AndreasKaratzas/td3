
import sys
sys.path.append('../')

import numpy as np


class Metric:
    def __init__(
        self,
        auto_update: bool = True,
        update_with_reset: bool = False,
        k_first: int = None,
        k_last: int = None,
        name: str = '',
        selection_metric: str = 'avg',
        comp_operator: str = 'greater'
    ):
        self.auto_update = auto_update
        self.update_with_reset = update_with_reset
        self.k_first = k_first
        self.k_last = k_last
        self.name = name.lower()

        self.selection_metric = selection_metric.lower()
        self.comp_operator = comp_operator.lower()

        assert self.selection_metric in ['min', 'max', 'avg', 'mean']
        assert self.comp_operator in ['lower', 'lower_eq', 'greater', 'greater_eq']

        self.reset(reset_elite=True)

    def reset(self, reset_elite: bool = False):
        self.min = np.inf
        self.max = -np.inf
        self.avg = np.nan
        self.reg = []

        if reset_elite:
            self.elite = None
    
    def operator(self):
        if self.comp_operator == 'greater':
            return lambda a, b: a > b
        if self.comp_operator == 'less':
            return lambda a, b: a < b
        if self.comp_operator == 'greater_eq':
            return lambda a, b: a >= b
        if self.comp_operator == 'less_eq':
            return lambda a, b: a <= b
        raise NotImplementedError()
    
    def select(self):
        if self.selection_metric in ['avg', 'mean']:
            return self.avg_fn
        if self.selection_metric == 'min':
            return self.min_fn
        if self.selection_metric == 'max':
            return self.max_fn
        raise NotImplementedError()
    
    def add(self, value):
        self.reg.append(value)

        if self.auto_update:
            self.update()
    
    def attr_fn(self, operator_fn) -> float:
        if self.k_first is not None:
            if self.k_first < len(self.reg):
                return operator_fn(self.reg[:self.k_first])
        if self.k_last is not None:
            if self.k_last < len(self.reg):
                return operator_fn(self.reg[self.k_last:])
        return operator_fn(self.reg)

    def min_fn(self):
        self.min = self.attr_fn(np.min)

    def max_fn(self):
        self.max = self.attr_fn(np.max)

    def avg_fn(self):
        self.avg = self.attr_fn(np.mean)

    def update(self):
        self.min_fn()
        self.max_fn()
        self.avg_fn()

        if self.update_with_reset:
            self.reset()

    def status(self, reset_elite: bool = False, status_reset: bool = False) -> bool:
        if self.elite is None: 
            self.elite = self.select()
            return True
        
        register = self.select()
        if self.operator(register, self.elite):
            self.elite = register
            return True
        
        if reset_elite:
            self.elite = None
        
        if status_reset:
            self.reset(reset_elite=reset_elite)
        
        return False
