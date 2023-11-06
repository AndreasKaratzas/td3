
import sys
sys.path.append('../')

import os
import yaml
import logging
import datetime
import numpy as np

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Tuple

from utils.functions import colorstr


class HardLogger(logging.Logger):
    def __init__(self, output_dir: str = '../data', output_fname: str = None, exp_name: Path = None, demo: bool = False):

        self.datetime_tag = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.export_data_path = Path(output_dir) if output_dir is not None else Path('logs')
        self.name = Path(exp_name + '_' + self.datetime_tag) if exp_name is not None else Path(self.datetime_tag)
        self.logger = logging.getLogger(__name__)

        self.parent_dir = self.export_data_path / self.name if not demo else Path(output_dir).parents[0]
        self.project_path = os.path.abspath(os.path.join(__file__, output_dir))

        self.parent_dir_printable_version = str(os.path.abspath(
            self.parent_dir)).replace(':', '').replace('\\', ' > ')
        self.project_path_printable_version = str(
            self.project_path).replace(':', '').replace('\\', ' > ')

        self.model_dir = self.parent_dir / "model"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = self.parent_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.parent_dir / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.demo_dir = self.parent_dir / "demo"
        self.demo_dir.mkdir(parents=True, exist_ok=True)

        self.plot_dir = self.parent_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.log_f_name = self.log_dir / Path(output_fname + ".log") if output_fname else Path("logger.log")

        if not demo:
            try:
                f = open(self.log_f_name, "x")
                f.close()
            except:
                raise PermissionError(
                    f"Could not create the file {self.log_f_name}.")

            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                datefmt='%Y-%m-%dT%H-%M-%S', filename=self.log_f_name, filemode='w')

    def log_message(self, message):
        self.logger.info(message.rstrip())

    def export_yaml(self, d=dict(), filename='exp'):
        if filename is None:
            filename = 'exp'
        with open(os.path.join(self.log_dir, filename + '.yaml'), 'w') as yaml_file:
            yaml.dump(d, yaml_file, default_flow_style=False)


class ProgressStatus:
    def __init__(self, format: Tuple[str], names: Dict, operators: List):
        self.status = SimpleNamespace(**names)
        self.register = SimpleNamespace(**names)
        self.names = list(self.status.__dict__)
        self.n_elements = len(self.names)
        self.flags = np.zeros(shape=(self.n_elements))
        self.is_first = True
        self.format = format
        self.operators = operators
    
    def set(self, metrics):
        for _metric in self.names:
            self.status.__dict__[_metric] = metrics[_metric]

    def update(self):
        for idx, _metric in enumerate(self.names):
            if self.operators[idx](self.status.__dict__[_metric], self.register.__dict__[_metric]):
                self.register.__dict__[_metric] = self.status.__dict__[_metric]
                self.flags[idx] = 1
            elif self.status.__dict__[_metric] == self.register.__dict__[_metric]:
                self.flags[idx] = 0
            else:
                self.register.__dict__[_metric] = self.status.__dict__[_metric]
                self.flags[idx] = -1
        
    def progress(self):
        self.str = ""

        # fixed metrics
        self.str += f"{self.format[0]}"
        self.str += f"{self.format[1]}"
        self.str += f"{self.format[2]}"
        self.str += f"{self.format[3]}"

        # agent tailored metrics (e.g reward {ddpg, td3}, success {her}, length {ddpg, td3, her}, e.t.c)
        for metric in range(self.n_elements):
            if self.flags[metric] == 1:
                self.str += f"{colorstr(options=['green'], string_args=list([self.format[metric + 4]]))}"
            elif self.flags[metric] == 0:
                self.str += self.format[metric + 4]
            elif self.flags[metric] == -1:
                self.str += f"{colorstr(options=['red'], string_args=list([self.format[metric + 4]]))}"

        # reset progress flags
        self.flags = np.zeros(shape=(self.n_elements))
    
    def mem_desc(self, cuda_mem):
        self.desc = 'M'
        if len(str(cuda_mem)) - 1 > 6:
            cuda_mem = round(cuda_mem / 1E3, 3)
            self.desc = 'G'
    
    def get_metric(self, metrics, idx):
        return list(metrics.items())[idx][1]
    
    def build_msg(self, metrics, show_cuda: bool):
        # initialize message placeholder
        self.msg = tuple()
        
        # fixed metrics
        self.msg = self.msg + (f'{metrics["epoch"] + 1}/{metrics["epochs"]}',)
        self.msg = self.msg + (f'{metrics["bench"]:13.3g}',)
        self.msg = self.msg + (f'{metrics["cuda_mem"] if show_cuda else 0:.3g} ' + self.desc,)
        self.msg = self.msg + (f'{metrics["ram_util"]}',)

        # agent tailored metrics (e.g reward {ddpg, td3}, success {her}, length {ddpg, td3, her}, e.t.c)
        for _metric in range(5, self.n_elements + 5):
            self.msg = self.msg + \
                (f'{self.get_metric(metrics=metrics, idx=_metric):12.3g}',)
        
        # finally, print out the compiled progress message 
        print((self.str) % self.msg)

    def sync_hard_logger(self):
        raw_msg = StringIO()
        print((''.join(map(str, self.format))) % self.msg, file=raw_msg)
        raw_msg = raw_msg.getvalue()
        return raw_msg

    def compile(self, metrics, show_cuda):
        self.set(metrics)
        
        if self.is_first:
            self.is_first = False
            for _metric in self.names:
                self.register.__dict__[_metric] = self.status.__dict__[_metric]
        else:
            self.update()
        
        self.progress()
        self.mem_desc(metrics["cuda_mem"])
        self.build_msg(metrics, show_cuda)

        return self.sync_hard_logger()
