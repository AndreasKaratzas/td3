
# TD3

### Abstract


### Introduction


### Installation

```bash
conda create -n td3 python=3.9 pip
conda activate td3
conda install swig
pip install "gymnasium[all]"
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install numpy
pip install pyyaml
conda install pandas
conda install matplotlib
pip install psutil
pip install torcheval
pip install chardet
```

### Usage

```bash
python main.py --env 'MountainCarContinuous-v0' --max-ep-len 1000 --hidden-sizes 256 256 256 --learning-starts 1000 --device 'cuda' --debug-mode --name 'MountainCarContinuous-v0' --auto-save --info --logger-name 'MountainCarContinuous-v0' --checkpoint-dir 'data/experiments' --batch-size 256
```

### Demo


### Experiments

Table with pretrained model stats

### Acknowledgement

- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/index.html)

### Future Work

- [ ] Add `tensorboard`
- [ ] Add `docstrings`
- [ ] Experiments
- [ ] Include pretrained models
- [ ] Include examples
- [ ] Complete `README`
