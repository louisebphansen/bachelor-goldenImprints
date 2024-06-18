Install conda using
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

Initialize conda
```
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

Install project requirements with GPU support in a conda env.
Tell keras to use torch as backend.
```
conda create -y -n torch-gpu
conda activate torch-gpu
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia
pip install datasets scikit_learn timm matplotlib tqdm
export KERAS_BACKEND="torch"
```

Test that GPU is available
```{python}
import torch
torch.cuda.is_available()
```