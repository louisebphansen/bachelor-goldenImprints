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

Install project requirements with GPU support in a conda env
```
conda create -y -n tf-gpu
conda activate tf-gpu
conda install tensorflow-gpu
pip install -r requirements.txt
```

Test that tensorflow sees GPU
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```