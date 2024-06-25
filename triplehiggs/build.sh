#!/bin/bash

# Source the conda environment setup script
source /lustre/collider/wangxinzhu/conda_xinzhu.env

# Specify the name of the virtual environment
environment_name="pytorch_trihiggs"

new_env_path="/lustre/collider/wangxinzhu/miniconda3/envs/$environment_name"
# Create a virtual environment
conda create --prefix "$new_env_path" python=3.8.16 

# Activate the virtual environment
conda activate "$new_env_path"

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# List of packages to install using conda
conda_packages=("numpy==1.21.6" "pandas==1.3.5" "matplotlib==3.5.2" "scikit-learn==1.0.2" "h5py==3.7.0" "scipy==1.7.3")

# List of packages to install using pip
pip_packages=("awkward0==0.15.5" "cachetools==5.2.0" "cycler==0.11.0" "EnergyFlow==1.3.2" "fonttools==4.33.3" "joblib==1.1.0" 
              "kiwisolver==1.4.2" "lz4==4.0.1" "Pillow==9.1.1" "pip==22.1.2" "pyparsing==3.0.9" "python-dateutil==2.8.2" 
              "pytz==2022.1" "setuptools==62.3.3" "six==1.16.0" "threadpoolctl==3.1.0" "torch==1.11.0+cu113" 
              "typing_extensions==4.2.0" "uproot==4.2.3" "uproot3==3.14.4" "uproot3-methods==0.10.1" "Wasserstein==1.0.1" 
              "wheel==0.37.1" "wurlitzer==3.0.2" "xxhash==3.0.0")

# Install specified packages using conda
conda install "${conda_packages[@]}" -y
# Install pytorch_lightning don't use conda
# Install specified packages using pip
pip install "${pip_packages[@]}"
conda install scikit-learn
conda config --add channels conda-forge
conda install uproot
conda install tqdm
# Display information
echo "Virtual environment '$environment_name' has been created, and packages including PyTorch have been installed."
echo "Activate the environment using 'conda activate $new_env_path'."
