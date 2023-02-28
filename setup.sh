#!/usr/bin/bash

# setting up conda
wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
conda init bash


# getting the source code for both models
git clone https://github.com/oceantracker/oceantracker.git
git clone https://github.com/OpenDrift/opendrift.git

# creating a virtual environment for for oceantracker
# and installing it
conda create --name oceantracker python=3.10
conda activate oceantracker
pip install -r oceantracker/requirements.txt
pip install -e oceantracker/setup.py

# creating a virtual environment for for Opendrift
# and installing it
conda config --add channels conda-forge
conda env create -f opendrift/environment.yml
conda activate opendrift
pip install --no-deps -e ./opendrift