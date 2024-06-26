# Uncertainty Quantification for AMPL Point Predictions

## Overview
This repository contains the routines and functions used for the MSSE capstone project. The goal of this project was to build and compare various classification models’ and their ability to classify NEK2 binders and inhibitors. This project, was guided by the Open Data and Models group at the Accelerating Therapeutics for Opportunities in Medicine (ATOM) Consortium. The data is provided by Chembl, but it is not included here as an extra precaution. 

## Installation
To set up this project locally, follow the steps below:
```bash
git clone https://github.com/jaycee-pang/nek2_final
cd nek2_final
```
To install and activate the conda virtual environment: 
```bash
install conda
conda env create -f environment.yaml
conda activate nek2_final
python -m ipykernel install --user --name nek2_final 
``` 
To deactivate: 
```bash
conda deactivate
```
## Directory 
Utility functions used throughout this project are located in the root directory. These are Python scripts including: 
- `utils.py`: for molecular normalization 
- `RF_Utils.py`: building RF models and evaulating metrics 
- `split_data.py`: for creating random splits 
- `VisUtils.py`: plotting functions

the nek2_final directory contains the notebooks to run. 
In this directory, there are 3 subdirectories: models, data, and notebooks. You should add the data directory here. The models directory is where the models will be saved to. The data directory contains folders for binding and inhibition data separately. The notebooks is the main folder where the splits are generated, models are built, trained, and tested, and visualizations are produced.

## Contact
- **Jaycee Pang** - jayceempang@gmail.com
