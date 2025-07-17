# Post-Processing python package for ABP software
This repo provides a python executables to process the structure and dynamics 
of ABP simulations.

## Installation
You can install our package in a dedicated python environment. To do so, you need to possess a Python 3 distribution. 
```
python -m venv postproc
pip install --upgrade pip
pip install -r requirements.txt
```
We recommand to install the module with the edit flag so that it automatically tracks remote updates:
```
git clone https://github.com/NikitaY69/structure-and-dynamics.git
cd structure-and-dynamics/
pip install -e .
```

## Usage
After installation, our package provides a set of executable precisely designed to
extract and compute a vast list of observables from finished ABP simulations. The data is stored for all snapshots. In detail:
- `abp-mov`: Make a movie
- `abp-obs`: Plot observables computed on the fly during the C++ run
- `abp-corr`: Compute 2-dimensional pair correlation functions in the reference frame of particles
- `abp-rdf`: Compute the Radial Distribution Function
- `abp-den`: Compute the local density field
- `abp-clust`: Complete cluster analysis (labels, radius of gyration)

We strongly encourage the user to discover our executables with the --help flag, ie:
```
abp-clust --help
```