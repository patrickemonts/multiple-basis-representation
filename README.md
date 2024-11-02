Simulation of the Transverse Field Ising Model
==============================================

This repository contains the code to simulate a two-dimensional transverse field Ising model with the Hamiltonian

$$
H = J \sum_{\langle i,j \rangle} X_{i} X_{j} + h \sum_i Z_{i},
$$

where $J$ is the Ising coupling and $h$ is the transverse field. 
The matrices $X_i$ and $Z_i$ are the respective Pauli matrices acting on site $i$.

## Installation
The code is written for Python 3 and tested to work with Python 3.8.

To make sure that all requirements of the package are fulfilled, the easiest way to use the code is to create a virtual environment and install the `mbrsim` package.

1. Creation of a virtual environment  
You can create the environment in a folder of your choice. 
For the rest of the tutorial, we assume it to be in `~/.pyenv/`
```
cd ~/.pyenv
python -m venv mbrsimenv
```
Assuming you are using bash or zsh, you can activate the environment with `source ~/.pyenv/mbrsimenv/bin/activate`.
Upon activation, you will notice that your prompt changes.
As long as it is prefixed with `(mbrsimenv)` the virtual environment is active.
The virtual environment can be deactivated with `deactivate`.

2. Cloning the code  
You can obtain the code by cloning the repo with
```
git clone git@github.com:patrick.emonts/TBD
```

Note that you currently have to be a member of the project to clone it.
Cloning via SSH works only if you have added a (public) SSH key to the repository.

3. Install the `mbrsim` package
The `mbrsim` package is still in the development phase.
We recommend an editable install, since the main reason to install the package is development.
Execute the following command while standing in the root directory of the repository
```
pip install -e .
```

Pip installs all dependencies as specified in the `pyproject.toml` and yields a working environment.

## Structure for the Code

The repository is split into two main parts: the package `mbrsim` (at `src/mbrsim`) and utility scripts in the main folder.

The package `mbrsim` contains the simulation code, i.e. the actual implementation of the physical problem, the transverse field Ising model.
All scripts in the main folder call parts of the package and provide the infrastructure to manage the simulations.

The package `mbrsim` is divided into several modules:
- `graph.py`: Build a graph representation of a given lattice
- `hamiltonian.py`: Simulation tools for the TFIM with different techniques (exact diagonalization, PEPS, MPS)
- `utils.py`: Utility functions for data management and debugging

## The Manager

The script `manager.py` is the main script that manages all simulations.
The `manager.py` can start different simulations.
Currently, there are three supported simulations: MPS, PEPS and exact diagonalization (ED).

An example call for a 2x2 lattice with open boundary conditions, solved with ED looks like
```
python manager.py --nx 2 --ny 2 --hmin 0.2 --hmax 1.0 --nsteps 9 --type ed
```
The call executes 9 simulations from $h=0.2$ until $h=1.0$ with an Ising coupling of $J=1$.


### Options

A full list of all options can be found with
```
python manager.py --help
```

## Future directions and known issues

- TBD
