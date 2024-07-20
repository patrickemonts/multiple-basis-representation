Simulation of the Transverse Field Ising Model
==============================================

This repository contains the code to simulate a two-dimensional transverse field Ising model with the Hamiltonian

$$
H = J \sum_{\langle i,j \rangle} S_{x,i} S_{x,j} + h \sum_i S_{z,i},
$$

where $J$ is the Ising coupling and $h$ is the transverse field. 
The matrices $S_{x,i}$ and $S_{z,i}$ are the spin operators $S_x$ and $S_z$ acting on site $i$.
We follow the condensed matter convention of 

$$
S_x=\frac{1}{2} \sigma_x,\quad 
S_z=\frac{1}{2} \sigma_z,
$$

where $\sigma_x$ and $\sigma_z$ are the Pauli matrices.

## Installation
The code is written for Python 3 and tested to work with Python 3.8.

To make sure that all requirements of the package are fulfilled, the easiest way to use the code is to create a virtual environment and install the `tfimsim` package.

1. Creation of a virtual environment  
You can create the environment in a folder of your choice. 
For the rest of the tutorial, we assume it to be in `~/.pyenv/`
```
cd ~/.pyenv
python -m venv tfimsimenv
```
Assuming you are using bash or zsh, you can activate the environment with `source ~/.pyenv/tfimsimenv/bin/activate`.
Upon activation, you will notice that your prompt changes.
As long as it is prefixed with `(tfimsimenv)` the virtual environment is active.
The virtual environment can be deactivated with `deactivate`.

2. Cloning the code  
You can obtain the code by cloning the repo with
```
git clone git@github.com:patrick.emonts/TBD
```

Note that you currently have to be a member of the project to clone it.
Cloning via SSH works only if you have added a (public) SSH key to the repository.

3. Install the `tfimsim` package
The `tfimsim` package is still in the development phase.
We recommend an editable install, since the main reason to install the package is development.
Execute the following command while standing in the root directory of the repository
```
pip install -e .
```

Pip installs all dependencies as specified in the `pyproject.toml` and yields a working environment.

## Structure for the Code

The repository is split into two main parts: the package `tfimsim` (at `src/tfimsim`) and utility scripts in the main folder.

The package `tfimsim` contains the simulation code, i.e. the actual implementation of the physical problem, the transverse field Ising model.
All scripts in the main folder call parts of the package and provide the infrastructure to manage the simulations.

The package `tfimsim` is divided into several modules:
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
