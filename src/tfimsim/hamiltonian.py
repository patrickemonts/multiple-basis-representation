import numpy as np
import logging
import time
from .graph import LatticeGraph
from enum import Enum

from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
from tenpy.algorithms import dmrg
from tenpy.networks.site import SpinHalfSite
from tenpy.algorithms.exact_diag import ExactDiag

import quimb as qu

class EDSimulatorConfig:
    """Configuration for the ED Simulator"""
    def __init__(self):
        self.use_sparse = True

class EDSimulatorQuimb:
    """Exact Diagonalization Simulator using Quimb"""

    def __init__(self, config: EDSimulatorConfig, graph, J: float, h: float):
        self.config = config
        self.graph = graph
        self.J = J
        self.h = h
        self._hamiltonian = None
        self._gs_energy = None
        self._groundstate = None

    @property
    def gs_energy(self):
        if self._gs_energy is None:
            self._gs_energy, self._groundstate = self.compute_gs_energy()
        return self._gs_energy

    @property
    def groundstate(self):
        if self._groundstate is None:
            self._gs_energy, self._groundstate = self.compute_gs_energy()
        return self._groundstate

    @property
    def hamiltonian(self):
        if self._hamiltonian is None:
            self._hamiltonian=self.build_hamiltonian()
        return self._hamiltonian

    def build_hamiltonian(self):
        """Build the Hamiltonian for the transverse field Ising model"""
        dest = qu.gen.operators.ham_heis_2D(self.graph.nx, self.graph.ny, j=(
            self.J, 0, 0), bz=self.h, cyclic=False, sparse=self.config.use_sparse)
        return dest
    
    def compute_gs_energy(self):
        """Compute the groundstate energy using exact diagonalization"""
        ham = self.hamiltonian
        el = qu.eigvalsh(ham, k=1, which="SA")
        # The return value is a tuple, the first element is the energy.
        # We are just returning a tuple to stay consistent with the other sister classes.
        return el, None

class PEPSSimulatorConfig:
    """Configuration for the PEPS Simulator"""
    def __init__(self):
        self.nsteps = 100
        self.schedule = [0.3, 0.1, 0.03, 0.01]
        self.bd_mps_chi = 10
        self.D = 6

class PEPSSimulator:
    """PEPS Simulator using Quimb"""

    def __init__(self, config: PEPSSimulatorConfig, graph: LatticeGraph, J: float, h: float):
        self.config = config
        self.graph = graph
        self.J = J
        self.h = h
        self._hamiltonian = None
        self._gs_energy = None
        self._groundstate = None

    @property
    def gs_energy(self):
        if self._gs_energy is None:
            self._gs_energy, self._groundstate = self.compute_gs_energy()
        return self._gs_energy

    @property
    def groundstate(self):
        if self._groundstate is None:
            self._gs_energy, self._groundstate = self.compute_gs_energy()
        return self._groundstate

    @property
    def hamiltonian(self):
        if self._hamiltonian is None:
            self._hamiltonian=self.build_hamiltonian()
        return self._hamiltonian

    def build_hamiltonian(self):
        """Build the Hamiltonian for the dimer model"""
        import quimb.tensor as qtn
        if self._hamiltonian is None:
            ham_2 = qu.ham_heis(2, j=(1,0,0))
            ham_1 = {None: self.h * qu.spin_operator('Z')}
            self._hamiltonian = qtn.LocalHam2D(self.graph.nx, self.graph.ny, H2=ham_2, H1=ham_1)
        return self._hamiltonian

    def compute_gs_energy(self, seed=None):
        """Compute the groundstate energy using PEPS"""
        import quimb.tensor as qtn
        nx = self.graph.nx
        ny = self.graph.ny
        ham = self.hamiltonian
        if seed is None:
            seed = round(time.time()*1000)
        logging.info(f"Seed for PEPS state: {seed}")
        psi0 = qtn.PEPS.rand(nx, ny, bond_dim=self.config.D, seed=seed)

        su = qtn.SimpleUpdate(
            psi0,
            ham,
            chi=self.config.bd_mps_chi,  # boundary contraction bond dim for computing energy
            compute_energy_every=10,
            compute_energy_per_site=False,
            keep_best=True,
        )

        for tau in self.config.schedule:
            su.evolve(self.config.nsteps, tau=tau)

        return np.real_if_close(su.best["energy"]), su.best["state"]


class MPSInitialState(Enum):
    """Enum to manage the initital state of the MPS"""

    UP = "up"
    DOWN = "down"
    RND_PROD= "rnd_prod"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s:str):
        try:
            return MPSInitialState[s.upper()]
        except KeyError:
            raise ValueError()


class MPSSimulatorConfig:
    """Configuration for the MPS Simulator"""

    def __init__(self):
        # Set some default values
        self.chi_max = 80 # Virtual bond dimension
        self.svd_min = 1.e-10 # Minimum cut-off of the SVD
        self.max_E_err = 1.e-8 # Convergence if the change of the energy in each step satisfies -Delta E / max(|E|, 1) < max_E_err, Default: 1e-8
        self.max_S_err = 1.e-5 # Convergence if the change of the energy in each step satisfies -Delta S / S < max_S_err, Default: 1e-5
        self.max_sweeps = 100
        self.use_mixer = True
        self.initial_state = MPSInitialState.from_string("up")


class MPSSimulator:
    """MPS Simulator using Tenpy"""

    def __init__(self, config: MPSSimulatorConfig, graph: LatticeGraph, J: float, h: float):
        self.config = config
        self.graph = graph
        self.J = J
        self.h = h
        self._hamiltonian = None
        self._gs_energy = None
        self._groundstate = None
    
    @property
    def gs_energy(self):
        if self._gs_energy is None:
            self._gs_energy, self._groundstate = self.compute_gs_energy()
        return self._gs_energy

    @property
    def groundstate(self):
        if self._groundstate is None:
            self._gs_energy, self._groundstate = self.compute_gs_energy()
        return self._groundstate

    @property
    def hamiltonian(self):
        if self._hamiltonian is None:
            self._hamiltonian=self.build_hamiltonian()
        return self._hamiltonian

    def ed_groundstate_from_MPO(self):
        """ Compute the groundstate using exact diagonalization of the MPO"""
        exact_diag = ExactDiag(self.hamiltonian)
        exact_diag.build_full_H_from_mpo()
        exact_diag.full_diagonalization()
        return exact_diag.groundstate()
    
    def build_hamiltonian(self):
        """Build the Hamiltonian for the dimer model"""
        if self._hamiltonian is None:
            # We do not want to use 2d coordinates here, but link indices which yields a pseudo-1d Hamiltonian
            model_params = dict(L=self.graph.size, h=self.h, graph=self.graph,
                                bc_MPS='finite', conserve=None)
            model = TFIModel(model_params)
            self._hamiltonian = model
        return self._hamiltonian

    def build_initial_state(self):
        """Build the initial state for the MPS"""
        if self.config.initial_state==MPSInitialState.UP:
            dest= ["up"] * self.hamiltonian.lat.N_sites
        elif self.config.initial_state==MPSInitialState.DOWN:
            dest= ["down"] * self.hamiltonian.lat.N_sites
        elif self.config.initial_state==MPSInitialState.RND_PROD:
            dest = []
            for _ in range(self.hamiltonian.lat.N_sites):
                dest.append(np.random.choice(["up", "down"]))
        else:
            dest = []
            for _ in range(self.hamiltonian.lat.N_sites):
                dest.append(np.random.choice(["up", "down"]))
            logging.warning("Unkown option in MPSInitialState, defaulting to random product state.")
        return dest

    def compute_gs_energy(self):
        """Compute the groundstate energy using MPS"""
        #Initialization of the state
        initial_state = self.build_initial_state()
        psi = MPS.from_product_state(self.hamiltonian.lat.mps_sites(), initial_state, bc=self.hamiltonian.lat.bc_MPS)
        dmrg_params = {
            'mixer': self.config.use_mixer,  # setting this to True helps to escape local minima
            'max_E_err': self.config.max_E_err, 
            'max_S_err': self.config.max_S_err,
            'max_sweeps': 100,
            'N_sweeps_check': 1,
            'trunc_params': {
                'chi_max': self.config.chi_max,
                'svd_min': self.config.svd_min
            },
            'combine': True
        }

        # Rund DMRG
        info = dmrg.run(psi, self.hamiltonian, dmrg_params) 
        energy = info['E']
        return energy, psi

class TFIModel(CouplingMPOModel):
    r"""Transverse Field Ising Model on an nx x ny lattice.
    The model is not aware of the boundary conditions. 

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`TFIModel` below.

    Options
    -------
    .. cfg:config :: DimerModel
        :include: CouplingMPOModel

        graph : LatticeGraph
            Representation of the graph
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        h : float
            Transverse field strength.
        J : float
            Coupling strength of the Ising term.

    """
    def init_sites(self, model_params):
        sort_charge = model_params.get('sort_charge', None)
        site = SpinHalfSite(conserve=None, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        J = float(model_params.get('J', 1))
        h = float(model_params.get('h', 0.5))
        graph = model_params.get("graph",None)

        #Initialization with separate terms
        for ind,link in enumerate(graph.adj_list):
            # We have to make sure that the link is oriented correctly
            i = min(link)
            j = max(link)
            # The model implemented here is XX + bz Z
            self.add_coupling_term(J, i, j,'Sigmax','Sigmax')
        for ind in range(graph.size):
            self.add_onsite_term(h, ind, 'Sigmaz')
