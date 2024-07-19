import unittest
import numpy as np
from tfimsim.hamiltonian import PEPSSimulator, MPSSimulator, MPSSimulatorConfig, PEPSSimulatorConfig, EDSimulatorConfig,EDSimulatorQuimb
from tfimsim.graph import LatticeGraph, Boundary

class TestEDQuimbSimulator(unittest.TestCase):
    def setUp(self):
        self.N = 3
        #These two dimers come from the same class
        self.graph = LatticeGraph(self.N, self.N, Boundary.OBC)
        self.config = EDSimulatorConfig()

    def test_full_herm(self):
        for i in range(100):
            J,h = np.random.rand(2)
            eds1 = EDSimulatorQuimb(self.config, self.graph, J,h)
            self.assertTrue(np.allclose(eds1.hamiltonian.H.A,eds1.hamiltonian.A))


class TestPEPSSimulator(unittest.TestCase):

    def setUp(self):
        self.N = 3
        #These two dimers come from the same class
        self.graph = LatticeGraph(self.N, self.N, Boundary.OBC)
        self.J = -1
        self.peps_config = PEPSSimulatorConfig()
        self.peps_config.D = 3
        self.ed_config = EDSimulatorConfig()

    def test_gs_energy_obc_ed_quimb(self):
        h = 0.34
        mpssim = PEPSSimulator(self.peps_config, self.graph, self.J, h)
        mps_gs_energy = mpssim.gs_energy

        edsim = EDSimulatorQuimb(self.ed_config, self.graph, self.J, h)
        ed_gs_energy = edsim.gs_energy
        rel_error = np.abs((mps_gs_energy-ed_gs_energy)/ed_gs_energy)
        self.assertLess(rel_error,0.001)


class TestMPSSimulator(unittest.TestCase):

    def setUp(self):
        self.N = 3
        #These two dimers come from the same class
        self.graph = LatticeGraph(self.N, self.N, Boundary.OBC)
        self.mps_config = MPSSimulatorConfig()
        self.ed_config= EDSimulatorConfig()
        self.J = -1

    def test_gs_energy_obc_ed_quimb(self):
        h = 0.34
        mpssim = MPSSimulator(self.mps_config, self.graph, self.J, h)
        mps_gs_energy = mpssim.gs_energy

        edsim = EDSimulatorQuimb(self.ed_config, self.graph, self.J, h)
        ed_gs_energy = edsim.gs_energy
        rel_error = np.abs((mps_gs_energy-ed_gs_energy)/ed_gs_energy)
        self.assertLess(rel_error,0.001)


    def test_gs_energy_pbc_ed_mpo(self):
        h = 0.39
        mpssim = MPSSimulator(self.mps_config, self.graph, self.J, h)
        mps_gs_energy = mpssim.gs_energy

        ed_gs_energy, ed_gs = mpssim.ed_groundstate_from_MPO()
        self.assertAlmostEqual(mps_gs_energy,ed_gs_energy,places=4)