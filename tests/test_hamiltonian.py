import unittest
import numpy as np
from tfimsim.hamiltonian import PEPSSimulator, MPSSimulator, MPSSimulatorConfig, PEPSSimulatorConfig, EDSimulatorConfig, EDSimulatorQuimb
from tfimsim.graph import LatticeGraph, Boundary


class TestEDQuimbSimulator(unittest.TestCase):
    def setUp(self):
        self.N = 3
        # These two dimers come from the same class
        self.graph = LatticeGraph(self.N, self.N, Boundary.OBC)
        self.config = EDSimulatorConfig()

    def test_full_herm(self):
        for i in range(100):
            J, h = np.random.rand(2) * np.random.choice([-1, 1], 2)
            eds1 = EDSimulatorQuimb(self.config, self.graph, J, h)
            self.assertTrue(np.allclose(
                eds1.hamiltonian.H.A, eds1.hamiltonian.A))

    def test_return_type(self):
        J, h = np.random.rand(2) * np.random.choice([-1, 1], 2)
        eds1 = EDSimulatorQuimb(self.config, self.graph, J, h)
        en = eds1.gs_energy
        self.assertIsInstance(en, float)


class TestPEPSSimulator(unittest.TestCase):

    def setUp(self):
        self.N = 3
        # These two dimers come from the same class
        self.graph = LatticeGraph(self.N, self.N, Boundary.OBC)
        self.peps_config = PEPSSimulatorConfig()
        self.peps_config.D = 3
        self.ed_config = EDSimulatorConfig()

    def test_gs_energy_obc_ed_quimb(self):
        J, h = np.random.rand(2) * np.random.choice([-1, 1], 2)
        mpssim = PEPSSimulator(self.peps_config, self.graph, J, h)
        mps_gs_energy = mpssim.gs_energy

        edsim = EDSimulatorQuimb(self.ed_config, self.graph, J, h)
        ed_gs_energy = edsim.gs_energy
        rel_error = np.abs((mps_gs_energy-ed_gs_energy)/ed_gs_energy)
        self.assertLess(rel_error, 0.001)


class TestMPSSimulator(unittest.TestCase):

    def setUp(self):
        self.N = 3
        # These two dimers come from the same class
        self.graph = LatticeGraph(self.N, self.N, Boundary.OBC)
        self.mps_config = MPSSimulatorConfig()
        self.ed_config = EDSimulatorConfig()

    def test_return_type(self):
        J, h = np.random.rand(2) * np.random.choice([-1, 1], 2)
        mpssim = MPSSimulator(self.mps_config, self.graph, J, h)
        mps_gs_energy = mpssim.gs_energy
        ed_mpo_gs_energy, _ = mpssim.ed_groundstate_from_MPO()
        self.assertIsInstance(mps_gs_energy, float)
        self.assertIsInstance(ed_mpo_gs_energy, float)

    def test_gs_energy_obc_ed_quimb(self):
        J, h = np.random.rand(2) * np.random.choice([-1, 1], 2)
        mpssim = MPSSimulator(self.mps_config, self.graph, J, h)
        mps_gs_energy = mpssim.gs_energy

        edsim = EDSimulatorQuimb(self.ed_config, self.graph, J, h)
        ed_gs_energy = edsim.gs_energy
        rel_error = np.abs((mps_gs_energy-ed_gs_energy)/ed_gs_energy)
        self.assertLess(rel_error, 0.001)

    def test_gs_energy_obc_ed_quimb_mpo(self):
        J, h = np.random.rand(2) * np.random.choice([-1, 1], 2)
        edsim = EDSimulatorQuimb(self.ed_config, self.graph, J, h)
        ed_gs_energy = edsim.gs_energy

        mpssim = MPSSimulator(self.mps_config, self.graph, J, h)
        ed_mpo_gs_energy, ed_gs = mpssim.ed_groundstate_from_MPO()
        self.assertAlmostEqual(ed_gs_energy, ed_mpo_gs_energy, places=4)

    def test_gs_energy_obc_ed_mpo(self):
        J, h = np.random.rand(2) * np.random.choice([-1, 1], 2)
        mpssim = MPSSimulator(self.mps_config, self.graph, J, h)
        mps_gs_energy = mpssim.gs_energy

        ed_gs_energy, ed_gs = mpssim.ed_groundstate_from_MPO()
        self.assertAlmostEqual(mps_gs_energy, ed_gs_energy,places=4)