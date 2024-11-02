import unittest
import numpy as np
from .hamiltonian import PEPSSimulator, MPSSimulator, MPSSimulatorConfig, PEPSSimulatorConfig, EDSimulatorConfig, EDSimulatorQuimb
from mbrsim.graph import LatticeGraph, Boundary


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
        pepssim = PEPSSimulator(self.peps_config, self.graph, J, h)
        peps_gs_energy = pepssim.gs_energy

        edsim = EDSimulatorQuimb(self.ed_config, self.graph, J, h)
        ed_gs_energy = edsim.gs_energy
        rel_error = np.abs((peps_gs_energy-ed_gs_energy)/ed_gs_energy)
        self.assertLess(rel_error, 0.001)

    @unittest.skip("This test does not always reach the necessary accuracy")
    def test_mag_x_energy_obc_ed_quimb_small(self):
        graph = LatticeGraph(2, 2, Boundary.OBC)
        J = 1
        h = np.random.rand()
        pepssim = PEPSSimulator(self.peps_config, graph, J, h)
        peps_mag_x = pepssim.magnetization_x

        edsim = EDSimulatorQuimb(self.ed_config, graph, J, h)
        ed_mag_x = edsim.magnetization_x
        self.assertAlmostEqual(peps_mag_x, ed_mag_x, places=4)

    @unittest.skip("This test does not always reach the necessary accuracy")
    def test_mag_z_energy_obc_ed_quimb(self):
        J = 1
        h = 0.8
        pepssim = PEPSSimulator(self.peps_config, self.graph, J, h)
        peps_mag_z = pepssim.magnetization_z

        edsim = EDSimulatorQuimb(self.ed_config, self.graph, J, h)
        ed_mag_z = edsim.magnetization_z
        self.assertAlmostEqual(peps_mag_z, ed_mag_z, places=4)

    @unittest.skip("This test does not always reach the necessary accuracy")
    def test_mag_x_stag_energy_obc_ed_quimb(self):
        J = 1
        h = np.random.rand()
        pepssim = PEPSSimulator(self.peps_config, self.graph, J, h)
        peps_mag_x_stag = pepssim.magnetization_x_stag

        edsim = EDSimulatorQuimb(self.ed_config, self.graph, J, h)
        ed_mag_x_stag = edsim.magnetization_x_stag
        self.assertAlmostEqual(peps_mag_x_stag, ed_mag_x_stag,places=4)


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

    def test_mag_x_energy_obc_ed_quimb(self):
        J = 1
        h = np.random.rand()
        mpssim = MPSSimulator(self.mps_config, self.graph, J, h)
        mps_mag_x = mpssim.magnetization_x

        edsim = EDSimulatorQuimb(self.ed_config, self.graph, J, h)
        ed_mag_x = edsim.magnetization_x
        self.assertAlmostEqual(mps_mag_x, ed_mag_x, places=4)

    def test_mag_z_energy_obc_ed_quimb_N_2(self):
        J = 1
        h = 0.8
        graph = LatticeGraph(2, 2, Boundary.OBC)
        mpssim = MPSSimulator(self.mps_config, graph, J, h)
        mps_mag_z = mpssim.magnetization_z

        edsim = EDSimulatorQuimb(self.ed_config, graph, J, h)
        ed_mag_z = edsim.magnetization_z
        self.assertAlmostEqual(mps_mag_z, ed_mag_z, places=4)

    def test_mag_z_energy_obc_ed_quimb_N_3(self):
        J = 1
        h = 0.8
        graph = LatticeGraph(3, 3, Boundary.OBC)
        mpssim = MPSSimulator(self.mps_config, graph, J, h)
        mps_mag_z = mpssim.magnetization_z

        edsim = EDSimulatorQuimb(self.ed_config, graph, J, h)
        ed_mag_z = edsim.magnetization_z
        self.assertAlmostEqual(mps_mag_z, ed_mag_z, places=4)

    def test_mag_x_stag_energy_obc_ed_quimb(self):
        J = 1
        h = np.random.rand()
        mpssim = MPSSimulator(self.mps_config, self.graph, J, h)
        mps_mag_x_stag = mpssim.magnetization_x_stag

        edsim = EDSimulatorQuimb(self.ed_config, self.graph, J, h)
        ed_mag_x_stag = edsim.magnetization_x_stag
        self.assertAlmostEqual(mps_mag_x_stag, ed_mag_x_stag,places=4)