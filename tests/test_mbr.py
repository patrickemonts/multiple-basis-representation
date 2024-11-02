import unittest
import numpy as np
import mbrsim.mbr as mbr

class TestMBR(unittest.TestCase):

    def setUp(self) -> None:
        self.nx = 2
        self.ny = 2
        self.degree = 1
        self.edges = mbr.create_edges(self.nx, self.ny)

    def test_bitstring(self):
        ref = mbr.generate_bitstrings_str(self.nx * self.ny, self.degree, mode='0')

        val = mbr.generate_bitstrings(self.nx * self.ny, self.degree, mode='0')

        val = [''.join(map(str,b)) for b in val]

        self.assertEqual(len(val), len(ref))
        self.assertEqual(val,ref)

    def test_neel(self):
        ref = mbr._neel_str(self.nx, self.ny)
        val = mbr._neel(self.nx, self.ny)
        val = ''.join(map(str,val)) 

        self.assertEqual(len(val), len(ref))
        self.assertEqual(val,ref)


    def test_bitstring_x(self):
        ref = mbr.create_x_list_str(self.nx, self.ny, self.degree, ferro=False)

        val = mbr.create_x_list(self.nx, self.ny, self.degree, ferro=False)

        val = [''.join(map(str,b)) for b in val]

        self.assertEqual(len(val), len(ref))
        self.assertEqual(val,ref)


    def test_bitstring_z(self):
        ref = mbr.create_z_list_str(self.nx, self.ny, self.degree, ferro=False)

        val = mbr.create_z_list(self.nx, self.ny, self.degree, ferro=False)

        val = [''.join(map(str,b)) for b in val]

        self.assertEqual(len(val), len(ref))
        self.assertEqual(val,ref)


    def test_overlap(self):
        b_x = mbr.create_x_list_str(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list_str(self.nx, self.ny, self.degree)

        ref = mbr.compute_overlap_matrix_str(b_x, b_z)

        b_x = mbr.create_x_list(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list(self.nx, self.ny, self.degree)

        val = mbr.compute_overlap_matrix(b_x, b_z)

        self.assertEqual(val.shape, ref.shape)
        self.assertLess(np.max(np.abs(val - ref)), 1e-5)

    def test_energies_xx_xx(self):
        b_x = mbr.create_x_list_str(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list_str(self.nx, self.ny, self.degree)

        ref = mbr._evaluate_energies_xx_xx_str(b_x, b_z, self.edges)

        b_x = mbr.create_x_list(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list(self.nx, self.ny, self.degree)

        val = mbr._evaluate_energies_xx_xx(b_x, b_z, self.edges)

        self.assertEqual(val.shape, ref.shape)
        self.assertLess(np.max(np.abs(val - ref)), 1e-5)


    def test_energies_xx_xz(self):
        b_x = mbr.create_x_list_str(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list_str(self.nx, self.ny, self.degree)

        ref = mbr._evaluate_energies_xx_xz_str(b_x, b_z, self.edges)

        b_x = mbr.create_x_list(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list(self.nx, self.ny, self.degree)

        val = mbr._evaluate_energies_xx_xz(b_x, b_z, self.edges)

        self.assertEqual(val.shape, ref.shape)
        self.assertLess(np.max(np.abs(val - ref)), 1e-5)


    def test_energies_xx_zz(self):
        b_x = mbr.create_x_list_str(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list_str(self.nx, self.ny, self.degree)

        ref = mbr._evaluate_energies_xx_zz_str(b_x, b_z, self.edges)

        b_x = mbr.create_x_list(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list(self.nx, self.ny, self.degree)

        val = mbr._evaluate_energies_xx_zz(b_x, b_z, self.edges)
        
        self.assertEqual(val.shape, ref.shape)
        self.assertLess(np.max(np.abs(val - ref)), 1e-5)

    
    def test_energies_z_xx(self):
        b_x = mbr.create_x_list_str(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list_str(self.nx, self.ny, self.degree)

        ref = mbr._evaluate_energies_z_xx_str(b_x, b_z)

        b_x = mbr.create_x_list(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list(self.nx, self.ny, self.degree)

        val = mbr._evaluate_energies_z_xx(b_x, b_z)
        
        self.assertEqual(val.shape, ref.shape)
        self.assertLess(np.max(np.abs(val - ref)), 1e-5)

    def test_energies_z_xz(self):
        b_x = mbr.create_x_list_str(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list_str(self.nx, self.ny, self.degree)

        ref = mbr._evaluate_energies_z_xz_str(b_x, b_z)

        b_x = mbr.create_x_list(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list(self.nx, self.ny, self.degree)

        val = mbr._evaluate_energies_z_xz(b_x, b_z)
        
        self.assertEqual(val.shape, ref.shape)
        self.assertLess(np.max(np.abs(val - ref)), 1e-5)

    def test_energies_z_zz(self):
        b_x = mbr.create_x_list_str(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list_str(self.nx, self.ny, self.degree)

        ref = mbr._evaluate_energies_z_zz_str(b_x, b_z)

        b_x = mbr.create_x_list(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list(self.nx, self.ny, self.degree)

        val = mbr._evaluate_energies_z_zz(b_x, b_z)
        
        self.assertEqual(val.shape, ref.shape)
        self.assertLess(np.max(np.abs(val - ref)), 1e-5)

    
    def test_magnetization_x_xx(self):
        b_x = mbr.create_x_list_str(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list_str(self.nx, self.ny, self.degree)

        ref = mbr._evaluate_magnetization_x_xx_str(b_x, b_z)

        b_x = mbr.create_x_list(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list(self.nx, self.ny, self.degree)

        val = mbr._evaluate_magnetization_x_xx(b_x, b_z)
        
        self.assertEqual(val.shape, ref.shape)
        self.assertLess(np.max(np.abs(val - ref)), 1e-5)

    def test_magnetization_x_xz(self):
        b_x = mbr.create_x_list_str(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list_str(self.nx, self.ny, self.degree)

        ref = mbr._evaluate_magnetization_x_xz_str(b_x, b_z)

        b_x = mbr.create_x_list(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list(self.nx, self.ny, self.degree)

        val = mbr._evaluate_magnetization_x_xz(b_x, b_z)
        
        self.assertEqual(val.shape, ref.shape)
        self.assertLess(np.max(np.abs(val - ref)), 1e-5)

    
    def test_magnetization_x_zz(self):
        b_x = mbr.create_x_list_str(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list_str(self.nx, self.ny, self.degree)

        ref = mbr._evaluate_magnetization_x_zz_str(b_x, b_z)

        b_x = mbr.create_x_list(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list(self.nx, self.ny, self.degree)

        val = mbr._evaluate_magnetization_x_zz(b_x, b_z)
        
        self.assertEqual(val.shape, ref.shape)
        self.assertLess(np.max(np.abs(val - ref)), 1e-5)


    def test_magnetization_staggered_x_xx(self):
        b_x = mbr.create_x_list_str(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list_str(self.nx, self.ny, self.degree)

        ref = mbr._evaluate_magnetization_staggered_x_xx_str(b_x, b_z, self.nx, self.ny)

        b_x = mbr.create_x_list(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list(self.nx, self.ny, self.degree)

        val = mbr._evaluate_magnetization_staggered_x_xx(b_x, b_z,  self.nx, self.ny)
        
        self.assertEqual(val.shape, ref.shape)
        self.assertLess(np.max(np.abs(val - ref)), 1e-5)

    def test_magnetization_staggered_x_xz(self):
        b_x = mbr.create_x_list_str(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list_str(self.nx, self.ny, self.degree)

        ref = mbr._evaluate_magnetization_staggered_x_xz_str(b_x, b_z,  self.nx, self.ny)

        b_x = mbr.create_x_list(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list(self.nx, self.ny, self.degree)

        val = mbr._evaluate_magnetization_staggered_x_xz(b_x, b_z,  self.nx, self.ny)
        
        self.assertEqual(val.shape, ref.shape)
        self.assertLess(np.max(np.abs(val - ref)), 1e-5)

    
    def test_magnetization_staggered_x_zz(self):
        b_x = mbr.create_x_list_str(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list_str(self.nx, self.ny, self.degree)

        ref = mbr._evaluate_magnetization_staggered_x_zz_str(b_x, b_z,  self.nx, self.ny)

        b_x = mbr.create_x_list(self.nx, self.ny, self.degree)
        b_z = mbr.create_z_list(self.nx, self.ny, self.degree)

        val = mbr._evaluate_magnetization_staggered_x_zz(b_x, b_z,  self.nx, self.ny)
        
        self.assertEqual(val.shape, ref.shape)
        self.assertLess(np.max(np.abs(val - ref)), 1e-5)


    @unittest.skip("API changed")
    def test_evaluate_energies_xx(self):
        edges = [(0, 1), (1, 2)]
        bitstring = '000'
        J = 1
        self.assertEqual(mbr.evaluate_energies_xx(edges, bitstring, J), -2)
    
    @unittest.skip("API changed")
    def test_evaluate_energies_z_down(self):
        bitstring = '000'
        self.assertEqual(mbr.evaluate_energies_z(bitstring), -3)

    @unittest.skip("API changed")
    def test_evaluate_energies_z_up(self):
        bitstring = '111'
        self.assertEqual(mbr.evaluate_energies_z(bitstring), 3)

