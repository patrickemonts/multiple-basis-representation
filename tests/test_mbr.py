import unittest
import numpy as np
import tfimsim.mbr as mbr

class TestMBR(unittest.TestCase):

    def setUp(self) -> None:
        self.nx = 2
        self.ny = 2
        self.degree = 1


    def test_bitstring_x_example(self):
        degree = 3
        val = mbr.create_x_list(self.nx, self.ny, degree, ferro=False)
        ref = ['0110',
        '1001',
        '1110',
        '0010',
        '0100',
        '0111',
        '0001',
        '1101',
        '1011',
        '1000',
        '1010',
        '1100',
        '1111',
        '0000',
        '0011',
        '0101',
        '0101',
        '0011',
        '0000',
        '1111',
        '1100',
        '1010',
        '1000',
        '1011',
        '1101',
        '0001',
        '0111',
        '0100',
        '0010',
        '1110']
        self.assertEqual(len(val), len(ref))
        self.assertEqual(val,ref)

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

        print(ref)
        print(val)


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

    def test_bitstrings_z(self):
        bitstrings_z = mbr.create_z_list(2, 2, 0)
        self.assertEqual(bitstrings_z[0].count('1'), 3)
