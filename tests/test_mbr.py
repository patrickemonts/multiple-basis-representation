import unittest

import tfimsim.mbr as mbr

class TestMBR(unittest.TestCase):

    def test_evaluate_energies_xx(self):
        edges = [(0, 1), (1, 2)]
        bitstring = '000'
        J = 1
        self.assertEqual(mbr.evaluate_energies_xx(edges, bitstring, J), -2)
    
    def test_evaluate_energies_z_down(self):
        bitstring = '000'
        self.assertEqual(mbr.evaluate_energies_z(bitstring), -3)

    def test_evaluate_energies_z_up(self):
        bitstring = '111'
        self.assertEqual(mbr.evaluate_energies_z(bitstring), 3)

    def test_bitstrings_z(self):
        bitstrings_z = mbr.create_z_list(2, 2, 0)
        self.assertEqual(bitstrings_z[0].count('1'), 3)
