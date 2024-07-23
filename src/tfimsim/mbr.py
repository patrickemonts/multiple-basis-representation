# DIAGONALIZATION OF A TRANSVERSE FIELD ISING MODEL AS GIVEN BY A GRAPH

import numpy as np
from itertools import combinations


def evaluate_energies_xx(edges, bitstring, J=1):
    # The XX hamiltonian is given by a sum of terms (-X_i X_j)
    # This function computes the energy of a bitstring in the basis H with respect to a Hamiltonian especified by the edges
    energy = 0
    for e in edges:
        if bitstring[e[0]] == bitstring[e[1]]:
            energy -= 1
        else:
            energy += 1

    return energy


def evaluate_energies_z(bitstring):
    # The Z hamiltonian is given by a sum of terms (-Z_i)
    # This function computes the energy of a bitstring in the with respect to a Hamiltonian given by the sum of all Z terms
    energy = bitstring.count('1') - bitstring.count('0')

    return energy


def _xor(b1, b2):
    # auxiliary function XOR, with proper formatting
    return bin(int(b1, 2) ^ int(b2, 2))[2:].zfill(len(b1))


def evaluate_all_energies_xx(bitstrings_x, bitstrings_z, edges):
    # Function to evaluate all energies given by the $XX$ hamiltonian, specified by bitstring in the computational and hadamard bases
    # Extension of evaluate_energies_xx
    energies_xx = np.zeros((len(bitstrings_x), len(bitstrings_x)))
    energies_xz = np.zeros((len(bitstrings_x), len(bitstrings_z)))
    energies_zz = np.zeros((len(bitstrings_z), len(bitstrings_z)))

    for i, xi in enumerate(bitstrings_x):
        for e in edges:
            if xi[e[0]] == xi[e[1]]:
                energies_xx[i, i] -= 1
            else:
                energies_xx[i, i] += 1

    for i, xi in enumerate(bitstrings_x):  # OVerlapping terms
        for j, zj in enumerate(bitstrings_z):
            for e in edges:
                v = 1
                for k, (x, z) in enumerate(zip(xi, zj)):
                    x = int(x)
                    z = int(z)
                    if k in e:
                        z = (z + 1) % 2

                    v *= (-1)**(z * x)/np.sqrt(2)

                energies_xz[i, j] -= v

    for i, zi in enumerate(bitstrings_z):  # ZZ terms, most terms are diagonal
        for j, zj in enumerate(bitstrings_z):
            for e in edges:
                v = 1

                for k, (z1, z2) in enumerate(zip(zi, zj)):
                    z1 = int(z1)
                    z2 = int(z2)
                    if k in e:
                        z2 = (z2 + 1) % 2

                    v *= int(z1 == z2)
                    if v == 0:
                        break

                energies_zz[i, j] -= v

    energies = np.block([[energies_xx, energies_xz], [
                        np.conj(energies_xz.T), energies_zz]])

    return energies


def evaluate_all_energies_z(bitstrings_x, bitstrings_z):
    # Function to evaluate all energies given by the $Z$ hamiltonian, specified by bitstring in the computational and hadamard bases
    # Extension of evaluate_energies_z
    energies_xx = np.zeros((len(bitstrings_x), len(bitstrings_x)))
    energies_xz = np.zeros((len(bitstrings_x), len(bitstrings_z)))
    energies_zz = np.zeros((len(bitstrings_z), len(bitstrings_z)))
    for i, xi in enumerate(bitstrings_x):  # XX terms
        for j, xj in enumerate(bitstrings_x):
            xor = _xor(xi, xj)
            if xor.count('1') == 1:
                energies_xx[i, j] -= 1

    for i, xi in enumerate(bitstrings_x):  # overlaps between bases
        for j, zj in enumerate(bitstrings_z):
            for q in range(len(xi)):
                v = 1
                for k, (x, z) in enumerate(zip(xi, zj)):
                    z = int(z)
                    x = int(x)
                    if k == q:
                        x = (x + 1) % 2
                    v *= (-1)**(x * z) / np.sqrt(2)
                energies_xz[i, j] -= v

    for i, zi in enumerate(bitstrings_z):  # contribution in the computational bases
        energies_zz[i, i] = zi.count('1') - zi.count('0')

    energies = np.block([[energies_xx, energies_xz], [
                        np.conj(energies_xz.T), energies_zz]])

    return energies


# Computation of the overlap matrix, for sets of bitstrings in the computational and Hadamard bases
def compute_overlap_matrix(bitstrings_x, bitstrings_z):
    f = np.ones((len(bitstrings_x), len(bitstrings_z))) / \
        (2**(.5 * len(bitstrings_x[0])))

    for i in range(len(bitstrings_x)):
        for j in range(len(bitstrings_z)):
            x = bitstrings_x[i]
            z = bitstrings_z[j]
            assert len(x) == len(z)
            for xi, zi in zip(x, z):
                f[i, j] *= (-1)**(int(xi) * int(zi))

    return f


def generate_bitstrings(n, k, mode='0'):
    if k > n:
        raise ValueError("Number of 1s (k) cannot be greater than length (n)")

    # Generate all combinations of indices for 1s in the bitstring
    index_combinations = combinations(range(n), k)

    # Initialize an empty list to store bitstrings
    bitstrings = []

    # Iterate over index combinations and generate bitstrings
    for indices in index_combinations:
        if mode == '0':
            bitstring = ['0'] * n  # Initialize with all 0s
        elif mode == '1':
            bitstring = ['1'] * n  # Initialize with all 1s
        for index in indices:
            if mode == '0':
                bitstring[index] = '1'  # Set 1s at specified indices
            if mode == '1':
                bitstring[index] = '0'  # Set 1s at specified indices
        # Join list to form a string and append to the result list
        bitstrings.append(''.join(bitstring))

    return bitstrings


def create_x_list(nqubits, degree):  # List of all bitstrings in the X bases that we consider
    bitstrings = []
    for k in range(degree+1):
        bitstrings += generate_bitstrings(nqubits, k, mode='0')
        bitstrings += generate_bitstrings(nqubits, k, mode='1')

    return bitstrings


# List of all bitstrings in the computational bases that we consider
def create_z_list(nqubits, degree):
    bitstrings = []
    for k in range(degree+1):
        bitstrings += generate_bitstrings(nqubits, k, mode='0')

    return bitstrings


def create_edges(nx, ny):  # Function to create the edges of a square lattice
    qubits = nx * ny
    edges = []
    for i in range(nx - 1):
        for j in range(ny):
            # Z(i + nx * j) * Z(i + 1 + nx * j)
            edges.append((i + nx * j, i + 1 + nx * j))
        for i in range(nx):
            for j in range(ny-1):
                # Z(i + nx * j) * Z(i + nx * (j + 1))
                edges.append((i + nx * j, i + nx * (j + 1)))
        return edges
