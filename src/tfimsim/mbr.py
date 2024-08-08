# DIAGONALIZATION OF A TRANSVERSE FIELD ISING MODEL AS GIVEN BY A GRAPH

import numpy as np
from itertools import combinations

def _xor(b1, b2):
    # auxiliary function XOR, with proper formatting
    return bin(int(b1, 2) ^ int(b2, 2))[2:].zfill(len(b1))

def _evaluate_energies_xx_xx(bitstrings_x, bitstrings_z, edges): 
    energies_xx = np.zeros((len(bitstrings_x), len(bitstrings_x)))
    for i, xi in enumerate(bitstrings_x):
        for e in edges:
            if xi[e[0]] == xi[e[1]]:
                energies_xx[i, i] += 1
            else:
                energies_xx[i, i] -= 1

    return energies_xx


def _evaluate_energies_xx_xz(bitstrings_x, bitstrings_z, edges):
    energies_xz = np.zeros((len(bitstrings_x), len(bitstrings_z)))
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

                energies_xz[i, j] += v

    return energies_xz


def _evaluate_energies_xx_zz(bitstrings_x, bitstrings_z, edges):
    energies_zz = np.zeros((len(bitstrings_z), len(bitstrings_z)))

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

                energies_zz[i, j] += v

    return energies_zz

def evaluate_all_energies_xx(bitstrings_x, bitstrings_z, edges):
    # Function to evaluate all energies given by the $XX$ hamiltonian, specified by bitstring in the computational and hadamard bases
    # Extension of evaluate_energies_xx
    energies_xx = _evaluate_energies_xx_xx(bitstrings_x, bitstrings_z, edges)
    energies_xz = _evaluate_energies_xx_xz(bitstrings_x, bitstrings_z, edges)
    energies_zz = _evaluate_energies_xx_zz(bitstrings_x, bitstrings_z, edges)
    

    energies = np.block([[energies_xx, energies_xz], [
                        np.conj(energies_xz.T), energies_zz]])

    return energies

def _evaluate_energies_z_xx(bitstrings_x, bitstrings_z):
    energies_xx = np.zeros((len(bitstrings_x), len(bitstrings_x)))
    for i, xi in enumerate(bitstrings_x):  # XX terms
        for j, xj in enumerate(bitstrings_x):
            xor = _xor(xi, xj)
            if xor.count('1') == 1:
                energies_xx[i, j] += 1

    return energies_xx

def _evaluate_energies_z_xz(bitstrings_x, bitstrings_z):
    energies_xz = np.zeros((len(bitstrings_x), len(bitstrings_z)))
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
                energies_xz[i, j] += v

    return energies_xz

def _evaluate_energies_z_zz(bitstrings_x, bitstrings_z):
    energies_zz = np.zeros((len(bitstrings_z), len(bitstrings_z)))
    for i, zi in enumerate(bitstrings_z):  # contribution in the computational bases
        energies_zz[i, i] = zi.count('0') - zi.count('1')

    return energies_zz

def evaluate_all_energies_z(bitstrings_x, bitstrings_z):
    # Function to evaluate all energies given by the $Z$ hamiltonian, specified by bitstring in the computational and hadamard bases
    # Extension of evaluate_energies_z
    energies_xx = _evaluate_energies_z_xx(bitstrings_x, bitstrings_z)
    energies_xz = _evaluate_energies_z_xz(bitstrings_x, bitstrings_z)
    energies_zz = _evaluate_energies_z_zz(bitstrings_x, bitstrings_z)

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
            # assert len(x) == len(z)
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

def _neel(nx, ny):
    neel = '0'
    for _ in range(nx - 1):
        neel += _xor(neel[-1], '1')
    for _ in range(ny - 1):
        neel += _xor(neel[-nx:], '1'*nx)
    
    return neel

def create_x_list(nx, ny, degree, ferro=True):  # List of all bitstrings in the X bases that we consider
    if not ferro:
        neel = _neel(nx, ny)

    
    nqubits = nx * ny
    bitstrings = []
    for k in range(degree+1):
        bitstrings += generate_bitstrings(nqubits, k, mode='0')
        bitstrings += generate_bitstrings(nqubits, k, mode='1')

    if not ferro:
        bitstrings = [_xor(b, neel) for b in bitstrings]

    return bitstrings


# List of all bitstrings in the computational bases that we consider
def create_z_list(nx, ny, degree, ferro=True):
    nqubits = nx * ny 
    bitstrings = []
    for k in range(degree+1):
        bitstrings += generate_bitstrings(nqubits, k, mode='0')

    return bitstrings


def create_edges(nx, ny):  # Function to create the edges of a square lattice
    qubits = nx * ny
    edges = []
    for i in range(nx-1):
        for j in range(ny):
            # Z(i + nx * j) * Z(i + 1 + nx * j)
            edges.append((i + nx * j, i + 1 + nx * j))

    for i in range(nx):
        for j in range(ny-1):
            # Z(i + nx * j) * Z(i + nx * (j + 1))
            edges.append((i + nx * j, i + nx * (j + 1)))

    return edges


def evaluate_magnetization_z(bitstrings_x, bitstrings_z):
    return evaluate_all_energies_z(bitstrings_x, bitstrings_z)


def evaluate_magnetization_x(bitstrings_x, bitstrings_z): 
    # Recycle previous calculation by noting that it is just a change of order with respect to previous function
    energies_xx = _evaluate_magnetization_x_xx(bitstrings_x, bitstrings_z)
    energies_xz = _evaluate_magnetization_x_xz(bitstrings_x, bitstrings_z)
    energies_zz = _evaluate_magnetization_x_zz(bitstrings_x, bitstrings_z)

    energies = np.block([[energies_xx, energies_xz],
                        [np.conj(energies_xz.T), energies_zz]])
    
    return energies

def _evaluate_magnetization_x_xx(bitstrings_x, bitstrings):
    energies_xx = np.zeros((len(bitstrings_x), len(bitstrings_x)))
    for i, xi in enumerate(bitstrings_x):  # XX terms
        energies_xx[i, i] = xi.count('0') - xi.count('1')

    return energies_xx

def _evaluate_magnetization_x_xz(bitstrings_x, bitstrings_z):
    energies_xz = np.zeros((len(bitstrings_x), len(bitstrings_z)))
    for i, xi in enumerate(bitstrings_x):  # overlaps between bases
        for j, zj in enumerate(bitstrings_z):
            for q in range(len(xi)):
                v = 1
                for k, (x, z) in enumerate(zip(xi, zj)):
                    z = int(z)
                    x = int(x)
                    if k == q:
                        z = (z + 1) % 2
                    v *= (-1)**(x * z) / np.sqrt(2)
                energies_xz[i, j] += v

    return energies_xz

def _evaluate_magnetization_x_zz(bitstrings_x, bitstrings_z): # To be finished!!!!
    energies_zz = np.zeros((len(bitstrings_z), len(bitstrings_z)))
    for i, zi in enumerate(bitstrings_z):  # XX terms
        for j, zj in enumerate(bitstrings_z):
            xor = _xor(zi, zj)
            if xor.count('1') == 1:
                energies_zz[i, j] += 1

    return energies_zz



def evaluate_magnetization_staggered_x(bitstrings_x, bitstrings_z, nx, ny):
    energies_xx = _evaluate_magnetization_staggered_x_xx(bitstrings_x, bitstrings_z, nx, ny)
    energies_xz = _evaluate_magnetization_staggered_x_xz(bitstrings_x, bitstrings_z, nx, ny)
    energies_zz = _evaluate_magnetization_staggered_x_zz(bitstrings_x, bitstrings_z, nx, ny)

    energies = np.block([[energies_xx, energies_xz],
                        [np.conj(energies_xz.T), energies_zz]])
    
    return energies

def _evaluate_magnetization_staggered_x_xx(bitstrings_x, bitstrings_z, nx, ny):
    energies_xx = np.zeros((len(bitstrings_x), len(bitstrings_x)))
    for i, xi in enumerate(bitstrings_x):  # XX terms
        xi = _xor(xi, _neel(nx, ny))

        energies_xx[i, i] = xi.count('0') - xi.count('1')

    return energies_xx

def _evaluate_magnetization_staggered_x_xz(bitstrings_x, bitstrings_z, nx, ny):
    energies_xz = np.zeros((len(bitstrings_x), len(bitstrings_z)))
    for i, xi in enumerate(bitstrings_x):  # overlaps between bases
        for j, zj in enumerate(bitstrings_z):
            for q in range(len(xi)):
                qx, qy = q // nx, q - q // nx
                v = 1
                for k, (x, z) in enumerate(zip(xi, zj)):
                    z = int(z)
                    x = int(x)
                    if k == q:
                        z = (z + 1) % 2
                    v *= (-1)**(x * z) / np.sqrt(2)
                energies_xz[i, j] += (-1)**(qx + qy) * v

    return energies_xz

def _evaluate_magnetization_staggered_x_zz(bitstrings_x, bitstrings_z, nx, ny): # To be finished!!!!
    energies_zz = np.zeros((len(bitstrings_z), len(bitstrings_z)))
    for i, zi in enumerate(bitstrings_z):  
        for j, zj in enumerate(bitstrings_z):
            xor = _xor(zi, zj)
            if xor.count('1') == 1:
                f = xor.find('1')
                qx, qy = f // nx, f - f // nx
                energies_zz[i, j] += (-1)**(qx + qy)

    return energies_zz
