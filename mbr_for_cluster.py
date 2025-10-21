import numpy as np
from src.tfimsim import mbr_cluster as mbr
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh, eigh


def main(nx, ny, k, J, h):

    bitstrings_x = mbr.create_list(nx, ny, k) # Create bitstrings in the cluster basis
    bitstrings_z = mbr.create_list(nx, ny, k - 1) # Create bitstrings in the computational basis

    ### THE SECOND BITSTRING HAS ORDER K - 1 FOR CONTROL. SWITCH TO K FOR FINAL RESULTS ###

    edges = mbr.create_edges(nx, ny) # Create edges of the lattice for the graph states

    f = mbr.compute_overlap_matrix(bitstrings_x, bitstrings_z, edges) # Compute overlap matrix - crossed terms

    F = np.block([[np.eye(len(bitstrings_z), len(bitstrings_z)), f.T], [
                        f, np.eye(len(bitstrings_x), len(bitstrings_x))]]) # Full overlap matrix

    F += 1e-10 * np.eye(F.shape[0])  # Regularization for inversion stability

    energies_z = mbr.evaluate_all_energies_z(bitstrings_x, bitstrings_z, edges) # Evaluate energies for the Z hamiltonian in 
    energies_cluster = mbr.evaluate_all_energies_cluster(bitstrings_x, bitstrings_z, edges)

    plt.imshow(energies_cluster) # Visual of the energy matrices for dimension control
    plt.show()
    plt.imshow(F)
    plt.show()

    H = J * energies_z + h * energies_cluster

    D, P = eigh(H, F) # Generalized eigenvalue solving 
    ground_state = P[:, 0]

    # ground_state_inv = np.linalg.inv(P)[0]
    # energy = ground_state_inv @ np.linalg.inv(F) @ H @ ground_state

    norm = np.conj(ground_state) @ F @ ground_state

if __name__ == "__main__":
    nx = 3
    ny = 2
    k = 2
    J = 1.0
    h = 0.5

    main(nx, ny, k, J, h)