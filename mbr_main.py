import mbr
import numpy as np
from scipy.linalg import eigvalsh
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument('--nx', type=int, default=3, help='Number of sites in the x-dimension')
parser.add_argument('--ny', type=int, default=3, help='Number of sites in the y-dimension')
parser.add_argument('--J', type=float, default=1, help='Coefficient for the XX term')
parser.add_argument('--h_min', type=float, default=0, help='Minimal value for the coefficient in the Z term')
parser.add_argument('--h_max', type=float, default=1, help='Maximal value for the coefficient in the Z term')
parser.add_argument('--steps', type=int, default=51, help='Number of steps to sweep the h parameter')



def main(nx, ny, J, h_min, h_max, steps):

    edges = mbr.create_edges(nx, ny)
    nqubits = nx * ny

    h_s = np.linspace(h_min,h_max, steps)[1:-1]
    degrees = [1, 2]
    if nqubits == 6: degrees = [1]

    energies = np.zeros((len(degrees), len(h_s)))
    f = []
    bitstrings_x = []
    bitstrings_z = []
    energies_z = []
    energies_xx = []
    F = []


    for i, degree in enumerate(degrees):
        bitstrings_x.append(mbr.create_x_list(nqubits, degree)) # Create bitstrings. X bitstring is twice as long as Z bitstrings
        bitstrings_z.append(mbr.create_z_list(nqubits, degree))
        f.append(mbr.compute_overlap_matrix(bitstrings_x[-1], bitstrings_z[-1]))
        energies_xx.append(np.array(mbr.evaluate_all_energies_xx(bitstrings_x[-1], bitstrings_z[-1], edges))) # Energy evaluation
        energies_z.append(np.array(mbr.evaluate_all_energies_z(bitstrings_x[-1], bitstrings_z[-1])))
        # Overlap matrix with all contributions
        F.append(np.block([[np.eye(len(bitstrings_x[-1]), len(bitstrings_x[-1])), f[-1]],[np.conj(f[-1]).T, np.eye(len(bitstrings_z[-1]), len(bitstrings_z[-1]))]]))

    for i, h in enumerate(h_s):
        
        for j, degree in enumerate(degrees):
            H = J * energies_xx[j] + h * energies_z[j]

            eigm = eigvalsh(H, F[j]) # Generalized eigenvalue solving
            
            print(h, degree, eigm[0]) #Just information
            energies[j, i] = eigm[0]



    filename = f'data_sparse_diag/full_ham_{(nx, ny)}q_'
    np.savetxt(filename + 'h.txt', h_s)
    np.savetxt(filename + 'energies.txt', energies)
    np.savetxt(filename + 'energies_site.txt', energies / nqubits)




if __name__ == '__main__':
    nx = vars(parser.parse_args())['nx']
    ny = vars(parser.parse_args())['ny']
    J = vars(parser.parse_args())['J']
    h_min = vars(parser.parse_args())['h_min']
    h_max = vars(parser.parse_args())['h_max']
    steps = vars(parser.parse_args())['steps']
    
    main(nx, ny, J, h_min, h_max, steps)
