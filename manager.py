import numpy as np
import logging
import sys
import os
from tqdm import tqdm
import pandas as pd

from tfimsim.hamiltonian import EDSimulatorConfig,EDSimulatorQuimb,PEPSSimulator,PEPSSimulatorConfig, MPSSimulatorConfig, MPSSimulator
from tfimsim.graph import Boundary,LatticeGraph
from tfimsim.utils import SimulationType
import tfimsim.mbr as mbr
from scipy.linalg import eigvalsh, eigh

def args2suffix(args):
    """Convert type to a suffix for the output file and the log"""
    if args.type == SimulationType.MPS:
        return f"_chi_{args.mps_max_chi:03d}"
    elif args.type == SimulationType.PEPS:
        return f"_chi_{args.peps_max_chi:03d}"
    elif args.type == SimulationType.MBR:
        return f"_chi_{args.mbr_degree:03d}"
    return ""

def args2logname(args):
    """Convert arguments to a name for the log file

    Args:
        args (namespace): Namespace of arguments as provided by argparse

    Returns:
        str: Filename of the log file
    """
    if args.nsteps is not None:
        fname = f"log_L_{args.nx:02d}-{args.ny:02d}_J_{args.J:0.2f}_hmin_{args.hmin:.2f}_hmax_{args.hmax:.2f}_nsteps_{args.nsteps:03d}_type_{args.type}{args2suffix(args)}.log"
    else:
        fname = f"log_L_{args.nx:02d}-{args.ny:02d}_J_{args.J:0.2f}_h_{args.hmin:.2f}_type_{args.type}{args2suffix(args)}.log"
    return os.path.join(args.output, fname)

def args2fname(args):
    """Convert arguments to a name for the output file

    Args:
        args (namespace): Namespace of arguments as provided by argparse

    Returns:
        str: Filename of the output file
    """
    if args.nsteps is not None:
        fname = f"data_L_{args.nx:02d}-{args.ny:02d}_J_{args.J:0.2f}_hmin_{args.hmin:.2f}_hmax_{args.hmax:.2f}_nsteps_{args.nsteps:03d}_type_{args.type}{args2suffix(args)}.csv"
    else:
        fname = f"data_L_{args.nx:02d}-{args.ny:02d}_J_{args.J:0.2f}_h_{args.hmin:.2f}_type_{args.type}{args2suffix(args)}.csv"
    return fname

def simulator_cls_from_args(args):
    if args.type == SimulationType.ED:
        return EDSimulatorQuimb
    elif args.type == SimulationType.MPS:
        return MPSSimulator
    elif args.type == SimulationType.PEPS:
        return PEPSSimulator
    else:
        raise ValueError("Unknown simulation type")

def config_from_args(args):
    if args.type == SimulationType.ED:
        config = EDSimulatorConfig()
        config.use_sparse = not args.ed_no_sparse
        logging.info("======== ED CONFIG =======")
        logging.info(f"Using sparse matrices: {config.use_sparse}")
        logging.info("==========================")
    elif args.type == SimulationType.MPS:
        config = MPSSimulatorConfig()
        config.chi_max = args.mps_max_chi
        config.rel_energy_delta = args.mps_rel_energy_delta
        config.rel_entropy_delta = args.mps_rel_entropy_delta
        config.no_mixer = args.mps_no_mixer
        logging.info("======== MPS CONFIG =======")
        logging.info(f"Maximal virtual bond dimension: {config.chi_max}")
        logging.info(f"Relative energy delta: {config.rel_energy_delta}")
        logging.info(f"Relative entropy delta: {config.rel_entropy_delta}")
        logging.info(f"Use mixer: {not config.no_mixer}")
        logging.info("==========================")
    elif args.type == SimulationType.PEPS:
        config = PEPSSimulatorConfig()
        config.bd_mps_chi = args.peps_bd_mps_max_chi
        config.D = args.peps_max_chi
        logging.info("======== PEPS CONFIG =======")
        logging.info(f"Maximal virtual bond dimension: {config.D}")
        logging.info(f"Maximal virtual bond dimension of boundary MPS: {config.bd_mps_chi}")
        logging.info(f"==========================")
    else:
        raise ValueError("Unknown simulation type")
    return config


def add_config_columns(df, args):
    df_dest = df.copy()
    if args.type == SimulationType.ED:
        df_dest["use_sparse"] = not args.ed_no_sparse
    elif args.type == SimulationType.MPS:
        df_dest["max_chi"] = args.mps_max_chi
        df_dest["rel_energy_delta"] = args.mps_rel_energy_delta
        df_dest["rel_entropy_delta"] = args.mps_rel_entropy_delta
        df_dest["mixer"] = not args.mps_no_mixer
    elif args.type == SimulationType.PEPS:
        df_dest["D"] = args.peps_max_chi
        df_dest["bd_mps_chi"] = args.peps_bd_mps_max_chi
    return df_dest

def main(args):
    """
    Main function of the program. 
    It sets up the logger and calls the appropriate function based on the mode specified via CLI.

    Args:
        args: The arguments as provided by argparse.
    """
    #Set up the logger
    h_stdout = logging.StreamHandler(stream=sys.stdout)
    h_stderr = logging.StreamHandler(stream=sys.stderr)
    h_stderr.addFilter(lambda record: record.levelno >= logging.WARNING)
    logging.basicConfig(
        level=args.level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args2logname(args)),
            h_stdout,
            h_stderr
        ]
    )

    fname_out = args2fname(args)
    path_out = os.path.join(args.output, fname_out) 
    if not os.path.exists(path_out) or args.overwrite:

        nx = args.nx
        ny = args.ny
        nsteps = args.nsteps
        hmin = args.hmin
        hmax = args.hmax
        J = args.J

        if hmin == hmax:
            hvec = [hmin]
        else:
            if nsteps == 0:
                logging.error("Number of steps must be greater than 0. Aborting here.")
                sys.exit(1)

            hvec = np.linspace(hmin,hmax,nsteps, endpoint=True)

        # logging.info("Git hash: {}".format(utils.get_git_hash()))
        logging.info("========= SYSTEM INFO ==========")
        logging.info(f"Size: {args.nx}x{args.ny}")
        logging.info(f"J: {args.J}")
        logging.info(f"nsteps: {nsteps}")
        logging.info(f"Boundary conditions: {args.bdc}")
        logging.info(f"Method: {args.type}")
        logging.info("================================")

        if args.type == SimulationType.MBR:

            dest_dict = {"nx":[],"ny":[],"J":[],"h":[],"energy":[], "degree":[], "mx":[], "mx_s":[], "mz":[]}

            degree = args.mbr_degree

            edges = mbr.create_edges(nx, ny)
            ferro = (J <= 0)

            # Create bitstrings. X bitstring is twice as long as Z bitstrings
            bitstrings_x = mbr.create_x_list(nx, ny, degree, ferro=ferro)
            bitstrings_z = mbr.create_z_list(nx, ny, degree)
            f = mbr.compute_overlap_matrix(bitstrings_x, bitstrings_z)
            energies_xx = np.array(mbr.evaluate_all_energies_xx(
                bitstrings_x, bitstrings_z, edges))  # Energy evaluation
            energies_z = np.array(mbr.evaluate_all_energies_z(
                bitstrings_x, bitstrings_z))
            # Overlap matrix with all contributions
            F = np.block([[np.eye(len(bitstrings_x), len(bitstrings_x)), f], [
                        np.conj(f).T, np.eye(len(bitstrings_z), len(bitstrings_z))]])
            
            magnetization_z = np.array(mbr.evaluate_magnetization_z(bitstrings_x, bitstrings_z))
            magnetization_x = np.array(mbr.evaluate_magnetization_x(bitstrings_x, bitstrings_z))
            magnetization_x_staggered = np.array(mbr.evaluate_magnetization_staggered_x(bitstrings_x, bitstrings_z, nx, ny))
            
            d, U = eigh(F) # to ensure numerical stability
            if np.min(np.abs(d)) < 1e-10: 
                F = U @ np.diag(np.clip(d, 1e-10, None)) @ np.conj(U.T)

            for i, h in enumerate(hvec):
                
                H = J * energies_xx + h * energies_z

                D, P = eigh(H, F) # Generalized eigenvalue solving 
                ground_state = P[:, 0]
                # ground_state_inv = np.linalg.inv(P)[0]
                # energy = ground_state_inv @ np.linalg.inv(F) @ H @ ground_state

                norm = np.conj(ground_state) @ F @ ground_state

                mz = np.conj(ground_state) @ magnetization_z @ ground_state / (nx*ny) / norm
                mx = np.conj(ground_state) @ magnetization_x @ ground_state / (nx*ny) / norm
                mx_s = np.conj(ground_state) @ magnetization_x_staggered @ ground_state / (nx*ny) /norm 
                
                logging.info(f"h: {h:0.2f}, degree: {degree}, energy: {D[0]:0.2f}") #Just information
                dest_dict["J"].append(J)
                dest_dict["h"].append(h)
                dest_dict["nx"].append(nx)
                dest_dict["ny"].append(ny)
                dest_dict["energy"].append(D[0])
                dest_dict["degree"].append(degree)
                dest_dict["mz"].append(mz)
                dest_dict["mx"].append(mx)
                dest_dict["mx_s"].append(mx_s)
                # 

            df = pd.DataFrame(dest_dict)
            df.astype({"nx":int, "ny":int, "J":float, "h":float, "energy":float, "degree":int})

        else:
            config = config_from_args(args)
            simulator_cls = simulator_cls_from_args(args)
            graph = LatticeGraph(nx,ny,Boundary.OBC)

            dest_dict = {"nx":[],"ny":[],"J":[],"h":[],"energy":[], "mx":[], "mx_s":[], "mz":[]}

            for h in tqdm(hvec):
                simulator = simulator_cls(config, graph, J, h)
                dest_dict["J"].append(J)
                dest_dict["h"].append(h)
                dest_dict["nx"].append(graph.nx)
                dest_dict["ny"].append(graph.ny)
                dest_dict["energy"].append(simulator.gs_energy)
                dest_dict["mx"].append(simulator.magnetization_x)
                dest_dict["mx_s"].append(simulator.magnetization_x_stag)
                dest_dict["mz"].append(simulator.magnetization_z)

            df_raw = pd.DataFrame(dest_dict)
            df_raw.astype({"nx":int, "ny":int, "J":float, "h":float, "energy":float})
            df = add_config_columns(df_raw, args) 

        df["energy_per_site"] = df["energy"]/(df["nx"]*df["ny"])
        df["type"] = str(args.type)
        df.to_csv(path_out)

    else:
        logging.warning(f"File {path_out} already exists. Skipping calculation.")


# This is the main entry point of the program
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output", default=".",
                        type=str, help="Output directory")
    parser.add_argument("--nx", type=int, required=True,
                        help="Size of the lattice in x direction")
    parser.add_argument("--ny", type=int, required=True,
                        help="Size of the lattice in y direction")
    parser.add_argument("--bdc", default=Boundary.OBC, type=Boundary,
                        choices=['obc'], help="Boundary conditions")
    parser.add_argument("--level", default="info", help="logging level")
    parser.add_argument("--overwrite", default=False,
                        action="store_true", help="Overwrite existing files")
    parser.add_argument("--nsteps", type=int, help="Number of steps")
    parser.add_argument("--hmin", type=float, help="Minimal transverse field")
    parser.add_argument("--hmax", type=float, help="Maximal transverse field")
    parser.add_argument("--J", type=float, default=-1., help="Ising coupling")
    parser.add_argument("--type", type=SimulationType, default=SimulationType.ED,
                        choices=list(SimulationType), help="Type of the simulation")


    # MPS options
    parser.add_argument("--mps-max-chi", type=int, default=80,
                        help="Maximal virtual bond dimension for MPS simulation")
    parser.add_argument("--mps-rel-energy-delta", type=float, default=1e-8,
                        help="Relative change of energy to consider convergence")
    parser.add_argument("--mps-rel-entropy-delta", type=float, default=1e-5,
                        help="Relative change of entropy to consider convergence")
    parser.add_argument("--mps-no-mixer", default=False,
                        action="store_true", help="Do not use the mixer in DMRG")

    # PEPS options
    parser.add_argument("--peps-max-chi", type=int, default=6,
                        help="Virtual bond dimension of the PEPS simulation")
    parser.add_argument("--peps-bd-mps-max-chi", type=int, default=32,
                        help="Maximal virtual bond dimension of the boundary MPS")

    # ED options
    parser.add_argument("--ed-no-sparse", default=False, action="store_true",
                        help="Do not use sparse matrices in ED simulation")

    # MBR options
    parser.add_argument("--mbr-degree", type=int, default=1,
                        help="Degree of approximation in the MBR diagonalization")

    args = parser.parse_args()

    main(args)
