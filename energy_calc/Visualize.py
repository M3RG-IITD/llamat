import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mace.calculators import MACECalculator
from tqdm import tqdm
from ase import Atoms, units
#from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.neighborlist import neighbor_list
from ase.md.nptberendsen import NPTBerendsen
from ase.md import MDLogger
import random
import sys

from Utils import *
from loguru import logger

from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
    FrameAveraging,
)
from matsciml.lightning import MatSciMLDataModule



class StabilityException(Exception):
    pass

def run_simulation(atoms, runsteps=1000, SimDir='./'):
    traj = []
    logger.info("Calculating initial RDFs ... ")
    _, initial_rdf = get_initial_rdf(atoms, perturb=20, noise_std=0.05, max_atoms=args.max_atoms, replicate=True)
    initial_bond_lengths, Initial_Pair_rdfs = get_bond_lengths_noise(atoms, perturb=20, noise_std=0.05, max_atoms=args.max_atoms, r_max=args.rdf_r_max)
    initial_temperature=args.temp
    # Replicate_system
    replication_factors, size = Symmetricize_replicate(len(atoms), max_atoms=args.max_atoms, box_lengths=atoms.get_cell_lengths_and_angles()[:3])
    atoms = replicate_system(atoms, replication_factors)

    # Set_calculator
    calculator = MACECalculator(model_path=args.model_path, device=args.device, default_dtype='float64')
    atoms.set_calculator(calculator)
    atoms = minimize_structure(atoms, steps=args.minimize_steps)

    # Set_simulation
    #NVE
    # MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
    # initial_energy = atoms.get_total_energy()
    # dyn = VelocityVerlet(atoms, dt=timestep * units.fs)

    #NPT
    dyn = NPTBerendsen(atoms, timestep=args.timestep * units.fs, temperature_K=args.temperature,
                       pressure_au=args.pressure * units.bar , compressibility_au=4.57e-5 / units.bar)
    
    dyn.attach(MDLogger(dyn, atoms, os.path.join(SimDir,"Simulation_thermo.log"), header=True, stress=True,
           peratom=False, mode="w"), interval=args.thermo_interval)

    def write_frame(a=atoms):
        if SimDir is not None:
            a.write(os.path.join(SimDir, f'MD_{atoms.get_chemical_formula()}_NPT.xyz'), append=True)
            
    dyn.attach(write_frame, interval=args.trajdump_interval)

    def append_traj(a=atoms):
        traj.append(a.copy())

    dyn.attach(append_traj, interval=1)

    # def energy_stability(a=atoms):
    #     logger.info("Checking energy stability...", end='\t')
    #     current_energy = atoms.get_total_energy()
    #     energy_error = abs((current_energy - initial_energy) / initial_energy)
    #     if energy_error > args.energy_tolerence:
    #         logger.error(f"Unstable : Energy_error={energy_error:.6g} (> {args.energy_tolerence:.6g})")
    #         raise StabilityException("Energy_criterion violated. Stopping the simulation.")
    #     else:
    #         logger.info(f"Stable : Energy_error={energy_error:.6g} (< {args.energy_tolerence:.6g})")

    # dyn.attach(energy_stability, interval=args.energy_criteria_interval)

    def temperature_stability(atoms, initial_temperature, temperature_tolerance):
        if len(traj) >= args.initial_equilibration_period:
            
            logger.info("Checking temperature stability...", end='\t')
            current_temperature = atoms.get_temperature()
            temperature_error = abs((current_temperature - initial_temperature) / initial_temperature)
            if temperature_error > temperature_tolerance:
                logger.error(f"Unstable : Temperature_error={temperature_error:.6g} (> {temperature_tolerance:.6g})")
                raise StabilityException("Temperature criterion violated. Stopping the simulation.")
            else:
                logger.info(f"Stable : Temperature_error={temperature_error:.6g} (< {temperature_tolerance:.6g})")
        
    # Attach the temperature stability check to the dynamics object
    dyn.attach(temperature_stability, interval=args.temperature_criteria_interval,
            atoms=atoms, initial_temperature=initial_temperature, 
            temperature_tolerance=args.temperature_tolerance)


    def calculate_rmsd(traj):
        initial_positions = traj[0].get_positions()
        N = len(traj[0])
        T = len(traj)
        displacements = np.zeros((N, T, 3))
        for t in range(T):
            current_positions = traj[t].get_positions()
            displacements[:, t, :] = current_positions - initial_positions
        msd = np.mean(np.sum(displacements**2, axis=2), axis=1)
        rmsd = np.sqrt(msd)
        return rmsd

    def calculate_average_nn_distance(atoms):
        i, j, _ = neighbor_list('ijd', atoms, cutoff=5.0)
        distances = atoms.get_distances(i, j, mic=True)
        return np.mean(distances)

    def lindemann_stability(a=atoms):
        if len(traj) >= args.lindemann_traj_length:
            logger.info("Checking lindemann stability...", end='\t')
            rmsd = calculate_rmsd(traj[-args.lindemann_traj_length:])
            avg_nn_distance = calculate_average_nn_distance(traj[0])
            lindemann_coefficient = np.mean(rmsd) / avg_nn_distance
            if lindemann_coefficient > args.max_linedmann_coefficient:
                logger.error(f"Unstable : Lindemann_coefficient={lindemann_coefficient:.6g} (> {args.max_linedmann_coefficient:.6g})")
                logger.error(f"Lindemann_stability criterion violated {lindemann_coefficient:.6g} > {args.max_linedmann_coefficient:.6g}, Stopping the simulation.")
                raise StabilityException()
            else:
                logger.info(f"Stable : Lindemann_coefficient={lindemann_coefficient:.6g} (< {args.max_linedmann_coefficient:.6g})")

    dyn.attach(lindemann_stability, interval=args.lindemann_criteria_interval)

    def rdf_stability(a=atoms):
        if len(traj) >= args.rdf_traj_length:
            logger.info("Checking RDF stability...", end='\t')
            r, rdf = get_rdf(traj[-args.rdf_traj_length:], r_max=args.rdf_r_max)
            RDF_len=min(len(rdf),len(initial_rdf))
            r=r[:RDF_len]
            rdf=rdf[:RDF_len]
            initial_rdf_=initial_rdf[:RDF_len]
            error_rdf = 100 * (((rdf - initial_rdf_)**2).sum()) / (((initial_rdf_)**2).sum())
            
            # Plotting the RDF
            plt.figure()
            plt.plot(r, initial_rdf_, label='Initial RDF')
            plt.plot(r, rdf, label='Simulated RDF')
            plt.xlabel('Distance (r)')
            plt.ylabel('RDF')
            plt.legend()
            plt.title(f'RDF Comparison\nInitial vs Simulated\nError={error_rdf:.6g}')
            plot_path = os.path.join(SimDir, f'RDF_{atoms.get_chemical_formula()}_{len(traj)}.png')
            plt.savefig(plot_path)
            logger.info("Saved figure at {}", plot_path)
            plt.close()
            if error_rdf > args.max_rdf_error_percent:
                logger.error(f"Unstable : RDF Error={error_rdf:.6g} (> {args.max_rdf_error_percent:.6g})")
                logger.error(f"RDF criterion violated. Stopping the simulation. WF={error_rdf:.6g}")
                raise StabilityException()
            else:
                logger.info(f"Stable : RDF Error={error_rdf:.6g} (< {args.max_rdf_error_percent:.6g})")

    dyn.attach(rdf_stability, interval=args.rdf_criteria_interval)

    def bond_lengths_stability(a=atoms):
        if len(traj) >= args.lindemann_traj_length:
            logger.info("Checking Bonds stability...", end='\t')
            curr_bond_lengths, Pair_rdfs = get_bond_lengths_TrajAvg(traj[-args.rdf_traj_length:], r_max=args.rdf_r_max)
            for key in curr_bond_lengths.keys():
                r, initial_rdf = Initial_Pair_rdfs[key]
                r, rdf = Pair_rdfs[key]  
                RDF_len=min(len(rdf),len(initial_rdf))
                r=r[:RDF_len]
                rdf=rdf[:RDF_len]
                initial_rdf_=initial_rdf[:RDF_len]
                error_percent = 100 * (((rdf - initial_rdf_)**2).sum()) / (((initial_rdf_)**2).sum())
                
                plt.figure()
                plt.plot(r, initial_rdf_, label='Initial RDF')
                plt.plot(r, rdf, label='Simulated RDF')
                plt.xlabel('Distance (r)')
                plt.ylabel('RDF')
                plt.legend()
                plt.title(f'RDF Comparison: Bond {key}={curr_bond_lengths[key]:.6g}, Initial={initial_bond_lengths[key]:.6g}, Error={error_percent:.6g}')
                plot_path = os.path.join(SimDir, f'PartialRDF_{atoms.get_chemical_formula()}_{key}_{len(traj)}.png')
                plt.savefig(plot_path)
                logger.info("Saved figure at {}", plot_path)
                if False: #error_percent > args.max_bond_error_percent:
                    logger.error(f"Unstable : Bond {key}={curr_bond_lengths[key]:.6g}, Initial={initial_bond_lengths[key]:.6g}, Error={error_percent:.6g} (> {args.max_bond_error_percent:.6g})")
                    logger.error(f"Bond length stability violated. Stopping the simulation. Bond {key}={curr_bond_lengths[key]:.6g}, Initial={initial_bond_lengths[key]:.6g}")                
                    raise StabilityException()
                else:
                    logger.info(f"Stable : Bond {key}: {error_percent: .6g} < {args.max_bond_error_percent:.6g} % Error")

    dyn.attach(bond_lengths_stability, interval=args.rdf_criteria_interval)

    try:
        logger.info(f"Simulating {atoms.get_chemical_formula()} {len(atoms)} atoms system ....")
        counter = 0
        for k in tqdm(range(runsteps)):
            dyn.run(1)
            counter += 1
        return runsteps  # Simulation completed successfully
    except StabilityException as e:
        logger.error(f"Simulation of {atoms.get_chemical_formula()} {len(atoms)} atoms system failed after {counter} steps")
        return len(traj)  # Return the number of steps completed before failure



class TestArgs:
    runsteps=20000
    model_path="/scratch/scai/phd/aiz238703/MDBENCHGNN/mace_universal_2.0/2024-01-07-mace-128-L2_epoch-199.model"
    timestep=1.0
    temp=298
    out_dir="/scratch/scai/phd/aiz238703/MDBENCHGNN/mace_universal_2.0/Stability/OutputTestingStability/"
    device='cuda'
    replicate=True
    max_atoms=200  #Replicate upto max_atoms (Min. will be max_atoms/2) (#Won't reduce if more than max_atoms)
    #energy_tolerence=0.1
    #energy_criteria_interval=100
    max_linedmann_coefficient=0.3
    lindemann_criteria_interval=1000
    lindemann_traj_length=1000
    max_rdf_error_percent=80
    max_bond_error_percent=80
    bond_criteria_interval=1000
    rdf_dr=0.02
    rdf_r_max=6.0
    rdf_traj_length=1000
    rdf_criteria_interval=1000
    trajdump_interval=10
    minimize_steps=200
    temperature=300
    temperature_tolerance=0.8
    thermo_interval=10
    pressure=1.01325
    temperature_criteria_interval=1000
    initial_equilibration_period=3000
args=TestArgs()

# Seed for the Python random module
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)  # if you are using multi-GPU.
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False



# Load Data
dm = MatSciMLDataModule(
    "MaterialsProjectDataset",
    train_path="/scratch/scai/phd/aiz238703/MDBENCHGNN/mace_universal_2.0/Stability/stability_new/",
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(cutoff_radius=6.0, adaptive_cutoff=True),
            PointCloudToGraphTransform(
                "pyg",
                node_keys=["pos", "atomic_numbers"],
            ),
        ],
    },
    batch_size=1,
)

dm.setup()
train_loader = dm.train_dataloader()
dataset_iter = iter(train_loader)

time_steps = []
unreadable_files = []
Range=[0,120]


index=int(sys.argv[1])
print("Index:",index)

counter_batch=0
for batch in train_loader:
    if counter_batch==index:
        atoms=convBatchtoAtoms(batch)
        SimDir=os.path.join(args.out_dir, f'Simulation_{atoms.get_chemical_formula()}')
        os.makedirs(SimDir, exist_ok=True)
        # Initialize logger
        logger.add(os.path.join(SimDir,"simulation.log"), rotation="500 MB")
        logger.info("All seeds set!")
        steps_completed = run_simulation(atoms,args.runsteps,SimDir)        
        time_steps.append(steps_completed)
        logger.info("System: {} : {} with originally {} atoms stopped at {} steps", counter_batch, atoms.get_chemical_formula(), len(atoms), steps_completed)
        counter_batch+=1
    else:
        counter_batch+=1
        continue

logger.info("Completed...")
logger.info("Time Steps: {}", time_steps)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run MD simulation with MACE model")
    
#     parser.add_argument("--model_path", type=str, default="/home/m3rg2000/Simulation/checkpoints-2024/FAENet_250k.ckpt", help="Path to the model")
#     parser.add_argument("--device", type=str, default="cpu", help="Device:['cpu','cuda']")
#     parser.add_argument("--out_dir", type=str, default="/home/m3rg2000/Simulation/OutputTestingStability", help="Output path")
#     parser.add_argument("--temp", type=float, default=298, help="Temperature in Kelvin")
#     parser.add_argument("--timestep", type=float, default=1.0, help="Timestep in fs units")
#     parser.add_argument("--runsteps", type=int, default=1000, help="No. of steps to run")
#     parser.add_argument("--sys_name", type=str, default='System', help="System name")
#     parser.add_argument("--energy_criteria_interval", type=int, default=10, help="Energy Criteria Interval")
#     parser.add_argument("--replicate", type=bool, default=True, help="Replicate the system")
#     parser.add_argument("--max_atoms", type=int, default=200, help="Max atoms (Min. will be max_atoms/2)")
#     parser.add_argument("--energy_tolerance", type=float, default=0.1, help="Energy tolerance")
#     parser.add_argument("--max_lindemann_coefficient", type=float, default=0.1, help="Max Lindemann coefficient")
#     parser.add_argument("--lindemann_criteria_interval", type=int, default=50, help="Lindemann criteria interval")
#     parser.add_argument("--lindemann_traj_length", type=int, default=50, help="Lindemann trajectory length")
#     parser.add_argument("--max_rdf_error_percent", type=float, default=10, help="Max RDF error percent")
#     parser.add_argument("--max_bond_error_percent", type=float, default=10, help="Max bond error percent")
#     parser.add_argument("--bond_criteria_interval", type=int, default=100, help="Bond criteria interval")
#     parser.add_argument("--rdf_dr", type=float, default=0.01, help="RDF dr")
#     parser.add_argument("--rdf_r_max", type=float, default=6.0, help="RDF r max")
#     parser.add_argument("--rdf_traj_length", type=int, default=500, help="RDF trajectory length")
#     parser.add_argument("--rdf_criteria_interval", type=int, default=100, help="RDF criteria interval")
#     parser.add_argument("--trajdump_interval", type=int, default=1, help="Trajectory dump interval")
    
#     args = parser.parse_args()
#     main(args)

