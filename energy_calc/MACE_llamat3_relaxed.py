import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mace.calculators import MACECalculator
from tqdm import tqdm
import ase
from ase import Atoms, units
from ase.neighborlist import neighbor_list
from ase.md import MDLogger
import random
import sys
from loguru import logger
import plotly

from ase import Atoms
from ase.optimize import FIRE
from ase.io.trajectory import Trajectory

def minimize_structure(atoms, calculator, fmax=0.01, steps=200):
    """
    Perform energy minimization on the given ASE Atoms object using the FIRE optimizer.

    Parameters:
    atoms (ase.Atoms): The Atoms object to be minimized.
    fmax (float): The maximum force tolerance for the optimization (default: 0.01 eV/Ang).
    steps (int): The maximum number of optimization steps (default: 1000).

    Returns:
    tuple: A tuple containing:
        - ase.Atoms: The minimized Atoms object.
        - ase.io.trajectory.Trajectory: The trajectory of the optimization process.
    """
    atoms.set_calculator(calculator)
    traj = []
    dyn = FIRE(atoms)
    #def append_traj(a=atoms):
    #    traj.append(a.copy())
    #dyn.attach(append_traj, interval=1)
    dyn.run(fmax=fmax, steps=steps) #Set logfile=None for no verbose output
    return atoms, traj

# Args
class TestArgs:
    model_pathUniv = "/scratch/scai/phd/aiz238703/MDBENCHGNN/mace_universal_2.0/2024-01-07-mace-128-L2_epoch-199.model"
    device = 'cuda'
    default_dtype = 'float64'
    minimize_steps = 200
configs = TestArgs()

# Initialize calculator
calculator = MACECalculator(model_path=configs.model_pathUniv, device=configs.device, default_dtype=configs.default_dtype)

# Define folder paths
Folders = {
    "llamat2": "/scratch/scai/phd/aiz238703/MDBENCHGNN/mace_universal_2.0/Visualisation/VisCode/llamat2_cif",
    "llamat3": "/scratch/scai/phd/aiz238703/MDBENCHGNN/mace_universal_2.0/Visualisation/VisCode/llamat3_cif/"
}

# Initialize output and file writing
counter = 1
file_write = open("llamat3_MACE_rlx_Engy_5.txt", 'w')
files = sorted(os.listdir(Folders["llamat3"]))
print("Len Files:",len(files))
# Loop through structures in the folder, minimize and compute energy
for file in tqdm(files[8000:]):
    counter += 1
    try:
        # Load structure
        LoadedStructure = ase.io.read(os.path.join(Folders["llamat3"], file))
        
        # Set the calculator and minimize structure
        minimized_structure, traj = minimize_structure(LoadedStructure, calculator, fmax=0.1, steps=200)
        
        # Calculate energy after minimization
        Energy = minimized_structure.get_total_energy()
        
        # Write results to file
        file_write.write(f"{counter},{int(file[:-4])},{Energy},\n")
    except Exception as e:
        # Handle any errors during processing
        file_write.write(f"{counter},{int(file[:-4])},NaN\n")
        logger.error(f"Error processing file {file}: {e}")

    file_write.flush()  # Ensure data is written after each file

file_write.close()  # Close the output file
