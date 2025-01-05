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

def minimize_structure(atoms,calculator, fmax=0.05, steps=50):
    """
    Perform energy minimization on the given ASE Atoms object using the FIRE optimizer.

    Parameters:
    atoms (ase.Atoms): The Atoms object to be minimized.
    fmax (float): The maximum force tolerance for the optimization (default: 0.05 eV/Å).
    steps (int): The maximum number of optimization steps (default: 50).

    Returns:
    tuple: A tuple containing:
        - ase.Atoms: The minimized Atoms object.
        - ase.io.trajectory.Trajectory: The trajectory of the optimization process.
    """
    atoms.set_calculator(calculator)
    traj = []
    dyn = FIRE(atoms)
    def append_traj(a=atoms):
        traj.append(a.copy())
    dyn.attach(append_traj, interval=1)
    dyn.run(fmax=fmax, steps=steps)
    return atoms, traj

#Args
class TestArgs:
    structure_path="/scratch/scai/phd/aiz238703/MDBENCHGNN/Repulsive/Data/lips/data/test/botnet.xyz"
    model_pathUniv="/scratch/scai/phd/aiz238703/MDBENCHGNN/mace_universal_2.0/2024-01-07-mace-128-L2_epoch-199.model"
    model_pathEgraFF="/home/scai/phd/aiz238703/MACE_model_run-123_swa.model"
    device='cuda'
    default_dtype='float64'
    minimize_steps=500
configs=TestArgs()

calculator = MACECalculator(model_path=configs.model_pathUniv, device=configs.device, default_dtype=configs.default_dtype)
Folders={"llamat2":"/scratch/scai/phd/aiz238703/MDBENCHGNN/mace_universal_2.0/Visualisation/VisCode/llamat2_cif",
        "llamat3":"/scratch/scai/phd/aiz238703/MDBENCHGNN/mace_universal_2.0/Visualisation/VisCode/llamat3_cif/"}


OutputF1=dict()
counter=1
file_write=open("llamat3.txt",'w')
for file in tqdm(os.listdir(Folders["llamat3"])):
    counter+=1
    try:
        LoadedStructure=ase.io.read(os.path.join(Folders["llamat3"],file))
        LoadedStructure.calc=calculator
        Energy=LoadedStructure.get_total_energy()
        file_write.write(f"{counter},{int(file[:-4])},{Energy},\n")
    except:
        Energy='NaN'
        file_write.write(f"{counter},{int(file[:-4])},{Energy},\n")
    file_write.flush()
    # OutputF1[file]= Energy
file.close()
