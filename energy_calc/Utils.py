import torch
import numpy as np
from matsciml.preprocessing.atoms_to_graphs import *
from matsciml.datasets.trajectory_lmdb import data_list_collater
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
    FrameAveraging,
)


from ase import Atoms, units
from ase.optimize import FIRE
from ase.geometry.analysis import Analysis
import matplotlib.pyplot as plt


a2g=AtomsToGraphs(max_neigh=200,
            radius=6,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=True,
            r_fixed=True,)
f_avg=FrameAveraging(frame_averaging="3D", fa_method="stochastic")

def convAtomstoBatch(atoms):
    data_obj=a2g.convert(atoms)
    Reformatted_batch={
        'cell' : data_obj.cell,
        'natoms' :  torch.Tensor([data_obj.natoms]).unsqueeze(0),
        'edge_index' : [data_obj.edge_index.shape],
        'cell_offsets': data_obj.cell_offsets,
        'y' : None,
        'force' : None, 
        'fixed' : [data_obj.fixed],
        'tags' : None,
        'sid' :None,
        'fid' : None,
        'dataset' : 'S2EFDataset',
        'graph' : data_list_collater([data_obj]),
    }
    Reformatted_batch=f_avg(Reformatted_batch)
    return Reformatted_batch

def convBatchtoAtoms(batch):
    # data_obj=a2g.convert(atoms)
    curr_atoms = Atoms(
            positions=batch['graph'].pos,
            cell = batch['cell'][0],
            numbers=batch['graph'].atomic_numbers,
            pbc=True) # True or false
    
    return curr_atoms

def minimize_structure(atoms, fmax=0.05, steps=50):
    """
    Perform energy minimization on the given ASE Atoms object using the FIRE optimizer.
 
    Parameters:
    atoms (ase.Atoms): The Atoms object to be minimized.
    fmax (float): The maximum force tolerance for the optimization (default: 0.01 eV/Å).
    steps (int): The maximum number of optimization steps (default: 1000).
 
    Returns:
    ase.Atoms: The minimized Atoms object.
    """
    dyn = FIRE(atoms, trajectory=None)
    dyn.run(fmax=fmax, steps=steps)
    return atoms

def min_height(cell_matrix):
    """
    Calculate the perpendicular heights in three directions given a 3x3 cell matrix.
    """
    a,b,c = cell_matrix[:, 0],cell_matrix[:, 1],cell_matrix[:, 2]
    volume = abs(np.dot(a, np.cross(b, c)))
    # Calculate the cross products
    a_cross_b,b_cross_c,c_cross_a = np.linalg.norm(np.cross(a, b)),np.linalg.norm(np.cross(b, c)),np.linalg.norm(np.cross(c, a))
    # Calculate the perpendicular heights
    height_a,height_b,height_c = abs(volume / a_cross_b), abs(volume / b_cross_c),abs(volume / c_cross_a)
    return min(height_a, height_b, height_c)

def perturb_config(atoms,displacement_std=0.01):
    # Create a new Atoms object with the perturbed positions
    positions = atoms.get_positions()
    displacements = np.random.normal(scale=displacement_std, size=positions.shape)
    new_positions = positions + displacements
    new_perturbed_atoms = atoms.copy()
    new_perturbed_atoms.set_positions(new_positions)
    return new_perturbed_atoms

def plot_pair_rdfs(Pair_rdfs,shift=0):
    counter=0
    plt.figure()
    for key in Pair_rdfs.keys():
        plt.plot(Pair_rdfs[key][0],Pair_rdfs[key][1]+shift*counter,label=key)
        counter+=1
    plt.legend(loc=(1.2,0))
    plt.xlabel("r (Angstrom)")
    plt.ylabel("g(r)")
    plt.show()

def replicate_system(atoms, replicate_factors):
    """
    Replicates the given ASE Atoms object according to the specified replication factors.
    """
    nx, ny, nz = replicate_factors
    original_cell = atoms.get_cell()
    original_positions = atoms.get_positions()@original_cell  #Scaled or Unscaled ?
    original_numbers = atoms.get_atomic_numbers()
    x_cell,y_cell,z_cell=original_cell[0],original_cell[1],original_cell[2]
    new_numbers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                new_numbers+=[original_numbers]
    pos_after_x=np.concatenate([original_positions+i*x_cell for i in range(nx)])
    pos_after_y=np.concatenate([pos_after_x+i*y_cell for i in range(ny)])
    pos_after_z=np.concatenate([pos_after_y+i*z_cell for i in range(nz)])
    new_cell = [nx * original_cell[0],ny * original_cell[1],nz * original_cell[2]]
    new_atoms = Atoms(numbers=np.concatenate(new_numbers), positions=pos_after_z, cell=new_cell, pbc=atoms.get_pbc())
    return new_atoms

def write_xyz(Filepath,atoms):
    '''Writes ovito xyz file'''
    R=atoms.get_position()
    species=atoms.get_atomic_numbers()
    cell=atoms.get_cell()
    f=open(Filepath,'w')
    f.write(str(R.shape[0])+"\n")
    flat_cell=cell.flatten()
    f.write(f"Lattice=\"{flat_cell[0]} {flat_cell[1]} {flat_cell[2]} {flat_cell[3]} {flat_cell[4]} {flat_cell[5]} {flat_cell[6]} {flat_cell[7]} {flat_cell[8]}\" Properties=species:S:1:pos:R:3 Time=0.0")
    for i in range(R.shape[0]):
        f.write("\n"+str(species[i])+"\t"+str(R[i,0])+"\t"+str(R[i,1])+"\t"+str(R[i,2]))

def Symmetricize_replicate(curr_atoms, max_atoms, box_lengths):
    replication=[1,1,1]
    atom_count=curr_atoms
    lengths=box_lengths
    while atom_count<(max_atoms//2):
        direction=np.argmin(box_lengths)
        replication[direction]+=1
        lengths[direction]=box_lengths[direction]*replication[direction]
        atom_count=curr_atoms*replication[0]*replication[1]*replication[2]
    return replication,atom_count

def get_pairs(atoms):
    Atom_types=np.unique(atoms.get_chemical_symbols())
    Pairs=[]
    for i in range(len(Atom_types)):
        for j in range(i,len(Atom_types)):
            Pairs+=[[Atom_types[i],Atom_types[j]]]   
    return Pairs

def getfirstpeaklength(r,rdf, r_max=6.0):
    bin_size=(r[-1]-r[0])/len(r)
    cut_index=int(r_max/bin_size)
    cut_index=min(cut_index,len(r))
    Peak_index=np.argmax(rdf[:cut_index])
    #Returns : Peak index and Bond length
    return Peak_index , r[Peak_index] 

def get_partial_rdfs(Traj,r_max=6.0,dr=0.01):
    rmax=min(r_max,min_height(Traj[0].get_cell())/2.7)
    analysis = Analysis(Traj)
    dr=dr
    nbins=int(rmax/dr)
    pairs_list=get_pairs(Traj[0])
    Pair_rdfs=dict()
    for pair in pairs_list:
        rdf = analysis.get_rdf(rmax=rmax, nbins=nbins, imageIdx=None, elements=pair, return_dists=True)
        x=rdf[0][1]
        y=np.array([rdf[k][0] for k in range(len(rdf))]).mean(axis=0)
        Pair_rdfs['-'.join(pair)]=[x,y]
    return Pair_rdfs

def get_partial_rdfs_smoothened(inp_atoms,perturb=10,noise_std=0.01,max_atoms=300,r_max=6.0,dr=0.01):
    atoms=inp_atoms.copy()
    replication_factors,_=Symmetricize_replicate(len(atoms), max_atoms=max_atoms, box_lengths=atoms.get_cell_lengths_and_angles()[:3])
    atoms=replicate_system(atoms,replication_factors)
    Traj=[perturb_config(atoms,noise_std) for k in range(perturb)]
    return get_partial_rdfs(Traj,r_max=r_max,dr=dr)

def get_bond_lengths_noise(inp_atoms,perturb=10,noise_std=0.01,max_atoms=300,r_max=6.0,dr=0.01):
    Pair_rdfs=get_partial_rdfs_smoothened(inp_atoms,perturb=perturb,noise_std=noise_std,max_atoms=max_atoms,r_max=r_max,dr=dr)
    Bond_lengths=dict()
    for key in Pair_rdfs:
        r,rdf=Pair_rdfs[key]
        Bond_lengths[key]=getfirstpeaklength(r,rdf)[1]
    return Bond_lengths , Pair_rdfs

def get_bond_lengths_TrajAvg(Traj,r_max=6.0,dr=0.01):
    Pair_rdfs=get_partial_rdfs(Traj,r_max=r_max,dr=dr)
    Bond_lengths=dict()
    for key in Pair_rdfs:
        r,rdf=Pair_rdfs[key]
        Bond_lengths[key]=getfirstpeaklength(r,rdf)[1]
    return Bond_lengths , Pair_rdfs



def get_initial_rdf(inp_atoms,perturb=10,noise_std=0.01,max_atoms=300,replicate=False,Structid=0,r_max=6.0,dr=0.01):
    atoms=inp_atoms.copy()
    #write_xyz(f"StabilityXYZData2/{Structid}.xyz",atoms.get_positions(),atoms.get_chemical_symbols(),atoms.get_cell())
    if replicate:
        n_atoms=len(atoms)
        replication_factors,size=Symmetricize_replicate(len(atoms), max_atoms=max_atoms, box_lengths=atoms.get_cell_lengths_and_angles()[:3])
        atoms=replicate_system(atoms,replication_factors)
    rmax=min(r_max,min_height(atoms.get_cell())/2.7)
    #atoms.rattle(0.01)
    analysis = Analysis([perturb_config(atoms,noise_std) for k in range(perturb)])
    #write_xyz(f"StabilityXYZDataReplicated2/{Structid}.xyz",atoms.get_positions(),atoms.get_chemical_symbols(),atoms.get_cell())
    dr=dr
    nbins=int(rmax/dr)
    rdf = analysis.get_rdf(rmax=rmax, nbins=nbins, imageIdx=None, elements=None, return_dists=True)
    x=rdf[0][1]
    y=np.array([rdf[k][0] for k in range(len(rdf))]).mean(axis=0)
    return x,y

def get_rdf(Traj,r_max=6.0,dr=0.01):
    rmax=min(r_max,min_height(Traj[0].get_cell())/2.7)
    analysis = Analysis(Traj)
    dr=dr
    nbins=int(rmax/dr)
    rdf = analysis.get_rdf(rmax=rmax, nbins=nbins, imageIdx=None, elements=None, return_dists=True)
    x=rdf[0][1]
    y=np.array([rdf[k][0] for k in range(len(rdf))]).mean(axis=0)
    return x,y
