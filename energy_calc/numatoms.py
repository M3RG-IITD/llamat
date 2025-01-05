import os
import json
from ase.io import read
from tqdm import tqdm

# Get a sorted list of files in the current directory
files = sorted(os.listdir('./llamat3_cif'))

# Initialize a dictionary to store the number of atoms for each CIF file
cif2natom = {}

# Loop through all files and process CIF files
for f in tqdm(files):
    try:
        # Read the CIF file and get the number of atoms
        data = read(f'./llamat3_cif/{f}')
        cif2natom[f] = len(data)
    except Exception as e:
        # Print the name of the file that caused an error
        print(f"Error processing file: {f}, Error: {e}")

# Save the results to a JSON file
with open("llamat3_natoms.json", 'w') as f:
    json.dump(cif2natom, f)

# Reset the dictionary for any further use
#cif2natom = {}
