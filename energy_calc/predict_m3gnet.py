import os
from pymatgen.core import Structure
import matgl
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Load the prediction model once, outside of the parallel function
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")

# Directory containing CIF files and output file
input_dir = "llamat2_cif"                              
output_file = "M3GNet-MP-2021.2.8-PES_llamat2.txt"

def predict_formation_energy(cif_file):
    """Function to load a CIF file, create a structure, and predict formation energy."""
    try:
        struct = Structure.from_file(cif_file)
        eform = pot.model.predict_structure(struct)
        result = f"Results for {os.path.basename(cif_file)}: {float(eform.numpy()):.3f} eV/atom\n" + "="*50 + "\n"
    except Exception as e:
        result = f"Failed to process {os.path.basename(cif_file)}: {e}\n" + "="*50 + "\n"
    return result

def main():
    # List all CIF files in the input directory
    cif_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.cif')]
    
    # Initialize the output file
    with open(output_file, "w") as outfile:
        outfile.write("")

    # Use ProcessPoolExecutor for parallel predictions
    with ProcessPoolExecutor() as executor, open(output_file, "a") as outfile:
        # Submit all tasks to the executor and wrap them in tqdm for a progress bar
        futures = {executor.submit(predict_formation_energy, cif_file): cif_file for cif_file in cif_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CIF files", unit="file"):
            # Get the result from each completed future
            result = future.result()
            outfile.write(result)
            outfile.flush()  # Ensure data is written to disk immediately

if __name__ == "__main__":
    main()
