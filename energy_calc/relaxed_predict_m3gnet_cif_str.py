import os
from pymatgen.core import Structure
import matgl
from matgl.ext.ase import Relaxer
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Load the model once, outside of the parallel function
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")

# Directories for input CIF files, output file, and relaxed structures
input_dir = "llamat3_cif"
output_dir = "relaxed_structures_llamat3_m3gnet_str"
output_file = "prediction_results_relaxed_llamat3_m3gnet_str.txt"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def relax_and_predict(cif_file):
    """Load CIF file, relax the structure, predict formation energy, and save the relaxed structure."""
    try:
        # Load the structure from CIF file
        structure = Structure.from_file(cif_file)

        # Initialize Relaxer with the model and perform relaxation
        relaxer = Relaxer(potential=pot)
        relax_results = relaxer.relax(structure, fmax=0.01)
        final_structure = relax_results["final_structure"]

        # Save the relaxed structure to the output directory
        relaxed_filename = os.path.join(output_dir, f"relaxed_{os.path.basename(cif_file)}")
        final_structure.to(fmt="cif", filename=relaxed_filename)

        # Predict formation energy on the relaxed structure
        eform = pot.model.predict_structure(final_structure)
        result = (
            f"Results for {os.path.basename(cif_file)}:\n"
            f"Relaxed structure saved to: {relaxed_filename}\n"
            f"Relaxed structure energy: {float(eform.numpy()):.3f} eV/atom\n" +
            "="*50 + "\n"
        )
    except Exception as e:
        result = f"Failed to process {os.path.basename(cif_file)}: {e}\n" + "="*50 + "\n"
    return result

def main():
    # List all CIF files in the input directory
    cif_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.cif')]
    
    # Initialize the output file
    with open(output_file, "w") as outfile:
        outfile.write("")

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor, open(output_file, "a") as outfile:
        # Submit all tasks to the executor and wrap them in tqdm for progress tracking
        futures = {executor.submit(relax_and_predict, cif_file): cif_file for cif_file in cif_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CIF files", unit="file"):
            # Get the result from each completed future and write it to the output file
            result = future.result()
            outfile.write(result)
            outfile.flush()  # Ensure immediate disk write

if __name__ == "__main__":
    main()
