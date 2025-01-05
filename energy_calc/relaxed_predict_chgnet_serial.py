import os
from pymatgen.core import Structure
import matgl
from matgl.ext.ase import Relaxer
from tqdm import tqdm

# Load the model once, outside of the function
#model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
pot = matgl.load_model("CHGNet-MPtrj-2024.2.13-PES-11M")

# Directory containing CIF files and output file
input_dir = "llamat3_cif"
output_file = "relaxed_llamat3_chgnet.txt"

def relax_and_predict(cif_file):
    """Load CIF file, relax the structure, and predict formation energy."""
    try:
        # Load the structure from CIF file
        structure = Structure.from_file(cif_file)
    
        # Initialize Relaxer with the model and perform relaxation
        relaxer = Relaxer(potential=pot)
        relax_results = relaxer.relax(structure, fmax=0.01)
        final_structure = relax_results["final_structure"]
    
        # Predict formation energy on the relaxed structure
        eform = pot.model.predict_structure(final_structure)
        result = (
            f"Results for {os.path.basename(cif_file)}:\n"
            f"Relaxed structure energy: {float(eform.detach().numpy()):.3f} eV/atom\n" +
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
    
    # Serial processing with tqdm for progress tracking
    with open(output_file, "a") as outfile:
        for cif_file in tqdm(cif_files, desc="Processing CIF files", unit="file"):
            result = relax_and_predict(cif_file)
            outfile.write(result)
            outfile.flush()  # Ensure immediate disk write

if __name__ == "__main__":
    main()
