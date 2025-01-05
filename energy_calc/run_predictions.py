import os
import subprocess
from tqdm import tqdm

# Directory containing CIF files and output file
input_dir = "llamat2_cif"
output_file = "prediction_results.txt"

# Open the output file in write mode to clear previous contents
with open(output_file, "w") as outfile:
    # Get list of all CIF files in the directory
    cif_files = [f for f in os.listdir(input_dir) if f.endswith('.cif')]
    
    # Initialize the progress bar with the total number of files
    for cif_file in tqdm(cif_files, desc="Processing CIF files", unit="file"):
        # Construct the command to be executed
        command = ["mgl", "predict", "--model", "M3GNet-MP-2018.6.1-Eform", "--infile", os.path.join(input_dir, cif_file)]
        
        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Write the command output to the file
        outfile.write(f"Results for {cif_file}:\n")
        outfile.write(result.stdout)
        outfile.write("\n" + "="*50 + "\n")  # Separator for readability
