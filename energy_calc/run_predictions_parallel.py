import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Directory containing CIF files and output file
input_dir = "llamat2_cif"
output_file = "prediction_results.txt"

def run_mgl_predict(cif_file):
    """Function to run mgl predict on a single CIF file."""
    command = ["mgl", "predict", "--model", "M3GNet-MP-2018.6.1-Eform", "--infile", os.path.join(input_dir, cif_file)]
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Prepare output with file name for readability
    output = f"Results for {cif_file}:\n{result.stdout}\n" + "="*50 + "\n"
    return output

def main():
    # Get list of all CIF files in the directory
    cif_files = [f for f in os.listdir(input_dir) if f.endswith('.cif')]
    
    # Initialize the output file to ensure it's empty before writing
    with open(output_file, "w") as outfile:
        outfile.write("")

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor, open(output_file, "a") as outfile:
        # Submit all tasks to the executor and wrap them in tqdm for progress tracking
        futures = {executor.submit(run_mgl_predict, cif_file): cif_file for cif_file in cif_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CIF files", unit="file"):
            # Get the result from each completed future and write it to the output file
            result = future.result()
            outfile.write(result)

if __name__ == "__main__":
    main()
