"""
Script to extract composition from CIF strings and compare with pretty_formula.
Based on the unconditional_eval_metrics.ipynb notebook.
"""

import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from collections import Counter
import re
from tqdm import tqdm
import warnings

def extract_composition_from_cif(cif_str):
    """
    Extract chemical composition from a CIF string using pymatgen.
    
    Args:
        cif_str (str): CIF string representation of the crystal structure
        
    Returns:
        str: Chemical formula (e.g., 'Ga2Te2') or None if parsing fails
    """
    try:
        # Parse the CIF string to get structure
        structure = Structure.from_str(cif_str, fmt="cif")
        
        # Get composition from the structure
        composition = structure.composition
        
        # Get the reduced formula (e.g., 'Ga2Te2')
        formula = composition.reduced_formula
        
        return formula
        
    except Exception as e:
        print(f"Error parsing CIF: {e}")
        return None

def normalize_formula(formula):
    """
    Normalize chemical formula for comparison by:
    1. Removing spaces
    2. Converting to lowercase
    3. Sorting elements alphabetically
    
    Args:
        formula (str): Chemical formula
        
    Returns:
        str: Normalized formula
    """
    if pd.isna(formula) or formula is None:
        return None
    
    # Convert to string and remove spaces
    formula = str(formula).strip().replace(" ", "")
    
    # Parse the formula to extract elements and counts
    # This regex finds element symbols followed by optional numbers
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    if not matches:
        return formula.lower()
    
    # Create a dictionary of element counts
    element_counts = {}
    for element, count in matches:
        count = int(count) if count else 1
        element_counts[element] = element_counts.get(element, 0) + count
    
    # Sort elements alphabetically and reconstruct formula
    sorted_elements = sorted(element_counts.keys())
    normalized_parts = []
    for element in sorted_elements:
        count = element_counts[element]
        if count == 1:
            normalized_parts.append(element)
        else:
            normalized_parts.append(f"{element}{count}")
    
    return "".join(normalized_parts)

def compare_compositions(extracted_formula, pretty_formula):
    """
    Compare extracted formula with pretty_formula.
    
    Args:
        extracted_formula (str): Formula extracted from CIF
        pretty_formula (str): Pretty formula from the dataset
        
    Returns:
        bool: True if formulas match, False otherwise
    """
    if extracted_formula is None or pretty_formula is None:
        return False
    
    # Normalize both formulas
    norm_extracted = normalize_formula(extracted_formula)
    norm_pretty = normalize_formula(pretty_formula)
    
    return norm_extracted == norm_pretty

def main():
    """
    Main function to process the CSV file and compare compositions.
    """
    # Load the CSV file
    csv_path = 'actual_pred_cif_llamat2_9064_llamat2_1758654262.csv'
    print(f"Loading data from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_path}")
        return
    
    # Check required columns
    required_columns = ['cif', 'pretty_formula']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Extract compositions from CIF strings
    print("Extracting compositions from CIF strings...")
    extracted_compositions = []
    successful_extractions = 0
    
    for idx, cif_str in enumerate(tqdm(df['cif'], desc="Processing CIFs")):
        if pd.isna(cif_str):
            extracted_compositions.append(None)
            continue
            
        composition = extract_composition_from_cif(cif_str)
        extracted_compositions.append(composition)
        
        if composition is not None:
            successful_extractions += 1
        
        # Print progress every 1000 iterations
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} CIFs, {successful_extractions} successful extractions")
    
    # Add extracted compositions to dataframe
    df['extracted_formula'] = extracted_compositions
    
    # Compare with pretty_formula
    print("Comparing extracted formulas with pretty_formula...")
    matches = []
    
    for idx, row in df.iterrows():
        match = compare_compositions(row['extracted_formula'], row['pretty_formula'])
        matches.append(match)
    
    df['formula_match'] = matches
    
    # Calculate statistics
    total_rows = len(df)
    valid_extractions = df['extracted_formula'].notna().sum()
    total_matches = df['formula_match'].sum()
    
    print("\n" + "="*60)
    print("COMPOSITION COMPARISON RESULTS")
    print("="*60)
    print(f"Total rows processed: {total_rows}")
    print(f"Successful CIF extractions: {valid_extractions} ({valid_extractions/total_rows*100:.2f}%)")
    print(f"Formula matches: {total_matches} ({total_matches/total_rows*100:.2f}%)")
    print(f"Match rate among valid extractions: {total_matches/valid_extractions*100:.2f}%" if valid_extractions > 0 else "No valid extractions")
    
    # Show some examples of matches and mismatches
    print("\n" + "="*60)
    print("SAMPLE RESULTS")
    print("="*60)
    
    # Show some matches
    matches_df = df[df['formula_match'] == True].head(10)
    if len(matches_df) > 0:
        print("\nSample matches:")
        for idx, row in matches_df.iterrows():
            print(f"Row {idx}: {row['pretty_formula']} == {row['extracted_formula']}")
    
    # Show some mismatches
    mismatches_df = df[(df['formula_match'] == False) & (df['extracted_formula'].notna())].head(10)
    if len(mismatches_df) > 0:
        print("\nSample mismatches:")
        for idx, row in mismatches_df.iterrows():
            print(f"Row {idx}: {row['pretty_formula']} != {row['extracted_formula']}")
    
    # Show failed extractions
    failed_df = df[df['extracted_formula'].isna()].head(10)
    if len(failed_df) > 0:
        print("\nSample failed extractions:")
        for idx, row in failed_df.iterrows():
            print(f"Row {idx}: pretty_formula = {row['pretty_formula']}, CIF length = {len(str(row['cif']))}")
    
    # Save results
    output_file = 'composition_comparison_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Create summary statistics
    summary_stats = {
        'total_rows': total_rows,
        'successful_extractions': valid_extractions,
        'extraction_success_rate': valid_extractions/total_rows*100,
        'total_matches': total_matches,
        'overall_match_rate': total_matches/total_rows*100,
        'match_rate_among_valid': total_matches/valid_extractions*100 if valid_extractions > 0 else 0
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_file = 'composition_comparison_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to {summary_file}")
    
    return df, summary_stats

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, stats = main()

