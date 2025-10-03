"""
Analysis of composition matching rates based on element count ranges.
Compares extracted formulas with pretty_formula for different element count categories.
"""

import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
import re
from tqdm import tqdm
import warnings
from collections import Counter

def extract_composition_from_cif(cif_str):
    """
    Extract chemical composition from a CIF string using pymatgen.
    """
    try:
        structure = Structure.from_str(cif_str, fmt="cif")
        composition = structure.composition
        formula = composition.reduced_formula
        return formula
    except Exception as e:
        return None

def normalize_formula(formula):
    """
    Normalize chemical formula for comparison.
    """
    if pd.isna(formula) or formula is None:
        return None
    
    formula = str(formula).strip().replace(" ", "")
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    if not matches:
        return formula.lower()
    
    element_counts = {}
    for element, count in matches:
        count = int(count) if count else 1
        element_counts[element] = element_counts.get(element, 0) + count
    
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
    """
    if extracted_formula is None or pretty_formula is None:
        return False
    
    norm_extracted = normalize_formula(extracted_formula)
    norm_pretty = normalize_formula(pretty_formula)
    
    return norm_extracted == norm_pretty

def count_elements_in_formula(formula):
    """
    Count the number of unique elements in a chemical formula.
    """
    if pd.isna(formula) or formula is None:
        return 0
    
    pattern = r'([A-Z][a-z]?)'
    matches = re.findall(pattern, str(formula))
    return len(set(matches))

def analyze_csv_file(csv_path, test_df):
    """
    Analyze a single CSV file and return results by element count ranges.
    """
    print(f"\nAnalyzing {csv_path}...")
    
    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_path}")
        return None
    
    # Check required columns
    required_columns = ['cif', 'pretty_formula']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None
    
    # Extract compositions from CIF strings
    print("Extracting compositions from CIF strings...")
    extracted_compositions = []
    
    for idx, cif_str in enumerate(tqdm(df['cif'], desc="Processing CIFs")):
        if pd.isna(cif_str):
            extracted_compositions.append(None)
            continue
            
        composition = extract_composition_from_cif(cif_str)
        extracted_compositions.append(composition)
    
    df['extracted_formula'] = extracted_compositions
    
    # Compare with pretty_formula
    print("Comparing extracted formulas with pretty_formula...")
    matches = []
    
    for idx, row in df.iterrows():
        match = compare_compositions(row['extracted_formula'], row['pretty_formula'])
        matches.append(match)
    
    df['formula_match'] = matches
    
    # Count elements in pretty_formula
    df['pretty_formula_element_count'] = df['pretty_formula'].apply(count_elements_in_formula)
    
    # Count elements in extracted formula
    df['extracted_formula_element_count'] = df['extracted_formula'].apply(count_elements_in_formula)
    
    # Create element count ranges
    df['element_count_range'] = pd.cut(
        df['pretty_formula_element_count'], 
        bins=[0, 5, 10, 25, float('inf')], 
        labels=['≤5', '6-10', '11-25', '>25'],
        include_lowest=True
    )
    
    # Calculate statistics by element count range
    results = {}
    
    for range_label in ['≤5', '6-10', '11-25', '>25']:
        range_data = df[df['element_count_range'] == range_label]
        
        if len(range_data) == 0:
            results[range_label] = {
                'total_count': 0,
                'successful_extractions': 0,
                'matches': 0,
                'extraction_success_rate': 0,
                'match_rate': 0,
                'match_rate_among_valid': 0
            }
            continue
        
        total_count = len(range_data)
        successful_extractions = range_data['extracted_formula'].notna().sum()
        matches = range_data['formula_match'].sum()
        
        extraction_success_rate = successful_extractions / total_count * 100 if total_count > 0 else 0
        match_rate = matches / total_count * 100 if total_count > 0 else 0
        match_rate_among_valid = matches / successful_extractions * 100 if successful_extractions > 0 else 0
        
        results[range_label] = {
            'total_count': total_count,
            'successful_extractions': successful_extractions,
            'matches': matches,
            'extraction_success_rate': extraction_success_rate,
            'match_rate': match_rate,
            'match_rate_among_valid': match_rate_among_valid
        }
    
    return results, df

def main():
    """
    Main analysis function.
    """
    print("ELEMENT COUNT ANALYSIS")
    print("="*80)
    
    # Load test.csv to get reference data
    test_path = '/Users/mohdzaki/Documents/github/llamat/src/cifs/crystal-text-llm/data/test.csv'
    print(f"Loading reference data from {test_path}...")
    
    try:
        test_df = pd.read_csv(test_path)
        print(f"Loaded {len(test_df)} reference rows")
    except FileNotFoundError:
        print(f"Error: Could not find file {test_path}")
        return
    
    # Find CSV files that start with "actual_pred_cif"
    import glob
    csv_files = glob.glob('actual_pred_cif*.csv')
    
    if not csv_files:
        print("No CSV files found that start with 'actual_pred_cif'")
        return
    
    print(f"Found {len(csv_files)} CSV files to analyze: {csv_files}")
    
    # Analyze each CSV file
    all_results = {}
    
    for csv_file in csv_files:
        results, df = analyze_csv_file(csv_file, test_df)
        if results is not None:
            all_results[csv_file] = results
    
    # Create summary report
    print("\n" + "="*80)
    print("ELEMENT COUNT ANALYSIS RESULTS")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    
    for csv_file, results in all_results.items():
        for range_label, stats in results.items():
            summary_data.append({
                'file': csv_file,
                'element_count_range': range_label,
                'total_count': stats['total_count'],
                'successful_extractions': stats['successful_extractions'],
                'matches': stats['matches'],
                'extraction_success_rate': stats['extraction_success_rate'],
                'match_rate': stats['match_rate'],
                'match_rate_among_valid': stats['match_rate_among_valid']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display results
    for csv_file in all_results.keys():
        print(f"\n{csv_file}:")
        print("-" * 60)
        file_results = summary_df[summary_df['file'] == csv_file]
        
        for _, row in file_results.iterrows():
            print(f"Element count {row['element_count_range']:>4}: "
                  f"Total={row['total_count']:>4}, "
                  f"Extracted={row['successful_extractions']:>4} "
                  f"({row['extraction_success_rate']:>5.1f}%), "
                  f"Matches={row['matches']:>4} "
                  f"({row['match_rate']:>5.1f}%), "
                  f"Valid Match Rate={row['match_rate_among_valid']:>5.1f}%")
    
    # Save detailed results
    summary_df.to_csv('element_count_analysis_summary.csv', index=False)
    print(f"\nDetailed results saved to element_count_analysis_summary.csv")
    
    # Create comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    # Pivot table for easier comparison
    pivot_df = summary_df.pivot(index='element_count_range', columns='file', values='match_rate')
    print("\nMatch Rate by Element Count Range:")
    print(pivot_df.round(2))
    
    # Save comparison table
    pivot_df.to_csv('element_count_comparison_table.csv')
    print(f"\nComparison table saved to element_count_comparison_table.csv")
    
    # Additional analysis: element count distribution
    print("\n" + "="*80)
    print("ELEMENT COUNT DISTRIBUTION")
    print("="*80)
    
    for csv_file in all_results.keys():
        print(f"\n{csv_file}:")
        file_data = summary_df[summary_df['file'] == csv_file]
        total = file_data['total_count'].sum()
        
        for _, row in file_data.iterrows():
            percentage = row['total_count'] / total * 100 if total > 0 else 0
            print(f"  {row['element_count_range']:>4} elements: {row['total_count']:>4} ({percentage:>5.1f}%)")
    
    return summary_df, all_results

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary_df, all_results = main()
