"""
Detailed analysis of composition matching rates based on exact element count (1, 2, 3, 4, 5, 6+).
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

def categorize_element_count(element_count):
    """
    Categorize element count into specific ranges.
    """
    if pd.isna(element_count) or element_count is None:
        return 'Unknown'
    
    count = int(element_count)
    
    if count == 1:
        return '1 element'
    elif count == 2:
        return '2 elements'
    elif count == 3:
        return '3 elements'
    elif count == 4:
        return '4 elements'
    elif count == 5:
        return '5 elements'
    elif count >= 6:
        return '6+ elements'
    else:
        return 'Unknown'

def analyze_csv_file(csv_path):
    """
    Analyze a single CSV file and return results by exact element count.
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
    
    # Categorize element counts
    df['element_count_category'] = df['pretty_formula_element_count'].apply(categorize_element_count)
    
    # Calculate statistics by element count category
    results = {}
    
    for category in ['1 element', '2 elements', '3 elements', '4 elements', '5 elements', '6+ elements', 'Unknown']:
        category_data = df[df['element_count_category'] == category]
        
        if len(category_data) == 0:
            results[category] = {
                'total_count': 0,
                'successful_extractions': 0,
                'matches': 0,
                'extraction_success_rate': 0,
                'match_rate': 0,
                'match_rate_among_valid': 0
            }
            continue
        
        total_count = len(category_data)
        successful_extractions = category_data['extracted_formula'].notna().sum()
        matches = category_data['formula_match'].sum()
        
        extraction_success_rate = successful_extractions / total_count * 100 if total_count > 0 else 0
        match_rate = matches / total_count * 100 if total_count > 0 else 0
        match_rate_among_valid = matches / successful_extractions * 100 if successful_extractions > 0 else 0
        
        results[category] = {
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
    print("DETAILED ELEMENT COUNT ANALYSIS")
    print("="*80)
    
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
        results, df = analyze_csv_file(csv_file)
        if results is not None:
            all_results[csv_file] = results
    
    # Create summary report
    print("\n" + "="*80)
    print("DETAILED ELEMENT COUNT ANALYSIS RESULTS")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    
    for csv_file, results in all_results.items():
        for category, stats in results.items():
            summary_data.append({
                'file': csv_file,
                'element_count_category': category,
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
        print("-" * 80)
        file_results = summary_df[summary_df['file'] == csv_file]
        
        for _, row in file_results.iterrows():
            if row['total_count'] > 0:  # Only show categories with data
                print(f"Element Count {row['element_count_category']:>12}: "
                      f"Total={row['total_count']:>4}, "
                      f"Extracted={row['successful_extractions']:>4} "
                      f"({row['extraction_success_rate']:>5.1f}%), "
                      f"Matches={row['matches']:>4} "
                      f"({row['match_rate']:>5.1f}%), "
                      f"Valid Match Rate={row['match_rate_among_valid']:>5.1f}%")
    
    # Save detailed results
    summary_df.to_csv('detailed_element_count_analysis_summary.csv', index=False)
    print(f"\nDetailed results saved to detailed_element_count_analysis_summary.csv")
    
    # Create comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE - MATCH RATES BY EXACT ELEMENT COUNT")
    print("="*80)
    
    # Pivot table for easier comparison
    pivot_df = summary_df[summary_df['total_count'] > 0].pivot(
        index='element_count_category', 
        columns='file', 
        values='match_rate'
    )
    print("\nMatch Rate by Exact Element Count:")
    print(pivot_df.round(2))
    
    # Save comparison table
    pivot_df.to_csv('detailed_element_count_comparison_table.csv')
    print(f"\nComparison table saved to detailed_element_count_comparison_table.csv")
    
    # Additional analysis: element count distribution
    print("\n" + "="*80)
    print("ELEMENT COUNT DISTRIBUTION")
    print("="*80)
    
    for csv_file in all_results.keys():
        print(f"\n{csv_file}:")
        file_data = summary_df[summary_df['file'] == csv_file]
        total = file_data['total_count'].sum()
        
        for _, row in file_data.iterrows():
            if row['total_count'] > 0:
                percentage = row['total_count'] / total * 100 if total > 0 else 0
                print(f"  {row['element_count_category']:>12}: {row['total_count']:>4} ({percentage:>5.1f}%)")
    
    return summary_df, all_results

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary_df, all_results = main()
