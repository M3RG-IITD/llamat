"""
Analysis of composition matching rates based on space group numbers.
Compares extracted formulas with pretty_formula for different space group categories.
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

def categorize_spacegroup(sg_number):
    """
    Categorize space group number into crystal system categories.
    """
    if pd.isna(sg_number) or sg_number is None:
        return 'Unknown'
    
    sg_num = int(sg_number)
    
    # Triclinic (1-2)
    if 1 <= sg_num <= 2:
        return 'Triclinic'
    # Monoclinic (3-15)
    elif 3 <= sg_num <= 15:
        return 'Monoclinic'
    # Orthorhombic (16-74)
    elif 16 <= sg_num <= 74:
        return 'Orthorhombic'
    # Tetragonal (75-142)
    elif 75 <= sg_num <= 142:
        return 'Tetragonal'
    # Trigonal (143-167)
    elif 143 <= sg_num <= 167:
        return 'Trigonal'
    # Hexagonal (168-194)
    elif 168 <= sg_num <= 194:
        return 'Hexagonal'
    # Cubic (195-230)
    elif 195 <= sg_num <= 230:
        return 'Cubic'
    else:
        return 'Unknown'

def analyze_csv_file(csv_path, test_df):
    """
    Analyze a single CSV file and return results by space group categories.
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
    required_columns = ['cif', 'pretty_formula', 'spacegroup_number']
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
    
    # Categorize space groups
    df['spacegroup_category'] = df['spacegroup_number'].apply(categorize_spacegroup)
    
    # Calculate statistics by space group category
    results = {}
    
    for category in ['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal', 'Hexagonal', 'Cubic', 'Unknown']:
        category_data = df[df['spacegroup_category'] == category]
        
        if len(category_data) == 0:
            results[category] = {
                'total_count': 0,
                'successful_extractions': 0,
                'matches': 0,
                'extraction_success_rate': 0,
                'match_rate': 0,
                'match_rate_among_valid': 0,
                'spacegroup_numbers': []
            }
            continue
        
        total_count = len(category_data)
        successful_extractions = category_data['extracted_formula'].notna().sum()
        matches = category_data['formula_match'].sum()
        
        extraction_success_rate = successful_extractions / total_count * 100 if total_count > 0 else 0
        match_rate = matches / total_count * 100 if total_count > 0 else 0
        match_rate_among_valid = matches / successful_extractions * 100 if successful_extractions > 0 else 0
        
        # Get unique space group numbers in this category
        unique_sgs = sorted(category_data['spacegroup_number'].dropna().unique().tolist())
        
        results[category] = {
            'total_count': total_count,
            'successful_extractions': successful_extractions,
            'matches': matches,
            'extraction_success_rate': extraction_success_rate,
            'match_rate': match_rate,
            'match_rate_among_valid': match_rate_among_valid,
            'spacegroup_numbers': unique_sgs
        }
    
    return results, df

def main():
    """
    Main analysis function.
    """
    print("SPACE GROUP ANALYSIS")
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
    print("SPACE GROUP ANALYSIS RESULTS")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    
    for csv_file, results in all_results.items():
        for category, stats in results.items():
            summary_data.append({
                'file': csv_file,
                'spacegroup_category': category,
                'total_count': stats['total_count'],
                'successful_extractions': stats['successful_extractions'],
                'matches': stats['matches'],
                'extraction_success_rate': stats['extraction_success_rate'],
                'match_rate': stats['match_rate'],
                'match_rate_among_valid': stats['match_rate_among_valid'],
                'spacegroup_numbers': str(stats['spacegroup_numbers'])[:100] + '...' if len(str(stats['spacegroup_numbers'])) > 100 else str(stats['spacegroup_numbers'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display results
    for csv_file in all_results.keys():
        print(f"\n{csv_file}:")
        print("-" * 80)
        file_results = summary_df[summary_df['file'] == csv_file]
        
        for _, row in file_results.iterrows():
            if row['total_count'] > 0:  # Only show categories with data
                print(f"Space Group {row['spacegroup_category']:>12}: "
                      f"Total={row['total_count']:>4}, "
                      f"Extracted={row['successful_extractions']:>4} "
                      f"({row['extraction_success_rate']:>5.1f}%), "
                      f"Matches={row['matches']:>4} "
                      f"({row['match_rate']:>5.1f}%), "
                      f"Valid Match Rate={row['match_rate_among_valid']:>5.1f}%")
    
    # Save detailed results
    summary_df.to_csv('spacegroup_analysis_summary.csv', index=False)
    print(f"\nDetailed results saved to spacegroup_analysis_summary.csv")
    
    # Create comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE - MATCH RATES BY SPACE GROUP")
    print("="*80)
    
    # Pivot table for easier comparison
    pivot_df = summary_df[summary_df['total_count'] > 0].pivot(
        index='spacegroup_category', 
        columns='file', 
        values='match_rate'
    )
    print("\nMatch Rate by Space Group Category:")
    print(pivot_df.round(2))
    
    # Save comparison table
    pivot_df.to_csv('spacegroup_comparison_table.csv')
    print(f"\nComparison table saved to spacegroup_comparison_table.csv")
    
    # Additional analysis: space group distribution
    print("\n" + "="*80)
    print("SPACE GROUP DISTRIBUTION")
    print("="*80)
    
    for csv_file in all_results.keys():
        print(f"\n{csv_file}:")
        file_data = summary_df[summary_df['file'] == csv_file]
        total = file_data['total_count'].sum()
        
        for _, row in file_data.iterrows():
            if row['total_count'] > 0:
                percentage = row['total_count'] / total * 100 if total > 0 else 0
                print(f"  {row['spacegroup_category']:>12}: {row['total_count']:>4} ({percentage:>5.1f}%)")
    
    # Detailed space group number analysis
    print("\n" + "="*80)
    print("DETAILED SPACE GROUP NUMBER ANALYSIS")
    print("="*80)
    
    for csv_file in all_results.keys():
        print(f"\n{csv_file}:")
        file_data = summary_df[summary_df['file'] == csv_file]
        
        for _, row in file_data.iterrows():
            if row['total_count'] > 0 and row['spacegroup_numbers'] != '[]':
                print(f"  {row['spacegroup_category']:>12}: SG numbers {row['spacegroup_numbers']}")
    
    return summary_df, all_results

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary_df, all_results = main()
