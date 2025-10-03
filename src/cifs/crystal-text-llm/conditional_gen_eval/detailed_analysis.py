"""
Detailed analysis of composition comparison results.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

def analyze_mismatches(df):
    """
    Analyze patterns in formula mismatches.
    """
    print("="*80)
    print("DETAILED MISMATCH ANALYSIS")
    print("="*80)
    
    # Get mismatches where both formulas exist
    mismatches = df[(df['formula_match'] == False) & 
                   (df['extracted_formula'].notna()) & 
                   (df['pretty_formula'].notna())]
    
    print(f"Total mismatches: {len(mismatches)}")
    print(f"Percentage of total: {len(mismatches)/len(df)*100:.2f}%")
    
    # Analyze common patterns in mismatches
    print("\n" + "="*50)
    print("COMMON MISMATCH PATTERNS")
    print("="*50)
    
    # Count differences in element counts
    count_diffs = []
    element_diffs = []
    
    for idx, row in mismatches.iterrows():
        pretty = str(row['pretty_formula'])
        extracted = str(row['extracted_formula'])
        
        # Extract element counts from both formulas
        pretty_elements = extract_element_counts(pretty)
        extracted_elements = extract_element_counts(extracted)
        
        # Find differences
        all_elements = set(pretty_elements.keys()) | set(extracted_elements.keys())
        for element in all_elements:
            pretty_count = pretty_elements.get(element, 0)
            extracted_count = extracted_elements.get(element, 0)
            if pretty_count != extracted_count:
                count_diffs.append({
                    'element': element,
                    'pretty_count': pretty_count,
                    'extracted_count': extracted_count,
                    'difference': extracted_count - pretty_count
                })
        
        # Check for completely different elements
        pretty_only = set(pretty_elements.keys()) - set(extracted_elements.keys())
        extracted_only = set(extracted_elements.keys()) - set(pretty_elements.keys())
        if pretty_only or extracted_only:
            element_diffs.append({
                'pretty_only': list(pretty_only),
                'extracted_only': list(extracted_only),
                'pretty_formula': pretty,
                'extracted_formula': extracted
            })
    
    # Analyze count differences
    if count_diffs:
        count_df = pd.DataFrame(count_diffs)
        print("\nMost common count differences:")
        diff_summary = count_df.groupby(['element', 'difference']).size().reset_index(name='count')
        diff_summary = diff_summary.sort_values('count', ascending=False)
        print(diff_summary.head(20))
    
    # Analyze element differences
    if element_diffs:
        print(f"\nCases with completely different elements: {len(element_diffs)}")
        print("\nSample cases with different elements:")
        for i, case in enumerate(element_diffs[:10]):
            print(f"{i+1}. Pretty: {case['pretty_formula']} -> Extracted: {case['extracted_formula']}")
            print(f"   Pretty only: {case['pretty_only']}, Extracted only: {case['extracted_only']}")
    
    return mismatches

def extract_element_counts(formula):
    """
    Extract element counts from a chemical formula.
    """
    # Simple regex to find element symbols and their counts
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    element_counts = {}
    for element, count in matches:
        count = int(count) if count else 1
        element_counts[element] = element_counts.get(element, 0) + count
    
    return element_counts

def analyze_failed_extractions(df):
    """
    Analyze cases where CIF extraction failed.
    """
    print("\n" + "="*80)
    print("FAILED EXTRACTION ANALYSIS")
    print("="*80)
    
    failed = df[df['extracted_formula'].isna()]
    print(f"Total failed extractions: {len(failed)}")
    print(f"Percentage of total: {len(failed)/len(df)*100:.2f}%")
    
    # Analyze CIF string lengths
    failed['cif_length'] = failed['cif'].astype(str).str.len()
    
    print(f"\nCIF string length statistics for failed extractions:")
    print(f"Mean length: {failed['cif_length'].mean():.2f}")
    print(f"Median length: {failed['cif_length'].median():.2f}")
    print(f"Min length: {failed['cif_length'].min()}")
    print(f"Max length: {failed['cif_length'].max()}")
    
    # Show examples of very short CIFs (likely empty or malformed)
    very_short = failed[failed['cif_length'] < 10]
    print(f"\nVery short CIFs (< 10 chars): {len(very_short)}")
    if len(very_short) > 0:
        print("Sample very short CIFs:")
        for idx, row in very_short.head(5).iterrows():
            print(f"  Row {idx}: '{row['cif']}' (length: {row['cif_length']})")
    
    # Show examples of longer CIFs that still failed
    longer_failed = failed[failed['cif_length'] > 100]
    print(f"\nLonger CIFs that failed (> 100 chars): {len(longer_failed)}")
    if len(longer_failed) > 0:
        print("Sample longer failed CIFs:")
        for idx, row in longer_failed.head(3).iterrows():
            print(f"  Row {idx}: pretty_formula = {row['pretty_formula']}")
            print(f"    CIF preview: {str(row['cif'])[:100]}...")
    
    return failed

def main():
    """
    Main analysis function.
    """
    # Load results
    df = pd.read_csv('composition_comparison_results.csv')
    
    print("COMPOSITION COMPARISON - DETAILED ANALYSIS")
    print("="*80)
    print(f"Total rows: {len(df)}")
    print(f"Successful extractions: {df['extracted_formula'].notna().sum()}")
    print(f"Formula matches: {df['formula_match'].sum()}")
    print(f"Overall match rate: {df['formula_match'].sum()/len(df)*100:.2f}%")
    
    # Analyze mismatches
    mismatches = analyze_mismatches(df)
    
    # Analyze failed extractions
    failed = analyze_failed_extractions(df)
    
    # Show some specific examples
    print("\n" + "="*80)
    print("SPECIFIC EXAMPLES")
    print("="*80)
    
    # Perfect matches
    perfect_matches = df[df['formula_match'] == True].head(10)
    print("\nPerfect matches:")
    for idx, row in perfect_matches.iterrows():
        print(f"  {row['pretty_formula']} == {row['extracted_formula']}")
    
    # Close matches (same elements, different counts)
    print("\nClose matches (same elements, different counts):")
    close_matches = []
    for idx, row in mismatches.iterrows():
        pretty_elements = set(extract_element_counts(str(row['pretty_formula'])).keys())
        extracted_elements = set(extract_element_counts(str(row['extracted_formula'])).keys())
        if pretty_elements == extracted_elements:
            close_matches.append(row)
            if len(close_matches) <= 10:
                print(f"  {row['pretty_formula']} != {row['extracted_formula']}")
    
    print(f"\nTotal close matches: {len(close_matches)}")
    
    # Completely different formulas
    print("\nCompletely different formulas:")
    different_formulas = []
    for idx, row in mismatches.iterrows():
        pretty_elements = set(extract_element_counts(str(row['pretty_formula'])).keys())
        extracted_elements = set(extract_element_counts(str(row['extracted_formula'])).keys())
        if pretty_elements != extracted_elements:
            different_formulas.append(row)
            if len(different_formulas) <= 10:
                print(f"  {row['pretty_formula']} != {row['extracted_formula']}")
    
    print(f"\nTotal completely different formulas: {len(different_formulas)}")

if __name__ == "__main__":
    main()

