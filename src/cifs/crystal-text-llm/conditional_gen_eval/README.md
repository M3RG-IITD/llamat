# Conditional Generation Evaluation

This directory contains comprehensive evaluation results and analysis for LLaMaT-2 and LLaMaT-3 models on conditional crystal structure generation tasks.

## Overview

The evaluation compares two models across multiple dimensions:
- **LLaMaT-2**: `actual_pred_cif_llamat2_9064_llamat2_1758654262.csv` (9,046 samples)
- **LLaMaT-3**: `actual_pred_cif_llamat3_9064_llamat3_1758654262.csv` (9,046 samples)

## Quick Start

To reproduce all evaluation results:

```bash
cd conditional_gen_eval/

# Run all analysis scripts
python composition_comparison.py
python element_count_analysis.py
python detailed_element_count_analysis.py
python spacegroup_analysis.py
```

## Analysis Reports

### 📊 [Composition Analysis Report](COMPOSITION_ANALYSIS_REPORT.md)
- **Focus**: Chemical formula extraction and matching
- **Key Metrics**: 79.1% match rate (LLaMaT-2) vs 5.5% (LLaMaT-3)
- **Script**: `composition_comparison.py`

### 🔢 [Element Count Analysis Report](ELEMENT_COUNT_ANALYSIS_REPORT.md)
- **Focus**: Performance by number of elements (≤5, 6-10, 11-25, >25)
- **Key Finding**: 99.9% of compounds have ≤5 elements
- **Script**: `element_count_analysis.py`

### 📈 [Detailed Element Count Analysis Report](DETAILED_ELEMENT_COUNT_ANALYSIS_REPORT.md)
- **Focus**: Exact element count performance (1, 2, 3, 4, 5, 6+ elements)
- **Key Finding**: 3-element compounds dominate (58.9%) with best LLaMaT-2 performance
- **Script**: `detailed_element_count_analysis.py`

### 🏗️ [Space Group Analysis Report](SPACEGROUP_ANALYSIS_REPORT.md)
- **Focus**: Performance by crystal system (Cubic, Orthorhombic, etc.)
- **Key Finding**: Cubic crystals show best performance (90.8% LLaMaT-2)
- **Script**: `spacegroup_analysis.py`

### 📋 [Overleaf Tables README](OVERLEAF_TABLES_README.md)
- **Focus**: LaTeX tables for publication
- **Files**: `evaluation_results_table.tex`, `compact_evaluation_table.tex`, `standalone_table.tex`

## Key Results Summary

| Metric | LLaMaT-2 | LLaMaT-3 | Gap |
|--------|----------|----------|-----|
| **Overall Match Rate** | 79.1% | 5.5% | 73.6% |
| **Extraction Success** | 90.8% | 69.3% | 21.5% |
| **Valid Match Rate** | 87.1% | 8.0% | 79.1% |

## Generated Files

### CSV Results
- `composition_comparison_results.csv` - Full composition analysis
- `composition_comparison_summary.csv` - Composition summary stats
- `element_count_analysis_summary.csv` - Element count analysis
- `detailed_element_count_analysis_summary.csv` - Detailed element count analysis
- `spacegroup_analysis_summary.csv` - Space group analysis

### LaTeX Tables
- `evaluation_results_table.tex` - Complete analysis table
- `compact_evaluation_table.tex` - Key results only
- `standalone_table.tex` - Copy-paste ready table

### Jupyter Notebooks
- `unconditional_eval_metrics.ipynb` - Interactive evaluation metrics

## Dependencies

```bash
pip install pandas pymatgen tqdm matplotlib seaborn
```

## Usage Examples

### Run Individual Analysis
```bash
# Composition analysis only
python composition_comparison.py

# Element count analysis only  
python element_count_analysis.py

# Space group analysis only
python spacegroup_analysis.py
```

### Generate LaTeX Tables
```bash
# Generate all LaTeX tables
python detailed_element_count_analysis.py
```

## File Structure

```
conditional_gen_eval/
├── README.md                                    # This file
├── actual_pred_cif_llamat2_*.csv              # LLaMaT-2 results
├── actual_pred_cif_llamat3_*.csv              # LLaMaT-3 results
├── *_ANALYSIS_REPORT.md                        # Detailed analysis reports
├── *.py                                        # Analysis scripts
├── *.csv                                       # Generated results
├── *.tex                                       # LaTeX tables
└── *.ipynb                                     # Jupyter notebooks
```

## Related Documentation

- [Main Crystal-Text-LLM README](../README.md) - Project overview
- [Inference README](../README_inference.md) - Model inference guide
- [Agent README](../../../agent/README.md) - Chat agent documentation
- [Evaluation Codes README](../../../evaluation_codes/README.md) - Downstream evaluation

## Citation

If you use these evaluation results, please cite the LLaMaT paper and reference the specific analysis reports used.
