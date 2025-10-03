# Overleaf Tables for Composition and Space Group Matching Evaluation

This directory contains LaTeX tables for Overleaf that summarize the composition and space group matching evaluation statistics for LLaMaT-2 and LLaMaT-3 models.

## Files Available

### 1. `evaluation_results_table.tex` - Complete Analysis
- **Purpose**: Comprehensive table with all detailed statistics
- **Content**: 
  - Full element count analysis (≤5, 6-10, 11-25, >25 elements)
  - Complete space group analysis (all 7 crystal systems)
  - Overall statistics and performance metrics
  - Detailed performance metrics by category
  - Summary statistics and key findings
- **Use case**: When you need complete documentation of all results

### 2. `compact_evaluation_table.tex` - Key Results Only
- **Purpose**: Focused table highlighting key performance metrics
- **Content**:
  - Element count analysis (≤5 and 6-10 elements only)
  - Space group analysis (all 7 crystal systems)
  - Key findings summary
- **Use case**: For papers where space is limited but you want comprehensive coverage

### 3. `standalone_table.tex` - Copy-Paste Ready
- **Purpose**: Single table that can be copied directly into any LaTeX document
- **Content**:
  - Essential performance metrics only
  - All categories included
  - Performance gap calculations
- **Use case**: Quick integration into existing documents

## How to Use

### Option 1: Complete Document
```latex
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{geometry}
\geometry{a4paper, margin=1cm}

\begin{document}
\input{evaluation_results_table.tex}
\end{document}
```

### Option 2: Copy-Paste into Existing Document
1. Copy the content from `standalone_table.tex`
2. Paste into your LaTeX document
3. Ensure you have the required packages in your preamble:
   ```latex
   \usepackage{booktabs}
   \usepackage{multirow}
   \usepackage{array}
   ```

### Option 3: Individual Tables
You can extract individual tables from `evaluation_results_table.tex` and use them separately.

## Key Statistics Highlighted

### Overall Performance
- **LLaMaT-2**: 79.1% match rate
- **LLaMaT-3**: 5.5% match rate
- **Performance Gap**: 73.6 percentage points

### Best Performance by Category
- **Element Count**: LLaMaT-2: 79.1% (≤5 elements), LLaMaT-3: 25.0% (6-10 elements)
- **Space Group**: LLaMaT-2: 90.8% (Cubic), LLaMaT-3: 8.5% (Triclinic)

### Data Distribution
- **Element Count**: 99.9% have ≤5 elements
- **Space Group**: Cubic (23.8%), Orthorhombic (19.1%), Tetragonal (17.3%)

## Table Features

### Visual Design
- **Bold formatting** for key performance metrics
- **Color coding** through bold text for easy comparison
- **Performance gap** column showing LLaMaT-2 advantage
- **Percentage distribution** in parentheses for context

### Data Organization
- **Element Count**: Ordered by frequency in dataset
- **Space Group**: Ordered by frequency in dataset
- **Consistent formatting** across all tables
- **Clear column headers** with units specified

## Customization Options

### Font Size
- Tables use `\resizebox{\textwidth}{!}{...}` for automatic scaling
- Remove `\resizebox` for fixed size
- Adjust `\textwidth` multiplier for custom sizing

### Column Width
- Modify `\resizebox{\textwidth}{!}{...}` to `\resizebox{0.8\textwidth}{!}{...}` for narrower tables
- Adjust column specifications in `tabular` environment

### Content Selection
- Remove rows for categories not needed
- Add additional metrics by modifying the table structure
- Combine tables for different presentation styles

## Required LaTeX Packages

```latex
\usepackage{booktabs}    % For professional table formatting
\usepackage{multirow}    % For multi-row cells
\usepackage{array}       % For advanced column formatting
\usepackage{geometry}    % For page margins (optional)
```

## Notes

1. **Dataset Size**: All tables based on 9,046 samples per model
2. **Precision**: Match rates rounded to 1 decimal place
3. **Consistency**: All percentages calculated consistently across categories
4. **Validation**: Results verified through multiple analysis scripts

## Contact

For questions about the data or table formatting, refer to the analysis scripts:
- `composition_comparison.py`
- `element_count_analysis.py`
- `spacegroup_analysis.py`
