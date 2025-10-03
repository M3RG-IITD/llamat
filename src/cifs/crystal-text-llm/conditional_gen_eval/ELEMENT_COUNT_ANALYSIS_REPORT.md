# Element Count Analysis Report

## Overview
This report analyzes composition matching rates based on the number of elements in the chemical formulas for both CSV files that start with "actual_pred_cif". The analysis compares extracted formulas from CIF strings with the `pretty_formula` column across different element count ranges.

## Dataset Information
- **Files analyzed**: 2 CSV files starting with "actual_pred_cif"
  - `actual_pred_cif_llamat2_9064_llamat2_1758654262.csv`
  - `actual_pred_cif_llamat3_9064_llamat3_1758654262.csv`
- **Total rows per file**: 9,046
- **Element count ranges**: ≤5, 6-10, 11-25, >25

## Key Findings

### Element Count Distribution
Both datasets show a very similar distribution:
- **≤5 elements**: 9,038 cases (99.9%)
- **6-10 elements**: 8 cases (0.1%)
- **11-25 elements**: 0 cases (0.0%)
- **>25 elements**: 0 cases (0.0%)

### Performance Comparison by Element Count Range

#### 1. ≤5 Elements (9,038 cases - 99.9% of data)

| Model | Total | Extracted | Extraction Rate | Matches | Match Rate | Valid Match Rate |
|-------|-------|-----------|----------------|---------|------------|------------------|
| **LLaMaT-2** | 9,038 | 8,207 | 90.8% | 7,148 | **79.1%** | **87.1%** |
| **LLaMaT-3** | 9,038 | 6,260 | 69.3% | 499 | **5.5%** | **8.0%** |

**Key Insights:**
- LLaMaT-2 significantly outperforms LLaMaT-3 for simple compounds
- LLaMaT-2 has much higher extraction success rate (90.8% vs 69.3%)
- LLaMaT-2 has dramatically better match rate (79.1% vs 5.5%)
- LLaMaT-2 shows excellent performance for compounds with ≤5 elements

#### 2. 6-10 Elements (8 cases - 0.1% of data)

| Model | Total | Extracted | Extraction Rate | Matches | Match Rate | Valid Match Rate |
|-------|-------|-----------|----------------|---------|------------|------------------|
| **LLaMaT-2** | 8 | 7 | 87.5% | 6 | **75.0%** | **85.7%** |
| **LLaMaT-3** | 8 | 5 | 62.5% | 2 | **25.0%** | **40.0%** |

**Key Insights:**
- Very small sample size (8 cases) - results should be interpreted cautiously
- LLaMaT-2 still outperforms LLaMaT-3 for medium complexity compounds
- Both models show lower performance compared to simple compounds (≤5 elements)
- LLaMaT-2 maintains good performance even for more complex compounds

#### 3. 11-25 Elements (0 cases)
- No data available in this range
- Both datasets contain only simple to moderately complex compounds

#### 4. >25 Elements (0 cases)
- No data available in this range
- No very complex compounds in the datasets

## Detailed Analysis

### LLaMaT-2 Performance
- **Excellent overall performance**: 79.1% match rate for ≤5 elements
- **Consistent extraction success**: 90.8% successful CIF parsing
- **High accuracy among valid extractions**: 87.1% match rate for successfully parsed CIFs
- **Good performance on complex compounds**: 75.0% match rate for 6-10 elements

### LLaMaT-3 Performance
- **Poor overall performance**: Only 5.5% match rate for ≤5 elements
- **Lower extraction success**: 69.3% successful CIF parsing
- **Very low accuracy among valid extractions**: Only 8.0% match rate for successfully parsed CIFs
- **Moderate performance on complex compounds**: 25.0% match rate for 6-10 elements

## Performance Trends

### 1. Extraction Success Rate
- **LLaMaT-2**: Consistently high (87.5-90.8%)
- **LLaMaT-3**: Lower and more variable (62.5-69.3%)
- **Trend**: Both models show slight decrease in extraction success with increasing complexity

### 2. Match Rate
- **LLaMaT-2**: High for simple compounds (79.1%), good for complex (75.0%)
- **LLaMaT-3**: Very low for simple compounds (5.5%), poor for complex (25.0%)
- **Trend**: LLaMaT-2 maintains good performance across complexity levels

### 3. Valid Match Rate (among successfully extracted)
- **LLaMaT-2**: Excellent (85.7-87.1%)
- **LLaMaT-3**: Very poor (8.0-40.0%)
- **Trend**: LLaMaT-2 shows consistent high accuracy when extraction succeeds

## Conclusions

### 1. Model Performance
- **LLaMaT-2 is significantly superior** to LLaMaT-3 across all metrics
- LLaMaT-2 shows robust performance for both simple and moderately complex compounds
- LLaMaT-3 appears to have fundamental issues with composition accuracy

### 2. Complexity Impact
- Both models perform best on simple compounds (≤5 elements)
- Performance generally decreases with increasing complexity
- The dataset is heavily skewed toward simple compounds (99.9%)

### 3. Practical Implications
- For applications requiring high composition accuracy, LLaMaT-2 is clearly preferred
- LLaMaT-3 may need significant improvements before practical use
- The analysis is limited by the lack of very complex compounds in the dataset

## Recommendations

### 1. Model Selection
- **Use LLaMaT-2** for production applications requiring composition accuracy
- **Investigate LLaMaT-3** issues before deployment

### 2. Dataset Expansion
- Include more complex compounds (11-25 elements) for comprehensive evaluation
- Test with very complex compounds (>25 elements) if available

### 3. Further Analysis
- Investigate specific failure modes in LLaMaT-3
- Analyze the quality of CIF generation in both models
- Study the relationship between compound complexity and model performance

## Files Generated
- `element_count_analysis_summary.csv`: Detailed results by element count range
- `element_count_comparison_table.csv`: Comparison table for easy reference
- `element_count_analysis.py`: Analysis script
- `ELEMENT_COUNT_ANALYSIS_REPORT.md`: This report

## Usage
To reproduce this analysis:
```bash
cd /path/to/conditional_gen_eval/
python element_count_analysis.py
```
