# Space Group Analysis Report

## Overview
This report analyzes composition matching rates based on crystal space group categories for both CSV files that start with "actual_pred_cif". The analysis compares extracted formulas from CIF strings with the `pretty_formula` column across different crystal systems.

## Dataset Information
- **Files analyzed**: 2 CSV files starting with "actual_pred_cif"
  - `actual_pred_cif_llamat2_9064_llamat2_1758654262.csv`
  - `actual_pred_cif_llamat3_9064_llamat3_1758654262.csv`
- **Total rows per file**: 9,046
- **Space group categories**: Triclinic, Monoclinic, Orthorhombic, Tetragonal, Trigonal, Hexagonal, Cubic

## Key Findings

### Space Group Distribution
Both datasets show identical distribution across crystal systems:
- **Cubic**: 2,157 cases (23.8%) - Most common
- **Orthorhombic**: 1,725 cases (19.1%) - Second most common
- **Tetragonal**: 1,561 cases (17.3%)
- **Monoclinic**: 1,288 cases (14.2%)
- **Hexagonal**: 975 cases (10.8%)
- **Trigonal**: 964 cases (10.7%)
- **Triclinic**: 376 cases (4.2%) - Least common

### Performance Comparison by Space Group Category

#### 1. Cubic (2,157 cases - 23.8% of data)

| Model | Total | Extracted | Extraction Rate | Matches | Match Rate | Valid Match Rate |
|-------|-------|-----------|----------------|---------|------------|------------------|
| **LLaMat-2** | 2,157 | 2,012 | 93.3% | 1,959 | **90.8%** | **97.4%** |
| **LLaMat-3** | 2,157 | 1,221 | 56.6% | 137 | **6.4%** | **11.2%** |

**Key Insights:**
- LLaMat-2 shows **excellent performance** for cubic crystals
- Highest match rate among all crystal systems for LLaMat-2
- LLaMat-3 shows very poor performance for cubic crystals

#### 2. Orthorhombic (1,725 cases - 19.1% of data)

| Model | Total | Extracted | Extraction Rate | Matches | Match Rate | Valid Match Rate |
|-------|-------|-----------|----------------|---------|------------|------------------|
| **LLaMat-2** | 1,725 | 1,529 | 88.6% | 1,276 | **74.0%** | **83.5%** |
| **LLaMat-3** | 1,725 | 1,341 | 77.7% | 77 | **4.5%** | **5.7%** |

**Key Insights:**
- LLaMat-2 shows **good performance** for orthorhombic crystals
- LLaMat-3 shows very poor performance

#### 3. Tetragonal (1,561 cases - 17.3% of data)

| Model | Total | Extracted | Extraction Rate | Matches | Match Rate | Valid Match Rate |
|-------|-------|-----------|----------------|---------|------------|------------------|
| **LLaMat-2** | 1,561 | 1,405 | 90.0% | 1,204 | **77.1%** | **85.7%** |
| **LLaMat-3** | 1,561 | 1,020 | 65.3% | 74 | **4.7%** | **7.3%** |

**Key Insights:**
- LLaMat-2 shows **good performance** for tetragonal crystals
- LLaMat-3 shows very poor performance

#### 4. Monoclinic (1,288 cases - 14.2% of data)

| Model | Total | Extracted | Extraction Rate | Matches | Match Rate | Valid Match Rate |
|-------|-------|-----------|----------------|---------|------------|------------------|
| **LLaMat-2** | 1,288 | 1,167 | 90.6% | 924 | **71.7%** | **79.2%** |
| **LLaMat-3** | 1,288 | 1,071 | 83.2% | 69 | **5.4%** | **6.4%** |

**Key Insights:**
- LLaMat-2 shows **good performance** for monoclinic crystals
- LLaMat-3 shows very poor performance

#### 5. Hexagonal (975 cases - 10.8% of data)

| Model | Total | Extracted | Extraction Rate | Matches | Match Rate | Valid Match Rate |
|-------|-------|-----------|----------------|---------|------------|------------------|
| **LLaMat-2** | 975 | 883 | 90.6% | 779 | **79.9%** | **88.2%** |
| **LLaMat-3** | 975 | 592 | 60.7% | 46 | **4.7%** | **7.8%** |

**Key Insights:**
- LLaMat-2 shows **excellent performance** for hexagonal crystals
- Second highest match rate for LLaMat-2
- LLaMat-3 shows very poor performance

#### 6. Trigonal (964 cases - 10.7% of data)

| Model | Total | Extracted | Extraction Rate | Matches | Match Rate | Valid Match Rate |
|-------|-------|-----------|----------------|---------|------------|------------------|
| **LLaMat-2** | 964 | 878 | 91.1% | 740 | **76.8%** | **84.3%** |
| **LLaMat-3** | 964 | 692 | 71.8% | 66 | **6.8%** | **9.5%** |

**Key Insights:**
- LLaMat-2 shows **good performance** for trigonal crystals
- LLaMat-3 shows very poor performance

#### 7. Triclinic (376 cases - 4.2% of data)

| Model | Total | Extracted | Extraction Rate | Matches | Match Rate | Valid Match Rate |
|-------|-------|-----------|----------------|---------|------------|------------------|
| **LLaMat-2** | 376 | 340 | 90.4% | 272 | **72.3%** | **80.0%** |
| **LLaMat-3** | 376 | 328 | 87.2% | 32 | **8.5%** | **9.8%** |

**Key Insights:**
- LLaMat-2 shows **good performance** for triclinic crystals
- LLaMat-3 shows very poor performance
- Smallest sample size among all categories

## Performance Trends by Crystal System

### 1. LLaMat-2 Performance Ranking (by match rate)
1. **Cubic**: 90.8% - Best performance
2. **Hexagonal**: 79.9% - Second best
3. **Tetragonal**: 77.1%
4. **Trigonal**: 76.8%
5. **Orthorhombic**: 74.0%
6. **Triclinic**: 72.3%
7. **Monoclinic**: 71.7% - Lowest performance

### 2. LLaMat-3 Performance Ranking (by match rate)
1. **Triclinic**: 8.5% - Best performance (still very poor)
2. **Cubic**: 6.4%
3. **Trigonal**: 6.8%
4. **Monoclinic**: 5.4%
5. **Tetragonal**: 4.7%
6. **Hexagonal**: 4.7%
7. **Orthorhombic**: 4.5% - Lowest performance

### 3. Extraction Success Rate Trends

#### LLaMat-2 (Consistently High)
- **Range**: 88.6% - 93.3%
- **Best**: Cubic (93.3%)
- **Worst**: Orthorhombic (88.6%)
- **Trend**: Relatively consistent across crystal systems

#### LLaMat-3 (Variable and Lower)
- **Range**: 56.6% - 87.2%
- **Best**: Triclinic (87.2%)
- **Worst**: Cubic (56.6%)
- **Trend**: Performance decreases with crystal system complexity

## Key Insights

### 1. Model Performance
- **LLaMat-2 significantly outperforms LLaMat-3** across all crystal systems
- LLaMat-2 shows **consistent high performance** (71.7% - 90.8% match rates)
- LLaMat-3 shows **consistently poor performance** (4.5% - 8.5% match rates)

### 2. Crystal System Complexity
- **Cubic crystals**: Best performance for LLaMat-2 (90.8%), worst for LLaMat-3 (6.4%)
- **High symmetry systems** (Cubic, Hexagonal): Generally better performance for LLaMat-2
- **Low symmetry systems** (Triclinic, Monoclinic): Lower but still good performance for LLaMat-2

### 3. Extraction Success Patterns
- **LLaMat-2**: High and consistent extraction success (88.6% - 93.3%)
- **LLaMat-3**: Variable extraction success, decreasing with complexity
- **Cubic crystals**: Most challenging for LLaMat-3 extraction (56.6%)

### 4. Valid Match Rate (among successfully extracted)
- **LLaMat-2**: Excellent accuracy when extraction succeeds (79.2% - 97.4%)
- **LLaMat-3**: Very poor accuracy even when extraction succeeds (5.7% - 11.2%)

## Conclusions

### 1. Model Selection
- **LLaMat-2 is clearly superior** for all crystal systems
- **Cubic crystals**: Best performance for LLaMat-2, worst for LLaMat-3
- **High symmetry systems**: Generally better for both models, but LLaMat-2 maintains advantage

### 2. Crystal System Impact
- **Cubic crystals**: Most represented (23.8%) and best performing for LLaMat-2
- **Orthorhombic crystals**: Second most represented (19.1%) with good LLaMat-2 performance
- **Triclinic crystals**: Least represented (4.2%) but still good LLaMat-2 performance

### 3. Practical Implications
- **Use LLaMat-2** for all crystal systems requiring composition accuracy
- **Cubic and hexagonal crystals** show best performance for LLaMat-2
- **LLaMat-3 needs significant improvements** before practical use

## Recommendations

### 1. Model Development
- **Focus on LLaMat-2** for production applications
- **Investigate LLaMat-3 issues** across all crystal systems
- **Study cubic crystal handling** in LLaMat-3 (worst performance)

### 2. Dataset Analysis
- **Cubic crystals dominate** the dataset (23.8%) - ensure balanced representation
- **Consider crystal system balance** in future datasets
- **Analyze specific space group numbers** within each category

### 3. Further Analysis
- **Investigate specific space group numbers** that perform poorly
- **Study the relationship** between crystal symmetry and model performance
- **Analyze CIF generation quality** by crystal system

## Files Generated
- `spacegroup_analysis_summary.csv`: Detailed results by space group category
- `spacegroup_comparison_table.csv`: Comparison table for easy reference
- `spacegroup_analysis.py`: Analysis script
- `SPACEGROUP_ANALYSIS_REPORT.md`: This report

## Usage
To reproduce this analysis:
```bash
cd /path/to/conditional_gen_eval/
python spacegroup_analysis.py
```
