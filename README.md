# Traffic Violations Analysis Project - Milestone 3

## Overview
This project analyzes traffic violation patterns to identify temporal and spatial hotspots. The analysis includes detailed time-based grouping and spatial clustering of violations.

## Features

### Time-based Pattern Analysis
- 3-hour window analysis of violations
- Weekday vs weekend pattern comparison
- Violation type correlation with peak times
- Statistical identification of peak hours

### Spatial Pattern Analysis
- Statistical hotspot identification
- Optional spatial clustering (when coordinates available)
- Location-based violation frequency analysis
- Severity analysis by location

## Directory Structure
```
src/
  M2_PatternDetection/
    analyze_patterns.py    - Main analysis script
    time_pattern_analysis.py - Time-based analysis
    location_analysis.py   - Spatial analysis
data/
  processed/
    csv/          - CSV format results
    parquet/      - Parquet format results
    visualizations/ - Analysis plots and charts
```

## Usage

1. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Run the complete analysis:
```bash
python src/M2_PatternDetection/analyze_patterns.py
```

3. View results in the `data/processed/` directory:
- CSV and Parquet files with detailed analysis
- Visualizations in the `visualizations` subdirectory
- Complete analysis report in `analysis_report.md`

## Analysis Outputs

### Time-based Analysis
- Violations by 3-hour windows
- Weekend vs weekday patterns
- Peak hour identification
- Violation type temporal distribution

### Spatial Analysis
- Statistically significant hotspots
- Cluster analysis (if coordinates available)
- Location-based violation frequencies
- Severity patterns by location

## Visualizations
- Time window violation distribution
- Weekday vs weekend comparison
- Violation type heatmap
- Hotspot location map
- Cluster visualization (if coordinates available)

## Dependencies
- PySpark
- Pandas
- Matplotlib
- Seaborn
- NumPy
- scikit-learn (for clustering)
