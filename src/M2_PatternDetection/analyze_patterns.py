import os
from time_pattern_analysis import run_time_pattern_analysis
from location_analysis import run_location_analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualization_dir():
    """Create directory for visualizations if it doesn't exist"""
    viz_dir = os.path.abspath(os.path.join("data", "processed", "visualizations"))
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir

def visualize_time_patterns(time_results, viz_dir):
    """Create visualizations for time-based patterns"""
    
    # Plot violations by time window
    violations_by_window = time_results["violations_by_window"].toPandas()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=violations_by_window, x="TimeWindow", y="Total_Violations")
    plt.title("Traffic Violations by Time Window")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "violations_by_timewindow.png"))
    plt.close()
    
    # Plot weekday vs weekend patterns
    weekday_weekend = time_results["weekday_weekend_analysis"].toPandas()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=weekday_weekend, x="Violation Type", y="Total_Violations", hue="IsWeekend")
    plt.title("Violations by Type: Weekday vs Weekend")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "weekday_weekend_patterns.png"))
    plt.close()
    
    # Heatmap of violation types by time window
    violation_window = time_results["violation_window_crosstab"].toPandas()
    plt.figure(figsize=(12, 8))
    # Get the first column name as it contains the violation types
    index_col = violation_window.columns[0]
    sns.heatmap(violation_window.set_index(index_col), cmap="YlOrRd", annot=True, fmt="g")
    plt.title("Violation Types by Time Window")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "violation_time_heatmap.png"))
    plt.close()

def visualize_location_patterns(location_results, viz_dir):
    """Create visualizations for location-based patterns"""
    
    # Plot top hotspots
    hotspots = location_results["hotspots"].toPandas()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=hotspots.head(10), x="Location", y="Total_Violations")
    plt.title("Top 10 Violation Hotspots")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "top_hotspots.png"))
    plt.close()
    
    # Plot cluster analysis if available
    if location_results["clusters"] is not None:
        clusters = location_results["clusters"].toPandas()
        plt.figure(figsize=(10, 6))
        plt.scatter(clusters["Cluster_Center_Long"], clusters["Cluster_Center_Lat"], 
                   s=clusters["Violations_in_Cluster"]/10, alpha=0.6)
        plt.title("Violation Clusters")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "violation_clusters.png"))
        plt.close()

def generate_analysis_report(time_results, location_results):
    """Generate a comprehensive analysis report"""
    
    report = []
    report.append("# Traffic Violation Pattern Analysis Report\n")
    
    # Time-based patterns
    report.append("## Temporal Patterns\n")
    
    # Peak hours analysis
    peak_hours = time_results["peak_hours"].toPandas()
    peak_hours_text = [
        "### Peak Hours Analysis",
        f"- Number of significant peak hours: {len(peak_hours[peak_hours['Is_Peak_Hour']])}",
        "- Top 3 peak hours:",
    ]
    for _, row in peak_hours.head(3).iterrows():
        peak_hours_text.append(f"  * Hour {int(row['Hour'])}: {row['Violations']} violations (Z-Score: {row['Z_Score']:.2f})")
    report.extend(peak_hours_text)
    report.append("")
    
    # Weekday vs Weekend patterns
    weekday_weekend = time_results["weekday_weekend_analysis"].toPandas()
    weekend_total = weekday_weekend[weekday_weekend["IsWeekend"]]["Total_Violations"].sum()
    weekday_total = weekday_weekend[~weekday_weekend["IsWeekend"]]["Total_Violations"].sum()
    report.extend([
        "### Weekday vs Weekend Patterns",
        f"- Average weekday violations: {weekday_total/5:.1f}",
        f"- Average weekend violations: {weekend_total/2:.1f}",
        ""
    ])
    
    # Location-based patterns
    report.append("## Spatial Patterns\n")
    
    # Hotspot analysis
    hotspots = location_results["hotspots"].toPandas()
    report.extend([
        "### Violation Hotspots",
        f"- Number of significant hotspots identified: {len(hotspots)}",
        "- Top 5 hotspot locations:"
    ])
    for _, row in hotspots.head(5).iterrows():
        report.append(f"  * {row['Location']}: {row['Total_Violations']} violations (Z-Score: {row['Z_Score']:.2f})")
    report.append("")
    
    # Cluster analysis
    if location_results["clusters"] is not None:
        clusters = location_results["clusters"].toPandas()
        report.extend([
            "### Spatial Clusters",
            f"- Number of clusters analyzed: {len(clusters)}",
            "- Cluster characteristics:"
        ])
        for _, row in clusters.iterrows():
            report.append(
                f"  * Cluster {int(row['prediction'])}: {row['Violations_in_Cluster']} violations, "
                f"Avg Severity: {row['Avg_Severity']:.2f}"
            )
    report.append("")
    
    # Save report
    report_path = os.path.join("data", "processed", "analysis_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    
    print(f"Analysis report saved to {report_path}")

def main():
    """Run complete pattern analysis"""
    print("Starting comprehensive pattern analysis...")
    
    # Create visualization directory
    viz_dir = create_visualization_dir()
    
    # Run time-based analysis
    print("\nRunning time pattern analysis...")
    time_results = run_time_pattern_analysis()
    
    # Run location-based analysis
    print("\nRunning location pattern analysis...")
    location_results = run_location_analysis()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_time_patterns(time_results, viz_dir)
    visualize_location_patterns(location_results, viz_dir)
    
    # Generate report
    print("\nGenerating analysis report...")
    generate_analysis_report(time_results, location_results)
    
    print("\n✅ Pattern analysis completed successfully!")
    print("Check data/processed/ directory for:")
    print("- CSV and Parquet files with detailed analysis")
    print("- Visualizations in the 'visualizations' subdirectory")
    print("- Complete analysis report in 'analysis_report.md'")

if __name__ == "__main__":
    main()