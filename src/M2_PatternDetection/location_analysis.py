import os
import sys
import pandas as pd
import numpy as np
from pyspark.sql.functions import (
    col, count, desc, expr, avg, stddev,
    sum as spark_sum, round as spark_round,
    when, lit, regexp_extract, to_timestamp,
    dayofweek
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.types import DoubleType

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.spark_utils import get_spark

# Configure Hadoop for Windows
hadoop_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'hadoop_home'))
os.environ['HADOOP_HOME'] = hadoop_home
os.environ['PATH'] = os.path.join(hadoop_home, 'bin') + os.pathsep + os.environ['PATH']

def extract_coordinates(df):
    """Extract latitude and longitude from location string if available"""
    return df.withColumn(
        "Latitude",
        when(
            regexp_extract(col("Location"), r"(-?\d+\.?\d*),\s*(-?\d+\.?\d*)", 1) != "",
            regexp_extract(col("Location"), r"(-?\d+\.?\d*),\s*(-?\d+\.?\d*)", 1).cast(DoubleType())
        ).otherwise(None)
    ).withColumn(
        "Longitude",
        when(
            regexp_extract(col("Location"), r"(-?\d+\.?\d*),\s*(-?\d+\.?\d*)", 2) != "",
            regexp_extract(col("Location"), r"(-?\d+\.?\d*),\s*(-?\d+\.?\d*)", 2).cast(DoubleType())
        ).otherwise(None)
    )

def identify_hotspots(df, threshold_stddev=2.0):
    """Identify statistically significant hotspots based on violation frequency"""
    
    # Calculate violation frequencies and statistics
    location_stats = (
        df.groupBy("Location")
          .agg(
              count("*").alias("Total_Violations"),
              avg("Severity").alias("Avg_Severity"),
              spark_sum(when(col("IsWeekend"), 1).otherwise(0)).alias("Weekend_Violations"),
              spark_sum(when(col("IsWeekend").isNull(), 1).otherwise(0)).alias("Weekday_Violations")
          )
    )
    
    # Calculate global statistics
    global_stats = location_stats.agg(
        avg("Total_Violations").alias("mean"),
        stddev("Total_Violations").alias("stddev")
    ).collect()[0]
    
    # Identify hotspots using z-score
    hotspots = (
        location_stats
        .withColumn(
            "Z_Score",
            (col("Total_Violations") - lit(global_stats["mean"])) / lit(global_stats["stddev"])
        )
        .withColumn(
            "Is_Hotspot",
            col("Z_Score") > threshold_stddev
        )
        .where(col("Is_Hotspot") == True)
        .orderBy(desc("Z_Score"))
    )
    
    return hotspots

def cluster_locations(df, k=5):
    """Perform K-means clustering on locations with coordinates"""
    
    # Filter locations with valid coordinates
    locations_with_coords = df.filter(
        col("Latitude").isNotNull() & 
        col("Longitude").isNotNull()
    )
    
    if locations_with_coords.count() == 0:
        print("No valid coordinates found for clustering analysis")
        return None
    
    # Prepare features for clustering
    assembler = VectorAssembler(
        inputCols=["Latitude", "Longitude"],
        outputCol="features"
    )
    
    # Transform the data
    features_df = assembler.transform(locations_with_coords)
    
    # Train K-means model
    kmeans = KMeans(k=k, seed=1)
    model = kmeans.fit(features_df)
    
    # Add cluster predictions
    clustered_locations = model.transform(features_df)
    
    # Analyze clusters
    cluster_stats = (
        clustered_locations
        .groupBy("prediction")
        .agg(
            count("*").alias("Violations_in_Cluster"),
            avg("Severity").alias("Avg_Severity"),
            avg("Latitude").alias("Cluster_Center_Lat"),
            avg("Longitude").alias("Cluster_Center_Long")
        )
        .orderBy(desc("Violations_in_Cluster"))
    )
    
    return cluster_stats

def run_location_analysis():
    # Step 1: Initialize Spark
    spark = get_spark("LocationAnalysis")

    # Step 2: Load the cleaned data
    df = spark.read.parquet("data/processed/parquet/cleaned_violations.parquet")
    
    # Convert string timestamp back to timestamp type
    df = df.withColumn("Timestamp", to_timestamp("Timestamp", "yyyy-MM-dd HH:mm:ss"))
    
    # Add weekend indicator
    df = df.withColumn("IsWeekend", dayofweek(col("Timestamp")).isin(1, 7))

    # Step 3: Extract coordinates if available
    df_with_coords = extract_coordinates(df)

    # Step 4: Identify hotspots
    print("Identifying violation hotspots...")
    hotspots = identify_hotspots(df)

    # Step 5: Check if we have any valid coordinates
    coords_count = df_with_coords.filter(
        col("Latitude").isNotNull() & 
        col("Longitude").isNotNull()
    ).count()

    # Step 6: Perform spatial clustering if coordinates are available
    cluster_analysis = None
    if coords_count > 0:
        print(f"Found {coords_count} locations with valid coordinates. Performing clustering...")
        cluster_analysis = cluster_locations(df_with_coords)
    else:
        print("No valid coordinates found in the dataset. Skipping spatial clustering.")

    # Step 7: Generate basic location statistics
    location_stats = (
        df.groupBy("Location")
          .agg(
              count("*").alias("Total_Violations"),
              avg("Severity").alias("Avg_Severity")
          )
          .orderBy(desc("Total_Violations"))
    )

    # Step 5: Ensure output directory exists
    output_dir = "data/processed/parquet"
    os.makedirs(output_dir, exist_ok=True)

    # Step 7: Save results in both CSV and Parquet formats
    csv_dir = os.path.abspath(os.path.join(os.getcwd(), "data", "processed", "csv"))
    print(f"Creating CSV directory: {csv_dir}")
    os.makedirs(csv_dir, exist_ok=True)

    parquet_dir = os.path.abspath(os.path.join(os.getcwd(), "data", "processed", "parquet"))
    print(f"Creating Parquet directory: {parquet_dir}")
    os.makedirs(parquet_dir, exist_ok=True)

    def save_dataframe(df, name):
        # Save as CSV
        csv_path = os.path.join(csv_dir, f"{name}.csv")
        # Save as Parquet
        parquet_path = os.path.join(parquet_dir, f"{name}.parquet")
        
        # Convert to pandas and save
        pandas_df = df.toPandas()
        pandas_df.to_csv(csv_path, index=False)
        pandas_df.to_parquet(parquet_path, index=False)
        print(f"Saved {name} to CSV and Parquet formats")

    # Save all analysis results
    print("Saving analysis results...")
    save_dataframe(location_stats, "location_stats")
    save_dataframe(hotspots, "hotspot_locations")
    if cluster_analysis is not None:
        save_dataframe(cluster_analysis, "location_clusters")

    print("✅ Advanced Location Analysis completed successfully!")
    return {
        "location_stats": location_stats,
        "hotspots": hotspots,
        "clusters": cluster_analysis
    }

if __name__ == "__main__":
    run_location_analysis()
