from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, expr
import os

# Initialize Spark session
spark = SparkSession.builder \
    .appName("IngestData") \
    .getOrCreate()

# Ensure output directories exist
os.makedirs("data/processed/parquet", exist_ok=True)
os.makedirs("data/processed/csv", exist_ok=True)

# Read input file (relative path in repo)
input_path = "raw_traffic_violations.csv"
df = spark.read.option("header", True).csv(input_path)

# Show initial data for verification
print(f"Total Records Loaded: {df.count()}")
df.show(5, truncate=False)

# Clean and standardize timestamps
print("Cleaning and standardizing timestamps...")
df = df.withColumn("Timestamp", 
    expr("""
        CASE 
            WHEN try_to_timestamp(Timestamp, 'yyyy-MM-dd HH:mm:ss') IS NOT NULL 
                THEN try_to_timestamp(Timestamp, 'yyyy-MM-dd HH:mm:ss')
            WHEN try_to_timestamp(Timestamp, 'MM/dd/yy HH:mm') IS NOT NULL 
                THEN try_to_timestamp(Timestamp, 'MM/dd/yy HH:mm')
            WHEN try_to_timestamp(Timestamp, 'yyyy/MM/dd HH:mm:ss') IS NOT NULL 
                THEN try_to_timestamp(Timestamp, 'yyyy/MM/dd HH:mm:ss')
            WHEN try_to_timestamp(Timestamp, 'yyyy-MMM-dd HH:mm:ss') IS NOT NULL 
                THEN try_to_timestamp(Timestamp, 'yyyy-MMM-dd HH:mm:ss')
            ELSE NULL
        END
    """))

# Remove rows with invalid timestamps
df = df.filter(col("Timestamp").isNotNull())

# Standardize severity values: map textual to numeric and cast
print("Standardizing severity values...")
df = df.withColumn("Severity", 
    when(col("Severity").isin("HIGH", "High"), 5)
    .when(col("Severity").isin("LOW", "Low"), 1)
    .otherwise(expr("try_cast(Severity AS INT)")))

# Handle missing locations
df = df.na.fill({"Location": "UNKNOWN"})

# Clean and validate violation types
print("Cleaning violation types...")
df = df.filter(col("Violation Type").isNotNull())
df = df.withColumn("Violation Type", expr("upper(trim(`Violation Type`))"))

# Map known violation types to codes and filter unknowns
df = df.withColumn(
    "Violation_Type_Code",
    when(col("Violation Type") == "ILLEGAL PARKING", 1)
    .when(col("Violation Type") == "NO SIGNAL", 2)
    .when(col("Violation Type") == "SPEEDING", 3)
    .when(col("Violation Type") == "ILLEGAL TURN", 4)
    .when(col("Violation Type") == "RED LIGHT", 5)
    .otherwise(None)
)

# Filter out records with invalid violation types
df = df.filter(col("Violation_Type_Code").isNotNull())

# Show cleaning results
print("\nCleaning Summary:")
print(f"Records after cleaning: {df.count()}")
df.show(5, truncate=False)

# Write cleaned data using pandas to avoid Hadoop native dependencies on Windows
print("\nSaving cleaned data via pandas (Parquet + CSV)...")
import pandas as pd

output_path_parquet = os.path.abspath("data/processed/parquet/cleaned_violations.parquet")
output_path_csv = os.path.abspath("data/processed/csv/cleaned_violations.csv")

# Convert to pandas
pandas_df = df.toPandas()

# Convert Timestamp to string to avoid Parquet timestamp precision issues when
# reading back with Spark on Windows environments
if 'Timestamp' in pandas_df.columns:
    try:
        pandas_df['Timestamp'] = pandas_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        pandas_df['Timestamp'] = pandas_df['Timestamp'].astype(str)

# Save parquet via pandas using fastparquet engine for better Spark compatibility
pandas_df.to_parquet(output_path_parquet, index=False, engine='fastparquet')
print(f"Cleaned data saved to {output_path_parquet}")

# Save CSV via pandas
pandas_df.to_csv(output_path_csv, index=False)
print(f"CSV version saved to {output_path_csv}")

# Print schema of cleaned data
print("\nCleaned Data Schema:")
df.printSchema()

spark.stop()
print("\n✅ Data ingestion and cleaning completed successfully!")
