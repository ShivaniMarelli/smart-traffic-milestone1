from pyspark.sql import SparkSession

# Path to the cleaned Parquet file created by ingest.py
PARQUET_INPUT_PATH = "processed_data/M1_Cleaned_Data.parquet"

# Create Spark session
spark = SparkSession.builder \
    .appName("Load_Cleaned_Data") \
    .master("local[*]") \
    .getOrCreate()

try:
    # Load the cleaned Parquet data
    df_clean = spark.read.parquet(PARQUET_INPUT_PATH)

    # Show top 5 records
    df_clean.show(5, truncate=False)

    # Print schema
    df_clean.printSchema()

except Exception as e:
    print(f"Error loading cleaned data: {e}")

finally:
    spark.stop()
import os

# Make sure processed_data folder exists
os.makedirs("processed_data", exist_ok=True)
