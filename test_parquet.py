from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os
import shutil

# -------------------------
# Paths
# -------------------------
input_csv = r"C:\Users\shiva\OneDrive\Desktop\smart-traffic\raw_traffic_violations.csv"
output_parquet = r"C:\temp\traffic_violations.parquet"  # Use a folder outside OneDrive

# -------------------------
# Clean previous output
# -------------------------
if os.path.exists(output_parquet):
    shutil.rmtree(output_parquet)

# -------------------------
# Set Hadoop for Windows
# -------------------------
os.environ['HADOOP_HOME'] = r"C:\hadoop"
os.environ['PATH'] = r"C:\hadoop\bin;" + os.environ['PATH']

# -------------------------
# Create Spark session
# -------------------------
spark = SparkSession.builder \
    .appName("Traffic Violations") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

try:
    # Load CSV
    df = spark.read.csv(input_csv, header=True, inferSchema=True)
    print(f"Total Records Loaded: {df.count()}")

    # Filter rows where Violation Type is not null
    df_filtered = df.filter(col("Violation Type").isNotNull())
    df_filtered.show(10, truncate=False)

    # Save as Parquet
    df_filtered.write.mode("overwrite").parquet(output_parquet)
    print(f"Data saved successfully to {output_parquet}")

    # -------------------------
    # Verify by reading back
    # -------------------------
    df_check = spark.read.parquet(output_parquet)
    print(f"Total Records in Parquet: {df_check.count()}")
    df_check.show(10, truncate=False)

except Exception as e:
    print(f"Error: {e}")

finally:
    spark.stop()
