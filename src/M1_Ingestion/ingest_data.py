from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, expr

# Initialize Spark session
spark = SparkSession.builder \
    .appName("IngestData") \
    .getOrCreate()

# Read input file (use the absolute path to your CSV)
input_path = r"C:\Users\shiva\OneDrive\Desktop\smart-traffic\raw_traffic_violations.csv"
df = spark.read.option("header", True).csv(input_path)

# Show initial data for verification
print(f"Total Records Loaded: {df.count()}")
df.show(10, truncate=False)

# Safely cast Severity to integer (it's already mostly int, but handle any strings/nulls)
df = df.withColumn("Severity", expr("try_cast(Severity AS INT)"))

# Optional: Filter out rows with null Violation Type (as in your original script)
df = df.filter(col("Violation Type").isNotNull())

# Optional: Add any other cleaning, e.g., map Violation Type to integers if needed
# Example: Map 'Illegal Parking' -> 1, 'Speeding' -> 2, etc. (adjust based on your needs)
violation_mapping = {
    "Illegal Parking": 1,
    "No Signal": 2,
    "Speeding": 3,
    "Illegal Turn": 4,
    "Red Light": 5
}
df = df.withColumn(
    "Violation_Type_Int",
    when(col("Violation Type") == "Illegal Parking", 1)
    .when(col("Violation Type") == "No Signal", 2)
    .when(col("Violation Type") == "Speeding", 3)
    .when(col("Violation Type") == "Illegal Turn", 4)
    .when(col("Violation Type") == "Red Light", 5)
    .otherwise(None)
)

# Write cleaned data to output (using Parquet for efficiency; change to CSV if needed)
output_path_csv = r"C:\Users\shiva\OneDrive\Desktop\smart-traffic\traffic_violations_cleaned.csv"
df.write.mode("overwrite").option("header", True).csv(output_path_csv)

print(f"Cleaned data saved to {output_path_csv}")

# Stop Spark session
spark.stop()
