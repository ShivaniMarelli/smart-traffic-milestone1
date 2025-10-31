from pyspark.sql.functions import hour, dayofweek, month, year, col, count
from utils.spark_utils import get_spark

# Step 1: Start Spark
spark = get_spark("TimeOffenseAnalysis")

# Step 2: Load your ingested CSV file from Milestone 1
df = spark.read.option("header", True).csv("data/raw/violations.csv")

# Step 3: Convert timestamp column to proper timestamp
# 👉 Change "Violation_Timestamp" to your real timestamp column name if different
df = df.withColumn("Violation_Timestamp", col("Violation_Timestamp").cast("timestamp"))

# Step 4: Derive new time features
df_time = (
    df.withColumn("Hour", hour(col("Violation_Timestamp")))
      .withColumn("DayOfWeek", dayofweek(col("Violation_Timestamp")))
      .withColumn("Month", month(col("Violation_Timestamp")))
      .withColumn("Year", year(col("Violation_Timestamp")))
)

# Step 5: Aggregations
violations_per_hour = df_time.groupBy("Hour").agg(count("*").alias("Total_Violations"))
violations_per_day = df_time.groupBy("DayOfWeek").agg(count("*").alias("Total_Violations"))
violations_by_type = df_time.groupBy("Offense_Type").agg(count("*").alias("Total_Violations"))
cross_tab = df_time.crosstab("Offense_Type", "Hour")

# Step 6: Save outputs as Parquet
violations_per_hour.write.mode("overwrite").parquet("data/processed/parquet/violations_per_hour")
violations_per_day.write.mode("overwrite").parquet("data/processed/parquet/violations_per_day")
violations_by_type.write.mode("overwrite").parquet("data/processed/parquet/violations_by_type")
cross_tab.write.mode("overwrite").parquet("data/processed/parquet/crosstab_offense_hour")

print("✅ Week 3 done — Time & Offense Analysis completed successfully!")
