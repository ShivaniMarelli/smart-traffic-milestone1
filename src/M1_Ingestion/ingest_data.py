from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from pyspark.sql.functions import col, when, to_timestamp, lower, regexp_replace

CSV_FILE_PATH = "raw_traffic_violations.csv"
PARQUET_OUTPUT_PATH = "processed_data/M1_Cleaned_Data.parquet"

TARGET_SCHEMA = StructType([
    StructField("Violation_ID", StringType(), False),
    StructField("Timestamp_Clean", TimestampType(), True),
    StructField("Location", StringType(), True),
    StructField("Violation_Type", StringType(), True),
    StructField("Vehicle_Type", StringType(), True),
    StructField("Severity_Clean", IntegerType(), True)
])

print("Attempting to create SparkSession...")
spark = SparkSession.builder \
    .appName("M1_DataCleaning") \
    .master("local[*]") \
    .getOrCreate()

print("\nSUCCESS: PySpark Session established.")

raw_schema = StructType([
    StructField("Violation ID", StringType(), True),
    StructField("Timestamp", StringType(), True),
    StructField("Location", StringType(), True),
    StructField("Violation Type", StringType(), True),
    StructField("Vehicle Type", StringType(), True),
    StructField("Severity", StringType(), True)
])

try:
    df_raw = spark.read.csv(
        CSV_FILE_PATH,
        header=True,
        schema=raw_schema,
        nullValue=""
    )
    print(f"Total Records Loaded for Cleaning: {df_raw.count()}")

    TS_FORMATS = [
        "yyyy-MM-dd HH:mm:ss",
        "MM/dd/yy HH:mm",
        "yyyy/dd/MM HH:mm:ss",
        "yyyy_MM_dd HH:mm:ss",
    ]
    
    df_clean = df_raw.withColumn(
        "Timestamp_Clean",
        to_timestamp(col("Timestamp"), *TS_FORMATS)
    )

    df_clean = df_clean.withColumn(
        "Severity_Clean",
        when(col("Severity").cast(IntegerType()).isNull(), None)
        .otherwise(col("Severity").cast(IntegerType()))
    )
    
    df_clean = df_clean.withColumn(
        "Violation_Type", lower(col("Violation Type"))
    ).drop("Violation Type")

    df_clean = df_clean.withColumnRenamed("Violation ID", "Violation_ID")
    
    df_clean = df_clean.select(
        "Violation_ID",
        "Timestamp_Clean",
        "Location",
        "Violation_Type",
        col("Vehicle Type").alias("Vehicle_Type"),
        "Severity_Clean"
    )

    print(f"\nWriting {df_clean.count()} cleaned records to {PARQUET_OUTPUT_PATH}...")
    
    df_clean.write.mode("overwrite").parquet(PARQUET_OUTPUT_PATH)
    
    print("\nSUCCESS: Data Cleaning Complete. Final Clean Schema:")
    df_clean.printSchema()

except Exception as e:
    print(f"\nCRITICAL ERROR during data cleaning: {e}")

finally:
    spark.stop()
