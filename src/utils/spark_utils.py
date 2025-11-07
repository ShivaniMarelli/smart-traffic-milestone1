import os
from pyspark.sql import SparkSession

def get_spark(app_name="smart-traffic-detector"):
    """Creates or retrieves a SparkSession."""
    # Create spark session
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    # Set log level to reduce verbose output
    spark.sparkContext.setLogLevel("WARN")
    
    return spark
