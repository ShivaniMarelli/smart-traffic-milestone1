from pyspark.sql import SparkSession

def get_spark(app_name="smart-traffic-detector"):
    """Creates or retrieves a SparkSession."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    return spark
