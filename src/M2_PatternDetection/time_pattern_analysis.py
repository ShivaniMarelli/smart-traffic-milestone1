import os
import sys
from pyspark.sql.functions import (
    hour, 
    dayofweek, 
    month, 
    year, 
    col, 
    count, 
    desc, 
    to_timestamp,
    regexp_replace,
    expr,
    window,
    sum as spark_sum,
    when,
    avg,
    stddev
)

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.spark_utils import get_spark

# Configure Hadoop for Windows
hadoop_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'hadoop_home'))
os.environ['HADOOP_HOME'] = hadoop_home
os.environ['PATH'] = os.path.join(hadoop_home, 'bin') + os.pathsep + os.environ['PATH']

def run_time_pattern_analysis():
    # Step 1: Initialize Spark
    spark = get_spark("TimePatternAnalysis")

    # Step 2: Load the cleaned data
    df = spark.read.parquet("data/processed/parquet/cleaned_violations.parquet")
    
    # Convert string timestamp back to timestamp type
    df = df.withColumn("Timestamp", to_timestamp("Timestamp", "yyyy-MM-dd HH:mm:ss"))

    # Step 3: Derive time-based features
    df_time = df.withColumn("Hour", hour(col("Timestamp"))) \
                .withColumn("DayOfWeek", dayofweek(col("Timestamp"))) \
                .withColumn("Month", month(col("Timestamp"))) \
                .withColumn("Year", year(col("Timestamp"))) \
                .withColumn("IsWeekend", col("DayOfWeek").isin(1, 7)) \
                .withColumn("TimeWindow", 
                    expr("""
                    case
                        when Hour between 0 and 2 then '00:00-03:00'
                        when Hour between 3 and 5 then '03:00-06:00'
                        when Hour between 6 and 8 then '06:00-09:00'
                        when Hour between 9 and 11 then '09:00-12:00'
                        when Hour between 12 and 14 then '12:00-15:00'
                        when Hour between 15 and 17 then '15:00-18:00'
                        when Hour between 18 and 20 then '18:00-21:00'
                        else '21:00-00:00'
                    end
                    """))

    # Step 4: Analyze violations by 3-hour windows
    violations_by_window = (
        df_time.groupBy("TimeWindow")
               .agg(
                   count("*").alias("Total_Violations"),
                   avg("Severity").alias("Avg_Severity")
               )
               .orderBy("TimeWindow")
    )

    # Step 5: Analyze weekday vs weekend patterns
    weekday_weekend_analysis = (
        df_time.groupBy("IsWeekend", "Violation Type")
               .agg(count("*").alias("Total_Violations"))
               .orderBy("IsWeekend", desc("Total_Violations"))
    )

    # Step 6: Time window patterns by violation type
    violation_time_patterns = (
        df_time.groupBy("TimeWindow", "Violation Type")
               .agg(count("*").alias("Violations"))
               .orderBy("TimeWindow", desc("Violations"))
    )

    # Step 7: Statistical analysis of peak times
    hourly_stats = (
        df_time.groupBy("Hour")
               .agg(
                   count("*").alias("Violations"),
                   avg("Severity").alias("Avg_Severity"),
                   stddev("Severity").alias("Stddev_Severity")
               )
    )

    # Calculate global statistics for z-score calculation
    global_avg = hourly_stats.agg(avg("Violations")).collect()[0][0]
    global_stddev = hourly_stats.agg(stddev("Violations")).collect()[0][0]

    # Identify statistically significant peak hours (z-score > 2)
    peak_hours = (
        hourly_stats.withColumn(
            "Z_Score",
            (col("Violations") - global_avg) / global_stddev
        )
        .withColumn(
            "Is_Peak_Hour",
            col("Z_Score") > 2
        )
        .orderBy(desc("Z_Score"))
    )

    # Step 8: Create detailed crosstab of violation type by time window
    violation_window_crosstab = df_time.crosstab("Violation Type", "TimeWindow")

    # Step 8: Ensure output directory exists
    output_dir = "data/processed/parquet"
    os.makedirs(output_dir, exist_ok=True)

    # Step 9: Save results in both CSV and Parquet formats
    # CSV output
    csv_dir = os.path.abspath(os.path.join(os.getcwd(), "data", "processed", "csv"))
    print(f"Creating CSV directory: {csv_dir}")
    os.makedirs(csv_dir, exist_ok=True)

    # Parquet output
    parquet_dir = os.path.abspath(os.path.join(os.getcwd(), "data", "processed", "parquet"))
    print(f"Creating Parquet directory: {parquet_dir}")
    os.makedirs(parquet_dir, exist_ok=True)

    # Convert to pandas and save
    def save_dataframe(df, name):
        # Save as CSV
        csv_path = os.path.join(csv_dir, f"{name}.csv")
        parquet_path = os.path.join(parquet_dir, f"{name}.parquet")
        
        # Convert to pandas
        pandas_df = df.toPandas()
        
        # Save as CSV
        pandas_df.to_csv(csv_path, index=False)
        
        # Save as Parquet using pandas
        pandas_df.to_parquet(parquet_path, index=False)
        
        print(f"Saved {name} to CSV and Parquet formats")

    # Save all dataframes
    save_dataframe(violations_by_window, "violations_by_window")
    save_dataframe(weekday_weekend_analysis, "weekday_weekend_analysis")
    save_dataframe(violation_time_patterns, "violation_time_patterns")
    save_dataframe(violation_window_crosstab, "violation_window_crosstab")
    save_dataframe(peak_hours, "peak_hours")

    print("✅ Advanced Time Pattern Analysis completed successfully!")
    return {
        "violations_by_window": violations_by_window,
        "weekday_weekend_analysis": weekday_weekend_analysis,
        "violation_time_patterns": violation_time_patterns,
        "violation_window_crosstab": violation_window_crosstab,
        "peak_hours": peak_hours
    }

if __name__ == "__main__":
    run_time_pattern_analysis()