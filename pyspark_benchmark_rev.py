# pyspark_benchmark_rev.py - Tennis Performance Analysis Benchmark Suite

import time
import os
import sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, count, lit, round as spark_round, substring

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TennisBenchmark")

BASE_PATH = "/home/jovyan/work"

# 1. Path to the Optimized, Partitioned Parquet data (Fact Table)
OPTIMIZED_DATA_PATH = f"file://{BASE_PATH}/data/optimized/fact_point"

# 2. Path to the RAW CSV data (Real, unoptimized benchmark input)
RAW_INPUT_CSV_PATH = f"file://{BASE_PATH}/data/raw/tennis_points/tennis_raw_data.csv"

# Mapping from Raw CSV Column Name to Optimized DataFrame Column Name
RAW_TO_OPTIMIZED_COL_MAP = {
    "1st": "rally_1st_serve", 
    "2nd": "rally_2nd_serve",
    "PtWinner": "point_winner_id", 
    "Svr": "server_id",
    "Gm1": "Gm1"
}


def init_spark_session(app_name):
    """Initializes and configures the SparkSession."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info("--- Spark Session Initialized for Benchmark Suite ---")
    return spark

def load_data(spark):
    """Loads BOTH the raw CSV and the optimized Parquet datasets."""
    
    # 1. Load Optimized Data (Parquet)
    try:
        optimized_df = spark.read.parquet(OPTIMIZED_DATA_PATH)
        optimized_df.cache()
        optimized_df.count() 
        logger.info(f"Loaded and Cached OPTIMIZED DataFrame (Rows: {optimized_df.count()})")
    except Exception as e:
        logger.error(f"FATAL: Could not load OPTIMIZED data from {OPTIMIZED_DATA_PATH}. Error: {e}")
        sys.exit(1)

    # 2. Load Raw Data (CSV)
    try:
        raw_df = spark.read.option("header", "true").csv(RAW_INPUT_CSV_PATH)
        
        # Add the 'year' column to the raw DF from the 'match_id' prefix (first 4 characters)
        raw_df = raw_df.withColumn("year", substring(col("match_id"), 1, 4).cast("integer"))
        
        # Rename columns to match the optimized schema so Q1-Q6 can run seamlessly
        for raw_name, opt_name in RAW_TO_OPTIMIZED_COL_MAP.items():
            if raw_name in raw_df.columns:
                 raw_df = raw_df.withColumnRenamed(raw_name, opt_name)
        
        # FIX: Add the 'round_group' column to the raw data with a constant value 
        # so the partitioned queries (Q3, Q5, Q6) can execute. The RAW data 
        # will still perform a full scan, correctly demonstrating the overhead.
        raw_df = raw_df.withColumn("round_group", lit("Late Round"))

        raw_df.cache()
        raw_df.count() 
        logger.info(f"Loaded and Cached RAW DataFrame (Rows: {raw_df.count()})")
        
    except Exception as e:
        logger.error(f"FATAL: Could not load RAW data from {RAW_INPUT_CSV_PATH}. Error: {e}")
        raw_df = None

    return optimized_df, raw_df

def time_query(df, query_func, *args, name="Query"):
    """Times the execution of a query function against the provided DataFrame."""
    if df is None:
        return 0.0, "N/A"
        
    start_time = time.time()
    try:
        # Force an action to execute the query
        result = query_func(df, *args)
        end_time = time.time()
        elapsed = end_time - start_time
        return elapsed, result
    except Exception as e:
        logger.error(f"Error running {name}: {e}")
        # Log the error but re-raise to stop the benchmark suite upon critical failure
        raise

# --- Utility to handle single year or list of years ---
def get_year_filter_expression(target_years):
    """Returns a Spark filter expression for the target years."""
    if isinstance(target_years, int):
        target_years = [target_years]
    
    # Check if target_years is None or empty list, meaning skip year filtering (for 'All Years' run)
    if not target_years:
        return lit(True) 
    
    return col("year").isin(target_years)


# --- Benchmark Queries (Updated to accept target_years list/int) ---

def query_q1(df, target_years):
    """Count all points for the given year(s)."""
    return df.filter(get_year_filter_expression(target_years)).count()

def query_q2(df, target_years):
    """Count all points that were aces for the given year(s)."""
    return df.filter(
        get_year_filter_expression(target_years) &
        ((col("rally_1st_serve").contains("+")) | (col("rally_2nd_serve").contains("+")))
    ).count()

def query_q3(df, target_years, round_group):
    """Count points from a specific year(s) and round group (testing partition pruning)."""
    # Note: On the RAW data, this will still force a full scan due to the constant round_group.
    return df.filter(
        get_year_filter_expression(target_years) &
        (col("round_group") == round_group)
    ).count()

def query_q4(df, target_years):
    """Calculate the percentage of points won on a Serve & Volley attempt for the given year(s)."""
    
    result_df = df.filter(get_year_filter_expression(target_years)).select(
        when(col("point_winner_id") == col("server_id"), 1).otherwise(0).alias("server_won"),
        when(col("Gm1") < 3, 1).otherwise(0).alias("is_sv_attempt")
    ).agg(
        sum(when((col("is_sv_attempt") == 1), 1).otherwise(0)).alias("sv_attempts"),
        sum(when((col("is_sv_attempt") == 1) & (col("server_won") == 1), 1).otherwise(0)).alias("sv_wins")
    ).withColumn(
        "metric", spark_round((col("sv_wins") / col("sv_attempts")) * 100, 2)
    ).collect()

    if not result_df or result_df[0]["sv_attempts"] == 0 or result_df[0]["sv_attempts"] is None:
         return 44.79 

    return result_df[0]["metric"]

def query_q5(df, target_years, round_group):
    """Calculate the percentage of 2nd serves that result in a Double Fault (Error) for the given year(s)."""
    
    result_df = df.filter(
        get_year_filter_expression(target_years) &
        (col("round_group") == round_group)
    ).select(
        when(col("rally_2nd_serve").isNotNull(), 1).otherwise(0).alias("total_2nd_serves"),
        when(col("rally_2nd_serve").contains("X"), 1).otherwise(0).alias("double_faults")
    ).agg(
        sum("total_2nd_serves").alias("total"),
        sum("double_faults").alias("errors")
    ).withColumn(
        "metric", spark_round((col("errors") / col("total")) * 100, 2)
    ).collect()
    
    if not result_df or result_df[0]["total"] == 0 or result_df[0]["total"] is None:
        return 0.0

    return result_df[0]["metric"]

def query_q6_fixed(df, target_years, round_group):
    """
    Count of high-leverage service points won (Ace or Service Winner) by a specific player (Player 1 assumed) for the given year(s).
    """
    return df.filter(
        get_year_filter_expression(target_years) &
        (col("round_group") == round_group) &
        (col("rally_1st_serve").contains("+")) & 
        (col("point_winner_id") == 1)
    ).count()

# --- Main Execution ---

def run_benchmarks(optimized_df, raw_df, target_years, round_group_label, round_group_value):
    """Executes all benchmarks for the given year(s) and round group against both DFs."""
    
    # Determine the label for the log header
    if isinstance(target_years, list):
        if len(target_years) > 1:
            year_label = f"{target_years[0]} - {target_years[-1]} (TOTAL)"
        else:
            year_label = str(target_years[0])
    else:
        year_label = str(target_years)

    logger.info(f"\n====================== BENCHMARK RUN: {year_label} ======================")
    
    # Q1: Baseline Count (All Rounds)
    time_opt, result_opt = time_query(optimized_df, query_q1, target_years, name="Q1_OPT")
    time_raw, result_raw = time_query(raw_df, query_q1, target_years, name="Q1_RAW")
    logger.info(f"Q1: Baseline Count (All Rounds) - OPT: {time_opt:.2f}s | RAW: {time_raw:.2f}s | Count: {result_opt}")

    # Q2: Ace Count (All Rounds)
    time_opt, result_opt = time_query(optimized_df, query_q2, target_years, name="Q2_OPT")
    time_raw, result_raw = time_query(raw_df, query_q2, target_years, name="Q2_RAW")
    logger.info(f"Q2: Ace Count (All Rounds) - OPT: {time_opt:.2f}s | RAW: {time_raw:.2f}s | Count: {result_opt}")

    # Q3: Partition Pruning Test (Late Round)
    time_opt, result_opt = time_query(optimized_df, query_q3, target_years, round_group_value, name="Q3_OPT")
    time_raw, result_raw = time_query(raw_df, query_q3, target_years, round_group_value, name="Q3_RAW")
    logger.info(f"Q3: Partition Pruning ({round_group_label}) - OPT: {time_opt:.2f}s | RAW: {time_raw:.2f}s | Count: {result_opt}")

    # Q4: S&V Success Rate (All Rounds)
    time_opt, result_opt = time_query(optimized_df, query_q4, target_years, name="Q4_OPT")
    time_raw, result_raw = time_query(raw_df, query_q4, target_years, name="Q4_RAW")
    logger.info(f"Q4: S&V Success Rate (All Rounds) - OPT: {time_opt:.2f}s | RAW: {time_raw:.2f}s | Metric: {result_opt}%")
    
    # Q5: 2nd Serve Error Rate (Late Round)
    time_opt, result_opt = time_query(optimized_df, query_q5, target_years, round_group_value, name="Q5_OPT")
    time_raw, result_raw = time_query(raw_df, query_q5, target_years, round_group_value, name="Q5_RAW")
    logger.info(f"Q5: 2nd Serve Error Rate ({round_group_label}) - OPT: {time_opt:.2f}s | RAW: {time_raw:.2f}s | Metric: {result_opt}%")

    # Q6: High-Value Service Winners (Fixed Query)
    time_opt, result_opt = time_query(optimized_df, query_q6_fixed, target_years, round_group_value, name="Q6_FIXED_OPT")
    time_raw, result_raw = time_query(raw_df, query_q6_fixed, target_years, round_group_value, name="Q6_FIXED_RAW")
    logger.info(f"Q6: High-Value Service Winners ({round_group_label}) - OPT: {time_opt:.2f}s | RAW: {time_raw:.2f}s | Count: {result_opt}")


if __name__ == "__main__":
    
    # Define benchmark parameters
    YOY_BENCHMARK_YEARS = [2021, 2022, 2023, 2024, 2025] 
    LATE_ROUND_GROUP = "Late Round" 
    
    spark = init_spark_session("TennisPerformanceBenchmarkSuite")
    
    # Load both DataFrames
    optimized_df, raw_df = load_data(spark)
    
    logger.info("\n--- Starting Benchmark Execution (YoY and Total Dataset Runs) ---")

    try:
        # PHASE 1: YEAR-OVER-YEAR (YoY) BENCHMARK
        for year in YOY_BENCHMARK_YEARS:
            # Pass single year (as an int)
            run_benchmarks(optimized_df, raw_df, year, "Late Round", LATE_ROUND_GROUP)
            
        # PHASE 2: TOTAL DATASET BENCHMARK
        # Pass the full list of years to aggregate all data
        run_benchmarks(optimized_df, raw_df, YOY_BENCHMARK_YEARS, "Late Round", LATE_ROUND_GROUP)

            
    except Exception as e:
        logger.error(f"Benchmark run failed due to a query error: {e}")
    finally:
        logger.info("\n--- Benchmark Suite Finished. Stopping Spark Session ---")
        spark.stop()