from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import split, col, to_date, concat, lit
import sys

# --- Configuration ---
# Set the base directory relative to the container's root working directory (/home/jovyan/work)
BASE_PATH = "/home/jovyan/work" 
RAW_INPUT_PATH = f"file://{BASE_PATH}/data/raw/tennis_points/tennis_raw_data.csv"
# The output path for the cleaned, baseline Parquet files
RAW_OUTPUT_PATH = f"file://{BASE_PATH}/data/raw/tennis_points_parquet"
LOG_FILE = f"{BASE_PATH}/raw_ingestion_log.txt"

# Define the full schema based on the actual CSV file structure
tennis_schema = StructType([
    StructField("match_id", StringType(), True), # e.g., 20220710-M-Wimbledon-F-Djokovic-Kyrgios (Updated structure)
    StructField("Pt", IntegerType(), True),       # Point number within match
    StructField("Set1", IntegerType(), True),
    StructField("Set2", IntegerType(), True),
    StructField("Gm1", IntegerType(), True),
    StructField("Gm2", IntegerType(), True),
    StructField("Pts", StringType(), True),       # Score (e.g., "40-AD")
    StructField("Gm#", IntegerType(), True),      # Game number within match
    StructField("Tbset", StringType(), True),     # Unneeded (to be dropped)
    StructField("Svr", IntegerType(), True),      # Server (1 or 2)
    StructField("1st", StringType(), True),       # 1st Serve Rally Description
    StructField("2nd", StringType(), True),       # 2nd Serve Rally Description
    StructField("Notes", StringType(), True),     # Unneeded (to be dropped)
    StructField("PtWinner", IntegerType(), True),  # Winner (1 or 2)
])

def run_raw_ingestion():
    """
    Spark job for Phase 1: Reads raw CSV, cleans, transforms key fields, 
    and saves the raw baseline data to the Raw Zone as Parquet.
    """
    
    # 1. Initialize Spark Session (Simulating a cluster session)
    try:
        spark = SparkSession.builder \
            .appName("TennisRawIngestion") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        
        print("--- Spark Session Initialized ---")
        
        # 2. Extract (Read Raw CSV using the defined schema)
        print(f"Reading raw CSV data from: {RAW_INPUT_PATH}")
        raw_df = spark.read \
            .csv(RAW_INPUT_PATH, header=True, schema=tennis_schema, inferSchema=False)
            
        print(f"Raw record count: {raw_df.count()}")
        raw_df.printSchema()

        # 3. Transform (Cleaning, Key Extraction, and Column Dropping)
        
        # A. Split the complex 'match_id' into required dimension columns
        # NEW STRUCTURE ASSUMPTION: date-gender-tournament-round-player1-player2
        df_split = raw_df.withColumn("match_split", split(col("match_id"), "-"))

        # NOTE: Indentation corrected below to align chained methods
        df_transformed = df_split \
            .withColumnRenamed("Pt", "point_id") \
            .withColumn("date_raw", col("match_split").getItem(0)) \
            .withColumn("date", to_date(col("date_raw"), "yyyyMMdd")) \
            .withColumn("tournament", col("match_split").getItem(2)) \
            .withColumn("round", col("match_split").getItem(3)) \
            .withColumn("player1_name", col("match_split").getItem(4)) \
            .withColumn("player2_name", col("match_split").getItem(5)) \
            .drop("match_split", "date_raw") # Drop temporary split column and raw date string
            
        # B. Drop unneeded columns as requested by the user
        df_transformed = df_transformed.drop("Tbset", "Notes", "match_id") # Dropping original match_id since we extracted its components

        # C. Minimal Cleaning (Drop rows with missing key fields)
        df_cleaned = df_transformed.na.drop(subset=['tournament', 'date', 'PtWinner'])
        
        # D. Remove Duplicates
        df_cleaned = df_cleaned.dropDuplicates()
        
        print(f"Cleaned record count after transformation: {df_cleaned.count()}")
        print("Final Raw Zone Schema:")
        df_cleaned.printSchema()


        # 4. Load (Ingest into Raw Zone as Parquet)
        print(f"Writing cleaned data to Raw Zone Parquet at: {RAW_OUTPUT_PATH}")
        df_cleaned.write \
            .mode("overwrite") \
            .parquet(RAW_OUTPUT_PATH)

        print("\n--- Phase 1: Raw Ingestion Complete ---")
        print(f"Data successfully saved to the Raw Zone in Parquet format: {RAW_OUTPUT_PATH}")
        
    except Exception as e:
        print(f"\nFATAL ERROR DURING INGESTION: {e}", file=sys.stderr)
        
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    run_raw_ingestion()