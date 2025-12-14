from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, monotonically_increasing_id, row_number, year, when, array_contains, explode, struct, sha2, concat_ws
from pyspark.sql.window import Window
import sys

# --- Configuration ---
BASE_PATH = "/home/jovyan/work" 
# Input: The cleaned Parquet files from Phase 1
RAW_PARQUET_PATH = f"file://{BASE_PATH}/data/raw/tennis_points_parquet"

# Output: The Optimized Zone paths
OPTIMIZED_FACT_PATH = f"file://{BASE_PATH}/data/optimized/fact_point"
OPTIMIZED_DIM_MATCH_PATH = f"file://{BASE_PATH}/data/optimized/dim_match"
OPTIMIZED_DIM_PLAYER_PATH = f"file://{BASE_PATH}/data/optimized/dim_player"

def get_round_group(round_col):
    """
    Categorizes the match round into 'Early Round' or 'Late Round'.
    Late Rounds: F (Finals), SF (Semi-Finals), Q (Quarter-Finals).
    """
    late_rounds = ['F', 'SF', 'Q']
    return when(col(round_col).isin(late_rounds), lit("Late Round")) \
        .otherwise(lit("Early Round"))

def run_etl_star_schema():
    """
    Spark job for Phase 2: Builds the Star Schema (Fact and Dimension Tables)
    from the Raw Zone data and applies Partitioning (by year and round group) 
    to the Fact Table.
    """
    
    # Initialize Spark Session
    try:
        spark = SparkSession.builder \
            .appName("TennisStarSchemaETL") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        
        print("--- Spark Session Initialized for ETL ---")
        
        # Extract: Read the cleaned Parquet data from the Raw Zone
        print(f"Reading cleaned Parquet data from: {RAW_PARQUET_PATH}")
        raw_df = spark.read.parquet(RAW_PARQUET_PATH)
        
        # DIMENSION 1: Match Dimension (DimMatch)
        print("\n--- Creating DimMatch ---")
        
        # Select unique match context fields
        match_df = raw_df.select(
            "date", 
            "tournament", 
            "round", 
            "player1_name", 
            "player2_name"
        ).distinct()
        
        # Add a unique, stable surrogate key (match_key)
        # Using SHA2 hash of concatenated match context fields for a stable ID
        dim_match = match_df.withColumn(
            "match_key", 
            sha2(concat_ws("|", col("date"), col("tournament"), col("player1_name"), col("player2_name")), 256)
        )
        
        # Add Year and Round Group for easy lookups and Fact Table joins
        dim_match = dim_match.withColumn("year", year(col("date")))
        dim_match = dim_match.withColumn("round_group", get_round_group("round"))
        
        # Write DimMatch to the Optimized Zone as Parquet
        print(f"Writing DimMatch (Count: {dim_match.count()}) to: {OPTIMIZED_DIM_MATCH_PATH}")
        dim_match.write.mode("overwrite").parquet(OPTIMIZED_DIM_MATCH_PATH)


        # DIMENSION 2: Player Dimension (DimPlayer)
        print("\n--- Creating DimPlayer ---")
        
        # Gather all unique player names from both player columns
        players1 = raw_df.select(col("player1_name").alias("player_name")).distinct()
        players2 = raw_df.select(col("player2_name").alias("player_name")).distinct()
        
        # Combine the lists and get unique player names
        dim_player_names = players1.union(players2).distinct()
        
        # Add a unique, stable surrogate key (player_key)
        dim_player = dim_player_names.withColumn(
            "player_key",
            sha2(col("player_name"), 256)
        ).withColumn("player_id", monotonically_increasing_id()) # Add simple ID for uniqueness demonstration
        
        # Write DimPlayer to the Optimized Zone as Parquet
        print(f"Writing DimPlayer (Count: {dim_player.count()}) to: {OPTIMIZED_DIM_PLAYER_PATH}")
        dim_player.write.mode("overwrite").parquet(OPTIMIZED_DIM_PLAYER_PATH)


        # FACT TABLE: Points Fact Table (FactPoint)
        print("\n--- Creating FactPoint (Partitioned by Year and Round Group) ---")
        
        # Join the Raw Data with DimMatch to get the match_key
        fact_df = raw_df.join(
            # Select required keys and partitioning columns from DimMatch
            dim_match.select("match_key", "date", "tournament", "player1_name", "player2_name", "year", "round_group"),
            on=["date", "tournament", "player1_name", "player2_name"],
            how="inner"
        )
        
        # Select the final fact columns and dimension keys
        fact_point = fact_df.select(
            # Keys
            col("match_key"),
            
            # Partitioning Columns
            col("year"), 
            col("round_group"),
            
            # Point Metrics/Facts
            col("point_id"),
            col("Set1"),
            col("Set2"),
            col("Gm1"),
            col("Gm2"),
            col("Pts").alias("game_score"),
            col("Gm#").alias("game_number_in_match"),
            col("Svr").alias("server_id"),
            col("1st").alias("rally_1st_serve"),
            col("2nd").alias("rally_2nd_serve"),
            col("PtWinner").alias("point_winner_id")
        )

        # Write FactPoint to the Optimized Zone with Partitioning
        print(f"Writing FactPoint (Count: {fact_point.count()}) to: {OPTIMIZED_FACT_PATH} and applying PARTITIONING...")
        
        # Partitioning by year AND round_group
        fact_point.write \
            .mode("overwrite") \
            .partitionBy("year", "round_group") \
            .parquet(OPTIMIZED_FACT_PATH)

        print("\n--- Phase 2: Star Schema ETL and Partitioning Complete ---")
        print("Optimized data is ready for benchmarking.")
        
    except Exception as e:
        print(f"\nFATAL ERROR DURING ETL: {e}", file=sys.stderr)
        
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    run_etl_star_schema()