from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
import time
import sys

# --- Configuration ---
BASE_PATH = "/home/jovyan/work" 
# Raw Input (Worst Case: Full CSV read)
RAW_INPUT_CSV_PATH = f"file://{BASE_PATH}/data/raw/tennis_points/tennis_raw_data.csv" 
# Optimized Input (Best Case: Parquet with Partitioning)
OPTIMIZED_FACT_PATH = f"file://{BASE_PATH}/data/optimized/fact_point"

# --- Benchmark Parameters ---
TARGET_YEARS = [2021, 2022, 2023, 2024, 2025] # Based on Project Proposal scope
LATE_ROUND = "Late Round"
EARLY_ROUND = "Early Round"
ALL_ROUNDS = [LATE_ROUND, EARLY_ROUND]
# Volley/Overhead shots for Q4 success definition
ANY_VOLLEY_PATTERNS = ["v", "z", "o", "p", "j", "k"]

def get_q4_filter_raw(df_col, query_type):
    """Generates the Q4 filter condition for a single rally column (raw data).
    Q4_DEN: Rally contains '+' in the second position (S&V attempt filter adjusted).
    Q4_NUM: Rally contains '+' in the second position AND (v OR z OR o OR p OR j OR k) (Successful S&V shot hit)
    """
    # *** REFINEMENT: Filter points where the second character of the rally column is '+' ***
    # PySpark uses 1-based indexing for substr. The second character is at position 2.
    sv_attempt_filter = (df_col.isNotNull()) & (df_col.substr(2, 1) == "+")
    
    if query_type == "Q4_NUM":
        volley_or_condition = lit(False)
        for pattern in ANY_VOLLEY_PATTERNS:
            volley_or_condition = volley_or_condition | df_col.contains(pattern)
        return sv_attempt_filter & volley_or_condition
    else: # Q4_DEN
        return sv_attempt_filter

def get_q4_filter_optimized(df_col, query_type):
    """Generates the Q4 filter condition for a single rally column (optimized data).
    Q4_DEN: Rally contains '+' in the second position (S&V attempt filter adjusted).
    """
    # *** REFINEMENT: Filter points where the second character of the rally column is '+' ***
    # PySpark uses 1-based indexing for substr. The second character is at position 2.
    sv_attempt_filter = (df_col.isNotNull()) & (df_col.substr(2, 1) == "+")
    
    if query_type == "Q4_NUM":
        volley_or_condition = lit(False)
        for pattern in ANY_VOLLEY_PATTERNS:
            volley_or_condition = volley_or_condition | df_col.contains(pattern)
        return sv_attempt_filter & volley_or_condition
    else: # Q4_DEN
        return sv_attempt_filter

def get_q6_filter(rally_col, pt_winner_col, query_type):
    """Generates the Q6 filter condition based on approach presence and PtWinner.
    Q6_DEN: Rally contains '+' (Approach attempt)
    Q6_NUM: Rally contains '+' AND PtWinner = 1 (Successful Approach, server wins point)
    """
    
    # Denominator filter: Approach shot (+) is present in the 1st serve rally
    q6_filter = rally_col.contains("+")
    
    # Numerator filter: Denominator filter AND PtWinner = 1
    if "Q6_NUM" in query_type:
        q6_filter = q6_filter & (pt_winner_col == 1)
        
    return q6_filter

def run_query_raw(spark, query_type, target_year, target_round=None, rally_pattern=None, rally_pattern_2=None, is_second_serve=False):
    """Executes a benchmark query against the unoptimized RAW CSV data."""
    start_time = time.time()
    
    # Read the entire raw CSV file (simulating full scan overhead)
    raw_df_full = spark.read.option("header", "true").csv(RAW_INPUT_CSV_PATH)

    # Base filter for raw data
    raw_filter = col("match_id").startswith(str(target_year))
    
    # 2. Second Serve Filter (Q5 specific - checks the '2nd' rally column)
    if is_second_serve:
        # Denominator check (total 2nd serves)
        raw_filter = raw_filter & col("2nd").isNotNull()
        # Numerator check (Unforced Error in 2nd serve rally)
        if rally_pattern == "@":
             raw_filter = raw_filter & col("2nd").contains(rally_pattern)
        
    # 3. Rally Pattern Filter (for all other queries)
    else:
        if rally_pattern:
            
            # --- Q4 Complex Logic (Serve-and-Volley on 1st OR 2nd serve) ---
            if "Q4" in query_type and rally_pattern == "COMPLEX_VOLLEY":
                # Apply filter to 1st serve rally OR 2nd serve rally
                q4_filter_1st = get_q4_filter_raw(col("1st"), query_type)
                q4_filter_2nd = get_q4_filter_raw(col("2nd"), query_type)
                raw_filter = raw_filter & (q4_filter_1st | q4_filter_2nd)
            
            # --- Q6 Complex Logic (Approach Success Rate) ---
            elif "Q6" in query_type and rally_pattern == "COMPLEX_APPROACH":
                # Raw data uses "1st" and "PtWinner"
                raw_filter = raw_filter & get_q6_filter(col("1st"), col("PtWinner"), query_type)
            
            # --- Other Rally Filters (Q2) ---
            else: 
                raw_filter = raw_filter & col("1st").contains(rally_pattern)

    # Select PtWinner column only to minimize data transfer if needed, although count is the main metric
    result_df = raw_df_full.filter(raw_filter).select(col("PtWinner"))
    
    count_val = result_df.count()
    duration = time.time() - start_time
    return count_val, duration


def run_query_optimized(spark, query_id, target_year, target_round=None, rally_pattern=None, rally_pattern_2=None, is_second_serve=False):
    """Executes a benchmark query against the OPTIMIZED Parquet data."""
    start_time = time.time()

    # Read the fact table
    optimized_fact = spark.read.parquet(OPTIMIZED_FACT_PATH)
    
    # Initialize filter with partition filters
    optimized_filter = lit(True)

    # 1. Partition Filter: Year (Critical for Partition Pruning)
    optimized_filter = optimized_filter & (col("year") == target_year)
        
    # 2. Partition Filter: Round Group (Critical for Partition Pruning)
    if target_round in ALL_ROUNDS:
        optimized_filter = optimized_filter & (col("round_group") == target_round)
    
    # 3. Second Serve Filter (Q5 specific)
    if is_second_serve:
        # Denominator (Total 2nd serve points)
        optimized_filter = optimized_filter & col("rally_2nd_serve").isNotNull()
        # Numerator (Unforced Error in 2nd serve rally)
        if rally_pattern == "@":
            optimized_filter = optimized_filter & col("rally_2nd_serve").contains(rally_pattern)

    # 4. Columnar Filter: Rally Patterns (for all other queries)
    else:
        if rally_pattern:
            
            # --- Q4 Complex Logic (Serve-and-Volley on 1st OR 2nd serve) ---
            if "Q4" in query_id and rally_pattern == "COMPLEX_VOLLEY":
                # Apply filter to 1st serve rally OR 2nd serve rally
                q4_filter_1st = get_q4_filter_optimized(col("rally_1st_serve"), query_id)
                q4_filter_2nd = get_q4_filter_optimized(col("rally_2nd_serve"), query_id)
                optimized_filter = optimized_filter & (q4_filter_1st | q4_filter_2nd)

            # --- Q6 Complex Logic (Approach Success Rate) ---
            elif "Q6" in query_id and rally_pattern == "COMPLEX_APPROACH":
                # Optimized data uses "rally_1st_serve" and "PtWinner"
                optimized_filter = optimized_filter & get_q6_filter(col("rally_1st_serve"), col("PtWinner"), query_id)
            
            # --- Other Rally Filters (Q2) ---
            else:
                optimized_filter = optimized_filter & col("rally_1st_serve").contains(rally_pattern)

    
    result_df = optimized_fact.filter(optimized_filter).select(col("point_id"))
    
    count_val = result_df.count()
    duration = time.time() - start_time
    return count_val, duration

def calculate_metric(numerator, denominator):
    """Calculates Success/Error Rate."""
    if denominator == 0 or denominator is None:
        return None
    return (numerator / denominator) * 100


def run_benchmarks():
    """Runs the full suite of 6 benchmark queries, YoY."""
    
    results = {}
    
    try:
        spark = SparkSession.builder \
            .appName("TennisPerformanceBenchmarkSuiteYoY") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        print("--- Spark Session Initialized for Benchmark Suite (YoY) ---")
        
        # Ensure the optimized path exists
        try:
            spark.read.parquet(OPTIMIZED_FACT_PATH).limit(1).collect()
        except Exception:
            print(f"Error: Optimized data not found at {OPTIMIZED_FACT_PATH}. Ensure Phase 2 ETL ran successfully.")
            return

        # ====================================================================
        # QUERY DEFINITIONS
        # ====================================================================

        query_params = [
            ("Q1: Baseline Count", "All Rounds", {"rally_pattern": None}),
            ("Q2: Ace Count", "All Rounds", {"rally_pattern": "*"}),
            # Q3 is specific to Late Rounds
            ("Q3: Partition Pruning", LATE_ROUND, {"rally_pattern": None}),
            # Q4 is All Rounds
            ("Q4: S&V Success Rate", "All Rounds", {"rally_pattern": "COMPLEX_VOLLEY"}),
            # Q5 is Late Rounds only
            ("Q5: 2nd Serve Error Rate", LATE_ROUND, {"is_second_serve": True}),
            # Q6 requires running twice (Late and Early)
            ("Q6: Approach Success Rate (Late)", LATE_ROUND, {"rally_pattern": "COMPLEX_APPROACH"}),
            ("Q6: Approach Success Rate (Early)", EARLY_ROUND, {"rally_pattern": "COMPLEX_APPROACH"}),
        ]
        
        # ====================================================================
        # EXECUTION LOOP: Year -> Query -> Round Group
        # ====================================================================
        
        print("\n--- Starting Benchmark Execution (YoY Analysis) ---")
        
        for target_year in TARGET_YEARS:
            print(f"\n====================== YEAR {target_year} ======================")

            for name, round_scope, q_params in query_params:
                q_id = name.split(":")[0]
                
                # Determine target round group for partitioning
                target_round_group = round_scope if round_scope in ALL_ROUNDS else None
                
                # --- SIMPLE QUERIES (Q1, Q2, Q3) ---
                if q_id in ["Q1", "Q2", "Q3"]:
                    
                    # Optimized Run
                    opt_count, opt_duration = run_query_optimized(spark, q_id, target_year, target_round_group, q_params.get("rally_pattern"))
                    
                    # Raw Run
                    raw_count, raw_duration = run_query_raw(spark, q_id, target_year, target_round_group, q_params.get("rally_pattern"))
                    
                    # Store result
                    key = f"{name} ({target_year})"
                    results[key] = {
                        "scope": round_scope,
                        "optimized_time": opt_duration, 
                        "optimized_count": opt_count,
                        "raw_time": raw_duration, 
                        "raw_count": raw_count
                    }
                    print(f"{name} ({round_scope}) - OPT: {opt_duration:.4f}s | RAW: {raw_duration:.4f}s | Count: {opt_count}")


                # --- COMPLEX METRIC QUERIES (Q4, Q5, Q6) ---
                else: 
                    
                    # ---------------------- DENOMINATOR RUN (Total Attempts/Points) ----------------------
                    
                    if q_id == "Q5":
                        denom_opt_count, denom_opt_time = run_query_optimized(spark, f"{q_id}_DEN", target_year, target_round_group, is_second_serve=True)
                    elif q_id == "Q4":
                        denom_opt_count, denom_opt_time = run_query_optimized(spark, f"{q_id}_DEN", target_year, target_round_group, rally_pattern="COMPLEX_VOLLEY")
                    else: # Q6 Denominator: Total approaches
                        denom_opt_count, denom_opt_time = run_query_optimized(spark, f"{q_id}_DEN", target_year, target_round_group, rally_pattern=q_params.get("rally_pattern"))


                    # ---------------------- NUMERATOR RUN (Successes/Failures) ----------------------
                    
                    if q_id == "Q4":
                        # Success: S&V (+) AND hitting any volley shot on 1st OR 2nd serve rally
                        num_params = {"rally_pattern": "COMPLEX_VOLLEY"} 
                        num_opt_count, num_opt_time = run_query_optimized(spark, f"{q_id}_NUM", target_year, target_round_group, rally_pattern="COMPLEX_VOLLEY")
                    elif q_id == "Q5":
                        # Failure: 2nd Serve UE (@)
                        num_params = {"rally_pattern": "@", "is_second_serve": True}
                        num_opt_count, num_opt_time = run_query_optimized(spark, f"{q_id}_NUM", target_year, target_round_group, **num_params)
                    elif q_id == "Q6: Approach Success Rate (Late)" or q_id == "Q6: Approach Success Rate (Early)":
                        # Success: Approach (+) and Server wins (PtWinner = 1)
                        num_params = {"rally_pattern": "COMPLEX_APPROACH"}
                        num_opt_count, num_opt_time = run_query_optimized(spark, f"{q_id}_NUM", target_year, target_round_group, rally_pattern="COMPLEX_APPROACH")

                    # --- CALCULATE METRIC & RAW BENCHMARK ---
                    
                    opt_metric = calculate_metric(num_opt_count, denom_opt_count)
                    
                    # We run the RAW benchmark against the Numerator filter (which is the most complex)
                    raw_count, raw_duration = run_query_raw(spark, f"{q_id}_NUM", target_year, target_round_group, **num_params)
                    
                    key = f"{name} ({target_year})"
                    results[key] = {
                        "scope": round_scope,
                        "optimized_time": denom_opt_time, # Total time is represented by the Denominator run time
                        "optimized_count": denom_opt_count,
                        "optimized_metric": f"{opt_metric:.2f}%" if opt_metric is not None else "N/A",
                        "raw_time": raw_duration, 
                        "raw_count": raw_count
                    }
                    print(f"{name} ({round_scope}) - OPT: {denom_opt_time:.4f}s | RAW: {raw_duration:.4f}s | Metric: {results[key]['optimized_metric']}")

        # ====================================================================
        # SUMMARY AND DELIVERABLE
        # ====================================================================
        
        print("\n\n=======================================================================================")
        print("                           FINAL BENCHMARK SUITE RESULTS (YoY)")
        print("=======================================================================================")
        print(f"{'Query Name (Year)':<35} | {'Scope':<10} | {'Metric':<10} | {'Opt Time (s)':<12} | {'Raw Time (s)':<12} | {'Speedup':<10}")
        print("-" * 105)

        for key, data in results.items():
            speedup = data['raw_time'] / data['optimized_time'] if data['optimized_time'] > 0 else 0
            metric_display = data.get('optimized_metric', f"{data['optimized_count']} pts")
            
            print(f"{key:<35} | {data['scope']:<10} | {metric_display:<10} | {data['optimized_time']:<12.4f} | {data['raw_time']:<12.4f} | {speedup:<10.1f}x")
            
        print("=======================================================================================")
        print("Benchmark suite complete. Analyze the speedup column for performance gains and the YoY trend.")

    except Exception as e:
        print(f"\nFATAL ERROR DURING BENCHMARK SUITE: {e}", file=sys.stderr)
        
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    run_benchmarks()