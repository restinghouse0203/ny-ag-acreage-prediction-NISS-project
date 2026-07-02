"""
generate_soil_mapping.py
========================
Queries the USDA Soil Data Access (SDA) REST API to resolve each unique
CSB polygon centroid to its gSSURGO MUKEY (soil map unit key), producing
the file:

    csbid_mukey_mapping.csv  —  columns: CSBID, MUKEY

This file is consumed by feature_engineering.py to join soil attributes
onto the crop dataset.

API reference
-------------
POST https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest
Parameter: QUERY = SQL using SDA_Get_Mukey_from_intersection_with_WktWgs84('POINT(lon lat)')
Format:    JSON+COLUMNNAME

Execution notes
---------------
- Deduplicates on CSBID so each polygon is queried exactly once.
- Processes polygons in batches using concurrent API requests (ThreadPoolExecutor).
- Writes a checkpoint file periodically so the script is resumable
  if interrupted (re-run and it will pick up where it left off).
- PERFORMANCE: Uses 6 concurrent workers by default for ~3-5x speedup.
- TEST MODE: Set TEST_MODE = True to process only 1000 polygons for testing.
- Estimated runtime: ~600k polygons with 6 workers ≈ 3-7 minutes (vs 10-20 minutes sequential).

Run this script once, then re-run feature_engineering.py.
"""

import os
import time
import requests
import pandas as pd
import pyproj
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from config import CSB_FEATHER_PATHS, OUTPUT_DIR, SOIL_CHECKPOINT, SOIL_MAP

OUTPUT_CSV = SOIL_MAP
CHECKPOINT_CSV = SOIL_CHECKPOINT

API_URL        = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"
BATCH_SIZE     = 350    # rows per API request; reduced for concurrent processing
SLEEP_SECONDS  = 0.1    # reduced delay since we're using threading
TIMEOUT        = 60     # seconds per request
MAX_WORKERS    = 6      # number of concurrent API requests
CHECKPOINT_FREQUENCY = 50  # save checkpoint every N completed batches

# Test mode configuration - set TEST_MODE = True to process only a small subset
TEST_MODE      = False   # set to True for testing with limited data
TEST_LIMIT     = 1000    # number of polygons to process in test mode

# Three CSB feather files that together cover 2008-2024
CSB_SOURCES = CSB_FEATHER_PATHS

BASE_COLS = ["CSBID", "INSIDE_X", "INSIDE_Y"]


# ---------------------------------------------------------------------------
# Step 1 — Load unique polygon centroids
# ---------------------------------------------------------------------------

def load_unique_centroids():
    """
    Read CSBID, INSIDE_X (longitude), INSIDE_Y (latitude) from every CSB
    feather file and return a deduplicated DataFrame with one row per polygon.

    Because the same CSBID can appear in multiple feather windows (e.g. the
    2009-2016 bridge file overlaps with 2008-2015), we deduplicate on CSBID
    and keep the first occurrence — centroid coordinates are stable across
    releases for the same polygon.
    """
    print("Loading CSB centroids from feather files...")
    frames = []
    for path in CSB_SOURCES:
        if not os.path.exists(path):
            print(f"  Warning: {path} not found — skipping.")
            continue
        df = pd.read_feather(path, columns=BASE_COLS)
        frames.append(df)
        print(f"  {os.path.basename(path)}: {len(df):,} polygons")

    if not frames:
        raise FileNotFoundError("No CSB feather files found. Check DATA_DIR.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["CSBID"])
    combined["CSBID"] = combined["CSBID"].astype(str)

    # Drop rows with missing coordinates (cannot query API for them)
    before = len(combined)
    combined = combined.dropna(subset=["INSIDE_X", "INSIDE_Y"])
    after = len(combined)
    if before != after:
        print(f"  Dropped {before - after:,} polygons with missing coordinates.")

    # Reproject from EPSG:5070 (CONUS Albers) to EPSG:4326 (WGS84 lat/lon)
    # The API requires decimal degrees: POINT(longitude latitude)
    print("  Reprojecting centroids from EPSG:5070 to WGS84...")
    transformer = pyproj.Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(
        combined["INSIDE_X"].values,
        combined["INSIDE_Y"].values,
    )
    combined = combined.copy()
    combined["LON_WGS84"] = lons
    combined["LAT_WGS84"] = lats

    print(f"  Total unique polygons to query: {len(combined):,}")
    return combined


# ---------------------------------------------------------------------------
# Step 2 — Build batch SQL query
# ---------------------------------------------------------------------------

def build_batch_query(batch_df):
    """
    Construct a UNION ALL SQL query that resolves each polygon centroid to
    its MUKEY using the SDA spatial function.

    Correct SDA function:
        SDA_Get_Mukey_from_intersection_with_WktWgs84('POINT(lon lat)')
    Note: WKT POINT uses (longitude latitude) order.
          Uses LON_WGS84 / LAT_WGS84 columns (already reprojected to EPSG:4326).
    """
    lines = []
    for _, row in batch_df.iterrows():
        csbid = str(row["CSBID"]).replace("'", "''")  # escape any apostrophes
        lat   = float(row["LAT_WGS84"])
        lon   = float(row["LON_WGS84"])
        lines.append(
            f"SELECT '{csbid}' AS csbid, mukey FROM mapunit\n"
            f"WHERE mukey IN (SELECT * FROM "
            f"SDA_Get_Mukey_from_intersection_with_WktWgs84('POINT({lon:.6f} {lat:.6f})'))"
        )
    return "\nUNION ALL\n".join(lines)


# ---------------------------------------------------------------------------
# Step 3 — Query the SDA REST API
# ---------------------------------------------------------------------------

def query_api(sql, attempt=1, max_attempts=3):
    """
    POST a SQL query to the SDA REST endpoint and return a DataFrame.
    Enhanced with better error handling and thread safety.

    Response format JSON+COLUMNNAME:
        {"Table": [["col1","col2"], [val1,val2], ...]}
    First row = column names; subsequent rows = data.

    Returns an empty DataFrame on repeated failure (non-fatal).
    """
    thread_id = threading.current_thread().ident
    payload = {"query": sql, "format": "JSON+COLUMNNAME"}
    
    try:
        resp = requests.post(API_URL, data=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        
        # Check for valid JSON response
        try:
            json_data = resp.json()
        except ValueError as e:
            raise ValueError(f"Invalid JSON response: {e}")
        
        table = json_data.get("Table", [])
        if len(table) < 2:
            # API returned headers only — no soil data for these coordinates
            return pd.DataFrame(columns=["csbid", "mukey"])
        
        cols = table[0]
        data_rows = table[1:]
        
        # Validate data structure
        if not cols or not data_rows:
            return pd.DataFrame(columns=["csbid", "mukey"])
            
        return pd.DataFrame(data_rows, columns=cols)

    except requests.exceptions.Timeout:
        print(f"    [Thread {thread_id}] Timeout on attempt {attempt}/{max_attempts}")
    except requests.exceptions.ConnectionError as exc:
        print(f"    [Thread {thread_id}] Connection error on attempt {attempt}/{max_attempts}: {exc}")
    except requests.exceptions.HTTPError as exc:
        print(f"    [Thread {thread_id}] HTTP error on attempt {attempt}/{max_attempts}: {exc}")
    except requests.exceptions.RequestException as exc:
        print(f"    [Thread {thread_id}] Request error on attempt {attempt}/{max_attempts}: {exc}")
    except (KeyError, ValueError, TypeError) as exc:
        print(f"    [Thread {thread_id}] Parse error on attempt {attempt}/{max_attempts}: {exc}")
    except Exception as exc:
        print(f"    [Thread {thread_id}] Unexpected error on attempt {attempt}/{max_attempts}: {exc}")

    if attempt < max_attempts:
        wait = min(2 ** attempt, 10)  # exponential back-off: 2s, 4s, 8s (max 10s)
        print(f"    [Thread {thread_id}] Retrying in {wait}s...")
        time.sleep(wait)
        return query_api(sql, attempt + 1, max_attempts)

    print(f"    [Thread {thread_id}] All attempts failed — recording empty result for this batch")
    return pd.DataFrame(columns=["csbid", "mukey"])


# ---------------------------------------------------------------------------
# Step 4 — Concurrent batch processing
# ---------------------------------------------------------------------------

def process_batch_concurrent(batch_data):
    """
    Process a single batch of polygons concurrently.
    
    Args:
        batch_data: tuple of (batch_num, batch_df, total_batches)
    
    Returns:
        tuple of (batch_num, result_df) for thread-safe result collection
    """
    batch_num, batch_df, total_batches = batch_data
    
    try:
        sql = build_batch_query(batch_df)
        result = query_api(sql)
        
        # Normalize column names to uppercase
        if not result.empty:
            result.columns = [c.upper() for c in result.columns]
        
        # Thread-safe progress tracking
        with progress_lock:
            global progress_counter
            progress_counter += 1
            if progress_counter % 50 == 0 or progress_counter == 1:
                pct = (progress_counter / total_batches) * 100
                print(f"  Progress: {progress_counter:,}/{total_batches:,} batches completed ({pct:.1f}%)")
        
        time.sleep(SLEEP_SECONDS)  # Polite delay
        return (batch_num, result)
        
    except Exception as e:
        print(f"  Error processing batch {batch_num + 1}: {e}")
        return (batch_num, pd.DataFrame(columns=["CSBID", "MUKEY"]))


# ---------------------------------------------------------------------------
# Step 5 — Thread-safe checkpoint management
# ---------------------------------------------------------------------------

# Global locks and progress tracking
checkpoint_lock = threading.Lock()
progress_lock = threading.Lock()
completed_batches = []
progress_counter = 0

def collect_results_with_checkpoint(futures, total_batches):
    """
    Collect results from concurrent futures and manage checkpointing.
    
    Args:
        futures: list of Future objects from ThreadPoolExecutor
        total_batches: total number of batches for progress tracking
    
    Returns:
        list of all result DataFrames
    """
    global completed_batches
    all_results = []
    
    for future in as_completed(futures):
        try:
            batch_num, result_df = future.result()
            
            if not result_df.empty:
                all_results.append(result_df)
            
            completed_batches.append(batch_num)
            
            # Save checkpoint every CHECKPOINT_FREQUENCY batches
            if len(completed_batches) % CHECKPOINT_FREQUENCY == 0:
                with checkpoint_lock:
                    save_checkpoint(all_results)
                    print(f"  Checkpoint saved: {len(completed_batches)}/{total_batches} batches completed")
                    
        except Exception as e:
            print(f"  Error collecting result from future: {e}")
    
    # Final checkpoint save
    with checkpoint_lock:
        save_checkpoint(all_results)
    
    return all_results

def save_checkpoint(results):
    """Thread-safe checkpoint saving."""
    if not results:
        return
        
    try:
        batch_df = pd.concat(results, ignore_index=True)
        
        # Ensure CSBID remains as string to prevent auto-conversion to int
        if "CSBID" in batch_df.columns:
            batch_df["CSBID"] = batch_df["CSBID"].astype(str)
        
        # Append to checkpoint file
        mode = "a" if os.path.exists(CHECKPOINT_CSV) else "w"
        header = not os.path.exists(CHECKPOINT_CSV)
        batch_df.to_csv(CHECKPOINT_CSV, mode=mode, header=header, index=False)
        
    except Exception as e:
        print(f"  Error saving checkpoint: {e}")


# ---------------------------------------------------------------------------
# Step 6 — Helper: write final output from checkpoint
# ---------------------------------------------------------------------------

def _write_final(checkpoint_df):
    """Deduplicate the checkpoint and write the final output CSV."""
    final = checkpoint_df.drop_duplicates(subset=["CSBID"])
    final["MUKEY"] = final["MUKEY"].replace("None", pd.NA)
    
    # Ensure CSBID remains as string to match feather file format
    final["CSBID"] = final["CSBID"].astype(str)
    
    final.to_csv(OUTPUT_CSV, index=False)
    matched = final["MUKEY"].notna().sum()
    print(f"\n=== Done ===")
    print(f"  Output saved to: {OUTPUT_CSV}")
    print(f"  Total polygons mapped: {len(final):,}")
    print(f"  Polygons with a valid MUKEY: {matched:,} ({matched / max(len(final), 1) * 100:.1f}%)")
    print(f"\nNext step: re-run feature_engineering.py to incorporate soil features.")


# ---------------------------------------------------------------------------
# Step 5 — Main pipeline
# ---------------------------------------------------------------------------

def main():
    centroids = load_unique_centroids()

    # ------------------------------------------------------------------
    # Resume from checkpoint if it exists
    # ------------------------------------------------------------------
    if os.path.exists(CHECKPOINT_CSV):
        done     = pd.read_csv(CHECKPOINT_CSV, dtype=str)
        done_ids = set(done["CSBID"].astype(str))
        remaining = centroids[~centroids["CSBID"].isin(done_ids)].reset_index(drop=True)
        print(f"\nResuming from checkpoint: {len(done_ids):,} already done, "
              f"{len(remaining):,} remaining.")
        if remaining.empty:
            print("All polygons already processed. Writing final output.")
            _write_final(done)
            return
    else:
        remaining = centroids.copy()
        print(f"\nNo checkpoint found. Starting fresh ({len(remaining):,} polygons.")

    # Apply test mode limit if enabled
    if TEST_MODE and len(remaining) > TEST_LIMIT:
        print(f"\n*** TEST MODE ENABLED ***")
        print(f"Processing only {TEST_LIMIT:,} polygons instead of {len(remaining):,}")
        remaining = remaining.head(TEST_LIMIT).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Process batches concurrently using ThreadPoolExecutor
    # ------------------------------------------------------------------
    n_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {n_batches:,} batches of up to {BATCH_SIZE} polygons each using {MAX_WORKERS} concurrent workers...\n")

    # Reset global variables for this run
    global completed_batches, progress_counter
    completed_batches = []
    progress_counter = 0

    # Prepare batch data for concurrent processing
    batch_tasks = []
    for batch_num, start in enumerate(range(0, len(remaining), BATCH_SIZE)):
        batch = remaining.iloc[start : start + BATCH_SIZE]
        batch_tasks.append((batch_num, batch, n_batches))

    # Process batches concurrently
    print(f"Starting concurrent processing with {MAX_WORKERS} workers...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all batch processing tasks
        futures = [executor.submit(process_batch_concurrent, batch_data) 
                  for batch_data in batch_tasks]
        
        # Collect results with checkpointing
        all_results = collect_results_with_checkpoint(futures, n_batches)
    
    elapsed_time = time.time() - start_time
    print(f"\nConcurrent processing completed in {elapsed_time:.1f} seconds")
    print(f"Average time per batch: {elapsed_time/n_batches:.2f} seconds")

    # ------------------------------------------------------------------
    # Read complete checkpoint and write final deduplicated output
    # ------------------------------------------------------------------
    if os.path.exists(CHECKPOINT_CSV):
        final = pd.read_csv(CHECKPOINT_CSV, dtype=str)
    else:
        final = pd.DataFrame(columns=["CSBID", "MUKEY"])

    _write_final(final)


if __name__ == "__main__":
    main()
