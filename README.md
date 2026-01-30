# National CSB Data Processing Pipeline

## Overview
This project processes the National CSB (2017-2024) geospatial dataset. It provides a reproducible pipeline to ingest raw Geodatabase (GDB) files, perform exploratory data analysis, and export region-specific subsets (e.g., New York State) into optimized formats for downstream research and analysis.

## Key Features
- **Data Ingestion**: Reads complex GDB layers using Geopandas.
- **EDA**: Includes steps for inspecting data structure, summary statistics, and geographic coverage.
- **Data Optimization**: Converts heavy spatial data into efficient formats:
  - **Parquet**: For efficient storage and big data analytics.
  - **Feather**: For high-speed I/O and interoperability between Python/R.
  - **CSV**: For easy access by non-technical stakeholders.

## Prerequisites

Ensure you have Python 3.8+ installed. The following libraries are required:

pip install geopandas pandas matplotlib fiona shapely pyproj pyarrow fastparquet*Note: On macOS with Apple Silicon, it is recommended to use `conda` or `mamba` to install `geopandas` and `pyarrow` to avoid binary compatibility issues.*

## Usage

1.  **Configure Paths**: Update the `gdb_path` variable in the script to point to your local `.gdb` directory.
2.  **Run the Notebook**: Execute the Jupyter notebook `script.ipynb`.
3.  **Outputs**: The script will generate the following files in the `data/` directory:
    - `ny_csb.parquet_20172024`
    - `ny_csb.feather_20172024`
    - `ny_csb.csv_20172024`

## Project Structure
.
├── NationalCSB_2017-2024_rev23/  # Raw Data (Not included in repo, availible on https://www.nass.usda.gov/Research_and_Science/Crop-Sequence-Boundaries/index.php)
├── data/                         # Processed Output
├── script.ipynb                  # Main processing pipeline
└── README
