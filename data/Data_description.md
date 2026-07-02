# Crop Sequence Boundaries (CSB) Data

## Overview
The Crop Sequence Boundaries (CSB) dataset is a vector-based geospatial product developed by the USDA National Agricultural Statistics Service (NASS) in cooperation with the Economic Research Service (ERS). It provides fully synthetic representations of agricultural fields, including their acreage and cropping rotation history.

## Data Type
*   **Format:** Vector polygons (Geodatabase/GDB source, converted here to Parquet/Feather/CSV).
*   **Source:** Primarily derived from the NASS Cropland Data Layer (CDL), combined with road and rail networks to define field boundaries.
*   **Coverage:** Contiguous United States (CONUS).

## Temporal Coverage & Structure
The dataset is organized into **8-year time frames** to capture crop rotation patterns over time.
*   **Total Range:** 2008–2024
*   **Windows:** The data is typically released in 8-year blocks (e.g., 2009–2016, 2017–2024) to provide a consistent historical view of land use.

## Usage
CSBs are designed to provide stable analysis units for agricultural research. Key applications include:
*   **Crop Rotation Analysis:** Tracking the sequence of crops planted on specific fields over multiple years.
*   **Acreage Estimation:** Aggregating pixel-based CDL data into field-level attribute tables.
*   **Land Use Modeling:** Studying relationships between crop choices, soil types, and environmental factors.

## Disclaimer
These boundaries are synthetic and algorithmically derived; they do not represent legal land ownership or administrative boundaries.

# Soil Data (gSSURGO)

## Overview

The soil dataset is derived from the USDA Natural Resources Conservation Service (NRCS) gSSURGO (Gridded Soil Survey Geographic) database for New York State. gSSURGO provides high-resolution, map unit–based soil information, including spatial boundaries and aggregated soil physical and hydrologic properties. The data are designed to support agricultural, environmental, and land-use analyses.

## Data Type

**Format:** Vector polygons (ESRI File Geodatabase / GDB source, processed and exported to CSV).

**Spatial Component:** Soil map unit polygons (MUPOLYGON), representing distinct soil map units.

**Attribute Component:** Aggregated soil attributes stored in the muaggatt table.

**Coverage:** New York State.

## Data Structure & Processing

The original gSSURGO data are distributed as a geodatabase in which spatial geometries and soil attributes are stored in separate tables. Using QGIS, the soil map unit polygon layer (MUPOLYGON) was joined with the aggregated soil attribute table (muaggatt) via the shared Map Unit Key (MUKEY). This join produced a polygon-level soil dataset where each soil polygon is associated with its corresponding aggregated soil properties.

After the join, soil variables were filtered to retain only attributes directly related to crop growth conditions and planting feasibility. Engineering-related and urban suitability variables were excluded. The resulting cleaned dataset was exported as a CSV file for downstream analysis and modeling.

## Selected Soil Attributes

The final dataset includes soil attributes that capture key physical, hydrologic, and terrain constraints on agricultural production:

**Drainage class:**
Indicates how well water moves through the soil, directly affecting root aeration and crop suitability.

**Water table depth (annual minimum / spring minimum):**
Captures shallow groundwater constraints that influence planting timing and crop stress.

**Flooding frequency:**
Represents the likelihood of periodic flooding, which affects crop risk and land usability.

**Ponding presence and frequency:**
Identifies areas where surface water accumulation may restrict planting or machinery access.

**Bedrock depth (minimum):**
Serves as a proxy for effective rooting depth and limitations on root development.

**Slope (dominant or weighted average):**
Reflects terrain constraints related to erosion risk, mechanization, and land use decisions.

**Hydrologic soil group:**
Describes runoff potential and infiltration capacity, interacting with precipitation and water availability.

**Available Water Capacity (selected depth intervals):**
Measures the soil’s ability to retain plant-available water, a key factor in drought resilience and crop choice.

## Usage

This soil dataset is intended to be used as a spatially explicit environmental covariate dataset in agricultural and land-use analyses. Typical applications include:

* Crop suitability and planting feasibility analysis

* Integration with crop sequence or field boundary datasets

* Environmental constraint modeling

* Regression or machine learning models incorporating soil controls
