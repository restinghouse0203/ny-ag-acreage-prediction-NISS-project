"""Plot NY CSB polygon centroids by grouped crop type and rotation strategy."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkb


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "merged_data"
    / "merged_CSB_polygon.parquet"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DEFAULT_CSB_2008_2015_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "csb_20082015" / "ny_csb.feather_20082015"
)
DEFAULT_CSB_2017_2024_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "csb_20172024" / "ny_csb.feather_20172024"
)

CROP_GROUP_ORDER = ["Corn", "Soybeans", "Alfalfa", "Hay/Grass", "Others"]
STRATEGY_ORDER = ["Rotation", "Mixed", "Continuous"]
WINDOW_TO_YEARS = {
    "2008-2015": [f"CDL{y}" for y in range(2008, 2016)],
    "2017-2024": [f"CDL{y}" for y in range(2017, 2025)],
}


def map_crop_code_to_group(code: float | int | str) -> str:
    if pd.isna(code):
        return "Others"
    c = int(code)
    if c == 1:
        return "Corn"
    if c == 5:
        return "Soybeans"
    if c == 36:
        return "Alfalfa"
    if c in (37, 176, 59):
        return "Hay/Grass"
    return "Others"


def classify_strategy(crop_groups: pd.Series) -> str:
    unique = {v for v in crop_groups.dropna().tolist()}
    n_unique = len(unique)
    if n_unique <= 1:
        return "Continuous"
    if n_unique == 2:
        return "Rotation"
    return "Mixed"


def classify_primary_crop(crop_groups: pd.Series) -> str:
    values = crop_groups.dropna()
    if values.empty:
        return "Others"
    counts = values.value_counts()
    top_count = counts.max()
    tied = counts[counts == top_count].index.tolist()
    if len(tied) == 1:
        return tied[0]
    # Tie-break with most recent year's grouped crop in this polygon history.
    for value in reversed(values.tolist()):
        if value in tied:
            return value
    return tied[0]


def build_window_labels(csb_path: Path, year_cols: list[str], window_label: str) -> pd.DataFrame:
    csb_df = pd.read_feather(csb_path)
    cols_needed = ["CSBID"] + [c for c in year_cols if c in csb_df.columns]
    if len(cols_needed) <= 1:
        raise ValueError(f"No expected year columns found in {csb_path}.")

    long_df = csb_df[cols_needed].melt(
        id_vars="CSBID", var_name="Year_Col", value_name="Crop_Code"
    )
    long_df["Crop_Group"] = long_df["Crop_Code"].apply(map_crop_code_to_group)
    grouped = (
        long_df.groupby("CSBID", as_index=False)
        .agg(
            crop_type=("Crop_Group", classify_primary_crop),
            rotation_strategy=("Crop_Group", classify_strategy),
        )
        .copy()
    )
    grouped["Source_Window"] = window_label
    return grouped


def load_label_table(csb_2008_2015: Path, csb_2017_2024: Path) -> pd.DataFrame:
    first = build_window_labels(csb_2008_2015, WINDOW_TO_YEARS["2008-2015"], "2008-2015")
    second = build_window_labels(csb_2017_2024, WINDOW_TO_YEARS["2017-2024"], "2017-2024")
    return pd.concat([first, second], ignore_index=True)


def load_geodata(
    data_path: Path,
    csb_2008_2015: Path,
    csb_2017_2024: Path,
    source_crs: str = "EPSG:5070",
    max_points: int | None = None,
) -> gpd.GeoDataFrame:
    df = pd.read_parquet(data_path)
    if "geometry" not in df.columns:
        raise ValueError("Input parquet must contain a 'geometry' column with WKB polygons.")
    if "CSBID" not in df.columns or "Source_Window" not in df.columns:
        raise ValueError("Input parquet must contain 'CSBID' and 'Source_Window' columns.")

    labels = load_label_table(csb_2008_2015, csb_2017_2024)
    df = df.merge(labels, on=["CSBID", "Source_Window"], how="left")
    df["crop_type"] = df["crop_type"].fillna("Others")
    df["rotation_strategy"] = df["rotation_strategy"].fillna("Mixed")

    if max_points and len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    df["geometry"] = df["geometry"].apply(lambda value: wkb.loads(value) if value is not None else None)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=source_crs)
    gdf = gdf[gdf.geometry.notnull()].copy()
    centroids = gpd.GeoSeries(gdf.geometry.centroid, crs=gdf.crs).to_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["point_geometry"] = centroids.values
    return gdf


def plot_map(
    gdf: gpd.GeoDataFrame,
    category_col: str,
    category_order: list[str],
    title: str,
    legend_title: str,
    output_path: Path,
    show_plot: bool = False,
) -> None:
    boundary = gdf.dissolve().boundary
    points = gpd.GeoDataFrame(
        gdf[[category_col]].copy(), geometry=gdf["point_geometry"], crs=gdf.crs
    )
    points[category_col] = pd.Categorical(points[category_col], categories=category_order, ordered=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=2)
    points.plot(
        ax=ax,
        column=category_col,
        cmap="tab10",
        markersize=6,
        alpha=0.75,
        legend=True,
        legend_kwds={"loc": "upper left", "title": legend_title},
        zorder=1,
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot NY polygon centroids with grouped crop type and rotational strategy labels."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Input parquet path.")
    parser.add_argument(
        "--csb-2008-2015",
        type=Path,
        default=DEFAULT_CSB_2008_2015_PATH,
        help="Path to 2008-2015 CSB feather data.",
    )
    parser.add_argument(
        "--csb-2017-2024",
        type=Path,
        default=DEFAULT_CSB_2017_2024_PATH,
        help="Path to 2017-2024 CSB feather data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for generated map images.",
    )
    parser.add_argument(
        "--source-crs",
        type=str,
        default="EPSG:5070",
        help="CRS used by the input WKB geometry.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120000,
        help="Optional cap for plotted points to improve rendering speed.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive plot window after saving.",
    )
    args = parser.parse_args()

    geodf = load_geodata(
        args.data,
        csb_2008_2015=args.csb_2008_2015,
        csb_2017_2024=args.csb_2017_2024,
        source_crs=args.source_crs,
        max_points=args.max_points,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_map(
        geodf,
        category_col="crop_type",
        category_order=CROP_GROUP_ORDER,
        title="New York State CSB Centroids by Crop Type",
        legend_title="Crop Type",
        output_path=args.output_dir / "ny_crop_type_map.png",
        show_plot=args.show,
    )
    plot_map(
        geodf,
        category_col="rotation_strategy",
        category_order=STRATEGY_ORDER,
        title="New York State CSB Centroids by Rotation Strategy",
        legend_title="Rotation Strategy",
        output_path=args.output_dir / "ny_rotation_strategy_map.png",
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()