#!/usr/bin/env python3

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import yirgacheffe as yg
from alive_progress import alive_bar

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')

def load_crosswalk_table(table_file_name: Path) -> dict[str, list[int]]:
    rawdata = pd.read_csv(table_file_name)
    result: dict[str, list[int]] = {}
    for _, row in rawdata.iterrows():
        code = str(row.code)
        try:
            result[code].append(int(row.value))
        except KeyError:
            result[code] = [int(row.value)]
    return result

def crosswalk_habitats(crosswalk_table: dict[str, list[int]], raw_habitats: set[str]) -> set[int]:
    result = set()
    for habitat in raw_habitats:
        try:
            crosswalked_habatit = crosswalk_table[habitat]
        except KeyError:
            continue
        result |= set(crosswalked_habatit)
    return result

def calculate_aoh(
    species_info_path: Path,
    habitat_path: Path,
    elevation_path: Path,
    area_path: Path,
    crosswalk_path: Path,
    output_path: Path,
    edge_proportion: float,
) -> None:

    os.makedirs(output_path.parent, exist_ok=True)

    crosswalk_table = load_crosswalk_table(crosswalk_path)

    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"
    try:
        filtered_species_info = gpd.read_file(species_info_path)
    except: # pylint:disable=W0702
        logger.error("Failed to read %s", species_info_path)
        sys.exit(1)
    assert filtered_species_info.shape[0] == 1

    try:
        elevation_lower = math.floor(float(filtered_species_info.elevation_lower.values[0]))
        elevation_upper = math.ceil(float(filtered_species_info.elevation_upper.values[0]))
        raw_habitats = set(filtered_species_info.full_habitat_code.values[0].split('|'))
    except (AttributeError, TypeError):
        logger.error("Species data missing one or more needed attributes: %s", filtered_species_info)
        sys.exit()

    habitat_list = list(crosswalk_habitats(crosswalk_table, raw_habitats))
    if len(habitat_list) == 0:
        logger.error("No habitats found in crosswalk! %s", raw_habitats)
        sys.exit()

    # Convolution matrices to detect each neighbor (4 cardinals + 4 diagonals)
    # 1 where the neighbor is, 0 elsewhere
    north_matrix = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    south_matrix = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    east_matrix = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    west_matrix = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    northeast_matrix = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    northwest_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    southeast_matrix = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    southwest_matrix = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    with (
        yg.read_raster(elevation_path) as elevation,
        yg.read_raster(habitat_path) as habitat,
        yg.read_narrow_raster(area_path) as area,
        yg.read_shape_like(species_info_path, elevation) as species_range,
    ):
        filtered_habitats = habitat.isin(habitat_list)
        habitat_float = filtered_habitats.astype(yg.DataType.Float32)

        # Detect which neighbors are present (1) or missing (0)
        has_north = habitat_float.conv2d(north_matrix)
        has_south = habitat_float.conv2d(south_matrix)
        has_east = habitat_float.conv2d(east_matrix)
        has_west = habitat_float.conv2d(west_matrix)
        has_northeast = habitat_float.conv2d(northeast_matrix)
        has_northwest = habitat_float.conv2d(northwest_matrix)
        has_southeast = habitat_float.conv2d(southeast_matrix)
        has_southwest = habitat_float.conv2d(southwest_matrix)

        # Count missing cardinals (0 to 4)
        missing_cardinals = (yg.constant(1.0) - has_north) + (yg.constant(1.0) - has_south) + \
                           (yg.constant(1.0) - has_east) + (yg.constant(1.0) - has_west)

        # Detect corners where BOTH adjacent cardinals are missing AND the diagonal is missing
        # In these cases, we need to add back the p² corner overlap
        # If the diagonal neighbor is present, both pixels lose the corner so no correction needed
        # NE corner: north AND east AND northeast all missing
        # NW corner: north AND west AND northwest all missing
        # SE corner: south AND east AND southeast all missing
        # SW corner: south AND west AND southwest all missing
        missing_ne_corner = (yg.constant(1.0) - has_north) * (yg.constant(1.0) - has_east) * \
                           (yg.constant(1.0) - has_northeast)
        missing_nw_corner = (yg.constant(1.0) - has_north) * (yg.constant(1.0) - has_west) * \
                           (yg.constant(1.0) - has_northwest)
        missing_se_corner = (yg.constant(1.0) - has_south) * (yg.constant(1.0) - has_east) * \
                           (yg.constant(1.0) - has_southeast)
        missing_sw_corner = (yg.constant(1.0) - has_south) * (yg.constant(1.0) - has_west) * \
                           (yg.constant(1.0) - has_southwest)
        corner_overlaps = missing_ne_corner + missing_nw_corner + missing_se_corner + missing_sw_corner

        # Calculate remaining fraction:
        # Start at 1.0, subtract edge_proportion for each missing cardinal,
        # add back edge_proportion² for each corner overlap (only when diagonal is also missing)
        edged_habitats = yg.constant(1.0) - (missing_cardinals * edge_proportion) + \
                        (corner_overlaps * (edge_proportion ** 2))

        # CRITICAL: Multiply by filtered_habitats to ensure non-habitat pixels (center=0) stay 0
        edged_habitats = edged_habitats * habitat_float

        # Clip to ensure values are between 0 and 1
        edged_habitats = yg.where(edged_habitats < 0.0, 0.0, edged_habitats)
        edged_habitats = yg.where(edged_habitats > 1.0, 1.0, edged_habitats)

        aoh = species_range * \
            ((elevation > elevation_lower) & (elevation < elevation_upper)) * \
            edged_habitats * area
        with alive_bar(manual=True) as bar:
            aoh.to_geotiff(output_path, callback=bar)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fractional edge AoH generator")
    parser.add_argument(
        '--species',
        type=Path,
        help='Path of map of species info geojson',
        required=True,
        dest='species_info_path',
    )
    parser.add_argument(
        '--habitat',
        type=Path,
        help='Path of map of habitat raster',
        required=True,
        dest='habitat_path',
    )
    parser.add_argument(
        '--elevation',
        type=Path,
        help='Path of elevation raster',
        required=True,
        dest='elevation_path',
    )
    parser.add_argument(
        '--area',
        type=Path,
        help='Path of area per pixel raster',
        required=True,
        dest='area_path',
    )
    parser.add_argument(
        '--crosswalk',
        type=Path,
        help="Path of habitat crosswalk table.",
        required=True,
        dest="crosswalk_path",
    )
    parser.add_argument(
        '--edge',
        type=float,
        help="Proportion of pixel edge to remove for each missing neighbor (e.g., 0.3 for 30m on 100m pixels)",
        required=True,
        dest="edge_proportion",
    )
    parser.add_argument(
        '--output',
        type=Path,
        help="Path of result raster",
        required=True,
        dest="output_path",
    )
    args = parser.parse_args()

    calculate_aoh(
        args.species_info_path,
        args.habitat_path,
        args.elevation_path,
        args.area_path,
        args.crosswalk_path,
        args.output_path,
        args.edge_proportion,
    )

if __name__ == "__main__":
    main()
