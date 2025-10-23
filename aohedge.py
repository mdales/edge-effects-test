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

    matrix = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    with (
        yg.read_raster(elevation_path) as elevation,
        yg.read_raster(habitat_path) as habitat,
        yg.read_narrow_raster(area_path) as area,
        yg.read_shape_like(species_info_path, elevation) as species_range,
    ):
        filtered_habitats = habitat.isin(habitat_list)
        edged_habitats = filtered_habitats.astype(yg.DataType.Float32).conv2d(matrix) == 9.0

        aoh = species_range * \
            ((elevation > elevation_lower) & (elevation < elevation_upper)) * \
            edged_habitats * area
        with alive_bar(manual=True) as bar:
            aoh.to_geotiff(output_path, callback=bar)


def main() -> None:
    parser = argparse.ArgumentParser(description="Binary AoH generator")
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
    )

if __name__ == "__main__":
    main()
