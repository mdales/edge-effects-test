import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yirgacheffe as yg
from alive_progress import alive_bar
# from yirgacheffe.window import Area

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

def findedgepixels(
    input_path: Path,
    output_path: Path,
) -> None:

    os.makedirs(output_path.parent, exist_ok=True)

    matrix = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])

    with yg.read_raster(input_path) as input_raster:
        # input_raster.set_window_for_intersection(Area(-73.9872354804, 5.24448639569, -34.7299934555, -33.7683777809))
        edge_pixels = (input_raster.astype(yg.DataType.Float32).conv2d(matrix) / 9.0) == input_raster
        with alive_bar(manual=True) as bar:
            edge_pixels.to_geotiff(output_path, callback=bar)


def main() -> None:
    parser = argparse.ArgumentParser(description="Binary AoH generator")
    parser.add_argument(
        '--input',
        type=Path,
        help='Path of map of input rastern',
        required=True,
        dest='input_path',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help="Path of result raster",
        required=True,
        dest="output_path",
    )
    args = parser.parse_args()

    findedgepixels(
        args.input_path,
        args.output_path,
    )

if __name__ == "__main__":
    main()
