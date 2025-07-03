import argparse
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import geopandas as gpd
from alive_progress import alive_bar
from osgeo import gdal
from yirgacheffe.layers import RasterLayer, VectorLayer, UniformAreaLayer
from yirgacheffe.operators import DataType
from yirgacheffe.window import Area

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')

def load_crosswalk_table(table_file_name: Path) -> Dict[str,List[int]]:
	rawdata = pd.read_csv(table_file_name)
	result = {}
	for _, row in rawdata.iterrows():
		code = str(row.code)
		try:
			result[code].append(int(row.value))
		except KeyError:
			result[code] = [int(row.value)]
	return result

def crosswalk_habitats(crosswalk_table: Dict[str, List[int]], raw_habitats: Set[str]) -> Set[int]:
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

	with RasterLayer.layer_from_file(input_path) as input:

		# input.set_window_for_intersection(Area(-73.9872354804, 5.24448639569, -34.7299934555, -33.7683777809))

		edge_pixels = (input.astype(DataType.Float32).conv2d(matrix) / 9.0) == input

		with RasterLayer.empty_raster_layer_like(edge_pixels, filename=output_path) as result:
			with alive_bar(manual=True) as bar:
				edge_pixels.save(result, callback=bar)


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