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

def aoh(
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
		[1.0, 1.0, 1.0]
	])


	with RasterLayer.layer_from_file(elevation_path) as elevation:
		with RasterLayer.layer_from_file(habitat_path) as habitat:
			with UniformAreaLayer.layer_from_file(area_path) as area:
				with VectorLayer.layer_from_file_like(species_info_path, elevation) as range:

					# quick work around for type coercion
					filtered_habitats = habitat.isin(habitat_list)

					edged_habitats = filtered_habitats.astype(DataType.Float32).conv2d(matrix) == 9.0

					aoh = range * \
						((elevation > elevation_lower) & (elevation < elevation_upper)) * \
						edged_habitats * area
					with RasterLayer.empty_raster_layer_like(aoh, filename=output_path, datatype=gdal.GDT_Float32) as result:
						with alive_bar(manual=True) as bar:
							aoh.save(result, callback=bar)


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

	aoh(
		args.species_info_path,
		args.habitat_path,
		args.elevation_path,
		args.area_path,
		args.crosswalk_path,
		args.output_path,
	)

if __name__ == "__main__":
	main()