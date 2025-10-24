#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

import pandas as pd
import yirgacheffe as yg


def extract_species_and_edge(filename: str) -> tuple[str, str]:
    """Extract species ID and edge value from filename like '22701083_RESIDENT_e0.0.tif'"""
    match = re.match(r'(.+)_[er]([0-9.]+)\.tif$', filename)
    if match:
        species_id = match.group(1)
        edge_value = match.group(2)
        return species_id, edge_value
    raise ValueError(f"Could not parse filename: {filename}")


def sum_raster(filepath: Path) -> float:
    """Sum all values in a raster file"""
    try:
        with yg.read_raster(filepath) as aoh:
            total_area = aoh.sum()
        return total_area
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: Failed to read {filepath}: {e}")
        return float('nan')


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AoH results into a CSV table")
    parser.add_argument(
        '--input',
        type=Path,
        help='Path to batch CSV file',
        required=True,
        dest='input_csv',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Path to output summary CSV',
        required=True,
        dest='output_csv',
    )
    args = parser.parse_args()

    # Read the batch CSV to get all output files
    print(f"Reading {args.input_csv}...")
    batch_data = pd.read_csv(args.input_csv)

    # Dictionary to store results: {species_id: {edge_value: total_area}}
    results: dict[str, dict[str, float]] = {}

    # Process each output file
    print(f"Processing {len(batch_data)} raster files...")
    for idx in range(len(batch_data)):
        output_file = Path(str(batch_data.iloc[idx]['--output']))

        # Extract species and edge from filename
        try:
            species_id, edge_value = extract_species_and_edge(output_file.name)
        except ValueError as e:
            print(f"Warning: {e}")
            continue

        # Sum the raster
        print(f"  [{idx+1}/{len(batch_data)}] Processing {output_file.name}...")
        total_area = sum_raster(output_file)

        # Store result
        if species_id not in results:
            results[species_id] = {}
        results[species_id][edge_value] = total_area

    # Get sorted list of edge values
    all_edge_values: set[str] = set()
    for species_data in results.values():
        all_edge_values.update(species_data.keys())
    edge_values = sorted(all_edge_values, key=float)

    # Write output CSV
    print(f"\nWriting results to {args.output_csv}...")
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ['species'] + [f'edge_{e}' for e in edge_values]
        writer.writerow(header)

        # Write data rows
        for species_id in sorted(results.keys()):
            data_row: list[str | float] = [species_id]
            for edge_value in edge_values:
                area = results[species_id].get(edge_value, float('nan'))
                data_row.append(area)
            writer.writerow(data_row)

    print(f"Done! Summary written to {args.output_csv}")
    print(f"  {len(results)} species")
    print(f"  {len(edge_values)} edge values: {', '.join(edge_values)}")


if __name__ == "__main__":
    main()
