#!/usr/bin/env python3

"""
Compute spatial correlations and their significance.
Usage:
  ./spatial.py

This file uses 2016.csv and writes its results to [key description].2016.csv.
"""

import csv
import csv_utils
from merge_data import DataKeys, MISMATCH_DELIMETER
from multiprocessing import Pool
import numpy as np
import random
from skbio import DistanceMatrix
from skbio.stats.distance import mantel
from scipy.spatial.distance import pdist, squareform

# The non-merged file already has GPS, rows and columns for all cells, so
# merging the cells does not increase coverage.
INPUT_PATH = '2016.csv'

# Bryan Manly says 5000 is minimum number to estimate a significance level of
# 0.01 in "Randomization, Bootstrap and Monte Carlo Methods in Biology" at the
# top of page 208.
MANTEL_PERMUTATIONS = 10000


def get_output_path(spatial_keys_description):
  return 'spatial.' + spatial_keys_description + '.' + INPUT_PATH


# Averge multiple numeric values, caused by mismatched data merging.
def average_mismatch(value):
  return np.mean([float(x) for x in value.split(MISMATCH_DELIMETER)])


# Computes spatial correlation relative to the two spatial keys provided
# (e.g. GPS_EASTINGS, GPS_NORTHINGS). To use a single spatial key, set spatial
# key 2 to None. This will then use 0 for all the corresponding values in that
# dimension.
# Returns (key name, number of data points, correlation coefficient, p value)
def get_spatial_correlation(arg):
  # Use a single argument since using pool.map().
  samples, spatial_key1, spatial_key2, data_key = arg
  samples = [x for x in samples if x[data_key.value] != '']

  spatial_data1 = [average_mismatch(x[spatial_key1.value]) for x in samples]
  spatial_data2 = [0.0] * len(samples)
  if spatial_key2 is not None:
    spatial_data2 = [average_mismatch(x[spatial_key2.value]) for x in samples]
  spatial_data = list(zip(spatial_data1, spatial_data2))
  # Add required second dimension, but set to 0.0, so no affect on distances.
  data = [(average_mismatch(x[data_key.value]), 0.0) for x in samples]

  # Sanity check: Random data should show no significant correlation.
  # Results: p-value of 0.905, so sanity check passed.
  #random.seed(10611)  # Does not seem to contain all randomness unfortunately.
  #data = [(random.random(), 0.0) for x in samples]

  spatial_distances = DistanceMatrix(squareform(pdist(spatial_data)))
  data_distances = DistanceMatrix(squareform(pdist(data)))
  coeff, p_value, n = mantel(spatial_distances, data_distances,
                             permutations=MANTEL_PERMUTATIONS)

  return (data_key.value, n, coeff, p_value)


def main():
  pool = Pool()
  samples = csv_utils.read_csv_as_dicts(INPUT_PATH)

  # Tuples of spatial_key1, spatial_key2, spatial_keys_description).
  mantel_runs = [
    (DataKeys.GPS_EASTINGS, DataKeys.GPS_NORTHINGS, 'eastings_and_northings'),
  ]

  for spatial_key1, spatial_key2, spatial_keys_description in mantel_runs:
    args = []
    for data_key in DataKeys:
      # This is a bit hacky way to skip values that are all text values.
      # But it works nicely right now.
      if (data_key == DataKeys.ROW or
          data_key == DataKeys.COLUMN or
          data_key == DataKeys.PLANT_ID):
        continue
      if data_key == DataKeys.GPS_EASTINGS:
        break  # Ignore this DataKey and all DataKeys after this one.

      args.append((samples, spatial_key1, spatial_key2, data_key))


    output_path = get_output_path(spatial_keys_description)
    print('Spawning jobs for:', output_path)
    results = pool.map(get_spatial_correlation, args)
    results.sort(key=lambda x: x[-1])  # Sort by p-value.

    with open(output_path, 'w') as f:
      csv_writer = csv.writer(f)
      csv_writer.writerow(['label', 'num_data_points', 'corr_coeff', 'p_value'])
      csv_writer.writerows(results)


if __name__ == '__main__':
    main()
