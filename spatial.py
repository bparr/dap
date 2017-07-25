#!/usr/bin/env python3

"""
Compute spatial correlations and their significance.
Usage:
  ./spatial.py

This file uses 2016.csv and writes its results in the spatial/ directory with
the name[key description].2016.csv.
"""

import csv
from csv_utils import average_mismatch, read_csv_as_dicts
from features import Features
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

# Number of closest plots to count as adjacent (not including the plot itself).
ADJACENT_COUNT = 8


def get_output_path(spatial_keys_description):
  return 'spatial/' + spatial_keys_description + '.' + INPUT_PATH


# Computes spatial correlation relative to the two spatial keys provided
# (e.g. GPS_EASTINGS, GPS_NORTHINGS). To use a single spatial key, set spatial
# key 2 to None. This will then use 0 for all the corresponding values in that
# dimension.
# Returns (key name, number of data points, correlation coefficient, p value)
def get_spatial_correlation(arg):
  # Use a single argument since using pool.map().
  samples, spatial_key1, spatial_key2, feature = arg
  samples = [x for x in samples if x[feature.value] != '']

  spatial_data1 = [average_mismatch(x[spatial_key1.value]) for x in samples]
  spatial_data2 = [0.0] * len(samples)
  if spatial_key2 is not None:
    spatial_data2 = [average_mismatch(x[spatial_key2.value]) for x in samples]
  spatial_data = list(zip(spatial_data1, spatial_data2))
  data = [average_mismatch(x[feature.value]) for x in samples]

  # Sanity check: Random data should show no significant correlation.
  # Results: p-value of 0.905, so sanity check passed.
  #random.seed(10611)  # Does not seem to contain all randomness unfortunately.
  #data = [random.random() for _ in samples]

  spatial_distances = squareform(pdist(spatial_data))
  # Add required second dimension, but set to 0.0, so no affect on distances.
  data_distances = squareform(pdist([(x, 0.0) for x in data]))

  n = len(samples)
  adjacent_data_distances = []
  nonadjacent_data_distances = []
  for i in range(n):
    sorted_row = sorted(zip(spatial_distances[i], range(n)))
    if sorted_row[0][0] != 0.0:
      raise Exception('The plot itself is NOT the nearest plot??')
    for j in range(1, ADJACENT_COUNT + 1):
      adjacent_data_distances.append(data_distances[i][sorted_row[j][1]])
    for j in range(ADJACENT_COUNT + 1, n):
      nonadjacent_data_distances.append(data_distances[i][sorted_row[j][1]])


  coeff, p_value, _ = mantel(DistanceMatrix(spatial_distances),
                             DistanceMatrix(data_distances),
                             permutations=MANTEL_PERMUTATIONS)

  return (feature.value, n, np.mean(data), np.mean(adjacent_data_distances),
          np.mean(nonadjacent_data_distances), coeff, p_value)


def main():
  pool = Pool()
  samples = read_csv_as_dicts(INPUT_PATH)

  # Tuples of spatial_key1, spatial_key2, spatial_keys_description).
  mantel_runs = [
    (Features.GPS_EASTINGS, Features.GPS_NORTHINGS, 'eastings_and_northings'),
    (Features.GPS_EASTINGS, None, 'eastings_only'),
    (Features.GPS_NORTHINGS, None, 'northings_only'),

    (Features.ROW, Features.COLUMN, 'plot_row_and_column'),
    (Features.ROW, None, 'plot_row_only'),
    (Features.COLUMN, None, 'plot_column_only'),
  ]

  for spatial_key1, spatial_key2, spatial_keys_description in mantel_runs:
    args = []
    for feature in Features:
      # This is a bit hacky way to skip values that are all text values.
      # But it works nicely right now.
      if (feature == Features.ROW or
          feature == Features.COLUMN or
          feature == Features.PLANT_ID):
        continue
      if feature == Features.GPS_EASTINGS:
        break  # Ignore this Feature and all Features after this one.

      args.append((samples, spatial_key1, spatial_key2, feature))


    output_path = get_output_path(spatial_keys_description)
    print('Spawning jobs for:', output_path)
    results = pool.map(get_spatial_correlation, args)
    results.sort(key=lambda x: x[-1])  # Sort by p-value.

    with open(output_path, 'w') as f:
      csv_writer = csv.writer(f)
      csv_writer.writerow(['label', 'num_data_points', 'avg_data_value',
                           'avg_diff_between_adjacent_plots',
                           'avg_diff_between_nonadjacent_plots',
                           'corr_coeff', 'p_value'])
      csv_writer.writerows(results)


if __name__ == '__main__':
  main()
