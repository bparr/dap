#!/usr/bin/env python3

"""
Compute spatial correlations and their significance.
"""

import csv
import csv_utils
import merge_data
from multiprocessing import Pool
import numpy as np
import random
from skbio import DistanceMatrix
from skbio.stats.distance import mantel
from scipy.spatial.distance import pdist, squareform

# The non-merged file already has GPS for all cells, so merging the cells does
# not increase coverage.
INPUT_PATHS = ['2016.csv'] #, '2016.merged.csv']
OUTPUT_PATHS = ['spatial.' + x for x in INPUT_PATHS]
EASTINGS_LABEL = merge_data.DataKeys.GPS_EASTINGS.value
NORTHINGS_LABEL = merge_data.DataKeys.GPS_NORTHINGS.value

# Manly says 5000 is minimum number to estimate a significance level of 0.01.
MANTEL_PERMUTATIONS = 10000


# Averge multiple numeric values, caused by mismatched data merging.
def average_mismatch(value):
  return np.mean([float(x) for x in value.split(merge_data.MISMATCH_DELIMETER)])


# Computes spatial correlation relative to each cell's GPS location.
# Returns (key name, number of data points, correlation coefficient, p value)
def get_spatial_correlation(arg):
  samples, data_key = arg
  samples = [x for x in samples if x[data_key.value] != '']
  eastings = [average_mismatch(x[EASTINGS_LABEL]) for x in samples]
  northings= [average_mismatch(x[EASTINGS_LABEL]) for x in samples]
  gps = list(zip(eastings, northings))
  # Add required second dimension, but set to 0.0, so no affect on distances.
  data = [(average_mismatch(x[data_key.value]), 0.0) for x in samples]

  # Sanity check: Random data should show no significant correlation.
  # Results: p-value of 0.905, so sanity check passed.
  #random.seed(10611)  # Does not seem to contain all randomness unfortunately.
  #data = [(random.random(), 0.0) for x in samples]

  gps_distances = DistanceMatrix(squareform(pdist(gps)))
  data_distances = DistanceMatrix(squareform(pdist(data)))
  coeff, p_value, n = mantel(gps_distances, data_distances,
                             permutations=MANTEL_PERMUTATIONS)

  return (data_key.value, n, coeff, p_value)


def main():
  pool = Pool()

  for input_path, output_path in zip(INPUT_PATHS, OUTPUT_PATHS):
    labels, *samples = csv_utils.read_csv(input_path)
    samples = [dict(zip(labels, line)) for line in samples]

    args = []
    for data_key in merge_data.DataKeys:
      # This is a bit hacky way to skip remaining values that are all text
      # values. But it works nicely right now.
      if (data_key == merge_data.DataKeys.ROW or
          data_key == merge_data.DataKeys.COLUMN or
          data_key == merge_data.DataKeys.PLANT_ID):
        continue
      if data_key == merge_data.DataKeys.GPS_EASTINGS:
        break  # Ignore all DataKeys after this one.

      args.append((samples, data_key))


    print('Spawning jobs for:', input_path)
    results = pool.map(get_spatial_correlation, args)
    with open(output_path, 'w') as f:
      csv_writer = csv.writer(f)
      csv_writer.writerow(['label', 'num_data_points', 'corr_coeff', 'p_value'])
      csv_writer.writerows(results)


if __name__ == '__main__':
    main()
