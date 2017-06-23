#!/usr/bin/env python3

"""
Compute spacial correlations and their significance.
"""

import csv
import csv_utils
import merge_data
import numpy as np
import random
from skbio import DistanceMatrix
from skbio.stats.distance import mantel
from scipy.spatial.distance import pdist, squareform

INPUT_PATHS = ['2016.csv', '2016.merged.csv']
OUTPUT_PATHS = ['spacial.' + x for x in INPUT_PATHS]
EASTINGS_LABEL = merge_data.DataKeys.GPS_EASTINGS.value
NORTHINGS_LABEL = merge_data.DataKeys.GPS_NORTHINGS.value

MAXIMUM_SIGNIFICANT_P_VALUE = 0.01

# Manly says 5000 is minimum number to estimate a significance level of 0.01.
MANTEL_PERMUTATIONS = 10# TODO revert10000


def average_mismatch(value):
  return np.mean([float(x) for x in value.split(merge_data.MISMATCH_DELIMETER)])


def write_spacial_correlation(csv_writer, samples, data_key):
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

  csv_writer.writerow([data_key.value, n, coeff, p_value])

  significant_str = 'Significant P Value'
  if p_value > MAXIMUM_SIGNIFICANT_P_VALUE:
    significant_str = 'NOT ' + significant_str
  print(significant_str + ':', data_key.value, '=', p_value)


def main():
  for input_path, output_path in zip(INPUT_PATHS, OUTPUT_PATHS):
    labels, *samples = csv_utils.read_csv(input_path)
    samples = [dict(zip(labels, line)) for line in samples]

    with open(output_path, 'w') as f:
      csv_writer = csv.writer(f)
      csv_writer.writerow(['label', 'num_data_points', 'corr_coeff', 'p_value'])

      for data_key in merge_data.DataKeys:
        if data_key == merge_data.DataKeys.PLANT_ID:
          print('Skipping:', data_key.value)
          continue
        if data_key == merge_data.DataKeys.ACCESSION_PHOTOPERIOD:
          # This is a bit hacky way to skip remaining values that are all text
          # values. But it works nicely right now.
          break

        write_spacial_correlation(csv_writer, samples, data_key)


if __name__ == '__main__':
    main()
