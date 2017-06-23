#!/usr/bin/env python3

"""
Compute spacial correlations and their significance.
"""

import csv
import csv_utils
import merge_data
import random
from skbio import DistanceMatrix
from skbio.stats.distance import mantel
from scipy.spatial.distance import pdist, squareform

INPUT_FILE_PATH = '2016.csv'
OUTPUT_FILE_PATH = 'spacial.' + INPUT_FILE_PATH
EASTINGS_LABEL = merge_data.DataKeys.GPS_EASTINGS.value
NORTHINGS_LABEL = merge_data.DataKeys.GPS_NORTHINGS.value

MAXIMUM_SIGNIFICANT_P_VALUE = 0.01

# Manly says 5000 is minimum number to estimate a significance level of 0.01.
MANTEL_PERMUTATIONS = 10000


def write_spacial_correlation(csv_writer, samples, data_key):
  samples = [x for x in samples if x[data_key.value] != '']
  eastings = [float(x[EASTINGS_LABEL]) for x in samples]
  northings= [float(x[EASTINGS_LABEL]) for x in samples]
  gps = list(zip(eastings, northings))
  data = [(x[data_key.value], 0.0) for x in samples]

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
  labels, *samples = csv_utils.read_csv(INPUT_FILE_PATH)
  samples = [dict(zip(labels, line)) for line in samples]

  with open(OUTPUT_FILE_PATH, 'w') as f:
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
