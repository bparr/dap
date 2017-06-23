#!/usr/bin/env python3

"""
Compute spacial correlations and their significance.
"""

import csv_utils
import merge_data
import random
from skbio import DistanceMatrix
from skbio.stats.distance import mantel
from scipy.spatial.distance import pdist, squareform

INPUT_FILE_PATH = '2016.csv'
EASTINGS_LABEL = merge_data.DataKeys.GPS_EASTINGS.value
NORTHINGS_LABEL = merge_data.DataKeys.GPS_NORTHINGS.value

# Manly says 5000 is minimum number to estimate a significance level of 0.01.
MANTEL_PERMUTATIONS = 10000


def compute_spacial_correlation(samples, data_key):
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
  return mantel(gps_distances, data_distances, permutations=MANTEL_PERMUTATIONS)


def main():
  labels, *samples = csv_utils.read_csv(INPUT_FILE_PATH)
  samples = [dict(zip(labels, line)) for line in samples]

  print(compute_spacial_correlation(samples, merge_data.DataKeys.SF16h_HGT1_120))


if __name__ == '__main__':
    main()
