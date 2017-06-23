#!/usr/bin/env python3

"""
Compute spacial correlations and their significance.
"""

import csv_utils
import merge_data
from skbio import DistanceMatrix
from skbio.stats.distance import mantel
from scipy.spatial.distance import pdist, squareform

INPUT_FILE_PATH = '2016.csv'
HGT1_LABEL = merge_data.DataKeys.SF16h_HGT1_120.value
EASTINGS_LABEL = merge_data.DataKeys.GPS_EASTINGS.value
NORTHINGS_LABEL = merge_data.DataKeys.GPS_NORTHINGS.value

def main():
  labels, *samples = csv_utils.read_csv(INPUT_FILE_PATH)
  samples = [dict(zip(labels, line)) for line in samples]

  height1_samples = [x for x in samples if x[HGT1_LABEL] != '']
  eastings = [float(x[EASTINGS_LABEL]) for x in height1_samples]
  northings= [float(x[EASTINGS_LABEL]) for x in height1_samples]
  gps = list(zip(eastings, northings))


  gps_distances = DistanceMatrix(squareform(pdist(gps)))
  print(gps_distances)


if __name__ == '__main__':
    main()
