#!/usr/bin/env python3

"""
Compute correlation matrix.
Usage:
  ./correlation_matrix.py

Writes output to results/correlation_matrix.2016.csv file.
"""

import csv
import csv_utils
import dataset
from features import Features
import numpy as np


INPUT_PATH = '2016.merged.csv'
OUTPUT_PATH = 'results/correlation_matrix.2016.csv'


def filter_labels(feature_starts_with_list):
  result = []
  for starts_with in feature_starts_with_list:
    result.extend([x.value for x in Features if x.name.startswith(starts_with)])
  if len(set(result)) != len(result):
    raise Exception('Duplicate features filtered!')
  return result


def main():
  samples = csv_utils.read_csv_as_dicts(INPUT_PATH)
  features_to_use = filter_labels(('ROW', 'COLUMN', 'GPS_', 'ROBOT_',
                                   'HARVEST_', 'SYNTHETIC_', 'COMPOSITION_'))
  dataset.convert_to_float_or_missing(samples, features_to_use)
  X = np.array([[sample[x] for x in features_to_use] for sample in samples])

  results = np.corrcoef(X, rowvar=False)

  with open(OUTPUT_PATH, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['feature_name'] + features_to_use)
    for feature_name, results_row in zip(features_to_use, results):
      writer.writerow([feature_name] + list(results_row))


if __name__ == '__main__':
  main()
