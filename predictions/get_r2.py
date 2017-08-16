#!/usr/bin/env python3

"""
For each dataset directory in the current directory:
Prints overal r2 score as well as the r2 score for each individual label.

Usage (must be run from the predictions/ directory!):
  ./get_r2.py > r2.csv

"""

import argparse
import collections
import csv
import numpy as np
import os
from sklearn.metrics import r2_score


def get_subdirs(d):
  return sorted([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))])


def convert(s):
  if s[0] == '[' and s[-1] == ']':
    s = s[1:-1]
  return float(s)


# Assert identical ordering of entries across all output predictions.
def verify_matched_rows_and_columns(all_lines):
  rows_and_columns = None
  for filename, lines in all_lines.items():
    current_rows_and_columns = [(x['row'], x['range']) for x in lines]
    if rows_and_columns is None:
      rows_and_columns = current_rows_and_columns
    if rows_and_columns != current_rows_and_columns:
      raise Exception('Rows and columns mismatch!')


def main():
  for d in get_subdirs('.'):
    results = collections.OrderedDict()
    for subdir in get_subdirs(d):
      all_lines = collections.OrderedDict()
      for filename in sorted(os.listdir(os.path.join(d, subdir))):
        with open(os.path.join(d, subdir, filename)) as f:
          all_lines[filename] = list(csv.DictReader(f))

      # The 2014 dataset does not have required 'row' or 'range' features.
      if not d.startswith('2014'):
        verify_matched_rows_and_columns(all_lines)

      results[subdir] = collections.OrderedDict()
      actual = []
      predicted = []
      for filename in sorted(os.listdir(os.path.join(d, subdir))):
        output_name = filename.replace('.csv', '')
        with open(os.path.join(d, subdir, filename)) as f:
          lines = list(csv.DictReader(f))
        actual_values = [convert(x['actual_' + output_name]) for x in lines]
        predicted_values = ([convert(x['predicted_' + output_name])
                             for x in lines])
        results[subdir][output_name] = r2_score(actual_values, predicted_values)

        actual.append(actual_values)
        predicted.append(predicted_values)


      # TODO remove this hack?
      #      The 2014 dataset does not have consistent number of entries, which
      #      breaks the overall r2_score.
      if d.startswith('2014'):
        continue

      actual = np.transpose(np.array(actual))
      predicted = np.transpose(np.array(predicted))
      results[subdir]['zOverall'] = r2_score(
          actual, predicted, multioutput='uniform_average')

    print('\n\n')
    print('Results_for_dataset_' + d)
    print(','.join(['output_label'] + list(results.keys())))
    output_labels = None
    for predictor_name, predictor_values in results.items():
      if output_labels is None:
        output_labels = sorted(predictor_values.keys())
      if sorted(predictor_values.keys()) != output_labels:
        raise Exception('Output labels mismatch')

    for output_label in output_labels:
      values = list(x[output_label] for x in results.values())
      print(','.join([output_label] + [str(x) for x in values]))


if __name__ == '__main__':
  main()