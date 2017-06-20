#!/usr/bin/env python3

"""
Parse and run boosted tree learner on 2014 data.
"""

import csv_utils
import numpy as np
import os

DATA_PATH = '2014/2014_Pheotypic_Data_FileS2.csv'
PERICARP_LABEL = 'Pericarp_pigmentation'


# TODO keep?
def pretty_label(label):
  label = label.replace('%', 'percent').replace('(', 'in ').replace(')', '')
  return label.replace(' ', '_')


def float_or_nan(s):
  try:
    return np.float(s)
  except:
    return np.nan


def parse_data():
  lines = csv_utils.read_csv(DATA_PATH, ignore_first_lines=2)
  labels = [pretty_label(x) for x in lines[0]]
  samples = [dict(zip(labels, line)) for line in lines[1:]]

  for sample in samples:
    del sample[labels[0]]  # Ignore plant id.
  labels = labels[1:]

  # Convert pericarp string to a number.
  pericarps = sorted(set([x[PERICARP_LABEL] for x in samples]))  # Includes ''.
  for sample in samples:
    if sample[PERICARP_LABEL]:
      sample[PERICARP_LABEL] = pericarps.index(sample[PERICARP_LABEL])

  result = []
  for sample in samples:
    result.append([float_or_nan(sample[x]) for x in labels])
  return np.array(result)


def main():
  samples = parse_data()
  print(samples)



if __name__ == '__main__':
    main()
