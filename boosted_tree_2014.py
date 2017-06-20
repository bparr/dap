#!/usr/bin/env python3

"""
Parse and run boosted tree learner on 2014 data.
"""

import csv_utils
import os

DATA_PATH = '2014/2014_Pheotypic_Data_FileS2.csv'
PERICARP_LABEL = 'Pericarp_pigmentation'


# TODO keep?
def pretty_label(label):
  label = label.replace('%', 'percent').replace('(', 'in ').replace(')', '')
  return label.replace(' ', '_')


# TODO replace with Imputer.
def float_or_empty(s):
  return None if s == '' else float(s)


def parse_data():
  lines = csv_utils.read_csv(DATA_PATH, ignore_first_lines=2)
  labels = [pretty_label(x) for x in lines[0]]
  samples = [dict(zip(labels, line)) for line in lines[1:]]

  for sample in samples:
    del sample[labels[0]]  # Ignore plant id.
  labels = labels[1:]

  # Includes the empty string.
  pericarps = sorted(set([x[PERICARP_LABEL] for x in samples]))

  for sample in samples:
    if sample[PERICARP_LABEL]:
      sample[PERICARP_LABEL] = pericarps.index(sample[PERICARP_LABEL])

  result = []
  for sample in samples:
    result.append(dict((k, float_or_empty(v)) for k, v in sample.items()))
  return labels, result


def main():
  labels, samples = parse_data()

  for sample in samples:
    print(','.join([str(sample[x]) for x in labels]))


if __name__ == '__main__':
    main()
