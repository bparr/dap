#!/usr/bin/env python3

"""
Parse and run boosted tree learner on 2014 data.
"""

import csv_utils
import os

DATA_PATH = '2014/2014_Pheotypic_Data_FileS2.csv'

def pretty_label(label):
  label = label.replace('%', 'percent').replace('(', 'in ').replace(')', '')
  return label.replace(' ', '_')


def main():
  lines = csv_utils.read_csv(DATA_PATH, ignore_first_lines=2)
  labels = [pretty_label(x) for x in lines[0]]
  print('\n'.join(labels))
  samples = [dict(zip(labels, line)) for line in lines[1:]]
  print(len(samples))

if __name__ == '__main__':
    main()
