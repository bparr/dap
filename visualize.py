#!/usr/bin/env python3

"""
Visualize specific labels of a CSV file.
"""

import argparse
from csv_utils import average_mismatch, read_csv_as_dicts
from features import Features
import matplotlib.pyplot as plt
import os

def main():
  parser = argparse.ArgumentParser(description='Visualize features')
  parser.add_argument('-f', '--file', required=True,
                      help='Path to input file.')
  parser.add_argument('-l', '--label', required=True,
                      help='Label name in input file to visualize.')
  args = parser.parse_args()

  samples = read_csv_as_dicts(args.file)
  samples = [x for x in samples if x[args.label] != '']

  rows = [average_mismatch(x[Features.GPS_EASTINGS.value]) for x in samples]
  columns = [average_mismatch(x[Features.GPS_NORTHINGS.value]) for x in samples]
  values = [average_mismatch(x[args.label]) for x in samples]

  plt.title('Visualization of ' + args.label + ' in ' +
            os.path.basename(args.file))
  xdim = 3
  ydim = 2
  plt.scatter(rows, columns, c=values, s=100,
              marker=[(-xdim, -ydim), (xdim, -ydim), (xdim, ydim),
                      (-xdim, ydim), (-xdim, -ydim)])
  cb = plt.colorbar()
  cb.ax.set_title('Value')
  plt.xlabel("Row")
  plt.ylabel("Range")
  plt.show()


if __name__ == '__main__':
  main()
