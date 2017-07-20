#!/usr/bin/env python3

"""
Visualize specific features of the field.
"""
# TODO improve file doc. Including a Usage:.
# TODO tests?

import argparse
import csv_utils
import matplotlib.pyplot as plt
from merge_data import DataKeys, average_mismatch

def main():
  samples = csv_utils.read_csv_as_dicts('2016.merged.csv')
  parser = argparse.ArgumentParser(description='Visualize features')
  parser.add_argument('-f', '--feature', required=True,
                      choices=samples[0].keys(),
                      help='Feature name to visualize.')
  args = parser.parse_args()

  samples = [x for x in samples if x[args.feature] != '']

  rows = [average_mismatch(x[DataKeys.ROW.value]) for x in samples]
  columns = [average_mismatch(x[DataKeys.COLUMN.value]) for x in samples]
  values = [average_mismatch(x[args.feature]) for x in samples]

  plt.title('Visualization of ' + args.feature)
  xdim = 3
  ydim = 2
  plt.scatter(rows, columns, c=values, s=100,
              marker=[(-xdim, -ydim), (xdim, -ydim), (xdim, ydim),
                      (-xdim, ydim), (-xdim, -ydim)])
  cb = plt.colorbar()
  # TODO fill in colorbar title.
  #cb.ax.set_title('')
  plt.xlabel("Row")
  plt.ylabel("Range")
  plt.show()


if __name__ == '__main__':
    main()