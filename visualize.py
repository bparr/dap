#!/usr/bin/env python3

"""
Visualize specific features of the field.
"""
# TODO improve file doc. Including a Usage:.
# TODO tests?

import csv_utils
import matplotlib.pyplot as plt
from merge_data import DataKeys, average_mismatch

# TODO allow specifying on command line?
FEATURE = '2016_09_light_interception'


def main():
  samples = csv_utils.read_csv_as_dicts('2016.merged.csv')
  samples = [x for x in samples if x[FEATURE] != '']

  rows = [average_mismatch(x[DataKeys.ROW.value]) for x in samples]
  columns = [average_mismatch(x[DataKeys.COLUMN.value]) for x in samples]
  values = [average_mismatch(x[FEATURE]) for x in samples]

  # TODO fill in title.
  plt.title('TODO fill in')
  xdim = 3
  ydim = 2
  plt.scatter(rows, columns, c=values, s=100,
              marker=[(-xdim, -ydim), (xdim, -ydim), (xdim, ydim),
                      (-xdim, ydim), (-xdim, -ydim)])
  cb = plt.colorbar()
  # TODO fill in colorbar title.
  cb.ax.set_title('TODO fill in')
  plt.xlabel("Row")
  plt.ylabel("Range")
  plt.show()


if __name__ == '__main__':
    main()
