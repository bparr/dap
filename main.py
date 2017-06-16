#!/usr/bin/env python

import csv
import os

EMPTY_VALUES = ['', ' ', 'FILL', 'NA']
DATA_DIRECTORY = '2016'

# TODO add file header docs.

# Simple container of all cells.
class Cells(object):
  def __init__(self):
    self._cells = dict()

  # Rw = row (case insensitive).
  # Ra = column (case insensitive).
  # TODO What does "Ra3" stand for? I'm using column now. Rename if needed.
  def get(self, row, column):
    # TODO keep? If remove, fix method doc too removing "(case insensitive)".
    row = row.lower()
    column = column.lower()

    columns =  self._cells.setdefault(row, dict())
    # TODO is it ok to create so many Cell objects?
    return columns.setdefault(column, Cell(row, column))


# Subplot in field containing all genetic siblings.
class Cell(object):
  def __init(self, row, column):
    self._row = row
    self._column = column
    self._data = dict()

  def add_data(self, name, value):
    if name in self._data:
      raise Exception('Unexpected existing value with name: ', name)
    self._data[name] = value


def read_csv(file_name):
  file_path = os.path.join(DATA_DIRECTORY, file_name)
  with open(file_path, 'r') as f:
    lines = []
    for line in csv.reader(f):
      lines.append(map(lambda v: '' if v in EMPTY_VALUES else v, line))

  return lines


def main():
  lines = read_csv('BAP16_PlotMap_Plant_IDs.csv')
  print(len(lines))


if __name__ == '__main__':
    main()
