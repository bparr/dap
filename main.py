#!/usr/bin/env python

import csv
import os

EMPTY_VALUES = ['', ' ', 'FILL', 'NA']
DATA_DIRECTORY = '2016'

# TODO add file header docs.

# Simple container of all cells.
class Cells(object):
  def __init__(self):
    self._cells = {}

  # Rw = row (case insensitive).
  # Ra = column (case insensitive).
  # TODO What does "Ra3" stand for? I'm using column now. Rename if needed.
  def get(self, row, column):
    # TODO keep? If remove, fix method doc too removing "(case insensitive)".
    row = row.lower()
    column = column.lower()

    columns =  self._cells.setdefault(row, {})
    # TODO is it ok to create so many Cell objects?
    return columns.setdefault(column, Cell(row, column))


# Subplot in field containing all genetic siblings.
class Cell(object):
  def __init(self, row, column):
    self._row = row
    self._column = column
    self._data = {}

  def add_data(self, name, value):
    if name in self._data:
      raise Exception('Unexpected existing value with name: ', name)
    self._data[name] = value


def read_csv(file_name):
  file_path = os.path.join(DATA_DIRECTORY, file_name)
  with open(file_path, 'r') as f:
    lines = []
    for line in csv.reader(f):
      lines.append(['' if v in EMPTY_VALUES else v for v in line])

  num_columns = len(lines[0])
  if len(set(lines[0])) != num_columns:
    raise Exception('Duplicate label in first row of csv: ', file_name)
  for i, line in enumerate(lines):
    if len(line) != num_columns:
      raise Exception('Unexpected amount of values in line: ',
                      i, len(line), num_columns)
  return lines


def parse_panel_accessions(lines):
  accessions = {}
  for line in lines[1:]:
    plant_id = line[0]  # File has plant id in first column.
    if plant_id in accessions:
      raise Exception('Duplicate entries for plant id: ', line[0])

    accession = {}
    for i, value in enumerate(line[1:], start=1):
      accession[lines[0][i]] = value
    accessions[plant_id] = accession

  return accessions


def main():
  accessions = parse_panel_accessions(read_csv('PanelAccessions-BAP.csv'))
  print(accessions['PI_63715'])


if __name__ == '__main__':
    main()
