#!/usr/bin/env python

# TODO add file header docs.

import csv
import os

# TODO don't ignore Check/CHECK values in BAP16_PlotMap_Plant_IDs.csv?
EMPTY_VALUES = ['', ' ', 'FILL', 'NA', 'CHECK', 'Check']
DATA_DIRECTORY = '2016'


# Simple container of all cells.
class Cells(object):
  def __init__(self):
    self._cells = {}

  # Rw = row (case insensitive).
  # Ra = column (case insensitive).
  # TODO What does "Ra3" stand for? I'm using column now. Rename if needed.
  def add(self, row, column):
    # TODO keep lower()? If remove, fix doc removing "(case insensitive)".
    row = row.lower()
    column = column.lower()

    columns =  self._cells.setdefault(row, {})
    if column in columns:
      raise Exception('Existing cell when adding: ', row, column)

    cell = Cell(row, column)
    columns[column] = cell
    return cell


  def get(self, row, column):
    return self._cells[row.lower()][column.lower()]


# Subplot in field containing all genetic siblings.
class Cell(object):
  def __init__(self, row, column):
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
  labels = ['accession_' + v for v in lines[0]]
  for line in lines[1:]:
    plant_id = line[0]  # File has plant id in first column.
    if plant_id in accessions:
      raise Exception('Duplicate entries for plant id: ', line[0])

    accession = {}
    for i, value in enumerate(line[1:], start=1):
      accession[labels[i]] = value
    accessions[plant_id] = accession

  return accessions, labels


def main():
  accessions, accessions_labels = parse_panel_accessions(
      read_csv('PanelAccessions-BAP.csv'))
  plot_map_plant_ids = read_csv('BAP16_PlotMap_Plant_IDs.csv')

  cells = Cells()
  for line in plot_map_plant_ids[1:]:
    for i, value in enumerate(line[1:], start=1):
      if value == '':
        continue

      cell = cells.add(line[0], plot_map_plant_ids[0][i])
      cell.add_data('plant_id', value)

      if value not in accessions:
        print('WARNING: No panel accessions for plant id: ',
              value, line[0], plot_map_plant_ids[0][i])
        continue

      for accession_key, accession_value in accessions[value].items():
        cell.add_data(accession_key, accession_value)


if __name__ == '__main__':
    main()
