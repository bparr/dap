#!/usr/bin/env python

# TODO add file header docs.

import csv
import os

DATA_DIRECTORY = '2016'
OUTPUT_FILENAME = DATA_DIRECTORY + '.csv'

# TODO don't ignore Check/CHECK values in BAP16_PlotMap_Plant_IDs.csv?
EMPTY_VALUES = ['', ' ', 'FILL', 'NA', 'CHECK', 'Check']


# Simple container of all cells.
class Cells(object):
  def __init__(self):
    self._cells = {}

  # Rw = row (case insensitive).
  # Ra = column (case insensitive).
  # TODO What does "Ra3" stand for? I'm using column now. Rename if needed.
  def add(self, row, column):
    # TODO keep upper()? If remove, fix doc removing "(case insensitive)".
    row = row.upper()
    column = column.upper()

    columns =  self._cells.setdefault(row, {})
    if column in columns:
      raise Exception('Existing cell when adding: ', row, column)

    cell = Cell(row, column)
    columns[column] = cell
    return cell


  def get(self, row, column):
    return self._cells[row.upper()][column.upper()]


  def sorted(self):
    cells = []
    for columns in self._cells.values():
      cells.extend(columns.values())
    cells.sort(key=lambda cell: cell.get_sort_key())
    return cells


# Subplot in field containing all genetic siblings.
class Cell(object):
  ROW_DATA_NAME = 'plot_row'
  COLUMN_DATA_NAME = 'plot_column'

  def __init__(self, row, column):
    self._row = row
    self._column = column
    self._data = {
        Cell.ROW_DATA_NAME: row,
        Cell.COLUMN_DATA_NAME: column,
    }

  def add_data(self, name, value):
    if name in self._data:
      raise Exception('Unexpected existing value with name: ', name)
    self._data[name] = value

  def get_data(self, name):
    return self._data.get(name, '')

  # List cells in a deterministic order using this.
  def get_sort_key(self):
    # TODO improve??
    return (self._row, self._column)


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

  # Drop label that was for the plant id.
  labels = labels[1:]
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


  # Write final contents.
  output_labels = ([Cell.ROW_DATA_NAME, Cell.COLUMN_DATA_NAME, 'plant_id'] +
                   accessions_labels)
  with open(OUTPUT_FILENAME, 'w') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(output_labels)
    for cell in cells.sorted():
      writer.writerow([cell.get_data(x) for x in output_labels])


if __name__ == '__main__':
    main()
