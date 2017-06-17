#!/usr/bin/env python3

"""
Combine all data files into a single one with each row describing a cell.

The field of sorghum is partitioned into cells of genetically siblings. Each
cell is located by its row (Rw###) and column (Ra###). Note that in the data
files, Ra### stands for "rank" which is called a "column" in this code to avoid
confusion with linear algebra ranks.
"""
# TODO add tests!

import csv
from enum import Enum
import os

DATA_DIRECTORY = '2016'
OUTPUT_FILENAME = DATA_DIRECTORY + '.csv'

# Values in original csv files that are interpreted as empty strings.
EMPTY_VALUES = ['', ' ', 'FILL', 'NA']

# The field begins and ends with 4 "FILL" rows that are not part of the
# experiement. They are used to avoid edge effects in the experiment. The
# PlotPlan and HarvestData files index their rows by excluding these rows. All
# other files include them. So, this code includes the FILL rows when indexing
# rows, and uses this offset to reindex rows in the PlotPlan and HarvestData.
NO_FILL_ROW_OFFSET = 4

# Store all values for a specific entry in output csv by delimitting them with
# this. For example, some plot ids are inconsistent across PlotMap and
# PlotPlan, so just store all the ones encountered in the single plot id entry
# for the inconsistent cells.  Some plot ids are inconsistent.
MISMATCH_DELIMETER = ' && '


# Parse a single row or column cell coordinate to an int.
def parse_coordinate(coordinate):
  if isinstance(coordinate, int):
    return coordinate
  if len(coordinate) < 3:
    return int(coordinate)

  coordinate_prefix = coordinate.upper()[:2]
  if coordinate_prefix == 'RW' or coordinate_prefix == 'RA':
    coordinate = coordinate[2:]
  return int(coordinate)


# Parse both the row and column cell coordinates to (int, int).
def parse_coordinates(row, column):
  return parse_coordinate(row), parse_coordinate(column)


# Simple container of all cells.
class Cells(object):
  def __init__(self):
    self._cells = {}

  # Row represented as an int or 'Rw###' (case insensitive).
  # Column represented as an int or 'Ra###' (case insensitive).
  def add(self, row, column):
    row, column = parse_coordinates(row, column)
    columns =  self._cells.setdefault(row, {})
    if column in columns:
      raise Exception('Existing cell when adding: ', row, column)

    cell = Cell(row, column)
    columns[column] = cell
    return cell

  def exists(self, row, column):
    row, column = parse_coordinates(row, column)
    return (row in self._cells and column in self._cells[row])

  def get(self, row, column):
    row, column = parse_coordinates(row, column)
    if not self.exists(row, column):
      raise Exception('Unknown cell: ', row, column)

    return self._cells[row][column]

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

  # row and column are both ints.
  def __init__(self, row, column):
    self._row = row
    self._column = column
    self._data = {
        DataKeys(Cell.ROW_DATA_NAME): row,
        DataKeys(Cell.COLUMN_DATA_NAME): column,
    }

  def __str__(self):
    return str(self._row) + ' ' + str(self._column)

  def __repr__(self):
    return self.__str__()

  # If there is already a different value stored for the key, then this method
  # will raise an Exception, unless append_if_mismatch is True (in which case
  # the given value is appended to the existing value).
  def add_data(self, key, value, append_if_mismatch=False):
    key = DataKeys(key)  # Ensure key is a DataKey instance.
    if key in self._data and value != self._data[key]:
      if append_if_mismatch:
        # Don't re-append a single value over and over.
        if value in self._data[key].split(MISMATCH_DELIMETER):
          return
        self._data[key] += MISMATCH_DELIMETER + value
        return

      raise Exception('Unexpected mismatch in existing value: ',
                      key, self._data[key], value)
    self._data[key] = value

  # Returns an empty string if there is no value for the given key.
  def get_data(self, key):
    return self._data.get(key, '')

  # List cells in a deterministic order using this.
  def get_sort_key(self):
    return (self._row, self._column)


# Use this to read a *.csv file. Includes sanity checks and converts values in
# EMPTY_VALUES to ''.
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
  labels = [DataKeys('accession_' + v.lower()) for v in lines[0][1:]]
  for line in lines[1:]:
    plant_id = line[0]  # File has plant id in first column.
    if plant_id in accessions:
      raise Exception('Duplicate entries for plant id: ', line[0])

    accession = {}
    for i, value in enumerate(line[1:]):
      accession[labels[i]] = value
    accessions[plant_id] = accession

  return accessions


def parse_rw_by_ra(lines, data_key, cells, get_extra_data_fn=None,
                   add_cells=False):
  added_cells = []
  for line in lines[1:]:
    row = line[0]

    for i, value in enumerate(line[1:], start=1):
      if value == '':
        continue

      column = lines[0][i]
      if not cells.exists(row, column):
        added_cells.append(cells.add(row, column))
      cell = cells.get(row, column)
      cell.add_data(data_key, value)

      if get_extra_data_fn is not None:
        for k, v in get_extra_data_fn(value).items():
          cell.add_data(k, v)

  if not add_cells and len(added_cells) != 0:
    print('WARNING: Added cell(s) that were missing: ', data_key, added_cells)


def parse_harvest_data(lines, cells):
  labels = [DataKeys(v) for v in lines[0][3:]]
  for line in lines[1:]:
    row, column = line[2], line[1]
    row = parse_coordinate(row) + NO_FILL_ROW_OFFSET
    cell = cells.get(row, column)
    cell.add_data(DataKeys.PLOT_ID, line[0], append_if_mismatch=True)
    for i, value in enumerate(line[3:]):
      cell.add_data(labels[i], value)


def parse_plot_plan(lines, cells):
  lines = lines[1:]  # Ignore labels.
  for plot_id, plant_id, column, row, x_of_y in lines:
    row = parse_coordinate(row) + NO_FILL_ROW_OFFSET
    cell = cells.get(row, column)
    cell.add_data(DataKeys.PLOT_ID, plot_id, append_if_mismatch=True)
    cell.add_data(DataKeys.PLANT_ID, plant_id)
    cell.add_data(DataKeys.X_OF_Y, x_of_y)


def parse_plot_plan_tags(lines, cells):
  lines = lines[1:]  # Ignore labels.
  for plot_id, plant_id, column, row, x_of_y, tag, con, barcode, end in lines:
    row = parse_coordinate(row) + NO_FILL_ROW_OFFSET
    cell = cells.get(row, column)
    cell.add_data(DataKeys.PLOT_ID, plot_id, append_if_mismatch=True)
    cell.add_data(DataKeys.PLANT_ID, plant_id)
    cell.add_data(DataKeys.X_OF_Y, x_of_y)
    cell.add_data(DataKeys.PLOT_PLAN_TAG, tag, append_if_mismatch=True)
    cell.add_data(DataKeys.PLOT_PLAN_CON, con, append_if_mismatch=True)
    cell.add_data(DataKeys.PLOT_PLAN_BARCODE, barcode, append_if_mismatch=True)
    cell.add_data(DataKeys.PLOT_PLAN_END, end, append_if_mismatch=True)


# TODO 2016_09_penetrometer_robot_Large_Stalks.csv has two lines for Rw22 Ra32
#      which seem to describe completely different plants. So ignoring.
# TODO reconsider using these row94 files?
#      - 2016_07_13_leaf_segmentation_leaf_fill_row94.csv
#      - 2016_09_penetrometer_manual_Row_94.csv
class DataKeys(Enum):
  ROW = Cell.ROW_DATA_NAME
  COLUMN = Cell.COLUMN_DATA_NAME
  PLANT_ID = 'plant_id'
  PLOT_ID = 'plot_id'

  # Harvest data.
  SF16h_HGT1_120 = 'SF16h_HGT1_120'
  SF16h_HGT2_120 = 'SF16h_HGT2_120'
  SF16h_HGT3_120 = 'SF16h_HGT3_120'
  SF16h_TWT_120 = 'SF16h_TWT_120'
  SF16h_WTP_120 = 'SF16h_WTP_120'
  SF16h_WTL_120 = 'SF16h_WTL_120'
  SF16h_PAN1_120 = 'SF16h_PAN1_120'
  SF16h_PAN2_120 = 'SF16h_PAN2_120'
  SF16h_PAN3_120 = 'SF16h_PAN3_120'

  # Robot data.
  LEAF_NECROSIS_07 = '2016_07_13-14_Leaf_Necrosis'
  VEGETATION_INDEX_07 = '2016_07_13-14_vegetation_index'
  VEGETATION_INDEX_08 = '2016_08_05-08_vegetation_index'
  LEAF_AREA_07 = '2016_07_13_BAP_Leaf_Area'
  LASER_PLANT_HEIGHT_07 = '2016_07_13_laser_plant_height'
  LIGHT_INTERCEPTION_07 = '2016_07_light_interception'
  LIGHT_INTERCEPTION_08 = '2016_08_light_interception'
  LIGHT_INTERCEPTION_09 = '2016_09_light_interception'

  # parse_panel_accessions depends on these exact ACCESSION_* string values.
  ACCESSION_PHOTOPERIOD = 'accession_photoperiod'
  ACCESSION_TYPE = 'accession_type'
  ACCESSION_ORIGIN = 'accession_origin'
  ACCESSION_RACE = 'accession_race'

  HARVEST_NOTES = 'Notes'
  X_OF_Y = 'x_of_y'
  PLOT_PLAN_TAG = 'plot_plan_tag'
  PLOT_PLAN_CON = 'plot_plan_con'
  PLOT_PLAN_BARCODE = 'plot_plan_barcode'
  PLOT_PLAN_END = 'plot_plan_end'


def main():
  accessions = parse_panel_accessions(read_csv('PanelAccessions-BAP.csv'))

  missing_accessions = set()
  def get_accessions_fn(plant_id):
    if plant_id not in accessions:
      if plant_id not in missing_accessions:
        missing_accessions.add(plant_id)
        print('WARNING: No panel accessions for plant id: ', plant_id)
      return {}

    return accessions[plant_id]


  cells = Cells()
  parse_rw_by_ra(read_csv('BAP16_PlotMap_Plant_IDs.csv'), DataKeys.PLANT_ID,
                 cells, get_extra_data_fn=get_accessions_fn, add_cells=True)
  parse_rw_by_ra(read_csv('BAP16_PlotMap_Plot_IDs.csv'),
                 DataKeys.PLOT_ID, cells)
  parse_harvest_data(read_csv('BAP16_HarvestData.csv'), cells)
  parse_plot_plan(read_csv('BAP16_PlotPlan_Plot_IDs.csv'), cells)
  parse_plot_plan_tags(read_csv('BAP16_PlotPlan_Plot_IDs_Tags.csv'), cells)

  parse_rw_by_ra(read_csv('2016_07_13-14_Leaf_Necrosis.csv'),
                 DataKeys.LEAF_NECROSIS_07, cells)
  parse_rw_by_ra(read_csv('2016_07_13-14_vegetation_index.csv'),
                 DataKeys.VEGETATION_INDEX_07, cells)
  parse_rw_by_ra(read_csv('2016_08_05-08_vegetation_index.csv'),
                 DataKeys.VEGETATION_INDEX_08, cells)
  parse_rw_by_ra(read_csv('2016_07_13_BAP_Leaf_Area.csv'),
                 DataKeys.LEAF_AREA_07, cells)
  parse_rw_by_ra(read_csv('2016_07_13_laser_plant_height.csv'),
                 DataKeys.LASER_PLANT_HEIGHT_07, cells)
  parse_rw_by_ra(read_csv('2016_07_light_interception.csv'),
                 DataKeys.LIGHT_INTERCEPTION_07, cells)
  parse_rw_by_ra(read_csv('2016_08_light_interception.csv'),
                 DataKeys.LIGHT_INTERCEPTION_08, cells)
  parse_rw_by_ra(read_csv('2016_09_light_interception.csv'),
                 DataKeys.LIGHT_INTERCEPTION_09, cells)

  # Write final contents.
  with open(OUTPUT_FILENAME, 'w') as output_file:
    writer = csv.writer(output_file)
    writer.writerow([x.value for x in DataKeys])
    for cell in cells.sorted():
      writer.writerow([cell.get_data(x) for x in DataKeys])


if __name__ == '__main__':
    main()
