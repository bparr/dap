#!/usr/bin/env python3

"""
Combine all data files into a single one with each row describing a cell.

The field of sorghum is partitioned into cells of genetically siblings. Each
cell is located by its row (Rw###) and range (Ra###). Note that in the data
files, Ra### stands for "range" which is called a "column" in this code to avoid
confusion with the Python range() function.

Usage:
  ./merge_data.py && ./merge_data_test.py

Running the merge_data_test file is not required but verifies the results,
which are written to 2016.csv and 2016.merged.csv.
"""

import csv
import csv_utils
from enum import Enum
import numpy as np
import os

DATA_DIRECTORY = '2016'
OUTPUT_FILENAME = DATA_DIRECTORY + '.csv'
MERGED_OUTPUT_FILENAME = DATA_DIRECTORY + '.merged.csv'

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

# The amount of rows a single cell contains.
# TODO better name than Cell?
ROWS_IN_CELL = 4


# Averge multiple numeric values, caused by mismatched data merging.
def average_mismatch(value):
  return np.mean([float(x) for x in value.split(MISMATCH_DELIMETER)])


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
      raise Exception('Existing cell when adding:', row, column)

    cell = Cell(row, column)
    columns[column] = cell
    return cell

  def exists(self, row, column):
    row, column = parse_coordinates(row, column)
    return (row in self._cells and column in self._cells[row])

  def get(self, row, column):
    row, column = parse_coordinates(row, column)
    if not self.exists(row, column):
      raise Exception('Unknown cell:', row, column)

    return self._cells[row][column]

  def sorted(self):
    cells = []
    for columns in self._cells.values():
      cells.extend(columns.values())
    cells.sort(key=lambda cell: cell.get_coordinates())
    return cells


# Subplot in field containing all genetic siblings.
class Cell(object):
  ROW_DATA_NAME = 'row'
  COLUMN_DATA_NAME = 'range'

  def __init__(self, row, column):
    self._data = {
        DataKeys(Cell.ROW_DATA_NAME): str(row),
        DataKeys(Cell.COLUMN_DATA_NAME): str(column),
    }

  def __str__(self):
    return (self._data[DataKeys(Cell.ROW_DATA_NAME)] + ' ' +
            self._data[DataKeys(Cell.COLUMN_DATA_NAME)])

  def __repr__(self):
    return self.__str__()

  # If there is already a different value stored for the key, then this method
  # will raise an Exception, unless append_if_mismatch is True (in which case
  # the given value is appended to the existing value).
  def add_data(self, key, value, append_if_mismatch=False):
    if value == '':
      return

    key = DataKeys(key)  # Ensure key is a DataKey instance.
    if key in self._data and value != self._data[key]:
      if append_if_mismatch:
        # Don't re-append a single value over and over.
        if value in self._data[key].split(MISMATCH_DELIMETER):
          return
        self._data[key] += MISMATCH_DELIMETER + value
        return

      raise Exception('Unexpected mismatch in existing value:',
                      key, self._data[key], value)
    self._data[key] = value

  # Returns an empty string if there is no value for the given key.
  def get_data(self, key):
    return self._data.get(key, '')

  # List cells in a deterministic order using this.
  def get_coordinates(self):
    return (int(self._data[DataKeys(Cell.ROW_DATA_NAME)]),
            int(self._data[DataKeys(Cell.COLUMN_DATA_NAME)]))

  def add_all_data_to_cell(self, cell_to_copy_to):
    for key, value in self._data.items():
      cell_to_copy_to.add_data(key, value, append_if_mismatch=True)


def read_csv(file_name):
  return csv_utils.read_csv(os.path.join(DATA_DIRECTORY, file_name))


# Return a dictionary whose keys are from the first column, and whose values
# are dictionaries from DataKey to corresponding value.
def parse_first_column_indexed(lines, get_label_fn=None, get_index_fn=None):
  if get_label_fn is None:
    get_label_fn = lambda x: x
  labels = [DataKeys(get_label_fn(x)) for x in lines[0][1:]]

  if get_index_fn is None:
    get_index_fn = lambda x: x

  results = {}
  for line in lines[1:]:
    index = get_index_fn(line[0])
    if index in results:
      raise Exception('Duplicate entry for index:', index)

    results[index] = dict(zip(labels, line[1:]))
  return results


def parse_rw_by_ra(lines, data_key, cells, extra_data=None,
                   warn_if_added_cells=True,
                   warn_if_missing_extra_data=True):
  added_cells = []
  missing_extra_data = set()
  used_extra_data_keys = set()

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

      if extra_data is None:
        continue

      if value not in extra_data:
        if value not in missing_extra_data and warn_if_missing_extra_data:
          missing_extra_data.add(value)
          print('WARNING: No extra data:', data_key, value)
        continue

      used_extra_data_keys.add(value)
      for k, v in extra_data[value].items():
        cell.add_data(k, v)

  if warn_if_added_cells and len(added_cells) != 0:
    print('WARNING: Added cell(s) that were missing:', data_key, added_cells)

  if extra_data is not None and len(used_extra_data_keys) != len(extra_data):
    unused_keys = set(extra_data.keys()) - used_extra_data_keys
    print('WARNING: Unused extra data:', data_key, sorted(unused_keys))


def parse_harvest_data(lines, cells):
  labels = [DataKeys(v) for v in lines[0][3:]]
  for line in lines[1:]:
    row, column = line[2], line[1]
    row = parse_coordinate(row) + NO_FILL_ROW_OFFSET
    cell = cells.get(row, column)
    cell.add_data(DataKeys.PLOT_ID, line[0])
    for i, value in enumerate(line[3:]):
      cell.add_data(labels[i], value)


def parse_plot_plan(lines, cells):
  lines = lines[1:]  # Ignore labels.
  for plot_id, plant_id, column, row, x_of_y in lines:
    row = parse_coordinate(row) + NO_FILL_ROW_OFFSET
    cell = cells.get(row, column)
    cell.add_data(DataKeys.PLOT_ID, plot_id)
    cell.add_data(DataKeys.PLANT_ID, plant_id)
    cell.add_data(DataKeys.X_OF_Y, x_of_y)


def parse_plot_plan_tags(lines, cells):
  lines = lines[1:]  # Ignore labels.
  for plot_id, plant_id, column, row, x_of_y, tag, con, barcode, end in lines:
    row = parse_coordinate(row) + NO_FILL_ROW_OFFSET
    cell = cells.get(row, column)
    cell.add_data(DataKeys.PLOT_ID, plot_id)
    cell.add_data(DataKeys.PLANT_ID, plant_id)
    cell.add_data(DataKeys.X_OF_Y, x_of_y)
    cell.add_data(DataKeys.PLOT_PLAN_TAG, tag, append_if_mismatch=True)
    cell.add_data(DataKeys.PLOT_PLAN_CON, con, append_if_mismatch=True)
    cell.add_data(DataKeys.PLOT_PLAN_BARCODE, barcode, append_if_mismatch=True)
    cell.add_data(DataKeys.PLOT_PLAN_END, end, append_if_mismatch=True)


def add_gps_to_cells(lines, cells):
  lines = lines[1:]  # Ignore labels.
  gps = {}
  for row, column, eastings, northings in lines:
    gps.setdefault(row, {})[column] = (float(eastings), float(northings))

  for cell in cells.sorted():
    row, column = cell.get_coordinates()
    eastings1, northings1 = gps[str(row - 1) + '.5'][str(column)]
    eastings2, northings2 = gps[str(row) + '.5'][str(column)]
    eastings = str(np.mean([eastings1, eastings2]))
    northings = str(np.mean([northings1, northings2]))
    cell.add_data(DataKeys.GPS_EASTINGS, eastings)
    cell.add_data(DataKeys.GPS_NORTHINGS, northings)


def add_synthetic_values(cells, data_keys, mean_data_key, std_data_key):
  for cell in cells.sorted():
    values = [cell.get_data(x) for x in data_keys]
    values = [float(x) for x in values if x != '']
    if len(values) > 0:
      cell.add_data(mean_data_key, str(np.mean(values)))
      cell.add_data(std_data_key, str(np.std(values)))


# TODO(bparr): 2016_09_penetrometer_robot_Large_Stalks.csv has two lines for
#              Rw22 Ra32 which seem to describe completely different plants. So
#              ignoring.
# TODO(bparr): Reconsider using these row94 files?
#      - 2016_07_13_leaf_segmentation_leaf_fill_row94.csv
#      - 2016_09_penetrometer_manual_Row_94.csv
class DataKeys(Enum):
  ROW = Cell.ROW_DATA_NAME
  COLUMN = Cell.COLUMN_DATA_NAME
  PLANT_ID = 'plant_id'

  # Harvest data.
  HARVEST_SF16h_HGT1_120 = 'SF16h_HGT1_120'
  HARVEST_SF16h_HGT2_120 = 'SF16h_HGT2_120'
  HARVEST_SF16h_HGT3_120 = 'SF16h_HGT3_120'
  HARVEST_SF16h_TWT_120 = 'SF16h_TWT_120'
  HARVEST_SF16h_WTP_120 = 'SF16h_WTP_120'
  HARVEST_SF16h_WTL_120 = 'SF16h_WTL_120'
  HARVEST_SF16h_PAN1_120 = 'SF16h_PAN1_120'
  HARVEST_SF16h_PAN2_120 = 'SF16h_PAN2_120'
  HARVEST_SF16h_PAN3_120 = 'SF16h_PAN3_120'

  # Composition data.
  COMPOSITION_ADF = 'ADF'
  COMPOSITION_AD_ICP = 'AD_ICP'
  COMPOSITION_ADJ_CP = 'Adj_CP'
  COMPOSITION_ANDFOM = 'aNDFom'
  COMPOSITION_ASH = 'Ash'
  COMPOSITION_CRUDE_PROTEIN = 'Crude_protein'
  COMPOSITION_DCAD = 'DCAD'
  COMPOSITION_DRY_MATTER = 'Dry_Matter'
  COMPOSITION_EE_FAT = 'EE_Fat'
  COMPOSITION_LIGNIN = 'Lignin'
  COMPOSITION_NEG_OARDC = 'NEG_OARDC'
  COMPOSITION_NEL3X_ADF = 'NEL3x_ADF'
  COMPOSITION_NEL3X_OARDC = 'NEL3x_OARDC'
  COMPOSITION_NEM_OARDC = 'NEM_OARDC'
  COMPOSITION_NFC = 'NFC'
  COMPOSITION_SPCP = 'SPCP'
  COMPOSITION_STARCH = 'Starch'
  COMPOSITION_TDN_OARDC = 'TDN_OARDC'
  COMPOSITION_WSC_SUGAR = 'WSC_Sugar'
  COMPOSITION_CELLULOSE = 'Cellulose'
  COMPOSITION_HEMICELLULOSE = 'Hemicellulose'

  # Robot data.
  ROBOT_LEAF_NECROSIS_07 = '2016_07_13-14_Leaf_Necrosis'
  ROBOT_VEGETATION_INDEX_07 = '2016_07_13-14_vegetation_index'
  ROBOT_VEGETATION_INDEX_08 = '2016_08_05-08_vegetation_index'
  ROBOT_LEAF_AREA_07 = '2016_07_13_BAP_Leaf_Area'
  ROBOT_LASER_PLANT_HEIGHT_07 = '2016_07_13_laser_plant_height'
  ROBOT_LIGHT_INTERCEPTION_07 = '2016_07_light_interception'
  ROBOT_LIGHT_INTERCEPTION_08 = '2016_08_light_interception'
  ROBOT_LIGHT_INTERCEPTION_09 = '2016_09_light_interception'

  # Synthetically created data.
  SYNTHETIC_HARVEST_SF16h_HGT_120_MEAN = 'SF16h_HGT_120_MEAN'
  SYNTHETIC_HARVEST_SF16h_HGT_120_STD = 'SF16h_HGT_120_STD'
  SYNTHETIC_HARVEST_SF16h_PAN_120_MEAN = 'SF16h_PAN_120_MEAN'
  SYNTHETIC_HARVEST_SF16h_PAN_120_STD = 'SF16h_PAN_120_STD'

  # GPS location, in UTM format.
  GPS_EASTINGS = 'gps_eastings_UTMzone17N'
  GPS_NORTHINGS = 'gps_northings_UTMzone17N'

  # Accessions Data.
  ACCESSION_PHOTOPERIOD = 'accession_photoperiod'
  ACCESSION_TYPE = 'accession_type'
  ACCESSION_ORIGIN = 'accession_origin'
  ACCESSION_RACE = 'accession_race'

  # Other lower priority data.
  PLOT_ID = 'plot_id'
  NOTES = 'Notes'
  X_OF_Y = 'x_of_y'
  PLOT_PLAN_TAG = 'plot_plan_tag'
  PLOT_PLAN_CON = 'plot_plan_con'
  PLOT_PLAN_BARCODE = 'plot_plan_barcode'
  PLOT_PLAN_END = 'plot_plan_end'


# Write output file.
def write_csv(file_path, cell_list):
  with open(file_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow([x.value for x in DataKeys])
    for cell in cell_list:
      writer.writerow([cell.get_data(x) for x in DataKeys])


def main():
  cells = Cells()

  accessions = parse_first_column_indexed(
      read_csv('PanelAccessions-BAP.csv'),
      get_label_fn=lambda x: 'accession_' + x.lower())
  # No cells yet, so do not warn when adding cells.
  parse_rw_by_ra(read_csv('BAP16_PlotMap_Plant_IDs.csv'), DataKeys.PLANT_ID,
                 cells, extra_data=accessions, warn_if_added_cells=False)

  compositions = parse_first_column_indexed(
      read_csv('2016_BAPClemsonGRDBBv2.csv'),
      get_index_fn=lambda x: x.replace('SF16', 'SF'),
      get_label_fn=lambda x: x.replace(' ', '_').replace('-', '_'))
  # Not all cells have composition data, do not warn about missing entries
  # in the composition dictionary.
  parse_rw_by_ra(read_csv('BAP16_PlotMap_Plot_IDs.csv'), DataKeys.PLOT_ID,
                 cells, extra_data=compositions,
                 warn_if_missing_extra_data=False)

  parse_harvest_data(read_csv('BAP16_HarvestData.csv'), cells)
  parse_plot_plan(read_csv('BAP16_PlotPlan_Plot_IDs.csv'), cells)
  parse_plot_plan_tags(read_csv('BAP16_PlotPlan_Plot_IDs_Tags.csv'), cells)

  parse_rw_by_ra(read_csv('2016_07_13-14_Leaf_Necrosis.csv'),
                 DataKeys.ROBOT_LEAF_NECROSIS_07, cells)
  parse_rw_by_ra(read_csv('2016_07_13-14_vegetation_index.csv'),
                 DataKeys.ROBOT_VEGETATION_INDEX_07, cells)
  parse_rw_by_ra(read_csv('2016_08_05-08_vegetation_index.csv'),
                 DataKeys.ROBOT_VEGETATION_INDEX_08, cells)
  parse_rw_by_ra(read_csv('2016_07_13_BAP_Leaf_Area.csv'),
                 DataKeys.ROBOT_LEAF_AREA_07, cells)
  parse_rw_by_ra(read_csv('2016_07_13_laser_plant_height.csv'),
                 DataKeys.ROBOT_LASER_PLANT_HEIGHT_07, cells)
  parse_rw_by_ra(read_csv('2016_07_light_interception.csv'),
                 DataKeys.ROBOT_LIGHT_INTERCEPTION_07, cells)
  parse_rw_by_ra(read_csv('2016_08_light_interception.csv'),
                 DataKeys.ROBOT_LIGHT_INTERCEPTION_08, cells)
  parse_rw_by_ra(read_csv('2016_09_light_interception.csv'),
                 DataKeys.ROBOT_LIGHT_INTERCEPTION_09, cells)
  add_gps_to_cells(read_csv('2016_all_BAP_gps_coords.csv'), cells)
  add_synthetic_values(cells, [DataKeys.HARVEST_SF16h_HGT1_120,
                               DataKeys.HARVEST_SF16h_HGT2_120,
                               DataKeys.HARVEST_SF16h_HGT3_120],
      DataKeys.SYNTHETIC_HARVEST_SF16h_HGT_120_MEAN,
      DataKeys.SYNTHETIC_HARVEST_SF16h_HGT_120_STD)
  add_synthetic_values(cells, [DataKeys.HARVEST_SF16h_PAN1_120,
                               DataKeys.HARVEST_SF16h_PAN2_120,
                               DataKeys.HARVEST_SF16h_PAN3_120],
      DataKeys.SYNTHETIC_HARVEST_SF16h_PAN_120_MEAN,
      DataKeys.SYNTHETIC_HARVEST_SF16h_PAN_120_STD)

  sorted_cells = cells.sorted()
  write_csv(OUTPUT_FILENAME, sorted_cells)

  # TODO all but one of the robot data has merge mismatch values.
  #      Try both averaging them and handling them deterministically seperately
  #      (low value = ROBOT_VALUE1, high value = ROBOT_VALUE2)? Sorting like
  #      thus will intermix the robot values so VALUE1 values might not always
  #      refer to same part of cell. Hmm. That seems bad. So no sorting if
  #      treating as two separate values?
  merged_cells = []
  for cell in sorted_cells:
    row, column = cell.get_coordinates()
    if row % ROWS_IN_CELL != 1:
      continue

    merged_cell = Cell(row, column)
    for i in range(ROWS_IN_CELL):
      cells.get(row + i, column).add_all_data_to_cell(merged_cell)
    merged_cells.append(merged_cell)

  if ROWS_IN_CELL * len(merged_cells) != len(sorted_cells):
    raise Exception('Unexpected number of merged cells:',
                    len(merged_cells), len(sorted_cells))
  write_csv(MERGED_OUTPUT_FILENAME, merged_cells)


if __name__ == '__main__':
    main()
