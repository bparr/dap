#!/usr/bin/env python3

import csv
import unittest
import main


# TODO check uses of main are small.

def read_csv_file(filepath):
  with open(filepath, 'r') as csv_file:
    return list(csv.DictReader(csv_file))

def read_output_file(filepath):
  dict_lines = read_csv_file(filepath)
  output = {}
  for d in dict_lines:
    row = int(d[main.Cell.ROW_DATA_NAME])
    column = int(d[main.Cell.COLUMN_DATA_NAME])
    if column in output.setdefault(row, {}):
      raise Exception('Duplicate row/column: ', row, column)
    output[row][column] = d

  return output


OUTPUT_FILE = '2016.csv'
OUTPUT_CONTENTS = read_output_file(OUTPUT_FILE)

def get_actual_value(row, column, key, include_fill_rows=True):
  row, column = main.parse_coordinates(row, column)
  if not include_fill_rows:
    row += main.NO_FILL_ROW_OFFSET
  return OUTPUT_CONTENTS.get(row, {}).get(column, {}).get(key, '')


class TestOutput(unittest.TestCase):
  def _assert_values_equal(self, expected, actual):
    expected = '' if expected in main.EMPTY_VALUES else expected
    self.assertIn(expected, actual.split(main.MISMATCH_DELIMETER))

  # Test values in an input file were correctly copied over to output file.
  def _assert_input(self, filename, data_key):
    dict_lines = read_csv_file('2016/' + filename)
    for d in dict_lines:
      for k, v in d.items():
        if d[' '] in main.EMPTY_VALUES or k in main.EMPTY_VALUES:
          continue

        self._assert_values_equal(v, get_actual_value(
            d[' '], k, data_key.value))


  def test_robot_files(self):
    self._assert_input('2016_07_13-14_Leaf_Necrosis.csv',
                       main.DataKeys.LEAF_NECROSIS_07)
    self._assert_input('2016_07_13-14_vegetation_index.csv',
                       main.DataKeys.VEGETATION_INDEX_07)
    self._assert_input('2016_08_05-08_vegetation_index.csv',
                       main.DataKeys.VEGETATION_INDEX_08)
    self._assert_input('2016_07_13_BAP_Leaf_Area.csv',
                       main.DataKeys.LEAF_AREA_07)
    self._assert_input('2016_07_13_laser_plant_height.csv',
                       main.DataKeys.LASER_PLANT_HEIGHT_07)
    self._assert_input('2016_07_light_interception.csv',
                       main.DataKeys.LIGHT_INTERCEPTION_07)
    self._assert_input('2016_08_light_interception.csv',
                       main.DataKeys.LIGHT_INTERCEPTION_08)
    self._assert_input('2016_09_light_interception.csv',
                       main.DataKeys.LIGHT_INTERCEPTION_09)

  def test_harvest_data(self):
    dict_lines = read_csv_file('2016/BAP16_HarvestData.csv')
    for d in dict_lines:
      for k, v in d.items():
        if k == 'RA1' or k == 'RW':
          continue
        if k == 'Plot ID':
          k = 'plot_id'

        self._assert_values_equal(v, get_actual_value(
            d['RW'], d['RA1'], k, include_fill_rows=False))


if __name__ == '__main__':
    unittest.main()
