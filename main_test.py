#!/usr/bin/env python3

import csv
import unittest
import main


def _read_csv_file(filepath):
  with open(filepath, 'r') as csv_file:
    return list(csv.DictReader(csv_file))

def _read_output_file(filepath):
  dict_lines = _read_csv_file(filepath)
  output = {}
  for d in dict_lines:
    row = int(d[main.Cell.ROW_DATA_NAME])
    column = int(d[main.Cell.COLUMN_DATA_NAME])
    if column in output.setdefault(row, {}):
      raise Exception('Duplicate row/column: ', row, column)
    output[row][column] = d

  return output


OUTPUT_FILE = '2016.csv'
OUTPUT_CONTENTS = _read_output_file(OUTPUT_FILE)

def get_actual_value(row, column, key):
  row, column = main.parse_coordinates(row, column)
  return OUTPUT_CONTENTS.get(row, {}).get(column, {}).get(key, '')


class TestOutput(unittest.TestCase):
  def test_2016_07_13_14_Leaf_Necrosis(self):
    dict_lines = _read_csv_file('2016/2016_07_13-14_Leaf_Necrosis.csv')
    for d in dict_lines:
      for k, v in d.items():
        if d[' '] in main.EMPTY_VALUES or k in main.EMPTY_VALUES:
          continue
        v = '' if v in main.EMPTY_VALUES else v
        row, column = main.parse_coordinates(d[' '], k)
        self.assertEqual(v, get_actual_value(
            d[' '], k, main.DataKeys.LEAF_NECROSIS_07.value))

if __name__ == '__main__':
    unittest.main()
