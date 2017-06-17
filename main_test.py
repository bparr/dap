#!/usr/bin/env python3

import csv
import main
import os
import unittest


# TODO check uses of main are small.

def read_input_file(filename):
  with open(os.path.join(main.DATA_DIRECTORY, filename), 'r') as input_file:
    return list(csv.DictReader(input_file))

def read_output_file():
  output = {}
  with open(main.OUTPUT_FILENAME, 'r') as output_file:
    for d in csv.DictReader(output_file):
      row = int(d[main.Cell.ROW_DATA_NAME])
      column = int(d[main.Cell.COLUMN_DATA_NAME])
      if column in output.setdefault(row, {}):
        raise Exception('Duplicate row/column: ', row, column)
      output[row][column] = d

  return output


OUTPUT_CONTENTS = read_output_file()

def get_actual_value(row, column, key, has_fill_rows=True):
  row, column = main.parse_coordinates(row, column)
  if not has_fill_rows:
    row += main.NO_FILL_ROW_OFFSET
  # Handle both a string key and a DataKeys key.
  key = main.DataKeys(key).value
  return OUTPUT_CONTENTS.get(row, {}).get(column, {}).get(key, '')


class TestOutput(unittest.TestCase):
  def _assert_values_equal(self, expected, actual):
    expected = '' if expected in main.EMPTY_VALUES else expected
    self.assertIn(expected, actual.split(main.MISMATCH_DELIMETER))

  # Test values in an input file were correctly copied over to output file.
  def _assert_input(self, filename, data_key, first_column_key=' '):
    dict_lines = read_input_file(filename)
    for d in dict_lines:
      for k, v in d.items():
        if d[first_column_key] in main.EMPTY_VALUES or k in main.EMPTY_VALUES:
          continue

        self._assert_values_equal(v, get_actual_value(
            d[first_column_key], k, data_key.value))


  def test_parse_coordinates(self):
    self.assertEqual((1,2), main.parse_coordinates(1, 2))
    self.assertEqual((12, 32), main.parse_coordinates(12, 32))
    self.assertEqual((12, 32), main.parse_coordinates('Rw12', 'Ra32'))
    self.assertEqual((12, 32), main.parse_coordinates('RW12', 'RA32'))
    self.assertEqual((12, 32), main.parse_coordinates('rw12', 'ra32'))

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
    dict_lines = read_input_file('BAP16_HarvestData.csv')
    for d in dict_lines:
      for k, v in d.items():
        if k == 'RA1' or k == 'RW':
          continue
        if k == 'Plot ID':
          k = main.DataKeys.PLOT_ID

        self._assert_values_equal(v, get_actual_value(
            d['RW'], d['RA1'], k, has_fill_rows=False))

  def test_plot_map_files(self):
    self._assert_input('BAP16_PlotMap_Plant_IDs.csv',
                       main.DataKeys.PLANT_ID, first_column_key='')
    self._assert_input('BAP16_PlotMap_Plot_IDs.csv',
                       main.DataKeys.PLOT_ID, first_column_key='')

  def test_BAP16_PlotPlan_Plot_IDs(self):
    dict_lines = read_input_file('BAP16_PlotPlan_Plot_IDs.csv')
    for d in dict_lines:
      self._assert_values_equal(d['Plot ID'], get_actual_value(
          d['RW'], d['RA1'], main.DataKeys.PLOT_ID, has_fill_rows=False))
      self._assert_values_equal(d['PI'], get_actual_value(
          d['RW'], d['RA1'], main.DataKeys.PLANT_ID, has_fill_rows=False))
      self._assert_values_equal(d['XofY'], get_actual_value(
          d['RW'], d['RA1'], main.DataKeys.X_OF_Y, has_fill_rows=False))

  def test_BAP16_PlotPlan_Plot_IDs_Tags(self):
    dict_lines = read_input_file('BAP16_PlotPlan_Plot_IDs_Tags.csv')
    for d in dict_lines:
      self._assert_values_equal(d['PlotID'], get_actual_value(
          d['Rw'], d['Ra'], main.DataKeys.PLOT_ID, has_fill_rows=False))
      self._assert_values_equal(d['PI'], get_actual_value(
          d['Rw'], d['Ra'], main.DataKeys.PLANT_ID, has_fill_rows=False))
      self._assert_values_equal(d['XofY'], get_actual_value(
          d['Rw'], d['Ra'], main.DataKeys.X_OF_Y, has_fill_rows=False))
      self._assert_values_equal(d['TAG'], get_actual_value(
          d['Rw'], d['Ra'], main.DataKeys.PLOT_PLAN_TAG, has_fill_rows=False))
      self._assert_values_equal(d['Con'], get_actual_value(
          d['Rw'], d['Ra'], main.DataKeys.PLOT_PLAN_CON, has_fill_rows=False))
      self._assert_values_equal(d['Barcode'], get_actual_value(
          d['Rw'], d['Ra'], main.DataKeys.PLOT_PLAN_BARCODE,
          has_fill_rows=False))
      self._assert_values_equal(d['End'], get_actual_value(
          d['Rw'], d['Ra'], main.DataKeys.PLOT_PLAN_END, has_fill_rows=False))

  def test_PanelAccessions_BAP(self):
    dict_lines = read_input_file('PanelAccessions-BAP.csv')
    accessions = dict((x['Taxa'], x) for x in dict_lines)
    for output_columns in OUTPUT_CONTENTS.values():
      for output_line in output_columns.values():
        plant_id = output_line[main.DataKeys.PLANT_ID.value]
        if plant_id not in accessions:
          continue

        accession = accessions[plant_id]
        row = output_line[main.DataKeys.ROW.value]
        column = output_line[main.DataKeys.COLUMN.value]
        self._assert_values_equal(accession['PHOTOPERIOD'], get_actual_value(
            row, column, main.DataKeys.ACCESSION_PHOTOPERIOD))
        self._assert_values_equal(accession['Type'], get_actual_value(
            row, column, main.DataKeys.ACCESSION_TYPE))
        self._assert_values_equal(accession['Origin'], get_actual_value(
            row, column, main.DataKeys.ACCESSION_ORIGIN))
        self._assert_values_equal(accession['Race'], get_actual_value(
            row, column, main.DataKeys.ACCESSION_RACE))


if __name__ == '__main__':
    unittest.main()
