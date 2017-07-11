#!/usr/bin/env python3

# Usage: ./merge_data_test.py

import csv
import csv_utils
import merge_data
import numpy as np
import os
import unittest


def read_input_file(filename):
  with open(os.path.join(merge_data.DATA_DIRECTORY, filename), 'r') as f:
    return list(csv.DictReader(f))

def read_output_file(file_path, merged=False):
  output = {}
  with open(file_path, 'r') as output_file:
    for d in csv.DictReader(output_file):
      row_str = d[merge_data.Cell.ROW_DATA_NAME]
      row_list = [row_str]
      if merged:
        row_list = row_str.split(merge_data.MISMATCH_DELIMETER)

      for row in row_list:
        row = int(row)
        column = int(d[merge_data.Cell.COLUMN_DATA_NAME])
        if column in output.setdefault(row, {}):
          raise Exception('Duplicate row/column: ', row, column)
        output[row][column] = d

  return output


OUTPUT_CONTENTS = read_output_file(merge_data.OUTPUT_FILENAME)
MERGED_OUTPUT_CONTENTS = read_output_file(
    merge_data.MERGED_OUTPUT_FILENAME, merged=True)

def get_actual_value(row, column, key, has_fill_rows=True):
  row, column = merge_data.parse_coordinates(row, column)
  if not has_fill_rows:
    row += merge_data.NO_FILL_ROW_OFFSET
  # Handle both a string key and a DataKeys key.
  key = merge_data.DataKeys(key).value
  return OUTPUT_CONTENTS.get(row, {}).get(column, {}).get(key, '')


class TestOutput(unittest.TestCase):
  def _assert_values_equal(self, expected, actual):
    if expected in csv_utils.EMPTY_VALUES:
      return
    actual_subentries = actual.split(merge_data.MISMATCH_DELIMETER)
    for expected_subentry in expected.split(merge_data.MISMATCH_DELIMETER):
      self.assertIn(expected_subentry, actual_subentries)

  # Test values in an input file were correctly copied over to output file.
  def _assert_input(self, filename, data_key, first_column_key=' '):
    dict_lines = read_input_file(filename)
    for d in dict_lines:
      for k, v in d.items():
        if (d[first_column_key] in csv_utils.EMPTY_VALUES or
            k in csv_utils.EMPTY_VALUES):
          continue

        self._assert_values_equal(v, get_actual_value(
            d[first_column_key], k, data_key.value))


  def test_parse_coordinates(self):
    self.assertEqual((1,2), merge_data.parse_coordinates(1, 2))
    self.assertEqual((12, 32), merge_data.parse_coordinates(12, 32))
    self.assertEqual((12, 32), merge_data.parse_coordinates('Rw12', 'Ra32'))
    self.assertEqual((12, 32), merge_data.parse_coordinates('RW12', 'RA32'))
    self.assertEqual((12, 32), merge_data.parse_coordinates('rw12', 'ra32'))

  def test_robot_files(self):
    self._assert_input('2016_07_13-14_Leaf_Necrosis.csv',
                       merge_data.DataKeys.ROBOT_LEAF_NECROSIS_07)
    self._assert_input('2016_07_13-14_vegetation_index.csv',
                       merge_data.DataKeys.ROBOT_VEGETATION_INDEX_07)
    self._assert_input('2016_08_05-08_vegetation_index.csv',
                       merge_data.DataKeys.ROBOT_VEGETATION_INDEX_08)
    self._assert_input('2016_07_13_BAP_Leaf_Area.csv',
                       merge_data.DataKeys.ROBOT_LEAF_AREA_07)
    self._assert_input('2016_07_13_laser_plant_height.csv',
                       merge_data.DataKeys.ROBOT_LASER_PLANT_HEIGHT_07)
    self._assert_input('2016_07_light_interception.csv',
                       merge_data.DataKeys.ROBOT_LIGHT_INTERCEPTION_07)
    self._assert_input('2016_08_light_interception.csv',
                       merge_data.DataKeys.ROBOT_LIGHT_INTERCEPTION_08)
    self._assert_input('2016_09_light_interception.csv',
                       merge_data.DataKeys.ROBOT_LIGHT_INTERCEPTION_09)

  def test_harvest_data(self):
    dict_lines = read_input_file('BAP16_HarvestData.csv')
    for d in dict_lines:
      for k, v in d.items():
        if k == 'RA1' or k == 'RW':
          continue
        if k == 'Plot ID':
          k = merge_data.DataKeys.PLOT_ID

        self._assert_values_equal(v, get_actual_value(
            d['RW'], d['RA1'], k, has_fill_rows=False))

  def test_composition_data(self):
    dict_lines = read_input_file('2016_BAPClemsonGRDBBv2.csv')
    compositions = dict((x['Sample ID'], x) for x in dict_lines)
    tested_plot_ids = set()
    for output_columns in OUTPUT_CONTENTS.values():
      for output_line in output_columns.values():
        plot_id = output_line[merge_data.DataKeys.PLOT_ID.value]
        plot_id = 'SF16' + plot_id[2:]
        if plot_id not in compositions:
          continue

        tested_plot_ids.add(plot_id)
        composition = compositions[plot_id]
        row = output_line[merge_data.DataKeys.ROW.value]
        column = output_line[merge_data.DataKeys.COLUMN.value]
        for k, v in composition.items():
          if k == 'Sample ID':
            continue

          k = k.replace(' ', '_').replace('-', '_')
          self._assert_values_equal(v, get_actual_value(
              row, column, merge_data.DataKeys(k)))

    self.assertSetEqual(tested_plot_ids, set(compositions.keys()))


  def test_plot_map_files(self):
    self._assert_input('BAP16_PlotMap_Plant_IDs.csv',
                       merge_data.DataKeys.PLANT_ID, first_column_key='')
    self._assert_input('BAP16_PlotMap_Plot_IDs.csv',
                       merge_data.DataKeys.PLOT_ID, first_column_key='')

  def test_BAP16_PlotPlan_Plot_IDs(self):
    dict_lines = read_input_file('BAP16_PlotPlan_Plot_IDs.csv')
    for d in dict_lines:
      self._assert_values_equal(d['Plot ID'], get_actual_value(
          d['RW'], d['RA1'], merge_data.DataKeys.PLOT_ID, has_fill_rows=False))
      self._assert_values_equal(d['PI'], get_actual_value(
          d['RW'], d['RA1'], merge_data.DataKeys.PLANT_ID, has_fill_rows=False))
      self._assert_values_equal(d['XofY'], get_actual_value(
          d['RW'], d['RA1'], merge_data.DataKeys.X_OF_Y, has_fill_rows=False))

  def test_BAP16_PlotPlan_Plot_IDs_Tags(self):
    dict_lines = read_input_file('BAP16_PlotPlan_Plot_IDs_Tags.csv')
    for d in dict_lines:
      self._assert_values_equal(d['PlotID'], get_actual_value(
          d['Rw'], d['Ra'], merge_data.DataKeys.PLOT_ID, has_fill_rows=False))
      self._assert_values_equal(d['PI'], get_actual_value(
          d['Rw'], d['Ra'], merge_data.DataKeys.PLANT_ID, has_fill_rows=False))
      self._assert_values_equal(d['XofY'], get_actual_value(
          d['Rw'], d['Ra'], merge_data.DataKeys.X_OF_Y, has_fill_rows=False))
      self._assert_values_equal(d['TAG'], get_actual_value(
          d['Rw'], d['Ra'], merge_data.DataKeys.PLOT_PLAN_TAG,
          has_fill_rows=False))
      self._assert_values_equal(d['Con'], get_actual_value(
          d['Rw'], d['Ra'], merge_data.DataKeys.PLOT_PLAN_CON,
          has_fill_rows=False))
      self._assert_values_equal(d['Barcode'], get_actual_value(
          d['Rw'], d['Ra'], merge_data.DataKeys.PLOT_PLAN_BARCODE,
          has_fill_rows=False))
      self._assert_values_equal(d['End'], get_actual_value(
          d['Rw'], d['Ra'], merge_data.DataKeys.PLOT_PLAN_END,
          has_fill_rows=False))

  def test_PanelAccessions_BAP(self):
    dict_lines = read_input_file('PanelAccessions-BAP.csv')
    accessions = dict((x['Taxa'], x) for x in dict_lines)
    for output_columns in OUTPUT_CONTENTS.values():
      for output_line in output_columns.values():
        plant_id = output_line[merge_data.DataKeys.PLANT_ID.value]
        if plant_id not in accessions:
          continue

        accession = accessions[plant_id]
        row = output_line[merge_data.DataKeys.ROW.value]
        column = output_line[merge_data.DataKeys.COLUMN.value]
        self._assert_values_equal(accession['PHOTOPERIOD'], get_actual_value(
            row, column, merge_data.DataKeys.ACCESSION_PHOTOPERIOD))
        self._assert_values_equal(accession['Type'], get_actual_value(
            row, column, merge_data.DataKeys.ACCESSION_TYPE))
        self._assert_values_equal(accession['Origin'], get_actual_value(
            row, column, merge_data.DataKeys.ACCESSION_ORIGIN))
        self._assert_values_equal(accession['Race'], get_actual_value(
            row, column, merge_data.DataKeys.ACCESSION_RACE))

  def test_GPS_manual_sanity_check(self):
    row = OUTPUT_CONTENTS[5][1]
    self.assertAlmostEquals(340658.5225,
        float(row[merge_data.DataKeys.GPS_EASTINGS.value]))
    self.assertAlmostEqual(3832647.415,
        float(row[merge_data.DataKeys.GPS_NORTHINGS.value]))

  def test_GPS_all(self):
    dict_lines = read_input_file('2016_all_BAP_gps_coords.csv')
    gps = dict((x['Row'], {}) for x in dict_lines)
    for line in dict_lines:
      gps[line['Row']][line['Range']] = line

    for output_columns in OUTPUT_CONTENTS.values():
      for output_line in output_columns.values():
        row = output_line[merge_data.DataKeys.ROW.value]
        column = output_line[merge_data.DataKeys.COLUMN.value]
        gps1 = gps[str(int(row) - 1) + '.5'][column]
        gps2 = gps[row + '.5'][column]

        expected_eastings = (0.5 * (float(gps1['Eastings(UTMzone17N)']) +
                                    float(gps2['Eastings(UTMzone17N)'])))
        actual_eastings = float(
            output_line[merge_data.DataKeys.GPS_EASTINGS.value])
        self.assertAlmostEqual(expected_eastings, actual_eastings)

        expected_northings = (0.5 * (float(gps1['Northings(UTMzone17N)']) +
                                     float(gps2['Northings(UTMzone17N)'])))
        actual_northings = float(
            output_line[merge_data.DataKeys.GPS_NORTHINGS.value])
        self.assertAlmostEqual(expected_northings, actual_northings)

  def test_synthetic_manual_sanity_check(self):
    row = OUTPUT_CONTENTS[7][1]
    self.assertAlmostEquals(74.0,
        float(row[merge_data.DataKeys.SYNTHETIC_SF16h_HGT_120_MEAN.value]))
    self.assertAlmostEquals(5.8878405775518976,
        float(row[merge_data.DataKeys.SYNTHETIC_SF16h_HGT_120_STD.value]))
    self.assertAlmostEquals(29.666666667,
        float(row[merge_data.DataKeys.SYNTHETIC_SF16h_PAN_120_MEAN.value]))
    self.assertAlmostEquals(4.1899350299921778,
        float(row[merge_data.DataKeys.SYNTHETIC_SF16h_PAN_120_STD.value]))

  def test_synthetic_all(self):
    dict_lines = read_input_file('BAP16_HarvestData.csv')
    for d in dict_lines:
      heights = [d['SF16h_HGT' + str(x) + '_120'] for x in range(1, 4)]
      heights = [float(x) for x in heights if x not in csv_utils.EMPTY_VALUES]
      self.assertTrue(len(heights) == 0 or len(heights) == 3)

      if len(heights) > 0:
        self._assert_values_equal(str(np.mean(heights)), get_actual_value(
            d['RW'], d['RA1'], merge_data.DataKeys.SYNTHETIC_SF16h_HGT_120_MEAN,
            has_fill_rows=False))
        self._assert_values_equal(str(np.std(heights)), get_actual_value(
            d['RW'], d['RA1'], merge_data.DataKeys.SYNTHETIC_SF16h_HGT_120_STD,
            has_fill_rows=False))

      pans = [d['SF16h_PAN' + str(x) + '_120'] for x in range(1, 4)]
      pans = [float(x) for x in pans if x not in csv_utils.EMPTY_VALUES]
      self.assertTrue(len(pans) == 0 or len(pans) == 3)

      if len(pans) > 0:
        self._assert_values_equal(str(np.mean(pans)), get_actual_value(
            d['RW'], d['RA1'], merge_data.DataKeys.SYNTHETIC_SF16h_PAN_120_MEAN,
            has_fill_rows=False))
        self._assert_values_equal(str(np.std(pans)), get_actual_value(
            d['RW'], d['RA1'], merge_data.DataKeys.SYNTHETIC_SF16h_PAN_120_STD,
            has_fill_rows=False))

  def test_mergeContentsIsSupersetOfNonMergedOutput(self):
    for output_columns in OUTPUT_CONTENTS.values():
      for output_line in output_columns.values():
        row = int(output_line[merge_data.DataKeys.ROW.value])
        column = int(output_line[merge_data.DataKeys.COLUMN.value])
        merged_output_line = MERGED_OUTPUT_CONTENTS[row][column]
        for k, v in output_line.items():
          self._assert_values_equal(v, merged_output_line[k])

if __name__ == '__main__':
    unittest.main()
