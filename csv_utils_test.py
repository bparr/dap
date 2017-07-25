#!/usr/bin/env python3

# Usage: ./csv_utils_test.py

import csv_utils
import tempfile
import unittest


class TestCsvUtils(unittest.TestCase):
  def setUp(self):
    self._temp_files = []

  def tearDown(self):
    for f in self._temp_files:
      f.close()

  # This function appends the '\n' char, so no need to include in lines.
  def _create_test_csv(self, lines):
    named_file = tempfile.NamedTemporaryFile(delete=False)
    with open(named_file.name, 'w') as f:
      for line in lines:
        f.write(line + '\n')

    self._temp_files.append(f)
    return f.name

  def test_average_mismatch(self):
    self.assertEqual(42.0, csv_utils.average_mismatch('42'))
    self.assertEqual(42.0, csv_utils.average_mismatch('42.0'))
    self.assertEqual(42.0, csv_utils.average_mismatch('40.5 && 43.5'))
    self.assertEqual(42.0, csv_utils.average_mismatch('40.5 && 42 && 43.5'))

  def test_append_value(self):
    self.assertEqual(' && new', csv_utils.append_value('', 'new'))
    self.assertEqual('old && new', csv_utils.append_value('old', 'new'))
    self.assertEqual('old1 && old2 && new', csv_utils.append_value(
        'old1 && old2', 'new'))
    self.assertEqual('old', csv_utils.append_value('old', 'old'))
    self.assertEqual('old1 && old2', csv_utils.append_value(
        'old1 && old2', 'old1'))
    self.assertEqual('old1 && old2', csv_utils.append_value(
        'old1 && old2', 'old2'))

  def test_read_csv_good_case(self):
    file_path = self._create_test_csv([
      'label1,label2,label3', '1,2,3', '4,5,6'])
    self.assertListEqual(
        [['label1', 'label2', 'label3'], ['1', '2', '3'], ['4', '5', '6']],
        csv_utils.read_csv(file_path))

  def test_read_csv_raises_exception_if_duplicate_label(self):
    # label1 is repeated twice.
    file_path = self._create_test_csv([
      'label1,label2,label1', '1,2,3', '4,5,6'])
    self.assertRaises(Exception, csv_utils.read_csv, file_path)

  def test_read_csv_raises_exception_if_mismatched_lengths(self):
    # Last row only has two entries.
    file_path = self._create_test_csv([
      'label1,label2,label3', '1,2,3', '4,5'])
    self.assertRaises(Exception, csv_utils.read_csv, file_path)

  def test_read_csv_as_dicts_good_case(self):
    file_path = self._create_test_csv([
      'label1,label2,label3', '1,2,3', '4,5,6'])
    self.assertListEqual(
        [{'label1': '1', 'label2': '2', 'label3': '3'},
         {'label1': '4', 'label2': '5', 'label3': '6'}],
        csv_utils.read_csv_as_dicts(file_path))

  def test_read_csv_as_dicts_raises_exception_if_duplicate_label(self):
    # label1 is repeated twice.
    file_path = self._create_test_csv([
      'label1,label2,label1', '1,2,3', '4,5,6'])
    self.assertRaises(Exception, csv_utils.read_csv_as_dicts, file_path)

  def test_read_csv_as_dicts_raises_exception_if_mismatched_lengths(self):
    # Last row only has two entries.
    file_path = self._create_test_csv([
      'label1,label2,label3', '1,2,3', '4,5'])
    self.assertRaises(Exception, csv_utils.read_csv_as_dicts, file_path)


if __name__ == '__main__':
  unittest.main()
