#!/usr/bin/env python3

# Usage: ./dataset_test.py

import collections
import dataset
import numpy as np
import unittest


class TestHelperFunctions(unittest.TestCase):
  def test_is_missing(self):
    self.assertTrue(dataset.is_missing(-1))

    self.assertFalse(dataset.is_missing(1))
    self.assertFalse(dataset.is_missing(np.nan))
    self.assertFalse(dataset.is_missing('foo'))
    self.assertFalse(dataset.is_missing(None))

  def test_converte_to_float_or_missing_good_case(self):
    samples = [{'foo': '42.1', 'untouched': '123'}]
    dataset.convert_to_float_or_missing(samples, ['foo'])
    self.assertEqual([{'foo': 42.1, 'untouched': '123'}], samples)

  def test_converte_to_float_or_missing_empty_case(self):
    samples = [{'foo': '', 'untouched': '123'}]
    dataset.convert_to_float_or_missing(samples, ['foo'])
    self.assertEqual([{'foo': -1, 'untouched': '123'}], samples)

  def test_converte_to_float_or_missing_multivalue_case(self):
    samples = [{'foo': '40.0 && 44.0', 'untouched': '123'}]
    dataset.convert_to_float_or_missing(samples, ['foo'])
    self.assertEqual([{'foo': 42.0, 'untouched': '123'}], samples)

  def test_converte_to_float_or_missing_raises_exception_if_conflict(self):
    samples = [{'foo': '-1', 'untouched': '123'}]
    self.assertRaises(Exception, dataset.convert_to_float_or_missing,
                      samples, ['foo'])


class TestDataset(unittest.TestCase):
  def setUp(self):
    self.samples = []
    self.input_labels = ['input_label1', 'input_label2']
    self.output_generators = collections.OrderedDict([
      ('key1', 'value1'), ('key2', 'value2')])
    self.dataset = dataset.Dataset(self.samples, self.input_labels,
                                   self.output_generators)

  def test_get_output_generators(self):
    self.assertListEqual([('key1', 'value1'), ('key2', 'value2')],
                         list(self.dataset.get_output_generators()))

if __name__ == '__main__':
    unittest.main()
