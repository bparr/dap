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

  def test_converte_to_float_or_missing_good_case_multiple(self):
    samples = [{'foo': '42.1', 'bar': '123'},
               {'foo': '43.4', 'bar': '124'}]
    dataset.convert_to_float_or_missing(samples, ['foo', 'bar'])
    self.assertEqual([{'foo': 42.1, 'bar': 123}, {'foo': 43.4, 'bar': 124}],
                     samples)

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
    self.input_labels = ['label1', 'label2']
    self.samples = [
      {'label1': 1.0, 'label2': 2.0, 'output': 3.0},
      {'label1': 4.0, 'label2': 5.0, 'output': 6.0},
      {'label1': 7.0, 'label2': 8.0, 'output': 9.0},
    ]
    self.samples_with_strings =[
      {'label1': 1.0, 'label2': 'foo', 'output': 3.0},
      {'label1': 4.0, 'label2': 'foo', 'output': 6.0},
      {'label1': 7.0, 'label2': '', 'output': 9.0},
    ]
    self.dataset = dataset.Dataset(
        self.samples, self.input_labels, 'output_generators')
    self.dataset_with_strings = dataset.Dataset(
        self.samples_with_strings, self.input_labels, 'output_generators')

  def test_generate_simple(self):
    output_generator = lambda x: 10.0 * x['output']
    # TODO actually test when shuffle is True?
    X_labels, X, y = self.dataset.generate(output_generator, shuffle=False)
    self.assertListEqual(['label1', 'label2'], X_labels)
    np.testing.assert_array_equal([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], X)
    np.testing.assert_array_equal([30.0, 60.0, 90.0], y)

  def test_generate_not_affect_if_input_labels_list_changed(self):
    output_generator = lambda x: x
    self.input_labels[0] = 'changed'
    X_labels, _, _ = self.dataset.generate(output_generator, shuffle=False)
    self.assertListEqual(['label1', 'label2'], X_labels)

  def test_get_output_generators(self):
    expected = [('key1', 'value1'), ('key2', 'value2')]
    ds = dataset.Dataset([], [], collections.OrderedDict(expected))
    self.assertListEqual(expected, list(ds.get_output_generators()))

if __name__ == '__main__':
    unittest.main()