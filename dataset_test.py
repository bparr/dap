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
    pass # TODO keep?

  def test_generate(self):
    samples = [{'label1': 1.0, 'label2': 2.0, 'output': 3.0}]
    input_labels = ['label1', 'label2']
    ds = dataset.Dataset(samples, input_labels, 'output_generators')

    output_generator = lambda x: 10.0 * x['output']
    # TODO actually test when shuffle is True?
    X_labels, X, y = ds.generate(output_generator, shuffle=False)
    self.assertListEqual(['label1', 'label2'], X_labels)
    np.testing.assert_array_equal([[1.0, 2.0]], X)
    np.testing.assert_array_equal([30.0], y)

  def test_get_output_generators(self):
    expected = [('key1', 'value1'), ('key2', 'value2')]
    ds = dataset.Dataset([], [], collections.OrderedDict(expected))
    self.assertListEqual(expected, list(ds.get_output_generators()))

if __name__ == '__main__':
    unittest.main()
