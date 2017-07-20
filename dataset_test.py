#!/usr/bin/env python3

# Usage: ./dataset_test.py

import collections
import dataset
import numpy as np
import tempfile
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
      {'label1': 1.0, 'label2': 2.0, 'output': 3.0, 'ignored': 1.0},
      {'label1': 4.0, 'label2': 5.0, 'output': 6.0, 'ignored': 1.0},
      {'label1': 7.0, 'label2': 8.0, 'output': 9.0, 'ignored': 1.0},
    ]
    self.samples_with_strings =[
      {'label1': 1.0, 'label2': 'foo', 'output': 3.0, 'ignored': 1.0},
      {'label1': 4.0, 'label2': 'foo', 'output': 6.0, 'ignored': 1.0},
      {'label1': 7.0, 'label2': '', 'output': 9.0, 'ignored': 1.0},
    ]
    self.dataset = dataset.Dataset(
        self.samples, self.input_labels, 'output_generators')
    self.dataset_with_strings = dataset.Dataset(
        self.samples_with_strings, self.input_labels, 'output_generators')

  def test_generate_simple(self):
    output_generator = lambda x: 10.0 * x['output']
    # TODO actually test when shuffle is True?
    X_labels, X, y = self.dataset._generate(output_generator, shuffle=False)
    self.assertListEqual(['label1', 'label2'], X_labels)
    np.testing.assert_array_equal([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], X)
    np.testing.assert_array_equal([30.0, 60.0, 90.0], y)

  def test_generate_not_affect_if_input_labels_list_changed(self):
    output_generator = lambda x: x
    self.input_labels[0] = 'changed'
    X_labels, _, _ = self.dataset._generate(output_generator, shuffle=False)
    self.assertListEqual(['label1', 'label2'], X_labels)

  def test_generate_ignores_missing_outputs(self):
    output_generator = lambda x: -1 if x['output'] > 3.0 else 42.0
    X_labels, X, y = self.dataset._generate(output_generator, shuffle=False)
    self.assertListEqual(['label1', 'label2'], X_labels)
    np.testing.assert_array_equal([[1.0, 2.0]], X)
    np.testing.assert_array_equal([42.0], y)

  def test_generate_vectorizes_strings(self):
    output_generator = lambda x: 10.0 * x['output']
    X_labels, X, y = self.dataset_with_strings._generate(
        output_generator, shuffle=False)
    self.assertListEqual(['label1', 'label2=', 'label2=foo'], X_labels)
    np.testing.assert_array_equal(
        [[1.0, 0.0, 1.0], [4.0, 0.0, 1.0], [7.0, 1.0, 0.0]], X)
    np.testing.assert_array_equal([30.0, 60.0, 90.0], y)

  def test_generate_is_robust_to_changing_results(self):
    output_generator = lambda x: 10.0 * x['output']
    X_labels, X, y = self.dataset_with_strings._generate(
        output_generator, shuffle=False)
    X_labels[0] = 'changed'
    X[0][0] = np.nan
    y[0] = np.nan
    X_labels, X, y = self.dataset_with_strings._generate(
        output_generator, shuffle=False)
    self.assertListEqual(['label1', 'label2=', 'label2=foo'], X_labels)
    np.testing.assert_array_equal(
        [[1.0, 0.0, 1.0], [4.0, 0.0, 1.0], [7.0, 1.0, 0.0]], X)
    np.testing.assert_array_equal([30.0, 60.0, 90.0], y)

  def test_generate_raises_exception_if_vectorization_labels_changed(self):
    output_generator1 = lambda x: 10.0 * x['output']
    self.dataset_with_strings._generate(output_generator1, shuffle=False)
    # With this generator, the empty string value for label2 no longer exists.
    output_generator2 = lambda x: -1 if x['output'] > 3.0 else 42.0
    self.assertRaises(Exception, self.dataset_with_strings._generate,
                      output_generator2, shuffle=False)

  def test_generate_no_raises_exception_if_vectorization_labels_changed(self):
    output_generator1 = lambda x: 10.0 * x['output']
    self.dataset_with_strings._generate(output_generator1, shuffle=False)
    # With this generator, all string values for label2 still exist.
    output_generator2 = lambda x: -1 if x['output'] <= 3.0 else 42.0
    self.dataset_with_strings._generate(output_generator2, shuffle=False)

  def test_get_input_labels(self):
    output_generator = lambda x: 10.0 * x['output']
    self.dataset._generate(output_generator, shuffle=False)
    self.assertEqual(['label1', 'label2'], self.dataset.get_input_labels())

  def test_get_input_labels_with_vectorized_strings(self):
    output_generator = lambda x: 10.0 * x['output']
    self.dataset_with_strings._generate(output_generator, shuffle=False)
    self.assertEqual(['label1', 'label2', 'label2'],
                     self.dataset_with_strings.get_input_labels())


class TestDataView(unittest.TestCase):
  def test_write_csv(self):
    data_view = dataset.DataView(
        ['label1', 'label2'], np.array([[1.0, 2.0], [3.0, 4.0]]),
        'output', np.array([5.0, 6.0]))
    csv_file = tempfile.NamedTemporaryFile(delete=False)
    data_view.write_csv(csv_file.name)

    with open(csv_file.name, 'r') as f:
      lines = f.readlines()
    self.assertListEqual(
        ['label1,label2,output\n', '1.0,2.0,5.0\n', '3.0,4.0,6.0\n'], lines)


if __name__ == '__main__':
    unittest.main()
