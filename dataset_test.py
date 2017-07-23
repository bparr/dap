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
    self.output_generators = collections.OrderedDict([
        ('times10', lambda x: 10.0 * x['output']),
        ('filterGt3', lambda x: -1 if x['output'] > 3.0 else 42.0)])

    self.dataset = dataset.Dataset(
        self.samples, self.input_labels, self.output_generators)
    self.dataset_with_strings = dataset.Dataset(
        self.samples_with_strings, self.input_labels, self.output_generators)

  def test_generate_views(self):
    results = list(self.dataset.generate_views())
    self.assertEqual(2, len(results))
    self.assertListEqual([2, 2], [len(x) for x in results])
    self.assertListEqual(['times10', 'filterGt3'], [x[0] for x in results])

    dv = results[0][1]
    self.assertListEqual(['label1', 'label2'], dv._X_labels)
    np.testing.assert_array_equal([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], dv._X)
    self.assertEqual('times10', dv._y_label)
    np.testing.assert_array_equal([30.0, 60.0, 90.0], dv._y)

    dv = results[1][1]
    self.assertListEqual(['label1', 'label2'], dv._X_labels)
    np.testing.assert_array_equal([[1.0, 2.0]], dv._X)
    self.assertEqual('filterGt3', dv._y_label)
    np.testing.assert_array_equal([42.0], dv._y)

  def test_get_input_labels(self):
    self.dataset._generate(self.output_generators['times10'])
    self.assertEqual(['label1', 'label2'], self.dataset.get_input_labels())

  def test_get_input_labels_with_vectorized_strings(self):
    self.dataset_with_strings._generate(self.output_generators['times10'])
    self.assertEqual(['label1', 'label2', 'label2'],
                     self.dataset_with_strings.get_input_labels())

  def test_generate_simple(self):
    X_labels, X, y = self.dataset._generate(self.output_generators['times10'])
    self.assertListEqual(['label1', 'label2'], X_labels)
    np.testing.assert_array_equal([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], X)
    np.testing.assert_array_equal([30.0, 60.0, 90.0], y)

  def test_generate_not_affect_if_input_labels_list_changed(self):
    self.input_labels[0] = 'changed'
    X_labels, _, _ = self.dataset._generate(self.output_generators['times10'])
    self.assertListEqual(['label1', 'label2'], X_labels)

  def test_generate_ignores_missing_outputs(self):
    X_labels, X, y = self.dataset._generate(self.output_generators['filterGt3'])
    self.assertListEqual(['label1', 'label2'], X_labels)
    np.testing.assert_array_equal([[1.0, 2.0]], X)
    np.testing.assert_array_equal([42.0], y)

  def test_generate_vectorizes_strings(self):
    X_labels, X, y = self.dataset_with_strings._generate(
        self.output_generators['times10'])
    self.assertListEqual(['label1', 'label2=', 'label2=foo'], X_labels)
    np.testing.assert_array_equal(
        [[1.0, 0.0, 1.0], [4.0, 0.0, 1.0], [7.0, 1.0, 0.0]], X)
    np.testing.assert_array_equal([30.0, 60.0, 90.0], y)

  def test_generate_is_robust_to_changing_results(self):
    X_labels, X, y = self.dataset_with_strings._generate(
        self.output_generators['times10'])
    X_labels[0] = 'changed'
    X[0][0] = np.nan
    y[0] = np.nan
    X_labels, X, y = self.dataset_with_strings._generate(
        self.output_generators['times10'])
    self.assertListEqual(['label1', 'label2=', 'label2=foo'], X_labels)
    np.testing.assert_array_equal(
        [[1.0, 0.0, 1.0], [4.0, 0.0, 1.0], [7.0, 1.0, 0.0]], X)
    np.testing.assert_array_equal([30.0, 60.0, 90.0], y)

  def test_generate_raises_exception_if_vectorization_labels_changed(self):
    output_generator1 = lambda x: 10.0 * x['output']
    self.dataset_with_strings._generate(output_generator1)
    # With this generator, the empty string value for label2 no longer exists.
    output_generator2 = lambda x: -1 if x['output'] > 3.0 else 42.0
    self.assertRaises(Exception, self.dataset_with_strings._generate,
                      output_generator2)

  def test_generate_no_raises_exception_if_vectorization_labels_changed(self):
    output_generator1 = lambda x: 10.0 * x['output']
    self.dataset_with_strings._generate(output_generator1)
    # With this generator, all string values for label2 still exist.
    output_generator2 = lambda x: -1 if x['output'] <= 3.0 else 42.0
    self.dataset_with_strings._generate(output_generator2)


class TestDataView(unittest.TestCase):
  def setUp(self):
    self.data_view = dataset.DataView(
        ['label1', 'label2'],
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        'output', np.array([7.0, 8.0, 9.0]))

  def test_get_num_samples(self):
    self.assertEqual(3, self.data_view.get_num_samples())

  def test_get_r2_score(self):
    self.assertEqual(1.0, self.data_view.get_r2_score([7.0, 8.0, 9.0]))
    self.assertEqual(0.0, self.data_view.get_r2_score([8.0, 8.0, 8.0]))
    self.assertEqual(0.75, self.data_view.get_r2_score([7.5, 8.0, 8.5]))

  def test_write_predictions(self):
    csv_file = tempfile.NamedTemporaryFile(delete=False)
    self.data_view.write_predictions(csv_file.name, [10.0, 12.0, 15.0],
                                     ['label2'])
    with open(csv_file.name, 'r') as f:
      lines = f.readlines()
    self.assertListEqual(
        ['label2,actual_output,predicted_output,prediction_diff\n',
         '2.0,7.0,10.0,3.0\n', '4.0,8.0,12.0,4.0\n', '6.0,9.0,15.0,6.0\n'],
        lines)

  def test_write_csv(self):
    csv_file = tempfile.NamedTemporaryFile(delete=False)
    self.data_view.write_csv(csv_file.name)

    with open(csv_file.name, 'r') as f:
      lines = f.readlines()
    self.assertListEqual(
        ['label1,label2,output\n', '1.0,2.0,7.0\n', '3.0,4.0,8.0\n',
         '5.0,6.0,9.0\n'], lines)


class TestKfoldDataView(unittest.TestCase):
  def setUp(self):
    self.kfold_data_view = dataset.KFoldDataView(
        ['label1', 'label2'],
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0]]),
        np.array([7.0, 8.0]))

  def test_get_all_X(self):
    np.testing.assert_array_equal(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        self.kfold_data_view.get_all_X())

  def test_augment_X(self):
    self.kfold_data_view.augment_X('new_label', [10.0, 11.0, 12.0])
    self.assertListEqual(['label1', 'label2', 'new_label'],
                         self.kfold_data_view.X_labels)
    np.testing.assert_array_equal(
        np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 11.0]]),
        self.kfold_data_view.X_train)
    np.testing.assert_array_equal(
        np.array([[5.0, 6.0, 12.0]]),
        self.kfold_data_view.X_test)
    np.testing.assert_array_equal(np.array([7.0, 8.0]),
                                  self.kfold_data_view.y_train)

  def test_augment_X_raises_exception_if_new_data_wrong_size(self):
    self.assertRaises(Exception, self.kfold_data_view.augment_X,
                      'new_label', [10.0, 11.0])
    self.assertRaises(Exception, self.kfold_data_view.augment_X,
                      'new_label', [10.0, 11.0, 12.0, 13.0])

  def test_create_filtered_data_view(self):
    filtered = self.kfold_data_view.create_filtered('label1')
    self.assertListEqual(['label1'], filtered.X_labels)
    np.testing.assert_array_equal(np.array([[1.0], [3.0]]), filtered.X_train)
    np.testing.assert_array_equal(np.array([[5.0]]), filtered.X_test)
    np.testing.assert_array_equal(np.array([7.0, 8.0]), filtered.y_train)


if __name__ == '__main__':
    unittest.main()
