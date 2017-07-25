#!/usr/bin/env python3

# Usage: ./missing_augment_test.py

import dataset as dataset_lib
import missing_augment
import numpy as np
import unittest

MV = dataset_lib.MISSING_VALUE

X_LABELs = tuple(['label1', 'label2', 'label3']) # Tuple so not mutable.

def create_test_input(X_train, X_test, y_train):
  return dataset_lib.KFoldDataView(
      list(X_LABELs), np.array(X_train), np.array(X_test), np.array(y_train))


class TestOutput(unittest.TestCase):
  def _assert_result(self, result, expected_X_train, expected_X_test,
                     expected_y_train, expected_sample_weight):
    kfold_data_view, sample_weight = result
    self.assertListEqual(list(X_LABELs), kfold_data_view.X_labels)
    np.testing.assert_array_equal(expected_X_train, kfold_data_view.X_train)
    np.testing.assert_array_equal(expected_X_test, kfold_data_view.X_test)
    np.testing.assert_array_equal(expected_y_train, kfold_data_view.y_train)
    np.testing.assert_array_equal(expected_sample_weight, sample_weight)

  def test_augment_if_none_missing(self):
    result = missing_augment.augment(create_test_input(
      [[1, 2, 3], [4, 5, MV]], [[7, 8, 9]], [42, 43]))
    self._assert_result(result, [[1, 2, 3], [4, 5, MV]], [[7, 8, 9]],
                        [42, 43], [1.0, 1.0])

  def test_augment_if_all_same_missing_feature(self):
    result = missing_augment.augment(create_test_input(
      [[1, 2, MV], [4, 5, MV]], [[7, 8, MV]], [42, 43]))
    self._assert_result(result, [[1, 2, MV], [4, 5, MV]], [[7, 8, MV]],
                        [42, 43], [1.0, 1.0])

  def test_augment_augments_simple(self):
    result = missing_augment.augment(create_test_input(
      [[1, 2, 3], [4, 5, MV]], [[7, 8, MV]], [42, 43]))
    self._assert_result(
        result,
        [[1, 2, 3], [1, 2, MV], [4, 5, MV]],
        [[7, 8, MV]],
        [42, 42, 43], [0.75, 0.25, 1.0])


  def test_augment_augments_all_samples(self):
    result = missing_augment.augment(create_test_input(
      [[1, 2, 3], [4, 5, MV]], [[7, 8, MV], [MV, 10, 11]], [42, 43]))
    self._assert_result(
        result,
        [[1, 2, 3], [1, 2, MV], [MV, 2, 3], [4, 5, MV], [MV, 5, MV]],
        [[7, 8, MV], [MV, 10, 11]],
        [42, 42, 42, 43, 43], [4.0 / 6, 1.0 / 6, 1.0 / 6, 0.75, 0.25])


  def test_augment_result_has_no_duplicates(self):
    result = missing_augment.augment(create_test_input(
      [[1, MV, 3], [4, 5, MV]], [[7, MV, MV]], [42, 43]))
    self._assert_result(
        result,
        [[1, MV, 3], [1, MV, MV], [4, 5, MV], [4, MV, MV]],
        [[7, MV, MV]],
        [42, 42, 43, 43], [0.75, 0.25, 0.75, 0.25])


if __name__ == '__main__':
    unittest.main()
