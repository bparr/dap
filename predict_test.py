#!/usr/bin/env python3

# Usage: ./predict_test.py

import predict
import unittest

MV = predict.MISSING_VALUE

class TestOutput(unittest.TestCase):
  def test_generate_augmented_returns_nothing_if_none_missing(self):
    X = [[1, 2], [3, 4]]
    y = [5, 6]

    X_augmented, y_augmented = predict.generate_augmented(X, y)
    self.assertListEqual([], X_augmented)
    self.assertListEqual([], y_augmented)

  def test_generate_augmented_augments_simple(self):
    X = [[1, MV], [3, 4]]
    y = [5, 6]

    X_augmented, y_augmented = predict.generate_augmented(X, y)
    self.assertListEqual([[3, MV]], X_augmented)
    self.assertListEqual([6], y_augmented)

  def test_generate_augmented_augments_all_samples(self):
    X = [[1, MV, 3], [MV, 5, 6], [7, 8, 9]]
    y = [10, 11, 12]

    X_augmented, y_augmented = predict.generate_augmented(X, y)
    self.assertListEqual([
        [MV, MV, 3],
        [MV, MV, 6],
        [MV, 8, 9], [7, MV, 9]
        ], X_augmented)
    self.assertListEqual([10, 11, 12, 12], y_augmented)

if __name__ == '__main__':
    unittest.main()
