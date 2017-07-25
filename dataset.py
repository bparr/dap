#!/usr/bin/env python3

"""
Dataset classes and helper functions.
"""

import csv
from csv_utils import average_mismatch
import numpy as np
import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


MISSING_VALUE = -1


def is_missing(value):
  return value == MISSING_VALUE


# For each sample[label], convert to float or MISSING_VALUE.
def convert_to_float_or_missing(samples, labels):
  for sample in samples:
    for label in labels:
      v = sample[label]
      if v == '':
        sample[label] = MISSING_VALUE
        continue

      sample[label] = average_mismatch(v)
      if is_missing(sample[label]):
        raise Exception('Bad value:', v)


# Contains entire dataset, and ways to view it.
class Dataset(object):
  _DICT_VECTORIZER_SEPERATOR = '='

  def __init__(self, samples, input_labels, output_generators):
    self._samples = samples
    self._input_labels = tuple(input_labels)
    self._output_generators = output_generators

    # Generated and verified in self._generate().
    self._vectorized_feature_names = None

  # Use kfold_random_state to make kfold splitting deterministic.
  # kfold_random_state can be None if want to use current numpy random state.
  def generate_views(self, kfold_random_state):
    kf = KFold(n_splits=10, shuffle=True, random_state=kfold_random_state)
    for output_label, output_generator in self._output_generators.items():
      X_labels, X, y = self._generate(output_generator)
      yield output_label, DataView(X_labels, X, output_label, y, kf)

  # Can contain a label multiple times if its values were strings, since
  # DictVectorizer converts those to one-hot vectors.
  # Raises an error if called before generating any views.
  def get_input_labels(self):
    if self._vectorized_feature_names is None:
      raise Exception('Can not call get_input_labels before generating a view.')
    sep = Dataset._DICT_VECTORIZER_SEPERATOR
    return [x.split(sep)[0] for x in self._vectorized_feature_names]

  # Returns: (X_labels, X, y)
  def _generate(self, output_generator):
    X_dicts = []
    y = []
    for sample in self._samples:
      output = output_generator(sample)
      if is_missing(output):
        # Ignore samples with missing output value.
        continue

      X_dicts.append(dict([(x, sample[x]) for x in self._input_labels]))
      y.append(output)

    vectorizer = DictVectorizer(separator=Dataset._DICT_VECTORIZER_SEPERATOR,
                                sort=True, sparse=False)
    X = vectorizer.fit_transform(X_dicts)
    if self._vectorized_feature_names is None:
      self._vectorized_feature_names = vectorizer.get_feature_names()
    if self._vectorized_feature_names != vectorizer.get_feature_names():
      # Equality is currently used to match up feature importance across kfold.
      # This could be removed if store mappings and merge correctly in code.
      raise Exception('Vectorized feature names changed!')

    return list(vectorizer.feature_names_), X, np.array(y)


# A single view of a subset of the data in a a Dataset.
class DataView(object):
  def __init__(self, X_labels, X, y_label, y, kfold):
    self._X_labels = X_labels
    self._X = X
    self._y_label = y_label
    self._y = y
    self._kfold = kfold

  # Returns total number of samples in the view.
  def get_num_samples(self):
    return self._X.shape[0]

  # Returns r2 score for provided predictions.
  def get_r2_score(self, y_pred):
    return r2_score(self._y, y_pred)

  # The predictor argument is a function that takes in a KFoldDataView and
  # outputs y test predictions.
  # TODO(bparr): Add tests for this method.
  def kfold_predict(self, predictor):
    y_pred = []
    for train_indexes, test_indexes in self._kfold.split(self._X):
      kfold_data_view = KFoldDataView(
          list(self._X_labels), np.copy(self._X[train_indexes]),
          np.copy(self._X[test_indexes]), np.copy(self._y[train_indexes]))
      y_pred.extend(zip(test_indexes, predictor(kfold_data_view)))

    y_pred_dict = dict(y_pred)
    if len(y_pred_dict) != len(y_pred):
      raise Exception('kfold splitting was bad.')
    return [y_pred_dict[i] for i in range(len(self._X))]

  # Write predictions, as well as actual values.
  # The include_X_labels is a list of other columns to write. Ensure that
  # the DataView has those columns, or else they won't be written.
  def write_predictions(self, file_path, y_pred, include_X_labels):
    filtered = [(i, x) for i, x in enumerate(self._X_labels)
                if x in include_X_labels]
    include_X_indexes = [x for x, _ in filtered]
    with open(file_path, 'w') as f:
      writer = csv.writer(f)
      writer.writerow([x for _, x in filtered] +
                      ['actual_' + self._y_label, 'predicted_' + self._y_label,
                       'prediction_diff'])
      for x_row, y_actual_row, y_pred_row in zip(self._X, self._y, y_pred):
        writer.writerow(list(x_row[include_X_indexes]) +
                        [y_actual_row, y_pred_row, y_pred_row - y_actual_row])

  # Currently useful for verifying results against lab's random forest code.
  def write_csv(self, file_path):
    with open(file_path, 'w') as f:
      writer = csv.writer(f)
      labels = self._X_labels + [self._y_label]
      writer.writerow(labels)
      for x_row, y_row in zip(self._X, self._y):
        row = list(x_row) + [y_row]
        if len(row) != len(labels):
          raise Exception('Inconsistent number of entries.')
        writer.writerow(row)


# Enforce not knowing true y_test when making predictions by not providing it.
class KFoldDataView(object):
  def __init__(self, X_labels, X_train, X_test, y_train):
    self.X_labels = X_labels
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train

