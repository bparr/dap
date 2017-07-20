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


class Dataset(object):
  _DICT_VECTORIZER_SEPERATOR = '='

  def __init__(self, samples, input_labels, output_generators):
    # Order modified (shuffled) by self.generate().
    self._samples = samples
    self._input_labels = tuple(input_labels)
    self._output_generators = output_generators

    # Generated and verified in self.generate().
    self._vectorized_feature_names = None

  # TODO add tests for this helper function?
  def generate_views(self, shuffle=True):
    for output_label, output_generator in self._output_generators.items():
      X_labels, X, y = self._generate(output_generator, shuffle)
      yield output_label, DataView(X_labels, X, output_label, y)

  # Returns: (X labels, X, y)
  def _generate(self, output_generator, shuffle=True):
    if shuffle:
      random.shuffle(self._samples)

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

  # Can contain a label multiple times if its values were strings, since
  # DictVectorizer converts those to one-hot vectors.
  # Raises an error if called before self.generate() is called.
  def get_input_labels(self):
    if self._vectorized_feature_names is None:
      raise Exception('Can not call get_input_labels before generate.')
    sep = Dataset._DICT_VECTORIZER_SEPERATOR
    return [x.split(sep)[0] for x in self._vectorized_feature_names]


# TODO document.
class DataView(object):
  def __init__(self, X_labels, X, y_label, y):
    self._X_labels = X_labels
    self._X = X
    self._y_label = y_label
    self._y = y

  # TODO add test.
  def get_num_samples(self):
    return self._X.shape[0]

  # TODO add test.
  def get_r2_score(self, y_pred):
    return r2_score(self._y, y_pred)

  # TODO add tests.
  def kfold_predict(self, regressor_generator):
    y_pred = []

    kf = KFold(n_splits=10)
    regressors = []
    for train_indexes, test_indexes in kf.split(self._X):
      X_train, X_test = self._X[train_indexes], self._X[test_indexes]
      y_train, y_test = self._y[train_indexes], self._y[test_indexes]

      # TODO reconsider using Imputer?
      regressor = regressor_generator().fit(X_train, y_train)
      y_pred.extend(zip(test_indexes, regressor.predict(X_test)))
      regressors.append(regressor)

    y_pred_dict = dict(y_pred)
    if len(y_pred_dict) != len(y_pred):
      raise Exception('kfold splitting was bad.')
    return [y_pred_dict[i] for i in range(len(self._X))], regressors

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

