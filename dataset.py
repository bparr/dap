#!/usr/bin/env python3

"""
Dataset classes and helper functions.
"""

import csv
from csv_utils import average_mismatch
import numpy as np
import random
from sklearn.feature_extraction import DictVectorizer


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

    print('INPUTS: ' + ','.join(self._input_labels))

  # output_generator must be one returned by get_output_generators().
  # Returns: (X labels, X, y)
  def generate(self, output_generator, shuffle=True):
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

  # TODO rework code so this method isn't needed?
  def get_output_generators(self):
    return self._output_generators.items()


# TODO document.
class DataView(object):
  def __init__(self, X_labels, X, y_label, y):
    self._X_labels = X_labels
    self._X = X
    self._y_label = y_label
    self._y = y

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

