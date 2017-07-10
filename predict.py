#!/usr/bin/env python3

"""
Parse full csv and predict harvest data.

Usage:
  ./predict.py > predict.out

"""
# TODO tests?

import argparse
import collections
import csv
import csv_utils
from merge_data import DataKeys, MISMATCH_DELIMETER
import numpy as np
import os
import random
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


# What percent of the whole dataset to use as the training set.
# TODO split test set into validation and test sets!?
TRAINING_SIZE = 0.8


def convert_column_to_number(samples, column_label):
  values = sorted(set([x[column_label] for x in samples]))
  for sample in samples:
    sample[column_label] = values.index(sample[column_label])


class Dataset(object):
  def __init__(self, samples, input_labels, output_generators):
    # Order modified (shuffled) by self.generate().
    self._samples = samples
    self._input_labels = tuple(input_labels)
    self._output_generators = output_generators

    print('Input labels:,', ', '.join(self._input_labels))

  # output_generator must be one returned by get_output_generators().
  def generate(self, output_generator, shuffle=True):
    # TODO double check ok to modify
    if shuffle:
      random.shuffle(self._samples)

    X = []
    y = []
    for sample in self._samples:
      output = float_or_missing(output_generator(sample))
      if is_missing(output):
        # Ignore samples with missing output value.
        continue

      X.append([float_or_missing(sample[x]) for x in self._input_labels])
      y.append(output)

    return np.array(X), np.array(y)

  def get_input_labels(self):
    return self._input_labels

  def get_output_generators(self):
    return self._output_generators.items()



def new2014Dataset():
  samples = csv_utils.read_csv_as_dicts('2014/2014_Pheotypic_Data_FileS2.csv')
  convert_column_to_number(samples, 'Pericarp pigmentation')

  # TODO include pericarp in input_labels? Is it known before harvest?
  input_labels = (
      'Anthesis date (days)',
      'Harvest date (days)',
      'Total fresh weight (kg)',
      'Brix (maturity)',
      'Brix (milk)',
      'Stalk height (cm)',
      # TODO allow using dry weight for predictions? When is it known?
      #'Dry weight (kg)',
      #'Dry tons per acre',
  )

  ADF_LABEL = 'ADF (% DM)'
  NDF_LABEL = 'NDF (% DM)'
  NFC_LABEL = 'NFC (% DM)'
  LIGNIN_LABEL = 'Lignin (% DM)'
  output_generators = collections.OrderedDict([
      ('adf', lambda sample: get_weight(sample, ADF_LABEL)),
      ('ndf', lambda sample: get_weight(sample, NDF_LABEL)),
      ('nfc', lambda sample: get_weight(sample, NFC_LABEL)),
      ('lignin', lambda sample: get_weight(sample, LIGNIN_LABEL)),
      ('c6', lambda sample: get_weight(sample, ADF_LABEL, minus=LIGNIN_LABEL)),
      ('c5', lambda sample: get_weight(sample, NDF_LABEL, minus=ADF_LABEL)),
  ])
  return Dataset(samples, input_labels, output_generators)


def filter_2016_labels(data_key_starts_with):
  return [x.value for x in DataKeys if x.name.startswith(data_key_starts_with)]

def create_2016_output_generator(key):
  return lambda sample: sample[key]

def new2016Dataset():
  samples = csv_utils.read_csv_as_dicts('2016.merged.csv')
  for label in filter_2016_labels('ACCESSION_'):
    convert_column_to_number(samples, label)

  # TODO what to include? Allow multiple subsets through commandline?
  input_data_keys_starts_with = (
      'ROBOT_',
      'HARVEST_',
      #'GPS_',
      #'ACCESSION_'
  )
  input_labels = filter_2016_labels(input_data_keys_starts_with)
  output_labels = sorted(filter_2016_labels('COMPOSITION_'))
  output_generators = collections.OrderedDict(
    [(x, create_2016_output_generator(x)) for x in output_labels]
  )

  return Dataset(samples, input_labels, output_generators)


RANDOM_SEED = 10611

# TODO is this just for 2014? Seems a bit bleg to be global scope then.
DRY_WEIGHT_LABEL = 'Dry weight (kg)'

# TODO tune??
MISSING_VALUE = np.nan
#MISSING_VALUE = -1  # Disables Imputer.



def float_or_missing(s):
  if isinstance(s, str):
    if not s:
      return MISSING_VALUE
    return np.mean([float(x) for x in s.split(MISMATCH_DELIMETER)])
  return np.float(s)


def is_missing(value):
  # Unfortunately np.nan == np.nan is False, so check both isnan and equality.
  return np.isnan(value) or value == MISSING_VALUE


# Returns result of percent DM value multiplied by dry weight.
# If given, the minus label's value is subtracted from label's value.
def get_weight(sample, label, minus=None):
  value = float_or_missing(sample[label])
  minus_value = 0.0 if minus is None else float_or_missing(sample[minus])
  dry_weight = float_or_missing(sample[DRY_WEIGHT_LABEL])
  if is_missing(value) or is_missing(minus_value) or is_missing(dry_weight):
    return MISSING_VALUE
  return dry_weight * (value - minus_value) / 100.0


def kfold_predict(X, y, regressor_generator):
  y_pred = []

  kf = KFold(n_splits=10)
  for train_indexes, test_indexes in kf.split(X):
    X_train, X_test = X[train_indexes], X[test_indexes]
    y_train, y_test = y[train_indexes], y[test_indexes]

    imp = Imputer()
    # Parser ignores rows with missing y, so no need to impute y.
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    regressor = regressor_generator().fit(X_train, y_train)
    y_pred.extend(zip(test_indexes, regressor.predict(X_test)))

  y_pred = dict(y_pred)
  return [y_pred[i] for i in range(len(X))]


# Output completely preprocessed CSV files.
# Currently useful for verifying results against lab's random forest code.
def write_csv(file_path, input_labels, X, output_label, y):
  with open(file_path, 'w') as f:
    writer = csv.writer(f)
    labels = list(input_labels) + [output_label]
    writer.writerow(labels)
    for x_row, y_row in zip(X, y):
      row = list(x_row) + [y_row]
      if len(row) != len(labels):
        raise Exception('Inconsistent number of entries.')
      writer.writerow(row)


def main():
  global MISSING_VALUE

  DATASET_FACTORIES = {
    '2014': new2014Dataset,
    '2016': new2016Dataset,
  }

  parser = argparse.ArgumentParser(description='Predict harvest data.')
  parser.add_argument('-d', '--dataset', default='2016',
                      choices=list(DATASET_FACTORIES.keys()),
                      help='Which dataset to use.')
  parser.add_argument('--write_dataviews_only', action='store_true',
                      help='No prediction. Just write data views.')
  args = parser.parse_args()

  if args.write_dataviews_only:
    print('Overwriting MISSING_VALUE because writing dataviews!')
    MISSING_VALUE = -1

  dataset = (DATASET_FACTORIES[args.dataset])()

  if args.write_dataviews_only:
    for output_name, output_generator in dataset.get_output_generators():
      X, y = dataset.generate(output_generator, shuffle=False)
      write_csv(os.path.join('dataviews', args.dataset, output_name + '.csv'),
                dataset.get_input_labels(), X, output_name, y)
    return

  regressors = collections.OrderedDict([
      ('random forests', lambda: RandomForestRegressor(n_estimators=100)),
      # TODO tune max_depth.
      ('boosted trees', lambda: GradientBoostingRegressor(max_depth=1)),
  ])

  for regressor_name, regressor_generator in regressors.items():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print('\n\n' + regressor_name)
    print('output_label,num_samples,r2_score')
    for output_name, output_generator in dataset.get_output_generators():
      X, y = dataset.generate(output_generator)
      y_pred = kfold_predict(X, y, regressor_generator)
      print(','.join([output_name, str(X.shape[0]), str(r2_score(y, y_pred))]))



if __name__ == '__main__':
    main()
