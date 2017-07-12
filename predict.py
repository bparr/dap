#!/usr/bin/env python3

"""
Parse full csv and predict harvest data.

Usage:
  ./predict.py > predict.2016.out
  ./predict.py -d 2014 > predict.2014.out

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


RANDOM_SEED = 10611

# TODO tune??
#MISSING_VALUE = np.nan
MISSING_VALUE = -1  # Disables Imputer.


def is_missing(value):
  # Unfortunately np.nan == np.nan is False, so check both isnan and equality.
  return np.isnan(value) or value == MISSING_VALUE


def convert_to_float_or_missing(samples, labels):
  for sample in samples:
    for label in labels:
      v = sample[label]
      if v == '':
        sample[label] = MISSING_VALUE
        continue

      sample[label] = np.mean([float(x) for x in v.split(MISMATCH_DELIMETER)])
      if is_missing(sample[label]):
        raise Exception('Bad value:', v)


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
      output = output_generator(sample)
      if is_missing(output):
        # Ignore samples with missing output value.
        continue

      X.append([sample[x] for x in self._input_labels])
      y.append(output)

    return np.array(X), np.array(y)

  def get_input_labels(self):
    return self._input_labels

  def get_output_generators(self):
    return self._output_generators.items()



def new2014Dataset(**kwargs):
  samples = csv_utils.read_csv_as_dicts('2014/2014_Pheotypic_Data_FileS2.csv')

  ADF = 'ADF (% DM)'
  NDF = 'NDF (% DM)'
  NFC = 'NFC (% DM)'
  LIGNIN = 'Lignin (% DM)'
  DRY_WEIGHT = 'Dry weight (kg)'

  input_labels = (
      'Anthesis date (days)',
      'Harvest date (days)',
      'Total fresh weight (kg)',
      'Brix (maturity)',
      'Brix (milk)',
      'Stalk height (cm)',
      # Including dry weight greatly increases predictive ability.
      #'Dry weight (kg)',
      #'Dry tons per acre',
  )

  convert_to_float_or_missing(samples, list(input_labels) + [
      ADF, NDF, NFC, LIGNIN, DRY_WEIGHT])

  output_generators = collections.OrderedDict([
      ('adf', lambda sample: get_weight(sample, DRY_WEIGHT, ADF)),
      ('ndf', lambda sample: get_weight(sample, DRY_WEIGHT, NDF)),
      ('nfc', lambda sample: get_weight(sample, DRY_WEIGHT, NFC)),
      ('lignin', lambda sample: get_weight(sample, DRY_WEIGHT, LIGNIN)),
      ('c6', lambda sample: get_weight(sample, DRY_WEIGHT, ADF, minus=LIGNIN)),
      ('c5', lambda sample: get_weight(sample, DRY_WEIGHT, NDF, minus=ADF)),
  ])
  return Dataset(samples, input_labels, output_generators)


def filter_2016_labels(data_key_starts_with):
  return [x.value for x in DataKeys if x.name.startswith(data_key_starts_with)]

def create_2016_output_generator(key):
  return lambda sample: sample[key]

# TODO remove all args here and in new2014Dataset.
def new2016Dataset(include_accession=False, **kwargs):
  samples = csv_utils.read_csv_as_dicts('2016.merged.csv')
  convert_to_float_or_missing(samples, filter_2016_labels((
      'HARVEST_', 'COMPOSITION_', 'ROBOT_', 'SYNTHETIC_', 'GPS_')) +
      [DataKeys.ROW.value, DataKeys.COLUMN.value])

  # TODO what to include? Allow multiple subsets through commandline?
  input_data_keys_starts_with = [
      'HARVEST_',
      'ROBOT_',
      'SYNTHETIC_',
      #'GPS_',
  ]
  if include_accession:
    input_data_keys_starts_with += ['ACCESSION_']

  input_labels = filter_2016_labels(tuple(input_data_keys_starts_with))
  output_labels = filter_2016_labels('COMPOSITION_')

  weight_datakeys = (
      DataKeys.COMPOSITION_LIGNIN,
      DataKeys.COMPOSITION_CELLULOSE,
      DataKeys.COMPOSITION_HEMICELLULOSE)

  def get_weight_generator(data_key):
    DRY_MATTER = DataKeys.COMPOSITION_DRY_MATTER.value
    generator = lambda sample: get_weight(sample, DRY_MATTER, data_key.value)
    return ('abs.' + data_key.value, generator)

  output_generators = collections.OrderedDict(sorted(
    [(x, create_2016_output_generator(x)) for x in output_labels] +
    [get_weight_generator(x) for x in weight_datakeys]
  ))

  return Dataset(samples, input_labels, output_generators)



# Returns result of percent DM value multiplied by dry weight.
# If given, the minus label's value is subtracted from label's value.
def get_weight(sample, dry_weight_label, label, minus=None):
  value = sample[label]
  minus_value = 0.0 if minus is None else sample[minus]
  dry_weight = sample[dry_weight_label]
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

  dataset = (DATASET_FACTORIES[args.dataset])(
      include_accession=args.write_dataviews_only)

  if args.write_dataviews_only:
    for output_name, output_generator in dataset.get_output_generators():
      X, y = dataset.generate(output_generator, shuffle=False)
      write_csv(os.path.join('dataviews', args.dataset, output_name + '.csv'),
                dataset.get_input_labels(), X, output_name, y)
    return

  regressors = collections.OrderedDict([
      ('random forests', lambda: RandomForestRegressor(n_estimators=100)),
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
