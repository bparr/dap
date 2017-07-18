#!/usr/bin/env python3

"""
Parse full csv and predict harvest data.

Usage:
  ./predict_test.py && ./predict.py > predict.2016.out
  ./predict_test.py && ./predict.py -d 2014 > predict.2014.out

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
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ARDRegression, HuberRegressor, LinearRegression, LogisticRegression, LogisticRegressionCV, PassiveAggressiveRegressor, RandomizedLogisticRegression, RANSACRegressor, SGDRegressor, TheilSenRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Imputer
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor


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

    print(','.join(['INPUTS'] + list(self._input_labels)))

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

    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(X_dicts)
    return vectorizer.feature_names_, X, np.array(y)

  def get_output_generators(self):
    return self._output_generators.items()



def new2014Dataset():
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

def new2016Dataset():
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
      'ACCESSION_',
  ]

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


# Generates new samples with input features missing in a way found in X.
# Note this only generates new samples, and not samples already in X.
def generate_augmented(X, y):
  missings = []
  for x_sample in X:
    missings.append(tuple(is_missing(a) for a in x_sample))
  missings_set = set(missings)

  X_augmented = []
  y_augmented = []
  for x_sample, x_missing, y_sample in zip(X, missings, y):
    augmented_missings = set([x_missing])
    for missing in missings_set:
      augmented_missing = tuple((a or b) for a, b in zip(x_missing, missing))
      # TODO disable this? What is the effect on performance?
      if augmented_missing in augmented_missings:
      # Ignore already seen sample augmentations.
        continue

      augmented_missings.add(augmented_missing)
      X_augmented.append(
          [(MISSING_VALUE if b else a) for a, b in zip(x_sample, missing)])
      y_augmented.append(y_sample)

  return X_augmented, y_augmented


def kfold_predict(X, y, regressor_generator):
  y_pred = []

  kf = KFold(n_splits=10)
  for train_indexes, test_indexes in kf.split(X):
    X_train, X_test = X[train_indexes], X[test_indexes]
    y_train, y_test = y[train_indexes], y[test_indexes]

    X_augmented, y_augmented = generate_augmented(X_train, y_train)
    X_train = np.append(X_train, X_augmented, axis=0)
    y_train = np.append(y_train, y_augmented, axis=0)

    imp = Imputer()
    # Parser ignores rows with missing y, so no need to impute y.
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    regressor = regressor_generator().fit(X_train, y_train)
    y_pred.extend(zip(test_indexes, regressor.predict(X_test)))

  y_pred_dict = dict(y_pred)
  if len(y_pred_dict) != len(y_pred):
    raise Exception('kfold splitting was bad.')
  return [y_pred_dict[i] for i in range(len(X))]


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
  parser.add_argument('--rf_only', action='store_true',
                      help='Only fit main random forest predictor.')
  parser.add_argument('--write_dataviews_only', action='store_true',
                      help='No prediction. Just write data views.')
  args = parser.parse_args()

  if args.write_dataviews_only:
    print('Overwriting MISSING_VALUE because writing dataviews!')
    MISSING_VALUE = -1

  dataset = (DATASET_FACTORIES[args.dataset])()

  if args.write_dataviews_only:
    for output_label, output_generator in dataset.get_output_generators():
      X_labels, X, y = dataset.generate(output_generator, shuffle=False)
      write_csv(os.path.join('dataviews', args.dataset, output_label + '.csv'),
                X_labels, X, output_label, y)
    return

  regressors = collections.OrderedDict([
      # Customized.
      # Based on lab code's configuration.
      ('random_forests', lambda: RandomForestRegressor(
          n_estimators=100, max_depth=10, max_features='sqrt',
          min_samples_split=10)),

      # Cross decomposition.
      ('PLSRegression', lambda: PLSRegression()),

      # Ensemble.
      ('AdaBoostRegressor', lambda: AdaBoostRegressor()),
      ('BaggingRegressor', lambda: BaggingRegressor()),
      ('ExtraTreesRegressor', lambda: ExtraTreesRegressor()),
      ('GradientBoostingRegressor', lambda: GradientBoostingRegressor()),
      ('RandomForestRegressor', lambda: RandomForestRegressor()),

      # Gaussian.
      ('GaussianProcessRegressor', lambda: GaussianProcessRegressor()),

      # Isotonic regression.
      # ValueError: X should be a 1d array
      #('IsotonicRegression', lambda: IsotonicRegression()),

      # Kernel ridge.
      ('KernelRidge', lambda: KernelRidge()),

      # Linear.
      # Way too slow.
      #('ARDRegression', lambda: ARDRegression()),
      ('HuberRegressor', lambda: HuberRegressor()),
      ('LinearRegression', lambda: LinearRegression()),
      # ValueError: Unknown label type: 'continuous'
      #('LogisticRegression', lambda: LogisticRegression()),
      # ValueError: Unknown label type: 'continuous'
      #('LogisticRegressionCV', lambda: LogisticRegressionCV()),
      ('PassiveAggressiveRegressor', lambda: PassiveAggressiveRegressor()),
      # ValueError: Unknown label type: 'continuous'
      #('RandomizedLogisticRegression', lambda: RandomizedLogisticRegression()),
      ('RANSACRegressor', lambda: RANSACRegressor()),
      ('SGDRegressor', lambda: SGDRegressor()),
      # Way too slow.
      #('TheilSenRegressor', lambda: TheilSenRegressor()),

      # Neighbors.
      ('KNeighborsRegressor', lambda: KNeighborsRegressor()),
      # Predicts Nan, infinity or too large of value.
      #('RadiusNeighborsRegressor', lambda: RadiusNeighborsRegressor()),

      # Neural network.
      ('MLPRegressor', lambda: MLPRegressor(max_iter=1000)),

      # Support vector machine.
      ('SVR', lambda: SVR()),
      ('LinearSVR', lambda: LinearSVR()),
      ('NuSVR', lambda: NuSVR()),

      # Tree.
      ('DecisionTreeRegressor', lambda: DecisionTreeRegressor()),
      ('ExtraTreeRegressor', lambda: ExtraTreeRegressor()),
  ])

  if args.rf_only:
    # Strip regressors dictionary to just the first entry (i.e. the main RF).
    regressors = collections.OrderedDict([next(iter(regressors.items()))])

  results = {}
  for regressor_name, regressor_generator in regressors.items():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    for output_label, output_generator in dataset.get_output_generators():
      _, X, y = dataset.generate(output_generator)
      y_pred = kfold_predict(X, y, regressor_generator)

      if not output_label in results:
        results[output_label] = {'num_samples': str(X.shape[0])}
      results[output_label][regressor_name] = str(r2_score(y, y_pred))
      # TODO speed up by reorganizing so augmentation happens once-ish.
      print(output_label, r2_score(y, y_pred))

  regressor_names = list(regressors.keys())
  print(','.join(['output_label', 'num_samples'] + regressor_names))
  for output_label in sorted(results.keys()):
    result = results[output_label]
    print(','.join([output_label, result['num_samples']] +
                   [result[x] for x in regressor_names]))


if __name__ == '__main__':
    main()
