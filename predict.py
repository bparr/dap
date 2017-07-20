#!/usr/bin/env python3

"""
Parse full csv and predict harvest data.

Usage:
  ./predict.py -d 2016
  ./predict.py -d 2016.noHarvest
  ./predict.py -d 2014

Outputs files to the results/ directory.
"""
# TODO tests?

import argparse
import collections
import csv
import csv_utils
import matplotlib.pyplot as plt
from merge_data import DataKeys, average_mismatch
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
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

RF_REGRESSOR_NAME = 'random_forest'

# Path to save results as CSV file.
CSV_OUTPUT_PATH = 'results/%s.out'
# Path to save plot, formatted with the dataset name.
FEATURE_IMPORTANCE_SAVE_PATH = 'results/feature_importance.%s.png'

RANDOM_SEED = 10611

def rprint(string_to_print):
  print(string_to_print)
  with open(CSV_OUTPUT_PATH, 'a') as f:
    f.write(string_to_print + '\n')

MISSING_VALUE = -1


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

    rprint(','.join(['INPUTS'] + list(self._input_labels)))

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


#######################
# 2014 Dataset logic. #
#######################
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

  # Returns result of percent DM value multiplied by dry weight.
  # If given, the minus label's value is subtracted from label's value.
  def get_weight(sample, dry_weight_label, label, minus=None):
    value = sample[label]
    minus_value = 0.0 if minus is None else sample[minus]
    dry_weight = sample[dry_weight_label]
    if is_missing(value) or is_missing(minus_value) or is_missing(dry_weight):
      return MISSING_VALUE
    return dry_weight * (value - minus_value) / 100.0


  output_generators = collections.OrderedDict([
      ('adf', lambda sample: get_weight(sample, DRY_WEIGHT, ADF)),
      ('ndf', lambda sample: get_weight(sample, DRY_WEIGHT, NDF)),
      ('nfc', lambda sample: get_weight(sample, DRY_WEIGHT, NFC)),
      ('lignin', lambda sample: get_weight(sample, DRY_WEIGHT, LIGNIN)),
      ('c6', lambda sample: get_weight(sample, DRY_WEIGHT, ADF, minus=LIGNIN)),
      ('c5', lambda sample: get_weight(sample, DRY_WEIGHT, NDF, minus=ADF)),
  ])
  return Dataset(samples, input_labels, output_generators)



#######################
# 2016 Dataset logic. #
#######################
def filter_2016_labels(data_key_starts_with):
  return [x.value for x in DataKeys if x.name.startswith(data_key_starts_with)]

def create_2016_output_generator(key):
  return lambda sample: sample[key]

def new2016Dataset(include_harvest=True):
  samples = csv_utils.read_csv_as_dicts('2016.merged.csv')
  convert_to_float_or_missing(samples, filter_2016_labels((
      'HARVEST_', 'COMPOSITION_', 'ROBOT_', 'SYNTHETIC_', 'GPS_')) +
      [DataKeys.ROW.value, DataKeys.COLUMN.value])

  # TODO what to include? Allow multiple subsets through commandline?
  input_data_keys_starts_with = [
      'ROBOT_',
      #'GPS_',
      'ACCESSION_',
  ]
  if include_harvest:
    input_data_keys_starts_with.append('HARVEST_')
    input_data_keys_starts_with.append('SYNTHETIC_HARVEST_')

  input_labels = filter_2016_labels(tuple(input_data_keys_starts_with))
  output_labels = filter_2016_labels('COMPOSITION_')

  weight_datakeys = (
      DataKeys.COMPOSITION_LIGNIN,
      DataKeys.COMPOSITION_CELLULOSE,
      DataKeys.COMPOSITION_HEMICELLULOSE)

  output_generators = collections.OrderedDict(sorted(
    [(x, create_2016_output_generator(x)) for x in output_labels]
  ))

  return Dataset(samples, input_labels, output_generators)


def new2016NoHarvestDataset():
  return new2016Dataset(include_harvest=False)



def kfold_predict(X, y, regressor_generator):
  y_pred = []

  kf = KFold(n_splits=10)
  regressors = []
  for train_indexes, test_indexes in kf.split(X):
    X_train, X_test = X[train_indexes], X[test_indexes]
    y_train, y_test = y[train_indexes], y[test_indexes]

    # TODO reconsider using Imputer?
    regressor = regressor_generator().fit(X_train, y_train)
    y_pred.extend(zip(test_indexes, regressor.predict(X_test)))
    regressors.append(regressor)

  y_pred_dict = dict(y_pred)
  if len(y_pred_dict) != len(y_pred):
    raise Exception('kfold splitting was bad.')
  return [y_pred_dict[i] for i in range(len(X))], regressors


# Merge (sum) importances that have the same input label.
def merge_importances(input_labels, feature_importances):
  feature_importances = np.array(feature_importances)
  sorted_input_labels = sorted(set(input_labels))
  input_label_to_index = dict((y, x) for x, y in enumerate(sorted_input_labels))
  merged_feature_importances = np.zeros((feature_importances.shape[0],
                                         len(input_label_to_index)))
  for i, input_label in enumerate(input_labels):
    merged_feature_importances[:,input_label_to_index[input_label]] += (
        feature_importances[:, i])

  return sorted_input_labels, merged_feature_importances


# Output completely preprocessed CSV files.
# Currently useful for verifying results against lab's random forest code.
# TODO remove!
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
  global CSV_OUTPUT_PATH

  DATASET_FACTORIES = {
    '2014': new2014Dataset,
    '2016': new2016Dataset,
    '2016.noHarvest': new2016NoHarvestDataset,
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
  CSV_OUTPUT_PATH = CSV_OUTPUT_PATH % args.dataset
  open(CSV_OUTPUT_PATH, 'w').close()  # Clear file.

  dataset = (DATASET_FACTORIES[args.dataset])()

  if args.write_dataviews_only:
    for output_label, output_generator in dataset.get_output_generators():
      X_labels, X, y = dataset.generate(output_generator, shuffle=False)
      write_csv(os.path.join('dataviews', args.dataset, output_label + '.csv'),
                X_labels, X, output_label, y)
    return

  regressor_generators = collections.OrderedDict([
      # Customized.
      # Based on lab code's configuration.
      (RF_REGRESSOR_NAME, lambda: RandomForestRegressor(
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
    regressor_generators = collections.OrderedDict([
        (RF_REGRESSOR_NAME, regressor_generators[RF_REGRESSOR_NAME])])

  results = {}
  feature_importances = []
  for regressor_name, regressor_generator in regressor_generators.items():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    for output_label, output_generator in dataset.get_output_generators():
      _, X, y = dataset.generate(output_generator)
      y_pred, regressors = kfold_predict(X, y, regressor_generator)

      if not output_label in results:
        results[output_label] = {'num_samples': str(X.shape[0])}
      results[output_label][regressor_name] = str(r2_score(y, y_pred))

      if regressor_name == RF_REGRESSOR_NAME:
        for regressor in regressors:
          feature_importances.extend(
              [tree.feature_importances_ for tree in regressor.estimators_])


  regressor_names = list(regressor_generators.keys())
  rprint(','.join(['output_label', 'num_samples'] + regressor_names))
  for output_label in sorted(results.keys()):
    result = results[output_label]
    rprint(','.join([output_label, result['num_samples']] +
                    [result[x] for x in regressor_names]))


  rprint('\n')
  rprint(','.join(['input_label', 'mean_importance', 'std_importance']))
  input_labels, feature_importances = merge_importances(
      dataset.get_input_labels(), feature_importances)
  mean_feature_importances = np.mean(feature_importances, axis=0)
  std_feature_importances = np.std(feature_importances, axis=0)
  for i, input_label in enumerate(input_labels):
    rprint(','.join([input_label, str(mean_feature_importances[i]),
                     str(std_feature_importances[i])]))


  # Plot the feature importances of the forest.
  # Based on http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html.
  indices = np.argsort(mean_feature_importances)[::-1]
  fig = plt.figure()
  plt.title("Feature Importances")
  plt.bar(range(len(indices)), mean_feature_importances[indices],
          color="r", yerr=std_feature_importances[indices], align="center")
  plt.xticks(range(len(indices)), [input_labels[x] for x in indices],
             rotation='vertical', fontsize=8)
  plt.xlim([-1, len(indices)])
  plt.tight_layout()
  plt.savefig(FEATURE_IMPORTANCE_SAVE_PATH % args.dataset)


if __name__ == '__main__':
    main()
