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
import dataset as dataset_lib
from features import Features
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ARDRegression, HuberRegressor, LinearRegression, LogisticRegression, LogisticRegressionCV, PassiveAggressiveRegressor, RandomizedLogisticRegression, RANSACRegressor, SGDRegressor, TheilSenRegressor
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


#######################
# 2014 Dataset logic. #
#######################
# Returns result of percent DM value multiplied by dry weight.
# If given, the minus label's value is subtracted from label's value.
def get_2014_weight(sample, dry_weight_label, label, minus=None):
  value = sample[label]
  minus_value = 0.0 if minus is None else sample[minus]
  dry_weight = sample[dry_weight_label]
  if (dataset_lib.is_missing(value) or dataset_lib.is_missing(minus_value) or
      dataset_lib.is_missing(dry_weight)):
    return dataset_lib.MISSING_VALUE
  return dry_weight * (value - minus_value) / 100.0

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

  dataset_lib.convert_to_float_or_missing(samples, list(input_labels) + [
      ADF, NDF, NFC, LIGNIN, DRY_WEIGHT])

  output_generators = collections.OrderedDict([
      ('adf', lambda sample: get_2014_weight(sample, DRY_WEIGHT, ADF)),
      ('ndf', lambda sample: get_2014_weight(sample, DRY_WEIGHT, NDF)),
      ('nfc', lambda sample: get_2014_weight(sample, DRY_WEIGHT, NFC)),
      ('lignin', lambda sample: get_2014_weight(sample, DRY_WEIGHT, LIGNIN)),
      ('c6', lambda sample: get_2014_weight(sample, DRY_WEIGHT, ADF,
                                            minus=LIGNIN)),
      ('c5', lambda sample: get_2014_weight(sample, DRY_WEIGHT, NDF,
                                            minus=ADF)),
  ])

  print('2014 Inputs: ' + ','.join(input_labels))
  return dataset_lib.Dataset(samples, input_labels, output_generators)



#######################
# 2016 Dataset logic. #
#######################
def filter_2016_labels(feature_starts_with):
  return [x.value for x in Features if x.name.startswith(feature_starts_with)]

# Used instead of lambda to avoid Python scoping issue.
def create_2016_output_generator(key):
  return lambda sample: sample[key]

def new2016Dataset(include_harvest=True):
  samples = csv_utils.read_csv_as_dicts('2016.merged.csv')
  dataset_lib.convert_to_float_or_missing(samples, filter_2016_labels((
      'HARVEST_', 'COMPOSITION_', 'ROBOT_', 'SYNTHETIC_', 'GPS_')) +
      [Features.ROW.value, Features.COLUMN.value])

  # TODO what to include? Allow multiple subsets through commandline?
  input_features_starts_with = [
      'ROBOT_',
      #'GPS_',
      'ACCESSION_',
  ]
  if include_harvest:
    input_features_starts_with.append('HARVEST_')
    input_features_starts_with.append('SYNTHETIC_HARVEST_')

  input_labels = filter_2016_labels(tuple(input_features_starts_with))
  output_labels = filter_2016_labels('COMPOSITION_')

  output_generators = collections.OrderedDict(sorted(
    [(x, create_2016_output_generator(x)) for x in output_labels]
  ))

  print('2016 Inputs: ' + ','.join(input_labels))
  return dataset_lib.Dataset(samples, input_labels, output_generators)


def new2016NoHarvestDataset():
  return new2016Dataset(include_harvest=False)


# Create a predictor that uses a single regressor to fit and predict.
def create_simple_predictor(regressor_generator):
  def simple_predictor(kfold_data_view):
    regressor = regressor_generator()
    regressor.fit(kfold_data_view.X_train, kfold_data_view.y_train)
    return regressor.predict(kfold_data_view.X_test)
  return simple_predictor


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


def main():
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

  dataset = (DATASET_FACTORIES[args.dataset])()

  if args.write_dataviews_only:
    for output_label, data_view in dataset.generate_views():
      data_view.write_csv(os.path.join(
          'dataviews', args.dataset, output_label + '.csv'))
    return


  global CSV_OUTPUT_PATH
  CSV_OUTPUT_PATH = CSV_OUTPUT_PATH % args.dataset
  open(CSV_OUTPUT_PATH, 'w').close()  # Clear file.

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

  predictors = collections.OrderedDict([
      (x, create_simple_predictor(y)) for x, y in regressor_generators.items()])

  results = {}
  feature_importances = []
  for predictor_name, predictor in predictors.items():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    for output_label, data_view in dataset.generate_views():
      y_pred = data_view.kfold_predict(predictor)

      if not output_label in results:
        results[output_label] = {'num_samples': data_view.get_num_samples()}
      results[output_label][predictor_name] = data_view.get_r2_score(y_pred)

      # TODO add back using a global blah in predictor instead of this hack.
      #if predictor_name == RF_REGRESSOR_NAME:
      #  for regressor in regressors:
      #    feature_importances.extend(
      #        [tree.feature_importances_ for tree in regressor.estimators_])


  # Print each predictors' r2 score results..
  predictor_names = list(predictors.keys())
  rprint(','.join(['output_label', 'num_samples'] + predictor_names))
  for output_label in sorted(results.keys()):
    result = results[output_label]
    rprint(','.join([output_label, str(result['num_samples'])] +
                    [str(result[x]) for x in predictor_names]))


  # TODO reenable feature importances!
  return

  # Print feature importances.
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
