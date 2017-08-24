#!/usr/bin/env python3

"""
Parse full csv and predict harvest data.

Usage:
  ./predict.py -d 2016
  ./predict.py -d 2016.noHarvest
  ./predict.py -d 2014

Outputs files to the results/ directory.
"""

import argparse
import collections
import csv
import csv_utils
import dataset as dataset_lib
from features import Features
import matplotlib.pyplot as plt
import missing_augment
import numpy as np
import os
import random
import scikit_regressors
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestRegressor


# Name prefix of main regressor. Random forest with AutonLab code settings.
RF_REGRESSOR_PREFIX = 'random_forest_'

# Where to store the actual predictions.
PREDICTIONS_DIR = 'predictions'

# Path to save results as CSV file.
CSV_OUTPUT_PATH = 'results/%s.out'

# Path to save plot, formatted with the dataset name.
FEATURE_IMPORTANCE_SAVE_PATH = 'results/feature_importance.%s.png'

DEFAULT_RANDOM_SEED = 10611

# Used to gather feature importance data across predictions.
global_feature_importances = []


# Print string to both stdout and CSV_OUTPUT_PATH.
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

# Returns list of Feature values whose names start with the given argument.
# feature_starts_with can be a string, or tuple of strings.
def filter_2016_labels(feature_starts_with):
  return [x.value for x in Features if x.name.startswith(feature_starts_with)]


# Used instead of lambda to avoid Python scoping issue.
def create_2016_output_generator(key):
  return lambda sample: sample[key]


def new2016Dataset(include_robot_and_aerial=True, include_harvest=True):
  samples = csv_utils.read_csv_as_dicts('2016.merged.csv')
  dataset_lib.convert_to_float_or_missing(samples, filter_2016_labels((
      'HARVEST_', 'COMPOSITION_', 'ROBOT_', 'AERIAL_', 'SYNTHETIC_', 'GPS_',
      'ROW', 'COLUMN')))

  input_features_starts_with = [
      'GPS_',
      'ACCESSION_',
  ]

  if include_robot_and_aerial:
    input_features_starts_with.extend(['ROBOT_', 'AERIAL_'])

  if include_harvest:
    input_features_starts_with.extend(['HARVEST_', 'SYNTHETIC_HARVEST_'])

  input_labels = filter_2016_labels(tuple(input_features_starts_with))

  output_labels = filter_2016_labels('COMPOSITION_')
  output_generators = collections.OrderedDict(sorted(
    [(x, create_2016_output_generator(x)) for x in output_labels]
  ))

  print('2016 Inputs: ' + ','.join(input_labels))
  return dataset_lib.Dataset(samples, input_labels, output_generators)


def new2016NoRobotAerialHarvestDataset():
  return new2016Dataset(include_robot_and_aerial=False, include_harvest=False)


def new2016NoHarvestDataset():
  return new2016Dataset(include_harvest=False)



# Create a predictor that uses a single regressor to fit and predict.
def create_simple_predictor(name, regressor_generator):
  def simple_predictor(kfold_data_view, sample_weight=None):
    regressor = regressor_generator()
    # Not all regressors have the sample_weight optional fit() argument.
    if name in scikit_regressors.REGRESSORS_NOT_SUPPORTING_SAMPLE_WEIGHT:
      regressor.fit(kfold_data_view.X_train, kfold_data_view.y_train)
    else:
      regressor.fit(kfold_data_view.X_train, kfold_data_view.y_train,
                    sample_weight=sample_weight)
    return regressor.predict(kfold_data_view.X_test)
  return simple_predictor


# Create a Random Forest predictor with AutonLab's code configuration.
# Also used for generating feature importances.
def rf_predictor(kfold_data_view, sample_weight=None):
  regressor = RandomForestRegressor(n_estimators=100, max_depth=10,
                                    max_features='sqrt', min_samples_split=10)
  regressor.fit(kfold_data_view.X_train, kfold_data_view.y_train,
                sample_weight=sample_weight)
  y_pred = regressor.predict(kfold_data_view.X_test)

  global global_feature_importances
  global_feature_importances.extend(
      [tree.feature_importances_ for tree in regressor.estimators_])

  return y_pred


# Create a new predictor that first augments the dataset using the missing
# value augmentation before predicting.
def create_missing_augmented_predictor(predictor):
  def missing_augmented_predictor(kfold_data_view):
    kfold_data_view, sample_weight = missing_augment.augment(kfold_data_view)
    return predictor(kfold_data_view, sample_weight=sample_weight)

  return missing_augmented_predictor


# Merge (sum) importances that have the same input label.
def get_merged_importances(input_labels):
  global global_feature_importances
  feature_importances = np.array(global_feature_importances)
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
    '2016.noRobotAerialHarvest': new2016NoRobotAerialHarvestDataset,
  }

  parser = argparse.ArgumentParser(description='Predict harvest data.')
  parser.add_argument('-d', '--dataset', default='2016',
                      choices=list(DATASET_FACTORIES.keys()),
                      help='Which dataset to use.')
  parser.add_argument('--rf_only', action='store_true',
                      help='Only fit main random forest predictor.')
  parser.add_argument('--rf_iterations', type=int, default=1,
                      help='Number of times to run main RF predictor.')
  parser.add_argument('--no_augment_missing', action='store_true',
                      help='Skip augmenting samples with more missing data.')
  parser.add_argument('--no_write_predictions', action='store_true',
                      help='Skip writing predicted values to predictions/.')
  parser.add_argument('--write_dataviews_only', action='store_true',
                      help='No prediction. Just write data views.')
  parser.add_argument('--start_iteration', type=int, default=0,
                      help='Which rf iteration number to start on.')
  args = parser.parse_args()

  dataset = (DATASET_FACTORIES[args.dataset])()
  dataset_name = args.dataset

  if args.write_dataviews_only:
    dataview_dir = os.path.join('dataviews', dataset_name)
    os.makedirs(dataview_dir, exist_ok=True)
    for output_label, data_view in dataset.generate_views(None):
      data_view.write_csv(os.path.join(dataview_dir, output_label + '.csv'))
    return

  # Append after the args.write_dataviews_only check, since 2016 dataviews are
  # identical to the 2016.noAugmentMissing dataviews, for example.
  if args.no_augment_missing:
    dataset_name += '.noAugmentMissing'

  global CSV_OUTPUT_PATH
  CSV_OUTPUT_PATH = CSV_OUTPUT_PATH % dataset_name
  open(CSV_OUTPUT_PATH, 'w').close()  # Clear file.

  predictors = collections.OrderedDict()
  random_seeds = []
  for i in range(args.start_iteration,
                 args.start_iteration + args.rf_iterations):
    predictors[RF_REGRESSOR_PREFIX + str(i)] = rf_predictor
    random_seeds.append(DEFAULT_RANDOM_SEED + i)

  if not args.rf_only:
    for name, regressor_generator in scikit_regressors.REGRESSORS.items():
      predictors[name] = create_simple_predictor(name, regressor_generator)
      random_seeds.append(DEFAULT_RANDOM_SEED)

  if not args.no_augment_missing:
    for predictor_name, predictor in predictors.items():
      predictors[predictor_name] = create_missing_augmented_predictor(predictor)


  # Make predictions.
  results = {}
  zipped_predictors = zip(predictors.items(), random_seeds)
  for (predictor_name, predictor), random_seed in zipped_predictors:
    random.seed(random_seed)
    np.random.seed(random_seed)
    kfold_random_state = np.random.randint(2 ** 32 - 1)

    predictor_dir = os.path.join(PREDICTIONS_DIR, dataset_name, predictor_name)
    os.makedirs(predictor_dir, exist_ok=True)

    for output_label, data_view in dataset.generate_views(kfold_random_state):
      y_pred = data_view.kfold_predict(predictor)
      if not args.no_write_predictions:
        data_view.write_predictions(
            os.path.join(predictor_dir, output_label + '.csv'), y_pred,
            [Features.GPS_EASTINGS.value, Features.GPS_NORTHINGS.value])

      if not output_label in results:
        results[output_label] = {'num_samples': data_view.get_num_samples()}
      results[output_label][predictor_name] = data_view.get_r2_score(y_pred)

      print(predictor_name, output_label, results[output_label][predictor_name])


  # Print each predictors' r2 score results.
  predictor_names = list(predictors.keys())
  rprint(','.join(['output_label', 'num_samples'] + predictor_names))
  for output_label in sorted(results.keys()):
    result = results[output_label]
    rprint(','.join([output_label, str(result['num_samples'])] +
                    [str(result[x]) for x in predictor_names]))

  # Print feature importances.
  rprint('\n')
  rprint(','.join(['input_label', 'mean_importance', 'std_importance']))
  input_labels, feature_importances = get_merged_importances(
      dataset.get_input_labels())
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
  plt.savefig(FEATURE_IMPORTANCE_SAVE_PATH % dataset_name)


if __name__ == '__main__':
  main()
