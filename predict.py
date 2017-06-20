#!/usr/bin/env python3

"""
Parse full csv and predict harvest data.
"""

import collections
import csv_utils
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

DATA_PATH = '2014/2014_Pheotypic_Data_FileS2.csv'

INPUT_LABELS = 'Anthesis date (days),Harvest date (days),Total fresh weight (kg),Brix (maturity),Brix (milk),Dry weight (kg),Stalk height (cm),Dry tons per acre'.split(',')

RANDOM_SEED = 10611

# TODO include in INPUT_LABELS? Is it known before harvest?
PERICARP_LABEL = 'Pericarp pigmentation'

ADF_LABEL = 'ADF (% DM)'
NDF_LABEL = 'NDF (% DM)'
NFC_LABEL = 'NFC (% DM)'
LIGNIN_LABEL = 'Lignin (% DM)'

# TODO tune??
MISSING_VALUE = np.nan
#MISSING_VALUE = -1



def pretty_label(label):
  label = label.replace('%', 'percent').replace('(', 'in ').replace(')', '')
  return label.replace(' ', '_')


def float_or_missing(s):
  try:
    return np.float(s)
  except:
    return MISSING_VALUE


def is_missing(value):
  # Unfortunately np.nan == np.nan is False, so check both isnan and equality.
  return np.isnan(value) or value == MISSING_VALUE


def subtract_or_missing(value1, value2):
  value1 = float_or_missing(value1)
  value2 = float_or_missing(value2)
  if is_missing(value1) or is_missing(value2):
    return MISSING_VALUE
  return value1 - value2


def parse_data(lines, input_labels, output_generator):
  labels, *samples = lines
  samples = [dict(zip(labels, line)) for line in samples]
  random.shuffle(samples)

  # Convert pericarp string to a number.
  pericarps = sorted(set([x[PERICARP_LABEL] for x in samples]))  # Includes ''.
  for sample in samples:
    if sample[PERICARP_LABEL]:
      sample[PERICARP_LABEL] = pericarps.index(sample[PERICARP_LABEL])

  X = []
  y = []
  for sample in samples:
    output = float_or_missing(output_generator(sample))
    if is_missing(output):
      continue

    X.append([float_or_missing(sample[x]) for x in input_labels])
    y.append(output)
  return np.array(X), np.array(y)


def predict(X, y, regressor_generator):
  y_true = []
  y_pred = []

  kf = KFold(n_splits=10)
  for train_indexes, test_indexes in kf.split(X):
    X_train, X_test = X[train_indexes], X[test_indexes]
    y_train, y_test = y[train_indexes], y[test_indexes]

    imp = Imputer()
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    regressor = regressor_generator().fit(X_train, y_train)
    regressor.fit(X_train, y_train)
    y_true.extend(y_test)
    y_pred.extend(regressor.predict(X_test))

  return y_true, y_pred


def main():
  random.seed(RANDOM_SEED)
  np.random.seed(RANDOM_SEED)

  lines = csv_utils.read_csv(DATA_PATH, ignore_first_lines=2)
  regressors = collections.OrderedDict([
      # TODO tune max_depth.
      ('boosted trees', lambda: GradientBoostingRegressor(max_depth=1, random_state=0)),
      ('random forests', lambda: RandomForestRegressor(n_estimators=100, random_state=0)),
  ])

  outputs = collections.OrderedDict([
      ('adf', lambda sample: sample[ADF_LABEL]),
      ('ndf', lambda sample: sample[NDF_LABEL]),
      ('nfc', lambda sample: sample[NFC_LABEL]),
      ('lignin', lambda sample: sample[LIGNIN_LABEL]),
      ('c6', lambda x: subtract_or_missing(x[ADF_LABEL], x[LIGNIN_LABEL])),
      ('c5', lambda x: subtract_or_missing(x[NDF_LABEL], x[ADF_LABEL])),
  ])

  for regressor_name, regressor_generator in regressors.items():
    print('\n\n' + regressor_name)
    for output_name, output_generator in outputs.items():
      X, y = parse_data(lines, INPUT_LABELS, output_generator)
      num_samples = X.shape[0]
      print('Total number of %s samples: %s' % (output_name, num_samples))

      y_true, y_pred = predict(X, y, regressor_generator)
      print('r2 score: ', r2_score(y_true, y_pred))



if __name__ == '__main__':
    main()
