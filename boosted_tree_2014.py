#!/usr/bin/env python3

"""
Parse and run boosted tree learner on 2014 data.
"""

import csv_utils
import numpy as np
import os
import random
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_spli
from sklearn.preprocessing import Imputer


# What percent of the whole dataset to use as the training set.
TRAINING_SET_RATIO = 0.8
# What percent of the *NON*-training set to use for test (vs. validation).
NON_TRAINING_SET_TEST_RATIO = 0.5

DATA_PATH = '2014/2014_Pheotypic_Data_FileS2.csv'

INPUT_LABELS = 'Anthesis date (days),Harvest date (days),Total fresh weight (kg),Brix (maturity),Brix (milk),Dry weight (kg),Stalk height (cm),Dry tons per acre'.split(',')
OUTPUT_LABEL = 'NFC (% DM)'


RANDOM_SEED = 10611

# TODO include in INPUT_LABELS? Is it known before harvest?
PERICARP_LABEL = 'Pericarp pigmentation'

# TODO keep?
def pretty_label(label):
  label = label.replace('%', 'percent').replace('(', 'in ').replace(')', '')
  return label.replace(' ', '_')


def float_or_nan(s):
  try:
    return np.float(s)
  except:
    return np.nan


def parse_data():
  labels, *samples = csv_utils.read_csv(DATA_PATH, ignore_first_lines=2)
  samples = [dict(zip(labels, line)) for line in samples]

  # Convert pericarp string to a number.
  pericarps = sorted(set([x[PERICARP_LABEL] for x in samples]))  # Includes ''.
  for sample in samples:
    if sample[PERICARP_LABEL]:
      sample[PERICARP_LABEL] = pericarps.index(sample[PERICARP_LABEL])

  X = []
  y = []
  for sample in samples:
    output = float_or_nan(sample[OUTPUT_LABEL])
    if np.isnan(output):
      continue

    X.append([float_or_nan(sample[x]) for x in INPUT_LABELS])
    y.append(output)
  return np.array(X), np.array(y)


def main():
  random.seed(RANDOM_SEED)
  np.random.seed(RANDOM_SEED)

  X, y = parse_data()
  num_samples = X.shape[0]
  training_set_size = int(num_samples * TRAINING_SET_RATIO)
  print('Total number of samples: ', num_samples)

  X_train, X_nontrain = X[:training_set_size], X[training_set_size:]
  y_train, y_nontrain = y[:training_set_size], y[training_set_size:]
  test_set_size = int(len(y_nontrain) * NON_TRAINING_SET_TEST_RATIO)
  X_validation, X_test = X_nontrain[test_set_size:], X[:test_set_size]
  y_validation, y_test = y_nontrain[test_set_size:], y[:test_set_size]
  print('Training set size: ', len(y_train))
  print('Validation set size: ', len(y_validation))
  print('Test set size: ', len(y_test))

  imp = Imputer()
  X_train = imp.fit_transform(X_train)
  X_validation
  print(X_train)

  # TODO tune max_depth.
  regressor = GradientBoostingRegressor(max_depth=1, random_state=0)
  regressor.fit(X_train, y_train)
  error = mean_squared_error(y_test, regressor.predict(X_test))
  print(error)



if __name__ == '__main__':
    main()
