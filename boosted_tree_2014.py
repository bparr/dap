#!/usr/bin/env python3

"""
Parse and run boosted tree learner on 2014 data.
"""

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
TRAINING_SIZE = 0.8
# What percent of the *NON*-training set to use for test (vs. validation).
NON_TRAINING_SET_TEST_SIZE = 0.5

DATA_PATH = '2014/2014_Pheotypic_Data_FileS2.csv'

INPUT_LABELS = 'Anthesis date (days),Harvest date (days),Total fresh weight (kg),Brix (maturity),Brix (milk),Dry weight (kg),Stalk height (cm),Dry tons per acre'.split(',')

#OUTPUT_LABEL = 'ADF (% DM)'
#OUTPUT_LABEL = 'NDF (% DM)'
#OUTPUT_LABEL = 'NFC (% DM)'
OUTPUT_LABEL = 'Lignin (% DM)'


RANDOM_SEED = 10611

# TODO include in INPUT_LABELS? Is it known before harvest?
PERICARP_LABEL = 'Pericarp pigmentation'

# TODO tune??
MISSING_VALUE = np.nan
#MISSING_VALUE = -1

# TODO keep?
def pretty_label(label):
  label = label.replace('%', 'percent').replace('(', 'in ').replace(')', '')
  return label.replace(' ', '_')


def float_or_missing(s):
  try:
    return np.float(s)
  except:
    return MISSING_VALUE


def parse_data():
  labels, *samples = csv_utils.read_csv(DATA_PATH, ignore_first_lines=2)
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
    output = float_or_missing(sample[OUTPUT_LABEL])
    if np.isnan(output) or output == MISSING_VALUE:
      continue

    X.append([float_or_missing(sample[x]) for x in INPUT_LABELS])
    y.append(output)
  return np.array(X), np.array(y)


def main():
  random.seed(RANDOM_SEED)
  np.random.seed(RANDOM_SEED)

  X, y = parse_data()
  num_samples = X.shape[0]
  print('Total number of samples: ', num_samples)

  #X_train, X_nontrain, y_train, y_nontrain = train_test_split(
  #    X, y, test_size=(1 - TRAINING_SIZE), random_state=RANDOM_SEED)
  #X_validation, X_test, y_validation, y_test = train_test_split(
  #    X_nontrain, y_nontrain, test_size=NON_TRAINING_SET_TEST_SIZE,
  #    random_state=RANDOM_SEED)

  y_true = []
  y_pred = []

  kf = KFold(n_splits=10)
  for train_indexes, test_indexes in kf.split(X):
    X_train, X_test = X[train_indexes], X[test_indexes]
    y_train, y_test = y[train_indexes], y[test_indexes]

    imp = Imputer()
    X_train = imp.fit_transform(X_train)
    #X_validation = imp.transform(X_validation)
    X_test = imp.transform(X_test)

    # TODO tune max_depth.
    regressor = GradientBoostingRegressor(max_depth=1, random_state=0)
    #regressor = RandomForestRegressor(n_estimators=num_samples, random_state=0)
    regressor.fit(X_train, y_train)
    y_true.extend(y_test)
    y_pred.extend(regressor.predict(X_test))


  print(r2_score(y_true, y_pred))


if __name__ == '__main__':
    main()
