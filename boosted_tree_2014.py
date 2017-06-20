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


TRAINING_SET_RATIO = 0.8
DATA_PATH = '2014/2014_Pheotypic_Data_FileS2.csv'

INPUT_LABELS = 'Anthesis date (days),Harvest date (days),Total fresh weight (kg),Brix (maturity),Brix (milk),Dry weight (kg),Stalk height (cm),Dry tons per acre'.split(',')
OUTPUT_LABEL = 'NFC (% DM)'


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
  random.shuffle(samples)

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
  random.seed(10611)
  np.random.seed(10611)

  X, y = parse_data()
  num_samples = X.shape[0]
  training_set_size = int(num_samples * TRAINING_SET_RATIO)
  print('Total number of samples: ', num_samples)
  print('Training set size: ', training_set_size)

  X_train, X_test = X[:training_set_size], X[training_set_size:]
  y_train, y_test = y[:training_set_size], y[training_set_size:]
  est = GradientBoostingRegressor(
      n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,
      loss='ls').fit(X_train, y_train)
  error = mean_squared_error(y_test, est.predict(X_test))
  print(error)



if __name__ == '__main__':
    main()
