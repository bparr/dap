#!/usr/bin/env python3

"""
Augment a dataset that has a high amount of missing data.

Example: A four sample dataset, where each sample has three input features and
one output feature. The first two samples are in the training set and the
remaining two samples are in the test set.

X-test sample #1 = [MISSING, 2.0, 3.0]
X-test sample #2 = [4.0, 5.0, MISSING]

X-train sample #1: = [7.0, 8.0, 9.0]
augment_sample(X-train sample #1) =
  1)  [7.0, 8.0, 9.0]
  2)  [MISSING, 8.0, 9.0]
  3)  [7.0, 8.0, MISSING]


Note: There are no duplicate entries in the result of augment_sample(). So if,
X-train sample #2: = [MISSING, 11.0, 12.0]
augment_sample(X-train sample #2) =
  1)  [MISSING, 11.0, 12.0]
  2)  [MISSING, 11.0, MISSING]


Each original X-train sample is given equal weight regardless of the size of
its augmentation by using scikit.fit's sample_weight optional argument.

Each augmented sample has its output value set to the original sample's output
value.
"""

import dataset as dataset_lib
import numpy as np


# TODO tests?! See git commit 447b84c48b4420ccfd8777e96a41fc6ef3b3039d
# TODO apply to 2014 dataset as well. Would be a better story that way if it
#     improved both.
# Return (new kfold_data_view, sample_weight) tuple where the new
# kfold_data_view contains the results of augment_sample() applied to all
# original training samples.
def augment(kfold_data_view):
  missings = set()  # Set up True/False tuples.
  for x in kfold_data_view.X_test:
    missings.add(tuple(dataset_lib.is_missing(value) for value in x))

  augmented_X_train = []
  augmented_y_train = []
  sample_weight = []
  for x, y in zip(kfold_data_view.X_train, kfold_data_view.y_train):
    # TODO add test for having set start with tuple(x).
    augmented_samples = set([tuple(x)])
    for missing in missings:
      augmented_samples.add(tuple([
          (dataset_lib.MISSING_VALUE if b else a) for a, b in zip(x, missing)]))

    augmented_X_train.append(x)
    augmented_y_train.append(y)
    # Note that sum(this sample's augmention weights) == 1.0.
    sample_weight.append(0.5)
    for augmented_sample in augmented_samples:
      augmented_X_train.append(augmented_sample)
      augmented_y_train.append(y)
      sample_weight.append(0.5 / len(augmented_samples))

  return dataset_lib.KFoldDataView(
      kfold_data_view.X_labels, np.array(augmented_X_train),
      kfold_data_view.X_test, np.array(augmented_y_train)), sample_weight

