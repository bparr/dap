#!/usr/bin/env python3

"""
scikit-learn regressors with (mainly) default settings.
"""

import collections
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

REGRESSORS = collections.OrderedDict([
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
  # Increase max_iter to avoid Warning about non-convergence within max_iter.
  ('MLPRegressor', lambda: MLPRegressor(max_iter=1000)),

  # Support vector machine.
  ('SVR', lambda: SVR()),
  ('LinearSVR', lambda: LinearSVR()),
  ('NuSVR', lambda: NuSVR()),

  # Tree.
  ('DecisionTreeRegressor', lambda: DecisionTreeRegressor()),
  ('ExtraTreeRegressor', lambda: ExtraTreeRegressor()),
])


# Regressors that do not support the sample_weight optional fit() argument.
REGRESSORS_NOT_SUPPORTING_SAMPLE_WEIGHT = set([
  'PLSRegression', 'GaussianProcessRegressor', 'PassiveAggressiveRegressor',
  'RandomizedLogisticRegression', 'SGDRegressor', 'TheilSenRegressor',
  'KNeighborsRegressor', 'MLPRegressor'])

