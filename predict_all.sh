#!/bin/sh
# Predict on all datasets with/without augmenting missingness.

time ./predict.py -d 2014
time ./predict.py -d 2014 --no_augment_missing
time ./predict.py -d 2016.noRobotAerialHarvest
time ./predict.py -d 2016.noRobotAerialHarvest --no_augment_missing
time ./predict.py -d 2016.noHarvest
time ./predict.py -d 2016.noHarvest --no_augment_missing
time ./predict.py -d 2016
time ./predict.py -d 2016 --no_augment_missing
