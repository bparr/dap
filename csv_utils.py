"""
Shared csv helper functions.

Use these functions to read a *.csv file because they include sanity checks and
convert values in EMPTY_VALUES to ''.
"""

import csv
import numpy as np

# Values in original csv files that are interpreted as empty strings.
EMPTY_VALUES = ('', ' ', 'FILL', 'NA', ' NA')

# Store all values for a specific entry in output csv by delimitting them with
# this. For example, some plots have multiple differing robotic values, so
# just store all the ones encountered in the single CSV entry.
_MISMATCH_DELIMETER = ' && '


# Averge multiple numeric values, caused by mismatched data merging.
def average_mismatch(value):
  return np.mean([float(x) for x in value.split(_MISMATCH_DELIMETER)])


# Combine two values in a single CSV entry.
def append_value(original, to_append):
  # Don't re-append a single value over and over.
  if to_append in original.split(_MISMATCH_DELIMETER):
    return original
  return original + _MISMATCH_DELIMETER + to_append


# Split appended values in a cingle CSV entry.
def split_values(string_to_split):
  return string_to_split.split(_MISMATCH_DELIMETER)


# Returns list of lines (represented as lists).
def read_csv(file_path):
  with open(file_path, 'r') as f:
    lines = []
    for line in csv.reader(f):
      lines.append(['' if v in EMPTY_VALUES else v for v in line])

  num_columns = len(lines[0])
  if len(set(lines[0])) != num_columns:
    raise Exception('Duplicate label in first line of csv: ', file_name)
  for i, line in enumerate(lines):
    if len(line) != num_columns:
      raise Exception('Unexpected amount of values in line: ',
                      i, len(line), num_columns)
  return lines


# Returns list of lines (represented as dictionaries where the labels for each
# dictionary come from the labels in the first line of the csv).
def read_csv_as_dicts(file_path):
  labels, *lines = read_csv(file_path)
  return [dict(zip(labels, line)) for line in lines]
