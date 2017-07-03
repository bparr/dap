"""
Shared csv helper functions.

Use these functions to read a *.csv file because they include sanity checks and
convert values in EMPTY_VALUES to ''.
"""

import csv

# Values in original csv files that are interpreted as empty strings.
EMPTY_VALUES = ['', ' ', 'FILL', 'NA', ' NA']

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
