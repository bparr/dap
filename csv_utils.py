# Shared csv helper functions.

import csv

# Values in original csv files that are interpreted as empty strings.
EMPTY_VALUES = ['', ' ', 'FILL', 'NA']

# Use this to read a *.csv file. Includes sanity checks and converts values in
# EMPTY_VALUES to ''.
def read_csv(file_path):
  with open(file_path, 'r') as f:
    lines = []
    for line in csv.reader(f):
      lines.append(['' if v in EMPTY_VALUES else v for v in line])

  num_columns = len(lines[0])
  if len(set(lines[0])) != num_columns:
    raise Exception('Duplicate label in first row of csv: ', file_name)
  for i, line in enumerate(lines):
    if len(line) != num_columns:
      raise Exception('Unexpected amount of values in line: ',
                      i, len(line), num_columns)
  return lines
