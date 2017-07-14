#!/usr/bin/env python3

"""
Investigate inbound scores.
"""

import csv
import numpy as np
import os

# Number of std away from mean use for filter.
M = 2

NUM_SAMPLES = 698

MIN_FEATURES_ANOMALOUS = 5

def write_filtered_csv(labels, samples, counts, filter_count):
  with open('2016.merged.no_' + str(filter_count) + '_anomalous.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(labels)
    for i, sample in enumerate(samples):
      if counts[i] < filter_count:
        writer.writerow(sample)


def main():
  dirs = sorted([x for x in os.listdir('.') if os.path.isdir(x)])
  counts = np.zeros(NUM_SAMPLES)
  for d in dirs:
    with open(os.path.join(d, 'values.csv'), 'r') as f:
      next(f)
      scores = np.array([float(x) for _, __, x in csv.reader(f)])

    if len(scores) != NUM_SAMPLES:
      raise Exception('Unexpected number of samples:', d, len(scores))

    counts += np.abs(scores - np.mean(scores)) >= M * np.std(scores)

  #results = [x for x in enumerate(counts) if x[1] > MIN_FEATURES_ANOMALOUS]
  #results.sort(reverse=True, key=lambda x: x[1])
  #print(results)

  with open('../2016.merged.csv') as f:
    reader = csv.reader(f)
    labels = next(reader)
    adf_label_index = labels.index('ADF')
    samples = [x for x in reader if x[adf_label_index]]

  if len(samples) != NUM_SAMPLES:
    raise Exception('Unexpected number of samples in 2016.merged.csv',
                    len(samples))

  with open('2016.merged.anomalous_counts.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(labels + ['anomalous_count'])
    for i, sample in enumerate(samples):
      writer.writerow(sample + [counts[i]])

  write_filtered_csv(labels, samples, counts, 5)
  write_filtered_csv(labels, samples, counts, 10)
  write_filtered_csv(labels, samples, counts, 20)


if __name__ == '__main__':
    main()

